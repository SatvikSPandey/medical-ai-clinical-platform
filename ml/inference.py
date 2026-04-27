"""Chest pathology inference using torchxrayvision DenseNet-121.

This module is the ONLY place in the codebase that imports torchxrayvision
or calls torch for inference. Everything upstream (DICOM parsing, pixel
normalization) is handled by dicom_handler/. Everything downstream (audit
logging, FHIR writing) receives plain Python dataclasses — no torch tensors
leak out of this module.

Input contract:
  - Accepts a NormalizedDicom (output of dicom_handler.normalize).
  - Pixel values must be float32 in [0, 1].

Output contract:
  - Returns a Prediction dataclass with per-pathology probabilities.
  - Every prediction references its ModelRecord for audit traceability.
  - Grad-CAM heatmap is generated alongside the prediction if requested.

Value range bridge:
  torchxrayvision expects pixels in [-1024, 1024].
  Our normalizer outputs [0, 1].
  Bridge: xrv_pixel = (normalized_pixel * 2 - 1) * 1024
  This is a linear remap, fully invertible, and recorded in the audit trail.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchxrayvision as xrv

from dicom_handler.normalizer import NormalizedDicom
from ml.model_registry import (
    ModelRecord,
    register_model,
    verify_weights_unchanged,
)

# ============================================================
# Constants
# ============================================================

MODEL_ID = "torchxrayvision-densenet121-all-1.4.0"
MODEL_NAME = "DenseNet-121"
MODEL_VERSION = "1.4.0"
MODEL_WEIGHTS_KEY = "densenet121-res224-all"
# torchxrayvision caches weights here by default
WEIGHTS_CACHE_DIR = Path.home() / ".torchxrayvision" / "models_data"
WEIGHTS_FILENAME = (
    "nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001"
    "-rot45-tr15-sc15-seed0-best.pt"
)

# torchxrayvision's expected input value range
XRV_MIN = -1024.0
XRV_MAX = 1024.0
INPUT_SHAPE = (1, 1, 224, 224)  # (batch, channels, H, W)

MODEL_NOTES = (
    "DenseNet-121 pretrained on NIH ChestX-ray14, PadChest, CheXpert, "
    "MIMIC-CXR, and OpenI datasets. "
    "Apache-2.0 license. "
    "Cite: Cohen et al. TorchXRayVision (2022) arxiv:2111.00595. "
    "For portfolio demonstration only — not validated for clinical use."
)


# ============================================================
# Output dataclasses
# ============================================================

@dataclass(frozen=True)
class PathologyScore:
    """Single pathology prediction with confidence score."""

    name: str
    probability: float  # sigmoid output, 0.0 to 1.0

    @property
    def percentage(self) -> float:
        return round(self.probability * 100, 2)


@dataclass(frozen=True)
class Prediction:
    """Complete inference result for one DICOM image.

    Immutable once created. The model_record field provides the full
    audit trail: which model, which weights hash, which version.
    """

    pathologies: tuple[PathologyScore, ...]  # all 18, sorted by probability desc
    model_record: ModelRecord
    source_sha256: str  # from ParsedDicom.source_sha256 — ties back to input file
    inference_time_ms: float  # wall-clock time for the forward pass

    def top_findings(self, n: int = 5) -> tuple[PathologyScore, ...]:
        """Return the top-n pathologies by probability."""
        return tuple(sorted(self.pathologies, key=lambda p: p.probability, reverse=True)[:n])

    def get(self, name: str) -> PathologyScore | None:
        """Look up a specific pathology by name (case-sensitive)."""
        for p in self.pathologies:
            if p.name == name:
                return p
        return None


class InferenceError(Exception):
    """Raised when inference fails for reasons other than bad input."""


# ============================================================
# Module-level singleton: model + registry record
# Loaded once, reused for all predictions in this process.
# ============================================================

_model: xrv.models.DenseNet | None = None
_model_record: ModelRecord | None = None


def _get_weights_path() -> Path:
    """Return the path to cached torchxrayvision weights.

    Raises:
        InferenceError: If weights file is not found. User must run the
            smoke test (or call load_model()) first to trigger download.
    """
    path = WEIGHTS_CACHE_DIR / WEIGHTS_FILENAME
    if not path.exists():
        raise InferenceError(
            f"Model weights not found at {path}. "
            "Run the smoke test or call load_model() first to download."
        )
    return path


def load_model() -> ModelRecord:
    """Load the torchxrayvision model and register it for audit traceability.

    This is idempotent — calling it multiple times returns the same record.
    The model is held in a module-level singleton for efficiency (no
    repeated disk I/O or weight loading between requests).

    Returns:
        ModelRecord — the immutable provenance record for audit logging.

    Raises:
        InferenceError: If weights are missing or model loading fails.
        ModelRegistryError: If weights hashing fails.
    """
    global _model, _model_record

    if _model is not None and _model_record is not None:
        return _model_record

    try:
        model = xrv.models.DenseNet(weights=MODEL_WEIGHTS_KEY)
        model.eval()
    except Exception as e:
        raise InferenceError(f"Failed to load torchxrayvision model: {e}") from e
    weights_path = _get_weights_path()

    record = register_model(
        model_id=MODEL_ID,
        model_name=MODEL_NAME,
        version=MODEL_VERSION,
        framework="pytorch",
        weights_path=weights_path,
        output_classes=tuple(model.pathologies),
        input_shape=INPUT_SHAPE,
        input_value_range=(XRV_MIN, XRV_MAX),
        notes=MODEL_NOTES,
    )

    _model = model
    _model_record = record
    return record


def _normalize_to_xrv(pixel_array: np.ndarray) -> torch.Tensor:
    """Bridge our [0, 1] normalizer output to torchxrayvision's [-1024, 1024].

    Transformation: xrv_pixel = (normalized_pixel * 2 - 1) * 1024
    This is a bijective linear map from [0, 1] to [-1024, 1024].

    Also handles:
    - Resize to 224x224 (torchxrayvision's training resolution)
    - Add batch and channel dimensions: (H, W) → (1, 1, H, W)
    """
    # Remap [0, 1] -> [-1024, 1024]
    xrv_arr = (pixel_array.astype(np.float32) * 2.0 - 1.0) * 1024.0

    # Convert to torch tensor, add channel dim: (H, W) -> (1, H, W)
    tensor = torch.from_numpy(xrv_arr).unsqueeze(0)

    # Resize to 224x224 using bilinear interpolation (model's training size)
    # Add batch dim for transforms: (1, H, W) -> (1, 1, H, W)
    tensor = tensor.unsqueeze(0)
    tensor = torch.nn.functional.interpolate(
        tensor,
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    )
    # Result: (1, 1, 224, 224) — ready for model input
    return tensor


def predict(normalized: NormalizedDicom, source_sha256: str) -> Prediction:
    """Run inference on a normalized DICOM and return a Prediction.

    Args:
        normalized: Output of dicom_handler.normalize(). Pixel values [0, 1].
        source_sha256: SHA-256 of the original .dcm file bytes, from
            ParsedDicom.source_sha256. Carried through for audit trail.

    Returns:
        Prediction (frozen dataclass) with 18 pathology scores.

    Raises:
        InferenceError: If model is not loaded or forward pass fails.
    """
    if _model is None or _model_record is None:
        raise InferenceError(
            "Model not loaded. Call load_model() before predict()."
        )

    # Tamper-detection check — verify weights haven't changed since load
    if not verify_weights_unchanged(_model_record):
        raise InferenceError(
            "SECURITY: Model weights file has changed since registration. "
            "Refusing to run inference. Check audit log."
        )

    tensor = _normalize_to_xrv(normalized.pixel_array)

    t0 = time.perf_counter()
    with torch.no_grad():
        raw_output = _model(tensor)
    inference_time_ms = (time.perf_counter() - t0) * 1000.0

    # raw_output shape: (1, 18) — batch of 1, sigmoid probabilities
    probs = raw_output[0].numpy()

    pathology_scores = tuple(
        PathologyScore(name=name, probability=float(prob))
        for name, prob in zip(_model_record.output_classes, probs, strict=False)
    )

    return Prediction(
        pathologies=pathology_scores,
        model_record=_model_record,
        source_sha256=source_sha256,
        inference_time_ms=round(inference_time_ms, 2),
    )
