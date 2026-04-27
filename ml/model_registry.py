"""Model registry — version + provenance tracking for ML models.

In regulated medical AI, every prediction must be traceable to a specific
model version. "What model produced this output, when, and from what
weights?" must have a single, exact answer. This module provides the
contract.

Design principles:
  - Models are registered with a semantic-version-like identifier
    (e.g., "torchxrayvision-densenet121-all-1.4.0").
  - The weights file's SHA-256 hash is computed and recorded — if anyone
    swaps the weights file, the hash changes, and the audit log catches it.
  - The registry is read-only after creation (frozen dataclass) — once a
    model is registered for a session, it can't be silently mutated.
  - This module knows nothing about torch, torchxrayvision, or any specific
    model. It is a generic registry. The ML/inference module wires the
    specific model into it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class ModelRecord:
    """Immutable record describing a registered model.

    Every prediction the platform produces will reference one of these.
    The fields are chosen to answer auditor questions:
      - "What model?"           -> model_id, model_name
      - "Which version?"        -> version
      - "From what weights?"    -> weights_sha256
      - "When was it loaded?"   -> registered_at_utc
      - "What does it predict?" -> output_classes
      - "What input shape?"     -> input_shape
    """

    model_id: str  # Unique identifier within this platform
    model_name: str  # Human-readable name (e.g., "DenseNet-121")
    version: str  # Semantic version of the source library (e.g., "1.4.0")
    framework: str  # "pytorch", "onnx", "tensorflow", etc.
    weights_path: str  # Absolute path on disk
    weights_sha256: str  # SHA-256 of the weights file bytes
    weights_size_bytes: int  # Size of the weights file
    output_classes: tuple[str, ...]  # Tuple = immutable; lists are mutable
    input_shape: tuple[int, ...]  # e.g., (1, 1, 224, 224) for batch x C x H x W
    input_value_range: tuple[float, float]  # e.g., (-1024.0, 1024.0)
    registered_at_utc: str  # ISO 8601 timestamp in UTC
    notes: str = ""  # Optional context: training data, license, citations


class ModelRegistryError(Exception):
    """Raised when registry operations fail (missing weights file, etc.)."""


def compute_weights_sha256(path: str | Path) -> str:
    """Compute SHA-256 of a weights file (chunked for memory efficiency).

    Returns:
        64-character lowercase hex string.

    Raises:
        ModelRegistryError: If the file is missing or unreadable.
    """
    path = Path(path)
    if not path.exists():
        raise ModelRegistryError(f"Weights file not found: {path}")
    if not path.is_file():
        raise ModelRegistryError(f"Weights path is not a file: {path}")

    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
    except OSError as e:
        raise ModelRegistryError(f"Failed to read weights {path}: {e}") from e

    return h.hexdigest()


def register_model(
    model_id: str,
    model_name: str,
    version: str,
    framework: str,
    weights_path: str | Path,
    output_classes: tuple[str, ...],
    input_shape: tuple[int, ...],
    input_value_range: tuple[float, float],
    notes: str = "",
) -> ModelRecord:
    """Register a model and produce an immutable ModelRecord.

    The weights file at `weights_path` is hashed once at registration time.
    The returned record is frozen — any modification creates a new record
    (which would be a different model from the audit trail's perspective).

    Args:
        model_id: Unique identifier within this platform.
        model_name: Human-readable name.
        version: Source library / model version (semver-like).
        framework: ML framework name.
        weights_path: Path to the on-disk weights file.
        output_classes: Tuple of class labels the model predicts.
        input_shape: Expected input tensor shape (batch, channels, H, W).
        input_value_range: (min, max) of expected input pixel values.
        notes: Optional human-readable context.

    Returns:
        ModelRecord (frozen).

    Raises:
        ModelRegistryError: If weights file is missing or unreadable.
    """
    weights_path = Path(weights_path).resolve()
    weights_sha256 = compute_weights_sha256(weights_path)
    weights_size_bytes = weights_path.stat().st_size

    return ModelRecord(
        model_id=model_id,
        model_name=model_name,
        version=version,
        framework=framework,
        weights_path=str(weights_path),
        weights_sha256=weights_sha256,
        weights_size_bytes=weights_size_bytes,
        output_classes=output_classes,
        input_shape=input_shape,
        input_value_range=input_value_range,
        registered_at_utc=datetime.now(UTC).isoformat(),
        notes=notes,
    )


def verify_weights_unchanged(record: ModelRecord) -> bool:
    """Re-hash the weights file and compare against the recorded hash.

    Used at inference time to detect tampering or accidental file replacement.
    A False return is a SECURITY-RELEVANT EVENT — the audit log should record
    it as a critical entry.

    Returns:
        True if the file's current SHA-256 matches the recorded one.
        False if the file is missing, unreadable, or has changed.
    """
    try:
        current_hash = compute_weights_sha256(record.weights_path)
    except ModelRegistryError:
        return False
    return current_hash == record.weights_sha256
