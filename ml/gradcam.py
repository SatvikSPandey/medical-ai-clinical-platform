"""Grad-CAM explainability for chest pathology predictions.

Generates a heatmap showing which regions of the input image most
influenced the model prediction for a given pathology. This is the
visual explainability artifact that FDA AI/ML guidance and the EU AI Act
(Annex IV) require for AI-assisted clinical decision support tools.

What Grad-CAM does (plain English):
  Gradient-weighted Class Activation Mapping looks at how much each
  spatial region of the last convolutional feature map contributes
  to the model confidence in a specific pathology. Regions with high
  gradient magnitude -> high CAM activation -> bright on the heatmap.
  This lets a radiologist see "the model was looking at the right lung"
  rather than trusting a black-box number.

Implementation uses the grad-cam library (pytorch-grad-cam) which
provides a clean API over PyTorch hooks. We use GradCAMPlusPlus
which is more robust than vanilla GradCAM for multi-class problems.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
# pytorch_grad_cam imported lazily inside generate_gradcam()
# ClassifierOutputTarget imported lazily inside generate_gradcam()

from ml.model_registry import ModelRecord


@dataclass(frozen=True)
class GradCAMResult:
    """Grad-CAM heatmap for a single pathology.

    The heatmap is a 2D float32 array in [0, 1] at the same spatial
    resolution as the model input (224x224). It is returned raw so
    the UI layer can overlay it on the original image at any resolution.
    """

    pathology: str
    heatmap: np.ndarray  # shape (224, 224), dtype float32, values in [0, 1]
    model_record: ModelRecord
    source_sha256: str  # ties back to the input DICOM file


class GradCAMError(Exception):
    """Raised when Grad-CAM generation fails."""


def generate_gradcam(
    normalized_pixels: np.ndarray,
    pathology_name: str,
    model_record: ModelRecord,
    source_sha256: str,
) -> GradCAMResult:
    """Generate a Grad-CAM++ heatmap for a specific pathology.

    Imports ml.inference lazily inside this function to avoid the
    circular-import / module-singleton timing issue that arises when
    gradcam.py captures _model=None at module-load time.

    Args:
        normalized_pixels: float32 array in [0, 1], shape (H, W).
        pathology_name: Name of the pathology to explain.
        model_record: The registered model record (for provenance).
        source_sha256: SHA-256 of the source DICOM file (for audit).

    Returns:
        GradCAMResult with heatmap at 224x224.

    Raises:
        GradCAMError: If the model is not loaded or pathology is invalid.
    """
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    # Lazy import — ensures we read the live singleton, not the None
    # that was present when this module was first imported.
    import ml.inference as _inf

    if _inf._model is None:
        raise GradCAMError(
            "Model not loaded. Call ml.inference.load_model() first."
        )

    if pathology_name not in model_record.output_classes:
        raise GradCAMError(
            f"Unknown pathology: '{pathology_name}'. "
            f"Valid: {model_record.output_classes}"
        )

    pathology_idx = model_record.output_classes.index(pathology_name)

    # Bridge [0,1] -> [-1024, 1024] and resize to 224x224
    tensor = _inf._normalize_to_xrv(normalized_pixels)

    # DenseNet-121 last convolutional layer — standard GradCAM target
    target_layers = [_inf._model.features.denseblock4.denselayer16.conv2]

    try:
        with GradCAMPlusPlus(
            model=_inf._model,
            target_layers=target_layers,
        ) as cam:
            targets = [ClassifierOutputTarget(pathology_idx)]
            heatmap = cam(input_tensor=tensor, targets=targets)
    except Exception as e:
        raise GradCAMError(f"Grad-CAM computation failed: {e}") from e

    # heatmap shape from library: (batch, H, W) = (1, 224, 224)
    heatmap_2d = heatmap[0].astype(np.float32)

    return GradCAMResult(
        pathology=pathology_name,
        heatmap=heatmap_2d,
        model_record=model_record,
        source_sha256=source_sha256,
    )
