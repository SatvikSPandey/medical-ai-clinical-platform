"""ML inference layer.

Loads pretrained models, runs inference, generates explainability artifacts.
The model_registry module records the exact model + weights hash used for
each prediction — the foundation of the audit trail in Phase 5.
"""

from ml.gradcam import GradCAMError, GradCAMResult, generate_gradcam
from ml.inference import (
    InferenceError,
    PathologyScore,
    Prediction,
    load_model,
    predict,
)
from ml.model_registry import (
    ModelRecord,
    ModelRegistryError,
    compute_weights_sha256,
    register_model,
    verify_weights_unchanged,
)

__all__ = [
    "GradCAMError",
    "GradCAMResult",
    "InferenceError",
    "ModelRecord",
    "ModelRegistryError",
    "PathologyScore",
    "Prediction",
    "compute_weights_sha256",
    "generate_gradcam",
    "load_model",
    "predict",
    "register_model",
    "verify_weights_unchanged",
]
