"""
DICOM pixel array normalization.

Converts raw DICOM pixel arrays into normalized float tensors suitable
for PyTorch model input. Handles three concerns that ChestVision did NOT
have to handle (because it consumed JPEGs, not DICOMs):

1. Modality LUT — apply RescaleSlope/RescaleIntercept to convert raw
   stored values to physically meaningful units (Hounsfield Units for CT,
   linear attenuation for X-ray, etc.).

2. Photometric inversion — MONOCHROME1 means "0 = white, max = black"
   (inverted display); MONOCHROME2 is normal. Models trained on
   MONOCHROME2 data must see MONOCHROME2 — we invert MONOCHROME1 on read.

3. Bit-depth normalization — DICOM pixels can be 8/12/16 bit, signed
   or unsigned. PyTorch wants float32 in [0, 1] (or [-1, 1]). We
   percentile-clip then min-max normalize to handle the long tail of
   outlier pixel values that destroy contrast if you naively divide.

Design principle: this module is pure NumPy. No torch import, no pydicom
import. It transforms one numpy array into another. Easy to test,
easy to reason about, framework-independent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dicom_handler.parser import ParsedDicom


@dataclass(frozen=True)
class NormalizationParams:
    """Parameters used during normalization — recorded for audit trail.

    The exact transformations applied to the pixel data must be reproducible.
    Storing these alongside the model output lets us replay the inference
    pipeline if a regulator asks "how did you produce this output?"
    """

    rescale_intercept_applied: float
    rescale_slope_applied: float
    photometric_inverted: bool
    clip_percentile_low: float
    clip_percentile_high: float
    clipped_min: float
    clipped_max: float
    output_shape: tuple[int, ...]
    output_dtype: str


@dataclass(frozen=True)
class NormalizedDicom:
    """Normalized pixel array + the parameters used to produce it."""

    pixel_array: np.ndarray
    params: NormalizationParams


def _apply_modality_lut(
    pixel_array: np.ndarray,
    rescale_slope: float | None,
    rescale_intercept: float | None,
) -> tuple[np.ndarray, float, float]:
    """Apply DICOM Modality LUT: output = pixel * slope + intercept.

    For CT: converts raw stored values to Hounsfield Units.
    For most X-ray: slope=1, intercept=0 (no-op).
    Returns the transformed array plus the actual slope/intercept used.
    """
    slope = float(rescale_slope) if rescale_slope is not None else 1.0
    intercept = float(rescale_intercept) if rescale_intercept is not None else 0.0

    if slope == 1.0 and intercept == 0.0:
        return pixel_array.astype(np.float32), slope, intercept

    return pixel_array.astype(np.float32) * slope + intercept, slope, intercept


def _apply_photometric_correction(
    pixel_array: np.ndarray, photometric: str
) -> tuple[np.ndarray, bool]:
    """Invert pixel values if photometric interpretation is MONOCHROME1.

    MONOCHROME1: 0 = displayed as white. Standard medical image.
    MONOCHROME2: 0 = displayed as black. Standard.
    For ML, we want MONOCHROME2 semantics (high value = bright = bone/dense).
    """
    if photometric == "MONOCHROME1":
        max_val = pixel_array.max()
        return max_val - pixel_array, True
    return pixel_array, False


def _percentile_clip_and_scale(
    pixel_array: np.ndarray,
    low_pct: float = 1.0,
    high_pct: float = 99.0,
) -> tuple[np.ndarray, float, float]:
    """Clip outlier pixels then min-max scale to [0, 1].

    Why percentile clipping instead of raw min/max:
    DICOM pixel data often has extreme outliers (metal implants saturate,
    detector errors produce spike pixels). Naive min-max divides everything
    by the outlier and crushes the actual image into a narrow band.
    Clipping to 1st-99th percentile preserves the diagnostically relevant
    contrast range.
    """
    low = float(np.percentile(pixel_array, low_pct))
    high = float(np.percentile(pixel_array, high_pct))

    if high <= low:
        # Degenerate case — uniform image. Return zeros to avoid divide-by-zero.
        return np.zeros_like(pixel_array, dtype=np.float32), low, high

    clipped = np.clip(pixel_array, low, high)
    scaled = (clipped - low) / (high - low)
    return scaled.astype(np.float32), low, high


def normalize(
    parsed: ParsedDicom,
    clip_low_pct: float = 1.0,
    clip_high_pct: float = 99.0,
) -> NormalizedDicom:
    """Normalize a parsed DICOM into a model-ready float32 array in [0, 1].

    Args:
        parsed: Output of dicom_handler.parser.parse_dicom().
        clip_low_pct: Lower percentile for outlier clipping (default 1.0).
        clip_high_pct: Upper percentile for outlier clipping (default 99.0).

    Returns:
        NormalizedDicom with float32 pixel_array in [0, 1] and full
        provenance metadata (NormalizationParams).
    """
    arr = parsed.pixel_array

    # Step 1 — apply Modality LUT (rescale to physically meaningful units)
    arr, slope, intercept = _apply_modality_lut(
        arr,
        parsed.metadata.rescale_slope,
        parsed.metadata.rescale_intercept,
    )

    # Step 2 — correct photometric interpretation
    arr, inverted = _apply_photometric_correction(
        arr, parsed.metadata.photometric_interpretation
    )

    # Step 3 — clip outliers and scale to [0, 1]
    arr, clipped_min, clipped_max = _percentile_clip_and_scale(
        arr, clip_low_pct, clip_high_pct
    )

    params = NormalizationParams(
        rescale_intercept_applied=intercept,
        rescale_slope_applied=slope,
        photometric_inverted=inverted,
        clip_percentile_low=clip_low_pct,
        clip_percentile_high=clip_high_pct,
        clipped_min=clipped_min,
        clipped_max=clipped_max,
        output_shape=arr.shape,
        output_dtype=str(arr.dtype),
    )

    return NormalizedDicom(pixel_array=arr, params=params)
