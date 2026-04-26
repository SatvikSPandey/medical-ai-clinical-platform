"""DICOM handling layer.

Wraps pydicom for parsing and normalization.
This is the ONLY place in the codebase that imports pydicom directly.
"""

from dicom_handler.normalizer import (
    NormalizationParams,
    NormalizedDicom,
    normalize,
)
from dicom_handler.parser import (
    DicomMetadata,
    DicomParseError,
    ParsedDicom,
    parse_dicom,
)

__all__ = [
    "DicomMetadata",
    "DicomParseError",
    "NormalizationParams",
    "NormalizedDicom",
    "ParsedDicom",
    "normalize",
    "parse_dicom",
]
