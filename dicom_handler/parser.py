"""
DICOM parser module.

Reads .dcm files via pydicom and extracts a curated set of metadata fields
plus the pixel array. The output is a pure-Python dataclass that downstream
modules (ML inference, FHIR writer, audit logger) can consume without
needing to know anything about pydicom's API.

Design principle: this module is the ONLY place in the codebase that
imports pydicom. If we ever swap pydicom for another DICOM library
(e.g., dcmtk via Python bindings, or a cloud DICOM service), only this
file changes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
from pydicom.dataset import FileDataset
from pydicom.errors import InvalidDicomError


@dataclass(frozen=True)
class DicomMetadata:
    """Curated DICOM metadata fields relevant to clinical AI workflows.

    Frozen (immutable) because once parsed, metadata represents a snapshot
    of the source file at ingestion time. Any changes would invalidate the
    audit chain that references this snapshot's hash.
    """

    # Identifiers (PHI-adjacent — handled with care downstream)
    patient_id: str
    patient_name: str
    study_instance_uid: str
    series_instance_uid: str
    sop_instance_uid: str

    # Clinical context
    modality: str
    sop_class_name: str
    manufacturer: str

    # Pixel format descriptors (needed to interpret pixel_array correctly)
    rows: int
    columns: int
    bits_allocated: int
    bits_stored: int
    pixel_representation: int  # 0 = unsigned, 1 = signed
    photometric_interpretation: str
    transfer_syntax_name: str

    # Optional rescale parameters (CT/PT use these for HU/SUV)
    rescale_intercept: float | None
    rescale_slope: float | None


@dataclass(frozen=True)
class ParsedDicom:
    """Complete parsed DICOM: metadata + pixel array + provenance hash."""

    metadata: DicomMetadata
    pixel_array: np.ndarray = field(repr=False)  # don't dump array in repr()
    source_sha256: str  # SHA-256 of the original .dcm file bytes
    source_path: str  # absolute path of the source file at parse time


class DicomParseError(Exception):
    """Raised when a DICOM file cannot be parsed or is missing required fields."""


def _safe_get(ds: FileDataset, attr: str, default: Any = None) -> Any:
    """Defensive attribute access — DICOM files in the wild are inconsistent."""
    return getattr(ds, attr, default)


def _compute_file_hash(path: Path) -> str:
    """SHA-256 hash of the raw file bytes — used as input provenance for the audit log."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_dicom(path: str | Path) -> ParsedDicom:
    """Parse a DICOM file into a ParsedDicom object.

    Args:
        path: Path to a .dcm file.

    Returns:
        ParsedDicom containing metadata, pixel array, and source hash.

    Raises:
        DicomParseError: If the file is missing, not a valid DICOM,
            or lacks pixel data.
    """
    path = Path(path).resolve()

    if not path.exists():
        raise DicomParseError(f"DICOM file not found: {path}")

    if not path.is_file():
        raise DicomParseError(f"Path is not a file: {path}")

    try:
        ds: FileDataset = pydicom.dcmread(str(path))
    except InvalidDicomError as e:
        raise DicomParseError(f"Not a valid DICOM file: {path}") from e
    except Exception as e:
        raise DicomParseError(f"Failed to read DICOM {path}: {e}") from e

    # Pixel data is required — we are an imaging platform
    try:
        pixel_array = ds.pixel_array
    except AttributeError as e:
        raise DicomParseError(f"DICOM has no pixel data: {path}") from e
    except Exception as e:
        raise DicomParseError(f"Failed to decode pixel data in {path}: {e}") from e

    # Build metadata — every field defaulted defensively
    metadata = DicomMetadata(
        patient_id=str(_safe_get(ds, "PatientID", "")),
        patient_name=str(_safe_get(ds, "PatientName", "")),
        study_instance_uid=str(_safe_get(ds, "StudyInstanceUID", "")),
        series_instance_uid=str(_safe_get(ds, "SeriesInstanceUID", "")),
        sop_instance_uid=str(_safe_get(ds, "SOPInstanceUID", "")),
        modality=str(_safe_get(ds, "Modality", "UNKNOWN")),
        sop_class_name=(
            ds.SOPClassUID.name if hasattr(ds, "SOPClassUID") else "UNKNOWN"
        ),
        manufacturer=str(_safe_get(ds, "Manufacturer", "")),
        rows=int(_safe_get(ds, "Rows", 0)),
        columns=int(_safe_get(ds, "Columns", 0)),
        bits_allocated=int(_safe_get(ds, "BitsAllocated", 0)),
        bits_stored=int(_safe_get(ds, "BitsStored", 0)),
        pixel_representation=int(_safe_get(ds, "PixelRepresentation", 0)),
        photometric_interpretation=str(
            _safe_get(ds, "PhotometricInterpretation", "")
        ),
        transfer_syntax_name=(
            ds.file_meta.TransferSyntaxUID.name
            if hasattr(ds, "file_meta") and hasattr(ds.file_meta, "TransferSyntaxUID")
            else "UNKNOWN"
        ),
        rescale_intercept=(
            float(ds.RescaleIntercept) if hasattr(ds, "RescaleIntercept") else None
        ),
        rescale_slope=(
            float(ds.RescaleSlope) if hasattr(ds, "RescaleSlope") else None
        ),
    )

    return ParsedDicom(
        metadata=metadata,
        pixel_array=pixel_array,
        source_sha256=_compute_file_hash(path),
        source_path=str(path),
    )
