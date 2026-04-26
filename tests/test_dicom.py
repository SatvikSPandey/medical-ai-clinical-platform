"""Tests for dicom_handler module.

Test naming follows the pattern: test_<unit>_<scenario>_<expected>.
We test against pydicom's bundled BSD-licensed sample DICOMs which
live in data/sample_dicoms/ (whitelisted in .gitignore as public_*.dcm).
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from dicom_handler import (
    DicomMetadata,
    DicomParseError,
    NormalizationParams,
    NormalizedDicom,
    ParsedDicom,
    normalize,
    parse_dicom,
)

SAMPLE_DIR = Path(__file__).parent.parent / "data" / "sample_dicoms"
CT_SAMPLE = SAMPLE_DIR / "public_CT_small.dcm"
JP2K_SAMPLE = SAMPLE_DIR / "public_examples_jpeg2k.dcm"
BAD_VR_SAMPLE = SAMPLE_DIR / "public_badVR.dcm"


# ============================================================
# parse_dicom — happy-path tests
# ============================================================

@pytest.mark.unit
class TestParseDicom:
    def test_returns_parseddicom_instance(self) -> None:
        result = parse_dicom(CT_SAMPLE)
        assert isinstance(result, ParsedDicom)

    def test_metadata_is_dicommetadata_instance(self) -> None:
        result = parse_dicom(CT_SAMPLE)
        assert isinstance(result.metadata, DicomMetadata)

    def test_pixel_array_is_numpy_array(self) -> None:
        result = parse_dicom(CT_SAMPLE)
        assert isinstance(result.pixel_array, np.ndarray)

    def test_ct_sample_has_expected_modality(self) -> None:
        result = parse_dicom(CT_SAMPLE)
        assert result.metadata.modality == "CT"

    def test_ct_sample_has_expected_dimensions(self) -> None:
        result = parse_dicom(CT_SAMPLE)
        assert result.metadata.rows == 128
        assert result.metadata.columns == 128
        assert result.pixel_array.shape == (128, 128)

    def test_ct_sample_has_signed_int16_pixels(self) -> None:
        result = parse_dicom(CT_SAMPLE)
        assert result.pixel_array.dtype == np.int16
        assert result.metadata.pixel_representation == 1

    def test_ct_sample_has_rescale_params(self) -> None:
        """Real CT files always have RescaleSlope and RescaleIntercept."""
        result = parse_dicom(CT_SAMPLE)
        assert result.metadata.rescale_slope == 1.0
        assert result.metadata.rescale_intercept == -1024.0

    def test_source_sha256_is_64_char_hex(self) -> None:
        result = parse_dicom(CT_SAMPLE)
        assert len(result.source_sha256) == 64
        assert all(c in "0123456789abcdef" for c in result.source_sha256)

    def test_source_sha256_is_deterministic(self) -> None:
        """Same file -> same hash. Critical for audit log integrity."""
        result1 = parse_dicom(CT_SAMPLE)
        result2 = parse_dicom(CT_SAMPLE)
        assert result1.source_sha256 == result2.source_sha256

    def test_metadata_is_immutable(self) -> None:
        """Frozen dataclass prevents post-parse tampering with audit data."""
        result = parse_dicom(CT_SAMPLE)
        with pytest.raises(FrozenInstanceError):
            result.metadata.patient_id = "TAMPERED"  # type: ignore[misc]

    def test_accepts_string_path(self) -> None:
        result = parse_dicom(str(CT_SAMPLE))
        assert result.metadata.modality == "CT"

    def test_accepts_pathlib_path(self) -> None:
        result = parse_dicom(CT_SAMPLE)
        assert result.metadata.modality == "CT"


# ============================================================
# parse_dicom — JPEG 2000 compressed file
# ============================================================

@pytest.mark.unit
class TestParseDicomCompressed:
    def test_jpeg2k_compressed_file_decodes(self) -> None:
        """Pixel data should decompress transparently."""
        result = parse_dicom(JP2K_SAMPLE)
        assert isinstance(result.pixel_array, np.ndarray)
        assert result.pixel_array.size > 0


# ============================================================
# parse_dicom — error handling
# ============================================================

@pytest.mark.unit
class TestParseDicomErrors:
    def test_missing_file_raises_dicomparseerror(self) -> None:
        with pytest.raises(DicomParseError, match="not found"):
            parse_dicom("nonexistent_file.dcm")

    def test_directory_path_raises_dicomparseerror(self, tmp_path: Path) -> None:
        with pytest.raises(DicomParseError):
            parse_dicom(tmp_path)

    def test_non_dicom_file_raises_dicomparseerror(self, tmp_path: Path) -> None:
        fake = tmp_path / "not_a_dicom.dcm"
        fake.write_bytes(b"this is not a DICOM file")
        with pytest.raises(DicomParseError):
            parse_dicom(fake)


# ============================================================
# normalize — happy-path tests
# ============================================================

@pytest.mark.unit
class TestNormalize:
    def test_returns_normalizeddicom_instance(self) -> None:
        result = normalize(parse_dicom(CT_SAMPLE))
        assert isinstance(result, NormalizedDicom)

    def test_params_is_normalizationparams_instance(self) -> None:
        result = normalize(parse_dicom(CT_SAMPLE))
        assert isinstance(result.params, NormalizationParams)

    def test_output_dtype_is_float32(self) -> None:
        result = normalize(parse_dicom(CT_SAMPLE))
        assert result.pixel_array.dtype == np.float32

    def test_output_range_is_zero_to_one(self) -> None:
        result = normalize(parse_dicom(CT_SAMPLE))
        assert result.pixel_array.min() >= 0.0
        assert result.pixel_array.max() <= 1.0

    def test_output_shape_matches_input(self) -> None:
        parsed = parse_dicom(CT_SAMPLE)
        result = normalize(parsed)
        assert result.pixel_array.shape == parsed.pixel_array.shape

    def test_modality_lut_applied_correctly(self) -> None:
        """For CT_small: slope=1, intercept=-1024 should be recorded."""
        result = normalize(parse_dicom(CT_SAMPLE))
        assert result.params.rescale_slope_applied == 1.0
        assert result.params.rescale_intercept_applied == -1024.0

    def test_monochrome2_not_inverted(self) -> None:
        """CT sample is MONOCHROME2 — should NOT trigger inversion."""
        result = normalize(parse_dicom(CT_SAMPLE))
        assert result.params.photometric_inverted is False

    def test_default_clip_percentiles_recorded(self) -> None:
        result = normalize(parse_dicom(CT_SAMPLE))
        assert result.params.clip_percentile_low == 1.0
        assert result.params.clip_percentile_high == 99.0

    def test_custom_clip_percentiles_recorded(self) -> None:
        result = normalize(parse_dicom(CT_SAMPLE), clip_low_pct=5.0, clip_high_pct=95.0)
        assert result.params.clip_percentile_low == 5.0
        assert result.params.clip_percentile_high == 95.0

    def test_normalization_is_deterministic(self) -> None:
        """Same input -> same output. Critical for reproducibility evidence."""
        parsed = parse_dicom(CT_SAMPLE)
        result1 = normalize(parsed)
        result2 = normalize(parsed)
        np.testing.assert_array_equal(result1.pixel_array, result2.pixel_array)

    def test_params_is_immutable(self) -> None:
        result = normalize(parse_dicom(CT_SAMPLE))
        with pytest.raises(FrozenInstanceError):
            result.params.rescale_slope_applied = 99.0  # type: ignore[misc]


# ============================================================
# normalize — synthetic edge cases
# ============================================================

@pytest.mark.unit
class TestNormalizeEdgeCases:
    def test_uniform_image_returns_zeros_without_crashing(self) -> None:
        """Degenerate case: image with single pixel value -> divide-by-zero risk."""
        from dicom_handler.normalizer import _percentile_clip_and_scale

        uniform = np.full((10, 10), 500, dtype=np.float32)
        result, _, _ = _percentile_clip_and_scale(uniform)
        assert np.all(result == 0.0)
        assert result.dtype == np.float32

    def test_synthetic_monochrome1_inversion_flag_set(self) -> None:
        """Verify MONOCHROME1 path triggers inversion (CT sample is MONOCHROME2)."""
        from dicom_handler.normalizer import _apply_photometric_correction

        arr = np.array([[0, 100, 200]], dtype=np.float32)
        result, inverted = _apply_photometric_correction(arr, "MONOCHROME1")
        assert inverted is True
        # max was 200, so 0 -> 200, 100 -> 100, 200 -> 0
        np.testing.assert_array_equal(result, np.array([[200, 100, 0]], dtype=np.float32))

    def test_monochrome2_no_inversion(self) -> None:
        from dicom_handler.normalizer import _apply_photometric_correction

        arr = np.array([[0, 100, 200]], dtype=np.float32)
        result, inverted = _apply_photometric_correction(arr, "MONOCHROME2")
        assert inverted is False
        np.testing.assert_array_equal(result, arr)
