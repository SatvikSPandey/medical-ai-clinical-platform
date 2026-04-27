"""Integration tests for fhir_client module.

These tests hit the live HAPI FHIR public test server at:
  http://hapi.fhir.org/baseR4

They are marked @pytest.mark.integration and are NOT run in the
default fast test suite. Run them explicitly with:
  pytest tests/test_fhir.py -m integration -v

HAPI public server behaviour:
  - Regularly purged and reloaded with synthetic test data.
  - Patient IDs change after purges. Tests that read a specific patient
    use search (by family name) rather than hardcoded IDs to be
    resilient to purges.
  - No auth required. Do NOT send real PHI.
"""

from __future__ import annotations

import asyncio
import warnings
from typing import Any

import pytest

warnings.filterwarnings("ignore")

HAPI_BASE = "http://hapi.fhir.org/baseR4"


# ============================================================
# Helpers
# ============================================================

def run(coro: Any) -> Any:
    """Run an async coroutine synchronously in tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ============================================================
# FHIRClient — connectivity and basic HTTP
# ============================================================

@pytest.mark.integration
class TestFHIRClientConnectivity:
    def test_metadata_returns_r4_capability_statement(self) -> None:
        from fhir_client import FHIRClient

        async def _test() -> None:
            async with FHIRClient(HAPI_BASE) as client:
                cap = await client.get_metadata()
                assert cap["resourceType"] == "CapabilityStatement"
                assert cap["fhirVersion"] == "4.0.1"
                assert cap["status"] == "active"

        run(_test())

    def test_client_requires_context_manager(self) -> None:
        from fhir_client import FHIRClient, FHIRClientError

        async def _test() -> None:
            client = FHIRClient(HAPI_BASE)
            with pytest.raises(FHIRClientError, match="context manager"):
                await client.get_metadata()

        run(_test())

    def test_get_nonexistent_resource_raises_fhir_client_error(self) -> None:
        from fhir_client import FHIRClient, FHIRClientError

        async def _test() -> None:
            async with FHIRClient(HAPI_BASE) as client:
                with pytest.raises(FHIRClientError) as exc_info:
                    await client.get("Patient", "this-id-does-not-exist-999999999")
                assert exc_info.value.status_code == 404

        run(_test())


# ============================================================
# Reader — Patient search and read
# ============================================================

@pytest.mark.integration
class TestPatientReader:
    def test_search_patients_returns_list(self) -> None:
        from fhir_client import FHIRClient, search_patients

        async def _test() -> None:
            async with FHIRClient(HAPI_BASE) as client:
                results = await search_patients(client, count=5)
                assert isinstance(results, list)
                assert len(results) <= 5

        run(_test())

    def test_search_result_is_patient_record(self) -> None:
        from fhir_client import FHIRClient, PatientRecord, search_patients

        async def _test() -> None:
            async with FHIRClient(HAPI_BASE) as client:
                results = await search_patients(client, count=1)
                if results:
                    p = results[0]
                    assert isinstance(p, PatientRecord)
                    assert p.fhir_id != ""
                    assert p.fhir_reference == f"Patient/{p.fhir_id}"

        run(_test())

    def test_patient_record_is_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        from fhir_client import FHIRClient, search_patients

        async def _test() -> None:
            async with FHIRClient(HAPI_BASE) as client:
                results = await search_patients(client, count=1)
                if results:
                    with pytest.raises(FrozenInstanceError):
                        results[0].fhir_id = "tampered"  # type: ignore[misc]

        run(_test())

    def test_get_patient_by_id_succeeds(self) -> None:
        from fhir_client import FHIRClient, PatientRecord, get_patient, search_patients

        async def _test() -> None:
            async with FHIRClient(HAPI_BASE) as client:
                # Search first to get a live ID (resilient to HAPI purges)
                results = await search_patients(client, count=1)
                if not results:
                    pytest.skip("HAPI returned no patients — server may be purged")
                patient_id = results[0].fhir_id

                patient = await get_patient(client, patient_id)
                assert isinstance(patient, PatientRecord)
                assert patient.fhir_id == patient_id

        run(_test())


# ============================================================
# Writer — DiagnosticReport creation
# ============================================================

@pytest.mark.integration
class TestDiagnosticReportWriter:
    def test_write_report_returns_created_resource(self) -> None:
        from fhir_client import FHIRClient, search_patients, write_diagnostic_report

        async def _test() -> None:
            async with FHIRClient(HAPI_BASE) as client:
                patients = await search_patients(client, count=1)
                if not patients:
                    pytest.skip("No patients available on HAPI")

                patient = patients[0]
                scores = {
                    "Pneumothorax": 0.53,
                    "Atelectasis": 0.45,
                    "Pneumonia": 0.38,
                    "Cardiomegaly": 0.22,
                }

                created = await write_diagnostic_report(
                    client=client,
                    patient_reference=patient.fhir_reference,
                    study_instance_uid="1.2.3.4.5.6.7.8.9.test",
                    model_id="test-model-1.0.0",
                    model_version="1.0.0",
                    weights_sha256="abc123def456" * 5,
                    pathology_scores=scores,
                    source_dicom_sha256="deadbeef" * 8,
                )

                assert created["resourceType"] == "DiagnosticReport"
                assert created["status"] == "preliminary"
                assert created.get("id") is not None

        run(_test())

    def test_written_report_is_readable(self) -> None:
        from fhir_client import FHIRClient, search_patients, write_diagnostic_report

        async def _test() -> None:
            async with FHIRClient(HAPI_BASE) as client:
                patients = await search_patients(client, count=1)
                if not patients:
                    pytest.skip("No patients available on HAPI")

                scores = {"Pneumothorax": 0.53, "Pneumonia": 0.40}

                created = await write_diagnostic_report(
                    client=client,
                    patient_reference=patients[0].fhir_reference,
                    study_instance_uid="1.2.3.test",
                    model_id="test-model",
                    model_version="1.0.0",
                    weights_sha256="abc123" * 10,
                    pathology_scores=scores,
                    source_dicom_sha256="deadbeef" * 8,
                )
                report_id = created["id"]

                retrieved = await client.get("DiagnosticReport", report_id)
                assert retrieved["id"] == report_id
                assert retrieved["status"] == "preliminary"
                assert "Patient/" in retrieved["subject"]["reference"]
                assert "AI-ASSISTED ANALYSIS" in retrieved["conclusion"]

        run(_test())

    def test_report_subject_matches_patient(self) -> None:
        from fhir_client import FHIRClient, search_patients, write_diagnostic_report

        async def _test() -> None:
            async with FHIRClient(HAPI_BASE) as client:
                patients = await search_patients(client, count=1)
                if not patients:
                    pytest.skip("No patients available on HAPI")

                patient = patients[0]
                scores = {"Atelectasis": 0.45}

                created = await write_diagnostic_report(
                    client=client,
                    patient_reference=patient.fhir_reference,
                    study_instance_uid="1.2.3.test2",
                    model_id="test-model",
                    model_version="1.0.0",
                    weights_sha256="abc123" * 10,
                    pathology_scores=scores,
                    source_dicom_sha256="deadbeef" * 8,
                )
                report_id = created["id"]

                retrieved = await client.get("DiagnosticReport", report_id)
                subject_ref = retrieved["subject"]["reference"]
                assert subject_ref == patient.fhir_reference

        run(_test())
