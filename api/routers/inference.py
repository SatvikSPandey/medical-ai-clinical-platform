"""Inference router - POST /infer."""
from __future__ import annotations

import io

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from api.core.config import Settings, get_settings
from api.core.security import get_current_user
from api.models.schemas import InferenceResponse, PathologyResult
from compliance.audit import AuditEventType, append_event, initialize_db
from dicom_handler import DicomParseError, normalize, parse_dicom
from fhir_client import FHIRClient, FHIRClientError, get_patient, write_diagnostic_report
from ml.inference import InferenceError, load_model, predict

router = APIRouter(prefix="/infer", tags=["inference"])


@router.post("/", response_model=InferenceResponse)
async def run_inference(
    file: UploadFile = File(..., description="DICOM .dcm file"),
    patient_fhir_id: str = Form(..., description="FHIR Patient ID to link the report to"),
    settings: Settings = Depends(get_settings),
    actor: str = Depends(get_current_user),
) -> InferenceResponse:
    """Upload a DICOM file, run AI inference, write a FHIR DiagnosticReport."""
    initialize_db(settings.audit_db_path)

    # 1. Parse DICOM from upload
    contents = await file.read()
    tmp = io.BytesIO(contents)
    tmp.name = file.filename or "upload.dcm"
    try:
        import pathlib
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tf:
            tf.write(contents)
            tmp_path = tf.name
        parsed = parse_dicom(tmp_path)
        pathlib.Path(tmp_path).unlink(missing_ok=True)
    except DicomParseError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)) from e

    normalized = normalize(parsed)

    # 2. Load model and run inference
    try:
        record = load_model()
        prediction = predict(normalized, source_sha256=parsed.source_sha256)
    except InferenceError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e)) from e

    # 3. Write FHIR DiagnosticReport
    scores = {p.name: p.probability for p in prediction.pathologies}
    try:
        async with FHIRClient(settings.fhir_base_url, timeout=settings.fhir_timeout_seconds) as client:
            patient = await get_patient(client, patient_fhir_id)
            created = await write_diagnostic_report(
                client=client,
                patient_reference=patient.fhir_reference,
                study_instance_uid=parsed.metadata.study_instance_uid,
                model_id=record.model_id,
                model_version=record.version,
                weights_sha256=record.weights_sha256,
                pathology_scores=scores,
                source_dicom_sha256=parsed.source_sha256,
            )
    except FHIRClientError as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e)) from e

    report_id = created.get("id", "unknown")

    # 4. Audit log
    audit_entry = append_event(
        AuditEventType.INFERENCE_COMPLETE,
        actor=actor,
        event_data={
            "report_fhir_id": report_id,
            "patient_fhir_id": patient_fhir_id,
            "top_findings": [
                {"name": p.name, "probability": p.probability}
                for p in prediction.top_findings(5)
            ],
            "inference_time_ms": prediction.inference_time_ms,
        },
        source_sha256=parsed.source_sha256,
        model_id=record.model_id,
        db_path=settings.audit_db_path,
    )

    top5 = prediction.top_findings(5)
    all_findings = sorted(prediction.pathologies, key=lambda p: p.probability, reverse=True)

    return InferenceResponse(
        report_fhir_id=report_id,
        patient_fhir_id=patient_fhir_id,
        source_dicom_sha256=parsed.source_sha256,
        model_id=record.model_id,
        model_version=record.version,
        weights_sha256=record.weights_sha256[:16] + "...",
        inference_time_ms=prediction.inference_time_ms,
        top_findings=[PathologyResult(name=p.name, probability=p.probability, percentage=p.percentage) for p in top5],
        all_findings=[PathologyResult(name=p.name, probability=p.probability, percentage=p.percentage) for p in all_findings],
        audit_sequence=audit_entry.sequence,
    )
