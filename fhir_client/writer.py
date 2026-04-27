"""FHIR R4 resource writer.

Constructs and POSTs DiagnosticReport resources that encode the AI
platform output in a standards-compliant clinical format.

What is a DiagnosticReport in clinical context?
  A DiagnosticReport is the FHIR resource type for structured results
  from diagnostic procedures — lab results, imaging reads, pathology
  reports. For AI-assisted imaging, it is the correct resource to carry:
    - Which patient (subject reference)
    - What imaging study was analysed (imagingStudy reference)
    - What the AI found (conclusion + conclusionCode)
    - Which model produced the result (extension carrying model version)
    - When the report was generated (effectiveDateTime)

  This is the resource type Siemens Healthineers AI-Rad Companion uses
  to write results back to the hospital EHR in production deployments.

SNOMED CT codes used:
  We use SNOMED CT codes from the standard clinical terminology for
  the 18 pathologies torchxrayvision detects. These codes are what
  allow a downstream EHR to query "show me all reports with Pneumothorax"
  rather than searching free text.

Framing in the report:
  All conclusions are explicitly framed as AI-generated findings requiring
  radiologist review. This is the honest, regulatory-safe framing for a
  SaMD decision support tool.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fhir_client.client import FHIRClient

# ============================================================
# SNOMED CT codes for torchxrayvision pathologies
# Source: SNOMED CT International Edition
# ============================================================

PATHOLOGY_SNOMED: dict[str, tuple[str, str]] = {
    "Atelectasis":               ("46621007",  "Atelectasis"),
    "Consolidation":             ("9494007",   "Pulmonary consolidation"),
    "Infiltration":              ("47515009",  "Pulmonary infiltrate"),
    "Pneumothorax":              ("36118008",  "Pneumothorax"),
    "Edema":                     ("19242006",  "Pulmonary edema"),
    "Emphysema":                 ("67992007",  "Pulmonary emphysema"),
    "Fibrosis":                  ("51615001",  "Pulmonary fibrosis"),
    "Effusion":                  ("60046008",  "Pleural effusion"),
    "Pneumonia":                 ("233604007", "Pneumonia"),
    "Pleural_Thickening":        ("196121004", "Pleural thickening"),
    "Cardiomegaly":              ("8186001",   "Cardiomegaly"),
    "Nodule":                    ("27550009",  "Pulmonary nodule"),
    "Mass":                      ("309529002", "Pulmonary mass"),
    "Hernia":                    ("73122008",  "Diaphragmatic hernia"),
    "Lung Lesion":               ("300999006", "Lung lesion"),
    "Fracture":                  ("46866001",  "Fracture of rib"),
    "Lung Opacity":              ("249674008", "Lung opacity"),
    "Enlarged Cardiomediastinum":("248515005", "Enlarged mediastinum"),
}

CONFIDENCE_THRESHOLD = 0.30  # Only include findings above this probability

# Platform extension URL for AI model provenance
MODEL_PROVENANCE_EXTENSION_URL = (
    "https://medical-ai-platform.example/fhir/StructureDefinition/ai-model-provenance"
)


def _build_diagnostic_report(
    patient_reference: str,
    study_instance_uid: str,
    model_id: str,
    model_version: str,
    weights_sha256: str,
    pathology_scores: dict[str, float],
    source_dicom_sha256: str,
    effective_datetime: str,
) -> dict[str, Any]:
    """Construct a FHIR DiagnosticReport dict for an AI chest X-ray analysis.

    Args:
        patient_reference: FHIR reference string e.g. "Patient/90270587".
        study_instance_uid: DICOM StudyInstanceUID from the parsed DICOM.
        model_id: Unique model identifier from ModelRecord.
        model_version: Model version string.
        weights_sha256: SHA-256 of the weights file (first 16 chars shown).
        pathology_scores: Dict of {pathology_name: probability} for all 18.
        source_dicom_sha256: SHA-256 of the input .dcm file.
        effective_datetime: ISO 8601 UTC string for when analysis was run.

    Returns:
        FHIR DiagnosticReport resource as a dict, ready to POST.
    """
    # Findings above the confidence threshold
    significant = {
        name: prob
        for name, prob in pathology_scores.items()
        if prob >= CONFIDENCE_THRESHOLD
    }

    # Build conclusion codes (SNOMED CT)
    conclusion_codes = []
    for name, prob in sorted(significant.items(), key=lambda x: x[1], reverse=True):
        snomed_code, snomed_display = PATHOLOGY_SNOMED.get(name, ("", name))
        conclusion_codes.append({
            "coding": [{
                "system": "http://snomed.info/sct",
                "code": snomed_code,
                "display": snomed_display,
            }],
            "text": f"{name} ({prob * 100:.1f}% AI confidence)",
        })

    # Human-readable conclusion
    if significant:
        findings_text = ", ".join(
            f"{n} ({p * 100:.1f}%)"
            for n, p in sorted(significant.items(), key=lambda x: x[1], reverse=True)
        )
        conclusion = (
            f"AI-ASSISTED ANALYSIS (requires radiologist review): "
            f"Findings above {CONFIDENCE_THRESHOLD * 100:.0f}% confidence threshold: "
            f"{findings_text}. "
            f"Model: {model_id} v{model_version}. "
            f"Source DICOM SHA-256: {source_dicom_sha256[:16]}..."
        )
    else:
        conclusion = (
            f"AI-ASSISTED ANALYSIS (requires radiologist review): "
            f"No findings above {CONFIDENCE_THRESHOLD * 100:.0f}% confidence threshold. "
            f"Model: {model_id} v{model_version}."
        )

    return {
        "resourceType": "DiagnosticReport",
        "status": "preliminary",  # NOT "final" — AI output is preliminary pending review
        "category": [{
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                "code": "RAD",
                "display": "Radiology",
            }]
        }],
        "code": {
            "coding": [{
                "system": "http://loinc.org",
                "code": "36643-5",
                "display": "Chest X-ray 2 views",
            }],
            "text": "AI-Assisted Chest Radiograph Analysis",
        },
        "subject": {"reference": patient_reference},
        "effectiveDateTime": effective_datetime,
        "issued": effective_datetime,
        "conclusion": conclusion,
        "conclusionCode": conclusion_codes,
        "extension": [{
            "url": MODEL_PROVENANCE_EXTENSION_URL,
            "extension": [
                {"url": "modelId",      "valueString": model_id},
                {"url": "modelVersion", "valueString": model_version},
                {"url": "weightsSha256","valueString": weights_sha256[:32]},
                {"url": "sourceDicomSha256", "valueString": source_dicom_sha256[:32]},
                {"url": "studyInstanceUid", "valueString": study_instance_uid},
            ],
        }],
    }


async def write_diagnostic_report(
    client: FHIRClient,
    patient_reference: str,
    study_instance_uid: str,
    model_id: str,
    model_version: str,
    weights_sha256: str,
    pathology_scores: dict[str, float],
    source_dicom_sha256: str,
) -> dict[str, Any]:
    """Build and POST a DiagnosticReport to the FHIR server.

    Args:
        client: An open FHIRClient context.
        patient_reference: e.g. "Patient/90270587".
        study_instance_uid: From ParsedDicom.metadata.study_instance_uid.
        model_id: From ModelRecord.model_id.
        model_version: From ModelRecord.version.
        weights_sha256: From ModelRecord.weights_sha256.
        pathology_scores: {name: probability} dict from Prediction.
        source_dicom_sha256: From ParsedDicom.source_sha256.

    Returns:
        The created DiagnosticReport resource dict (with server-assigned ID).

    Raises:
        FHIRClientError: If the POST fails.
    """
    effective_dt = datetime.now(UTC).isoformat()

    report = _build_diagnostic_report(
        patient_reference=patient_reference,
        study_instance_uid=study_instance_uid,
        model_id=model_id,
        model_version=model_version,
        weights_sha256=weights_sha256,
        pathology_scores=pathology_scores,
        source_dicom_sha256=source_dicom_sha256,
        effective_datetime=effective_dt,
    )

    created = await client.post("DiagnosticReport", report)
    return created
