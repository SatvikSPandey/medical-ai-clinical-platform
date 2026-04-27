"""Pydantic request/response schemas for the API layer.

These are the HTTP contract — what the API accepts and returns.
Kept separate from internal dataclasses (ParsedDicom, Prediction, etc.)
so the HTTP layer can evolve independently of the domain layer.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    """Body for POST /infer - which patient to link the report to."""
    patient_fhir_id: str = Field(..., description="FHIR Patient resource ID on the configured server")


class PathologyResult(BaseModel):
    """Single pathology score in an inference response."""
    name: str
    probability: float
    percentage: float


class InferenceResponse(BaseModel):
    """Full inference result returned by POST /infer."""
    report_fhir_id: str
    patient_fhir_id: str
    source_dicom_sha256: str
    model_id: str
    model_version: str
    weights_sha256: str
    inference_time_ms: float
    top_findings: list[PathologyResult]
    all_findings: list[PathologyResult]
    audit_sequence: int


class AuditEntryResponse(BaseModel):
    """Single audit log entry returned by GET /audit."""
    sequence: int
    event_type: str
    actor: str
    source_sha256: str
    model_id: str
    timestamp_utc: str
    entry_hash: str


class ChainStatusResponse(BaseModel):
    """Result of GET /audit/verify - hash chain integrity check."""
    is_valid: bool
    total_entries: int
    first_broken_sequence: int | None
    broken_reason: str | None


class TokenResponse(BaseModel):
    """OAuth2 token response from POST /auth/token."""
    access_token: str
    token_type: str
