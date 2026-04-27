"""FHIR router - GET /fhir/patient/{id}."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from api.core.config import Settings, get_settings
from api.core.security import get_current_user
from fhir_client import FHIRClient, FHIRClientError, get_patient

router = APIRouter(prefix="/fhir", tags=["fhir"])


class PatientResponse(BaseModel):
    fhir_id: str
    display_name: str
    gender: str
    birth_date: str
    fhir_reference: str


@router.get("/patient/{patient_id}", response_model=PatientResponse)
async def read_patient(
    patient_id: str,
    settings: Settings = Depends(get_settings),
    _user: str = Depends(get_current_user),
) -> PatientResponse:
    """Fetch a Patient resource from the FHIR server by ID."""
    try:
        async with FHIRClient(settings.fhir_base_url, timeout=settings.fhir_timeout_seconds) as client:
            patient = await get_patient(client, patient_id)
    except FHIRClientError as e:
        status_code = e.status_code or status.HTTP_502_BAD_GATEWAY
        raise HTTPException(status_code=status_code, detail=str(e)) from e
    return PatientResponse(
        fhir_id=patient.fhir_id,
        display_name=patient.display_name,
        gender=patient.gender,
        birth_date=patient.birth_date,
        fhir_reference=patient.fhir_reference,
    )
