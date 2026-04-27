"""FHIR R4 resource reader.

Reads Patient resources from the FHIR server and returns clean Python
dataclasses. The raw FHIR JSON never leaks past this module — callers
work with PatientRecord objects, not raw dicts.

Design principle: this module knows about FHIR resource structure.
The client.py module knows about HTTP. They are separate concerns.
"""

from __future__ import annotations

from dataclasses import dataclass

from fhir_client.client import FHIRClient


@dataclass(frozen=True)
class PatientRecord:
    """Curated patient demographics from a FHIR Patient resource.

    Frozen because patient identity at the time of the AI prediction
    is audit-relevant — it must not change after being recorded.
    """

    fhir_id: str               # Server-assigned resource ID
    family_name: str
    given_names: tuple[str, ...]
    gender: str                # "male" | "female" | "other" | "unknown"
    birth_date: str            # ISO 8601 date string e.g. "1975-04-23"
    active: bool

    @property
    def display_name(self) -> str:
        given = " ".join(self.given_names)
        return f"{given} {self.family_name}".strip()

    @property
    def fhir_reference(self) -> str:
        """FHIR reference string for linking resources e.g. DiagnosticReport.subject."""
        return f"Patient/{self.fhir_id}"


def _parse_patient(raw: dict) -> PatientRecord:
    """Parse a raw FHIR Patient JSON dict into a PatientRecord.

    Defensively handles missing fields — real-world FHIR servers
    frequently omit optional fields.
    """
    resource_id = raw.get("id", "")

    names = raw.get("name", [{}])
    primary_name = names[0] if names else {}
    family = primary_name.get("family", "")
    given = tuple(primary_name.get("given", []))

    return PatientRecord(
        fhir_id=resource_id,
        family_name=family,
        given_names=given,
        gender=raw.get("gender", "unknown"),
        birth_date=raw.get("birthDate", ""),
        active=raw.get("active", True),
    )


async def get_patient(client: FHIRClient, patient_id: str) -> PatientRecord:
    """Fetch a single Patient resource by FHIR ID.

    Args:
        client: An open FHIRClient context.
        patient_id: The server-assigned patient ID (e.g. "90270587").

    Returns:
        PatientRecord (frozen).

    Raises:
        FHIRClientError: If the patient is not found (404) or server error.
    """
    raw = await client.get("Patient", patient_id)
    return _parse_patient(raw)


async def search_patients(
    client: FHIRClient,
    family: str | None = None,
    given: str | None = None,
    count: int = 10,
) -> list[PatientRecord]:
    """Search for patients by name.

    Args:
        client: An open FHIRClient context.
        family: Family name search string.
        given: Given name search string.
        count: Maximum number of results (default 10).

    Returns:
        List of PatientRecord objects (may be empty if no matches).

    Raises:
        FHIRClientError: On server error.
    """
    params: dict[str, str] = {"_count": str(count)}
    if family:
        params["family"] = family
    if given:
        params["given"] = given

    bundle = await client.search("Patient", params)
    entries = bundle.get("entry", [])
    return [_parse_patient(e["resource"]) for e in entries if "resource" in e]
