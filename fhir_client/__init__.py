"""FHIR R4 client layer.

Reads Patient resources and writes DiagnosticReport resources
to a FHIR R4 server (default: HAPI public test server).
"""

from fhir_client.client import FHIRClient, FHIRClientError
from fhir_client.reader import PatientRecord, get_patient, search_patients
from fhir_client.writer import write_diagnostic_report

__all__ = [
    "FHIRClient",
    "FHIRClientError",
    "PatientRecord",
    "get_patient",
    "search_patients",
    "write_diagnostic_report",
]
