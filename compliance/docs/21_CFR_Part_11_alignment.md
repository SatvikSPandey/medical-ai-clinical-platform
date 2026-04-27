# 21 CFR Part 11 Alignment Document

**Project:** Medical AI Clinical Platform (Project 18)  
**Version:** 0.1.0  
**Date:** 2026-04-27  
**Status:** Draft - For Portfolio Demonstration  

## Disclaimer

This document maps the platform technical controls to FDA 21 CFR Part 11 requirements. This platform is a **portfolio demonstration project** and has not undergone formal FDA submission, 510(k) clearance, or De Novo classification. The controls described are aligned with Part 11 principles and are honestly framed as such.

---

## Regulation Overview

21 CFR Part 11 (Electronic Records; Electronic Signatures) establishes FDA criteria for accepting electronic records and signatures as equivalent to paper records. For AI/ML-based Software as a Medical Device (SaMD), Part 11 compliance is relevant wherever the software creates, modifies, maintains, archives, retrieves, or transmits records required by FDA regulations.

---

## Control Mapping

### SS11.10(a) - Validation

**Requirement:** Validation of systems to ensure accuracy, reliability, consistent intended performance, and the ability to discern invalid or altered records.

**Implementation:**
- `tests/test_audit.py` - 23 unit tests covering audit log correctness, chain integrity, and tamper detection
- `tests/test_dicom.py` - 30 unit tests covering DICOM parsing and normalization correctness
- `tests/test_fhir.py` - 10 integration tests covering FHIR read/write round-trips
- All tests run in CI on every commit (Phase 9)
- Determinism tests (`test_source_sha256_is_deterministic`, `test_normalization_is_deterministic`) verify consistent output for identical inputs

### SS11.10(d) - Access Control

**Requirement:** Limiting system access to authorized individuals.

**Implementation:**
- FastAPI layer (Phase 6) implements JWT authentication; every API request carries a signed token
- The audit log records the `actor` field on every event - unauthorized access attempts are logged
- The `verify_weights_unchanged()` function in `ml/inference.py` detects if the model weights file has been tampered with between authenticated sessions

### SS11.10(e) - Audit Trails

**Requirement:** Use of audit trails - computer-generated, date/time-stamped audit trails to independently record the date and time of operator entries and actions that create, modify, or delete electronic records.

**Implementation (`compliance/audit.py`):**
- Every inference request generates an `INFERENCE_COMPLETE` audit entry with: UTC timestamp, actor identity, source DICOM SHA-256, model ID and version, top pathology scores
- Every FHIR write generates a `FHIR_REPORT_WRITTEN` entry with: report ID, patient reference, timestamp
- Every model load generates a `MODEL_LOADED` entry with: model ID, weights SHA-256, weights size
- Timestamps are ISO 8601 UTC (no timezone ambiguity)
- The `actor` field records who triggered the event

### SS11.10(a) + (e) Combined - Tamper-Evident Records

**Requirement:** Ability to discern invalid or altered records; audit trail completeness.

**Implementation - Hash Chain:**
- Each audit entry computes `entry_hash = SHA-256(all_fields + prev_entry_hash)`
- The first entry uses a known genesis hash (`0 * 64`)
- `verify_chain()` recomputes every hash from scratch and detects any modification, deletion, or insertion
- **Tested:** `test_tampered_entry_detected` and `test_deleted_row_breaks_chain` verify detection even when SQLite triggers are bypassed by a direct-file attacker

### SS11.10(a) - Immutability Controls

**Requirement:** System shall be capable of producing accurate and complete copies of records.

**Implementation - SQLite Triggers:**
- `no_update_audit_log` trigger: `RAISE(ABORT, "21CFR11: audit_log rows are immutable - UPDATE forbidden")`
- `no_delete_audit_log` trigger: `RAISE(ABORT, "21CFR11: audit_log rows are immutable - DELETE forbidden")`
- These triggers operate at the database engine level, independent of application code
- **Tested:** `test_update_is_rejected_by_trigger` and `test_delete_is_rejected_by_trigger` verify database-level enforcement

### SS11.10(k) - Change Control

**Requirement:** Use of appropriate controls over systems documentation including adequate controls over the issuance of, access to, and use of documentation.

**Implementation - Model Registry:**
- `ml/model_registry.py` records every model loaded with: model ID, version, framework, weights path, weights SHA-256, weights size, output classes, input shape, registration timestamp
- `ModelRecord` is a frozen dataclass - immutable after creation
- `verify_weights_unchanged()` is called before every inference to detect unauthorized model substitution
- `requirements.txt` pins all dependency versions exactly - reproducible builds are a form of change control

---

## Evidence Artifacts

| Artifact | Location | Part 11 Relevance |
|---|---|---|
| Audit log module | `compliance/audit.py` | SS11.10(a)(e) |
| Audit log tests | `tests/test_audit.py` | SS11.10(a) validation |
| Model registry | `ml/model_registry.py` | SS11.10(k) |
| Inference pipeline | `ml/inference.py` | SS11.10(a)(d)(e) |
| FHIR DiagnosticReport | `fhir_client/writer.py` | Record creation |
| Requirements (pinned) | `requirements.txt` | SS11.10(k) |
| CI pipeline | `.github/workflows/ci.yml` | SS11.10(a) validation |

---

## Honest Limitations

This platform implements the **technical architectural controls** described above. It does not constitute:
- A formal 21 CFR Part 11 validation package (which requires IQ/OQ/PQ documentation)
- A Software Validation Plan per FDA General Principles of Software Validation guidance
- A Quality Management System (ISO 13485)
- Clinical validation of the AI model outputs

For a production SaMD deployment, these controls would form the technical foundation of a broader compliance program including formal validation protocols, SOPs, training records, and risk management per ISO 14971.
