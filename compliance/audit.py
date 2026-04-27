"""Append-only hash-chained audit log.

Every significant event in the platform (model load, inference, FHIR write,
tamper detection, errors) is recorded here. The log is:

  1. APPEND-ONLY: rows are never updated or deleted. SQLite enforces this
     via a trigger that raises an error on any UPDATE or DELETE.

  2. HASH-CHAINED: each row's hash is computed from its own content PLUS
     the previous row's hash. If any row is modified or deleted, every
     subsequent hash in the chain becomes invalid. Tamper-evident.

  3. IMMUTABLE ENTRIES: AuditEntry is a frozen dataclass. Once created,
     the Python object cannot be mutated. The SQLite row is similarly
     protected by the trigger.

  4. UTC TIMESTAMPS: all times are stored in UTC ISO 8601 format.
     No timezone ambiguity in a global clinical platform.

Regulatory basis:
  21 CFR Part 11 §11.10(a): Validation of systems to ensure accuracy,
    reliability, consistent intended performance, and the ability to
    discern invalid or altered records.
  21 CFR Part 11 §11.10(e): Use of audit trails — computer-generated,
    date/time-stamped audit trails to independently record the date and
    time of operator entries and actions that create, modify, or delete
    electronic records.
  21 CFR Part 11 §11.10(d): Limiting system access to authorized
    individuals — the tamper-detection chain supports this by making
    unauthorized modifications detectable.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

# ============================================================
# Event types
# ============================================================

class AuditEventType(str, Enum):
    """Categories of auditable events in the platform."""

    # Model lifecycle
    MODEL_LOADED       = "MODEL_LOADED"
    MODEL_TAMPER_DETECTED = "MODEL_TAMPER_DETECTED"

    # Inference
    INFERENCE_STARTED  = "INFERENCE_STARTED"
    INFERENCE_COMPLETE = "INFERENCE_COMPLETE"
    INFERENCE_FAILED   = "INFERENCE_FAILED"

    # FHIR operations
    FHIR_PATIENT_READ  = "FHIR_PATIENT_READ"
    FHIR_REPORT_WRITTEN = "FHIR_REPORT_WRITTEN"
    FHIR_ERROR         = "FHIR_ERROR"

    # Compliance checks
    CHAIN_VERIFIED     = "CHAIN_VERIFIED"
    CHAIN_BROKEN       = "CHAIN_BROKEN"

    # System
    SYSTEM_START       = "SYSTEM_START"
    SYSTEM_ERROR       = "SYSTEM_ERROR"


# ============================================================
# Audit entry dataclass
# ============================================================

@dataclass(frozen=True)
class AuditEntry:
    """A single immutable audit log record.

    row_id:         SQLite rowid (None before insertion).
    sequence:       Monotonically increasing integer — gaps indicate deletions.
    event_type:     Category from AuditEventType.
    actor:          Who triggered the event (user ID, "system", API key id).
    event_data:     JSON-serializable dict with event-specific details.
    source_sha256:  SHA-256 of the input DICOM (if inference event), else "".
    model_id:       Model identifier (if inference event), else "".
    timestamp_utc:  ISO 8601 UTC string.
    prev_hash:      SHA-256 of the previous entry's entry_hash field.
    entry_hash:     SHA-256 of this entry's content + prev_hash.
    """

    row_id: int | None
    sequence: int
    event_type: str
    actor: str
    event_data: dict[str, Any]
    source_sha256: str
    model_id: str
    timestamp_utc: str
    prev_hash: str
    entry_hash: str


# ============================================================
# Hash computation
# ============================================================

def _compute_entry_hash(
    sequence: int,
    event_type: str,
    actor: str,
    event_data: dict[str, Any],
    source_sha256: str,
    model_id: str,
    timestamp_utc: str,
    prev_hash: str,
) -> str:
    """Compute the SHA-256 hash for an audit entry.

    The hash is computed over ALL content fields plus prev_hash.
    Changing any field — including the timestamp — invalidates the hash.
    This is what makes the log tamper-evident.
    """
    content = json.dumps({
        "sequence":     sequence,
        "event_type":   event_type,
        "actor":        actor,
        "event_data":   event_data,
        "source_sha256":source_sha256,
        "model_id":     model_id,
        "timestamp_utc":timestamp_utc,
        "prev_hash":    prev_hash,
    }, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# ============================================================
# Database setup
# ============================================================

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS audit_log (
    row_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    sequence        INTEGER NOT NULL UNIQUE,
    event_type      TEXT    NOT NULL,
    actor           TEXT    NOT NULL,
    event_data      TEXT    NOT NULL,
    source_sha256   TEXT    NOT NULL DEFAULT '',
    model_id        TEXT    NOT NULL DEFAULT '',
    timestamp_utc   TEXT    NOT NULL,
    prev_hash       TEXT    NOT NULL,
    entry_hash      TEXT    NOT NULL
);
"""

_CREATE_NO_UPDATE_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS no_update_audit_log
BEFORE UPDATE ON audit_log
BEGIN
    SELECT RAISE(ABORT, '21CFR11: audit_log rows are immutable — UPDATE forbidden');
END;
"""

_CREATE_NO_DELETE_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS no_delete_audit_log
BEFORE DELETE ON audit_log
BEGIN
    SELECT RAISE(ABORT, '21CFR11: audit_log rows are immutable — DELETE forbidden');
END;
"""


def _get_connection(db_path: str | Path) -> sqlite3.Connection:
    """Open a SQLite connection with WAL mode for concurrent read safety."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def initialize_db(db_path: str | Path = "audit_log.db") -> None:
    """Create the audit_log table and immutability triggers if not present.

    Safe to call multiple times — uses CREATE IF NOT EXISTS.
    """
    with _get_connection(db_path) as conn:
        conn.execute(_CREATE_TABLE_SQL)
        conn.execute(_CREATE_NO_UPDATE_TRIGGER)
        conn.execute(_CREATE_NO_DELETE_TRIGGER)
        conn.commit()


# ============================================================
# Core append operation
# ============================================================

def append_event(
    event_type: AuditEventType | str,
    actor: str,
    event_data: dict[str, Any],
    source_sha256: str = "",
    model_id: str = "",
    db_path: str | Path = "audit_log.db",
) -> AuditEntry:
    """Append one event to the audit log and return the created entry.

    This is the ONLY way to write to the audit log. It:
      1. Reads the last entry's hash (or uses GENESIS_HASH for the first).
      2. Determines the next sequence number.
      3. Computes this entry's hash.
      4. Inserts the row atomically.

    Args:
        event_type: Category from AuditEventType.
        actor: Identity of the actor ("system", a user ID, etc.).
        event_data: Arbitrary JSON-serializable dict with event details.
        source_sha256: SHA-256 of the input DICOM if this is inference-related.
        model_id: Model identifier if this is inference-related.
        db_path: Path to the SQLite database file.

    Returns:
        The created AuditEntry (frozen).
    """
    GENESIS_HASH = "0" * 64  # The prev_hash for the very first entry

    timestamp_utc = datetime.now(UTC).isoformat()
    event_type_str = event_type.value if isinstance(event_type, AuditEventType) else event_type
    event_data_str = json.dumps(event_data, sort_keys=True, separators=(",", ":"))

    with _get_connection(db_path) as conn:
        # Read last entry to get prev_hash and next sequence
        cursor = conn.execute(
            "SELECT sequence, entry_hash FROM audit_log ORDER BY sequence DESC LIMIT 1"
        )
        last_row = cursor.fetchone()

        if last_row is None:
            prev_hash = GENESIS_HASH
            sequence = 1
        else:
            prev_hash = last_row[1]
            sequence = last_row[0] + 1

        entry_hash = _compute_entry_hash(
            sequence=sequence,
            event_type=event_type_str,
            actor=actor,
            event_data=json.loads(event_data_str),
            source_sha256=source_sha256,
            model_id=model_id,
            timestamp_utc=timestamp_utc,
            prev_hash=prev_hash,
        )

        conn.execute(
            """
            INSERT INTO audit_log
              (sequence, event_type, actor, event_data,
               source_sha256, model_id, timestamp_utc, prev_hash, entry_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (sequence, event_type_str, actor, event_data_str,
             source_sha256, model_id, timestamp_utc, prev_hash, entry_hash),
        )
        row_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.commit()

    return AuditEntry(
        row_id=row_id,
        sequence=sequence,
        event_type=event_type_str,
        actor=actor,
        event_data=json.loads(event_data_str),
        source_sha256=source_sha256,
        model_id=model_id,
        timestamp_utc=timestamp_utc,
        prev_hash=prev_hash,
        entry_hash=entry_hash,
    )


# ============================================================
# Chain verification
# ============================================================

@dataclass(frozen=True)
class ChainVerificationResult:
    """Result of a full audit log chain integrity check."""

    is_valid: bool
    total_entries: int
    first_broken_sequence: int | None
    broken_reason: str | None


def verify_chain(db_path: str | Path = "audit_log.db") -> ChainVerificationResult:
    """Verify the integrity of the entire audit log hash chain.

    Reads every row in sequence order and recomputes each hash.
    A mismatch means the chain is broken — tampering is detected.

    Returns:
        ChainVerificationResult — is_valid=True means the log is intact.
    """
    GENESIS_HASH = "0" * 64

    with _get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT sequence, event_type, actor, event_data,
                   source_sha256, model_id, timestamp_utc, prev_hash, entry_hash
            FROM audit_log ORDER BY sequence ASC
            """
        ).fetchall()

    if not rows:
        return ChainVerificationResult(
            is_valid=True, total_entries=0,
            first_broken_sequence=None, broken_reason=None,
        )

    expected_prev = GENESIS_HASH

    for row in rows:
        (seq, evt, actor, evt_data_str, src_sha, mod_id,
         ts_utc, prev_hash, stored_hash) = row

        # Check chain linkage
        if prev_hash != expected_prev:
            return ChainVerificationResult(
                is_valid=False,
                total_entries=len(rows),
                first_broken_sequence=seq,
                broken_reason=(
                    f"prev_hash mismatch at sequence {seq}: "
                    f"expected {expected_prev[:16]}... "
                    f"got {prev_hash[:16]}..."
                ),
            )

        # Recompute and compare entry hash
        recomputed = _compute_entry_hash(
            sequence=seq,
            event_type=evt,
            actor=actor,
            event_data=json.loads(evt_data_str),
            source_sha256=src_sha,
            model_id=mod_id,
            timestamp_utc=ts_utc,
            prev_hash=prev_hash,
        )

        if recomputed != stored_hash:
            return ChainVerificationResult(
                is_valid=False,
                total_entries=len(rows),
                first_broken_sequence=seq,
                broken_reason=f"entry_hash mismatch at sequence {seq}",
            )

        expected_prev = stored_hash

    return ChainVerificationResult(
        is_valid=True,
        total_entries=len(rows),
        first_broken_sequence=None,
        broken_reason=None,
    )


# ============================================================
# Read helpers
# ============================================================

def get_recent_events(
    n: int = 20,
    db_path: str | Path = "audit_log.db",
) -> list[AuditEntry]:
    """Return the most recent n audit entries (newest first)."""
    with _get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT row_id, sequence, event_type, actor, event_data,
                   source_sha256, model_id, timestamp_utc, prev_hash, entry_hash
            FROM audit_log ORDER BY sequence DESC LIMIT ?
            """,
            (n,),
        ).fetchall()

    return [
        AuditEntry(
            row_id=r[0], sequence=r[1], event_type=r[2],
            actor=r[3], event_data=json.loads(r[4]),
            source_sha256=r[5], model_id=r[6],
            timestamp_utc=r[7], prev_hash=r[8], entry_hash=r[9],
        )
        for r in rows
    ]
