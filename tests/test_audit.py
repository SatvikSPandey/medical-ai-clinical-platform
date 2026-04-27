"""Tests for compliance/audit.py - hash-chained audit log.

These are unit tests (no network, no external services).
A fresh in-memory SQLite DB is created for each test via tmp_path fixture.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from compliance.audit import (
    AuditEntry,
    AuditEventType,
    ChainVerificationResult,
    append_event,
    get_recent_events,
    initialize_db,
    verify_chain,
)

# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def db(tmp_path: Path) -> Path:
    """Fresh initialized audit DB for each test."""
    db_path = tmp_path / "test_audit.db"
    initialize_db(db_path)
    return db_path


# ============================================================
# Initialization
# ============================================================

@pytest.mark.unit
class TestInitializeDb:
    def test_creates_audit_log_table(self, db: Path) -> None:
        conn = sqlite3.connect(str(db))
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        conn.close()
        assert ("audit_log",) in tables

    def test_creates_no_update_trigger(self, db: Path) -> None:
        conn = sqlite3.connect(str(db))
        triggers = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger'"
        ).fetchall()
        conn.close()
        names = [t[0] for t in triggers]
        assert "no_update_audit_log" in names
        assert "no_delete_audit_log" in names

    def test_initialize_is_idempotent(self, db: Path) -> None:
        """Calling initialize_db twice must not raise."""
        initialize_db(db)
        initialize_db(db)


# ============================================================
# Append event
# ============================================================

@pytest.mark.unit
class TestAppendEvent:
    def test_returns_audit_entry(self, db: Path) -> None:
        entry = append_event(
            AuditEventType.SYSTEM_START, "system", {}, db_path=db
        )
        assert isinstance(entry, AuditEntry)

    def test_entry_is_frozen(self, db: Path) -> None:
        from dataclasses import FrozenInstanceError
        entry = append_event(
            AuditEventType.SYSTEM_START, "system", {}, db_path=db
        )
        with pytest.raises(FrozenInstanceError):
            entry.actor = "tampered"  # type: ignore[misc]

    def test_first_entry_uses_genesis_prev_hash(self, db: Path) -> None:
        entry = append_event(
            AuditEventType.SYSTEM_START, "system", {}, db_path=db
        )
        assert entry.prev_hash == "0" * 64

    def test_sequence_increments(self, db: Path) -> None:
        e1 = append_event(AuditEventType.SYSTEM_START, "system", {}, db_path=db)
        e2 = append_event(AuditEventType.MODEL_LOADED, "system", {}, db_path=db)
        e3 = append_event(AuditEventType.INFERENCE_COMPLETE, "user1", {}, db_path=db)
        assert e1.sequence == 1
        assert e2.sequence == 2
        assert e3.sequence == 3

    def test_prev_hash_chains_correctly(self, db: Path) -> None:
        e1 = append_event(AuditEventType.SYSTEM_START, "system", {}, db_path=db)
        e2 = append_event(AuditEventType.MODEL_LOADED, "system", {}, db_path=db)
        assert e2.prev_hash == e1.entry_hash

    def test_entry_hash_is_64_hex_chars(self, db: Path) -> None:
        entry = append_event(
            AuditEventType.INFERENCE_COMPLETE, "user1",
            {"result": "ok"}, db_path=db
        )
        assert len(entry.entry_hash) == 64
        assert all(c in "0123456789abcdef" for c in entry.entry_hash)

    def test_event_data_is_preserved(self, db: Path) -> None:
        data = {"model_id": "test-model", "score": 0.85, "pathology": "Pneumothorax"}
        entry = append_event(
            AuditEventType.INFERENCE_COMPLETE, "user1", data, db_path=db
        )
        assert entry.event_data == data

    def test_source_sha256_is_stored(self, db: Path) -> None:
        sha = "a" * 64
        entry = append_event(
            AuditEventType.INFERENCE_COMPLETE, "user1", {},
            source_sha256=sha, db_path=db
        )
        assert entry.source_sha256 == sha

    def test_model_id_is_stored(self, db: Path) -> None:
        entry = append_event(
            AuditEventType.INFERENCE_COMPLETE, "user1", {},
            model_id="densenet121-v1.4.0", db_path=db
        )
        assert entry.model_id == "densenet121-v1.4.0"

    def test_timestamp_is_utc_iso8601(self, db: Path) -> None:
        from datetime import datetime
        entry = append_event(
            AuditEventType.SYSTEM_START, "system", {}, db_path=db
        )
        # Must parse without error and include UTC offset
        dt = datetime.fromisoformat(entry.timestamp_utc)
        assert dt.tzinfo is not None


# ============================================================
# Immutability - SQLite triggers
# ============================================================

@pytest.mark.unit
class TestImmutability:
    def test_update_is_rejected_by_trigger(self, db: Path) -> None:
        append_event(AuditEventType.SYSTEM_START, "system", {}, db_path=db)
        conn = sqlite3.connect(str(db))
        with pytest.raises(sqlite3.Error, match="21CFR11"):
            conn.execute("UPDATE audit_log SET actor='hacker' WHERE sequence=1")
            conn.commit()
        conn.close()

    def test_delete_is_rejected_by_trigger(self, db: Path) -> None:
        append_event(AuditEventType.SYSTEM_START, "system", {}, db_path=db)
        conn = sqlite3.connect(str(db))
        with pytest.raises(sqlite3.Error, match="21CFR11"):
            conn.execute("DELETE FROM audit_log WHERE sequence=1")
            conn.commit()
        conn.close()


# ============================================================
# Chain verification
# ============================================================

@pytest.mark.unit
class TestVerifyChain:
    def test_empty_log_is_valid(self, db: Path) -> None:
        result = verify_chain(db_path=db)
        assert isinstance(result, ChainVerificationResult)
        assert result.is_valid is True
        assert result.total_entries == 0

    def test_intact_chain_is_valid(self, db: Path) -> None:
        for i in range(5):
            append_event(AuditEventType.INFERENCE_COMPLETE, f"user{i}", {}, db_path=db)
        result = verify_chain(db_path=db)
        assert result.is_valid is True
        assert result.total_entries == 5
        assert result.first_broken_sequence is None

    def test_tampered_entry_detected(self, db: Path) -> None:
        """Directly modify a row bypassing the trigger - verify_chain catches it."""
        append_event(AuditEventType.SYSTEM_START, "system", {}, db_path=db)
        append_event(AuditEventType.INFERENCE_COMPLETE, "user1", {}, db_path=db)

        # Bypass the trigger by using PRAGMA to disable it temporarily
        conn = sqlite3.connect(str(db))
        # Drop and recreate without trigger (simulates a sophisticated attacker
        # who has direct SQLite file access and disables triggers)
        conn.execute("DROP TRIGGER IF EXISTS no_update_audit_log")
        conn.execute("UPDATE audit_log SET actor='hacker' WHERE sequence=1")
        conn.commit()
        conn.close()

        result = verify_chain(db_path=db)
        assert result.is_valid is False
        assert result.first_broken_sequence is not None

    def test_deleted_row_breaks_chain(self, db: Path) -> None:
        """Deleting a middle row breaks the sequence continuity."""
        for _ in range(3):
            append_event(AuditEventType.SYSTEM_START, "system", {}, db_path=db)

        conn = sqlite3.connect(str(db))
        conn.execute("DROP TRIGGER IF EXISTS no_delete_audit_log")
        conn.execute("DELETE FROM audit_log WHERE sequence=2")
        conn.commit()
        conn.close()

        result = verify_chain(db_path=db)
        assert result.is_valid is False

    def test_verification_result_is_frozen(self, db: Path) -> None:
        from dataclasses import FrozenInstanceError
        result = verify_chain(db_path=db)
        with pytest.raises(FrozenInstanceError):
            result.is_valid = True  # type: ignore[misc]


# ============================================================
# Read helpers
# ============================================================

@pytest.mark.unit
class TestGetRecentEvents:
    def test_returns_list_of_audit_entries(self, db: Path) -> None:
        append_event(AuditEventType.SYSTEM_START, "system", {}, db_path=db)
        events = get_recent_events(n=10, db_path=db)
        assert isinstance(events, list)
        assert all(isinstance(e, AuditEntry) for e in events)

    def test_returns_newest_first(self, db: Path) -> None:
        for i in range(5):
            append_event(AuditEventType.SYSTEM_START, f"user{i}", {}, db_path=db)
        events = get_recent_events(n=5, db_path=db)
        sequences = [e.sequence for e in events]
        assert sequences == sorted(sequences, reverse=True)

    def test_respects_n_limit(self, db: Path) -> None:
        for _ in range(10):
            append_event(AuditEventType.SYSTEM_START, "system", {}, db_path=db)
        events = get_recent_events(n=3, db_path=db)
        assert len(events) == 3
