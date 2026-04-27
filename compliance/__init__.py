"""Compliance layer.

Provides append-only hash-chained audit logging and model versioning
controls aligned with FDA 21 CFR Part 11 and the FDA AI/ML Action Plan.
"""

from compliance.audit import (
    AuditEntry,
    AuditEventType,
    ChainVerificationResult,
    append_event,
    get_recent_events,
    initialize_db,
    verify_chain,
)

__all__ = [
    "AuditEntry",
    "AuditEventType",
    "ChainVerificationResult",
    "append_event",
    "get_recent_events",
    "initialize_db",
    "verify_chain",
]
