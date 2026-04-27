"""Audit router - GET /audit and GET /audit/verify."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from api.core.config import Settings, get_settings
from api.core.security import get_current_user
from api.models.schemas import AuditEntryResponse, ChainStatusResponse
from compliance.audit import get_recent_events, verify_chain

router = APIRouter(prefix="/audit", tags=["audit"])


@router.get("/", response_model=list[AuditEntryResponse])
async def list_audit_events(
    n: int = Query(default=20, ge=1, le=200),
    settings: Settings = Depends(get_settings),
    _user: str = Depends(get_current_user),
) -> list[AuditEntryResponse]:
    """Return the most recent n audit log entries (newest first)."""
    events = get_recent_events(n=n, db_path=settings.audit_db_path)
    return [
        AuditEntryResponse(
            sequence=e.sequence,
            event_type=e.event_type,
            actor=e.actor,
            source_sha256=e.source_sha256,
            model_id=e.model_id,
            timestamp_utc=e.timestamp_utc,
            entry_hash=e.entry_hash,
        )
        for e in events
    ]


@router.get("/verify", response_model=ChainStatusResponse)
async def verify_audit_chain(
    settings: Settings = Depends(get_settings),
    _user: str = Depends(get_current_user),
) -> ChainStatusResponse:
    """Verify the hash chain integrity of the entire audit log."""
    result = verify_chain(db_path=settings.audit_db_path)
    return ChainStatusResponse(
        is_valid=result.is_valid,
        total_entries=result.total_entries,
        first_broken_sequence=result.first_broken_sequence,
        broken_reason=result.broken_reason,
    )
