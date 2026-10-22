"""Exports: the files an agent built for a person (Word, Excel, PowerPoint, chart), collected here.

A file is served only to the person it was built for (``created_by`` equals the caller) or to an
Admin; anyone else gets 404, not 403, so the id reveals nothing. Listing shows the caller's own
exports. Deleting is the owner's act. No route writes content: the agent tools do that.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response

from api.auth import require_user
from src.services import export_store, rbac
from src.services.contract_tools import store

log = logging.getLogger(__name__)
router = APIRouter(prefix="/exports", tags=["Exports"])


def _subject(principal: Any) -> str:
    return str(getattr(principal, "subject", None) or getattr(principal, "user_id", None) or "")


def _mine_or_admin(record: dict, principal: Any) -> bool:
    if record.get("created_by") == _subject(principal):
        return True
    try:
        return rbac.effective_role(principal) == "Admin"
    except Exception:  # noqa: BLE001
        return False


@router.get("")
def list_exports(limit: int = 20, principal=Depends(require_user)):
    try:
        return {"exports": export_store.recent(_subject(principal), limit=limit)}
    except store.StoreUnavailable:
        raise HTTPException(status_code=503, detail="The export store could not be read just now.")


@router.get("/{export_id}")
def get_export(export_id: str, principal=Depends(require_user)):
    try:
        record = export_store.get(export_id)
    except store.StoreUnavailable:
        raise HTTPException(status_code=503, detail="The export store could not be read just now.")
    if record is None or not _mine_or_admin(record, principal):
        raise HTTPException(status_code=404, detail="No such export.")
    return Response(content=record["content"], media_type=record["media_type"],
                    headers={"Content-Disposition": f'attachment; filename="{record["filename"]}"',
                             "X-Content-Type-Options": "nosniff"})


@router.delete("/{export_id}")
def delete_export(export_id: str, principal=Depends(require_user)):
    try:
        gone = export_store.delete(export_id, created_by=_subject(principal))
    except store.StoreUnavailable:
        raise HTTPException(status_code=503, detail="The export store could not be reached just now.")
    if not gone:
        raise HTTPException(status_code=404, detail="No such export.")
    return {"deleted": export_id}
