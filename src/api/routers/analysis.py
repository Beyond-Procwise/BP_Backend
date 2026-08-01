"""Analysis events — the read surface the Analyse area and the deal history use.

Called directly by spendiq-ui against VITE_AI_API_URL; the Node gateway does not
proxy these. See docs/superpowers/specs/2026-08-01-analysis-events-design.md
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.db import get_conn
from src.services import analysis_store

log = logging.getLogger(__name__)

router = APIRouter(prefix="/analysis", tags=["Analysis"])

_LIST_COLS = """
    a.analysis_id, a.name, a.mode, a.session_id, a.status, a.failure_reason,
    a.started_at, a.completed_at, a.document_count, a.value_found, a.currency
"""


def _query(sql: str, params: tuple) -> list[dict]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(sql, params)
        cols = [c.name for c in (cur.description or [])]
        return [dict(zip(cols, r)) for r in (cur.fetchall() or [])]


class AnalysisStartIn(BaseModel):
    session_id: str = Field(min_length=1)
    name: Optional[str] = None
    mode: str = "new"
    created_by: Optional[str] = None


@router.post("", summary="Start an analysis event for an upload session")
def post_analysis(body: AnalysisStartIn) -> dict:
    try:
        analysis_id = analysis_store.start(
            session_id=body.session_id, name=body.name, mode=body.mode,
            created_by=body.created_by)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        log.exception("could not start analysis for session=%s", body.session_id)
        raise HTTPException(status_code=500, detail=str(exc))
    return {"analysis_id": analysis_id}


@router.get("", summary="List analysis events, newest first")
def list_analyses(status: Optional[str] = None, deal_id: Optional[str] = None,
                  limit: int = 50, offset: int = 0) -> dict:
    where, params = ["1=1"], []
    if status:
        where.append("a.status = %s")
        params.append(status)
    if deal_id:
        where.append("EXISTS (SELECT 1 FROM proc.bp_analysis_deal ad "
                     "WHERE ad.analysis_id = a.analysis_id AND ad.deal_id = %s)")
        params.append(deal_id)
    rows = _query(
        f"SELECT {_LIST_COLS}, "
        "  (SELECT COALESCE(json_agg(json_build_object("
        "        'deal_id', ad.deal_id, 'version', ad.version)), '[]'::json) "
        "     FROM proc.bp_analysis_deal ad "
        "    WHERE ad.analysis_id = a.analysis_id) AS deals "
        "  FROM proc.bp_analysis a "
        f" WHERE {' AND '.join(where)} "
        " ORDER BY a.started_at DESC LIMIT %s OFFSET %s",
        tuple(params) + (max(1, min(int(limit), 200)), max(0, int(offset))),
    )
    return {"analyses": rows, "total": len(rows)}


# Declared BEFORE /{analysis_id}: otherwise the path param captures 'by-deal'
# and 'by-session'. Same trap the gateway documents at spendiq.controller.ts:80.
@router.get("/by-deal/{deal_id}", summary="Version history for one deal")
def get_by_deal(deal_id: str) -> dict:
    rows = _query(
        f"SELECT {_LIST_COLS}, ad.version, ad.is_latest "
        "  FROM proc.bp_analysis a "
        "  JOIN proc.bp_analysis_deal ad ON ad.analysis_id = a.analysis_id "
        " WHERE ad.deal_id = %s ORDER BY ad.version DESC",
        (deal_id,),
    )
    # Delta against the NEXT row down (the previous version). The oldest version
    # has nothing to compare against, so its delta is None — not zero.
    for i, row in enumerate(rows):
        prev = rows[i + 1] if i + 1 < len(rows) else None
        row["delta"] = None if prev is None else {
            "document_count": _diff(row.get("document_count"),
                                    prev.get("document_count")),
            "value_found": _diff(row.get("value_found"), prev.get("value_found")),
        }
    return {"deal_id": deal_id, "versions": rows}


def _diff(now: Any, before: Any) -> Optional[float]:
    """None when either side is unknown — an unknown is not a zero change."""
    if now is None or before is None:
        return None
    return round(float(now) - float(before), 2)


@router.get("/by-session/{session_id}", summary="The analysis for one upload")
def get_by_session(session_id: str) -> dict:
    rows = _query(
        f"SELECT {_LIST_COLS} FROM proc.bp_analysis a WHERE a.session_id = %s",
        (session_id,))
    if not rows:
        raise HTTPException(status_code=404, detail="no analysis for that session")
    return _hydrate(rows[0])


@router.get("/{analysis_id}", summary="One analysis event in full")
def get_analysis(analysis_id: str) -> dict:
    rows = _query(
        f"SELECT {_LIST_COLS}, a.findings FROM proc.bp_analysis a "
        " WHERE a.analysis_id = %s", (analysis_id,))
    if not rows:
        raise HTTPException(status_code=404, detail="no such analysis")
    return _hydrate(rows[0])


def _hydrate(row: dict) -> dict:
    aid = row["analysis_id"]
    row["documents"] = _query(
        "SELECT doc_type, doc_pk, file_path, file_name, outcome "
        "  FROM proc.bp_analysis_document WHERE analysis_id = %s "
        " ORDER BY file_name", (aid,))
    row["deals"] = _query(
        "SELECT deal_id, version, is_latest, linked_at "
        "  FROM proc.bp_analysis_deal WHERE analysis_id = %s "
        " ORDER BY deal_id", (aid,))
    return row
