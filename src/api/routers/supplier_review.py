"""Supplier match-review API.

Humans confirm or reject close-call supplier-identity decisions. Confirm merges
the extracted variant into the candidate (alias); reject records it as a distinct
supplier. The document's literal extracted value is never changed — only the
canonical supplier_id link is governed.
"""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.auth import require_user
from src.services.db import get_conn
from src.services.extraction_v3 import supplier_resolver as SR

log = logging.getLogger(__name__)

router = APIRouter(prefix="/suppliers", tags=["Supplier Review"])

_COLS = (
    "review_id, created_date, extracted_name, decision, chosen_supplier_id, "
    "candidate_supplier_id, candidate_supplier_name, score, doc_type, doc_pk, "
    "status, reviewed_by, reviewed_date"
)


class ReviewBody(BaseModel):
    """What is being decided. The `reviewer` field is NOT who decided it.

    It is kept because clients send it and because it sometimes carries
    something a person meant, but the actor is the token -- see `_reviewer`.
    """

    reviewer: str = "api"


@router.get("/reviews/queue")
def reviews_queue(limit: int = 100):
    """One combined supplier-review queue for the UI: name-match/duplicate reviews
    AND web-research enrichment approvals, in a uniform shape with per-item actions."""
    from src.services.supplier_enrichment import research as R
    items: list[dict] = []
    with get_conn() as c, c.cursor() as cur:
        # 1) supplier-identity reviews (name-match + duplicate sweep)
        cur.execute(
            f"SELECT {_COLS} FROM proc.bp_supplier_review WHERE status='pending' "
            "ORDER BY created_date DESC LIMIT %s", (limit,),
        )
        mcols = [d[0] for d in cur.description]
        for row in cur.fetchall():
            r = dict(zip(mcols, row))
            dup = r["decision"] == "existing_dup"
            items.append({
                "review_type": "supplier_match",
                "id": r["review_id"],
                "supplier_id": r["chosen_supplier_id"],
                "supplier_name": r["extracted_name"],
                "title": (f"Possible duplicate: {r['extracted_name']!r} ~ {r['candidate_supplier_name']!r}"
                          if dup else
                          f"Supplier match: {r['extracted_name']!r} ~ {r['candidate_supplier_name']!r}"),
                "score": float(r["score"]) if r["score"] is not None else None,
                "created_date": r["created_date"],
                "detail": {"extracted_name": r["extracted_name"], "decision": r["decision"],
                           "candidate_supplier_id": r["candidate_supplier_id"],
                           "candidate_supplier_name": r["candidate_supplier_name"], "doc_pk": r["doc_pk"]},
                "actions": [
                    {"label": "Same supplier (merge)", "method": "POST", "path": f"/suppliers/reviews/{r['review_id']}/confirm"},
                    {"label": "Different supplier", "method": "POST", "path": f"/suppliers/reviews/{r['review_id']}/reject"},
                ],
            })

        # 2) web-research enrichment approvals
        cur.execute(
            "SELECT e.enrichment_id, e.supplier_id, s.supplier_name, e.raw->>'matched_name', "
            "NULLIF(e.raw->>'name_match','')::float, e.confidence, e.fields, e.citations, e.created_date "
            "FROM proc.bp_supplier_enrichment e LEFT JOIN proc.bp_supplier s ON s.supplier_id = e.supplier_id "
            "WHERE e.apply_status = 'pending' ORDER BY e.created_date DESC LIMIT %s", (limit,),
        )
        enrich_rows = cur.fetchall()
        for eid, sid, sname, matched, nm, conf, fields, citations, created in enrich_rows:
            fields = fields or {}
            if isinstance(fields, str):
                fields = json.loads(fields or "{}")
            cur.execute("SELECT " + ", ".join(R._APPLY_COLUMNS) + " FROM proc.bp_supplier WHERE supplier_id = %s", (sid,))
            sv = cur.fetchone()
            current = dict(zip(R._APPLY_COLUMNS, sv)) if sv else {}
            # The proposal itself, from the same function the approve button runs.
            # This was a third local re-derivation of "researched and currently
            # empty", which ignored content verification and column widths and so
            # could offer a fill that the approval would then decline to make.
            would_fill = sorted(R.fillable(cur, sid, fields))
            items.append({
                "review_type": "supplier_enrichment",
                "id": eid,
                "supplier_id": sid,
                "supplier_name": sname,
                "title": (f"Enrich {sname!r}: matched {matched!r}, would fill {would_fill}"
                          if would_fill else f"Enrich {sname!r}: nothing verified (matched {matched!r})"),
                "score": float(conf) if conf is not None else None,
                "created_date": created,
                "detail": {"matched_name": matched, "name_match": nm, "would_fill": would_fill,
                           "current": current, "fields": fields, "citations": citations or []},
                "actions": [
                    {"label": "Approve (fill empty fields)", "method": "POST", "path": f"/suppliers/enrichment/{eid}/apply"},
                    {"label": "Reject", "method": "POST", "path": f"/suppliers/enrichment/{eid}/reject"},
                ],
            })

    items.sort(key=lambda x: x["created_date"] or "", reverse=True)
    return {"count": len(items), "items": items}


@router.get("/reviews")
def list_reviews(status: str = "pending", limit: int = 100):
    with get_conn() as c, c.cursor() as cur:
        cur.execute(
            f"SELECT {_COLS} FROM proc.bp_supplier_review "
            "WHERE (%s = 'all' OR status = %s) ORDER BY created_date DESC LIMIT %s",
            (status, status, limit),
        )
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    return {"count": len(rows), "reviews": rows}


@router.post("/reviews/sweep")
def sweep_duplicates(min_score: float | None = None):
    """Scan existing suppliers for likely duplicates and flag them for review."""
    with get_conn() as c:
        return SR.sweep_supplier_duplicates(c, min_score=min_score)


def _reviewer(principal) -> str | None:
    """The token, never `body.reviewer`. A merge into the supplier master is not
    something a caller gets to sign in somebody else's name."""

    return getattr(principal, "subject", None) or None


@router.post("/reviews/{review_id}/confirm")
def confirm_review(review_id: int, body: ReviewBody, principal=Depends(require_user)):
    """The extracted name IS the candidate supplier — merge (alias to canonical)."""
    with get_conn() as c:
        try:
            return SR.confirm_review(review_id, _reviewer(principal), c)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))


@router.post("/reviews/{review_id}/reject")
def reject_review(review_id: int, body: ReviewBody, principal=Depends(require_user)):
    """The extracted name is a DISTINCT supplier — keep it separate (alias to its own id)."""
    with get_conn() as c:
        try:
            return SR.reject_review(review_id, _reviewer(principal), c)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
