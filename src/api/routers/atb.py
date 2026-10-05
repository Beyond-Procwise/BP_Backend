"""Imported style packs and the page layouts measured out of them.

An import READS a .pptx and writes candidates. Nothing it produces can be used by a report until
a human approves it, which is why importing is a `write` action and approving is `configure` —
the same class as approving a playbook, because an approved pack changes what every report built
from it looks like.

Responses carry ids and names. They never name a route or a table: the output-safety layer
replaces such fields with "[withheld]", which has silently emptied payloads here before.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from src.services.atb.pptx_import import store
from src.services.atb.pptx_import.contract import validate_rating_scale
from src.services.atb.pptx_import.import_pack import ImportRefused, import_pack, key_for
from src.services.db import get_conn

from api.auth import require_user
from api.endpoint_gate import require as gate
from api.routers.reports import readable_deck

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/atb", tags=["ATB"])
_AGENT = "AtbRouter"
_MAX_UPLOAD_BYTES = 64 * 1024 * 1024
_PPTX_SUFFIX = ".pptx"


class RenameBody(BaseModel):
    name: str


class RatingScaleBody(BaseModel):
    name: str
    chips: Dict[str, Dict[str, str]]
    promote: Optional[Dict[str, Any]] = None


def _subject(principal) -> str:
    return str(getattr(principal, "subject", None) or "unknown")


def _pack_summary(row: dict) -> dict:
    """What a listing says about a pack. Tokens and evidence are fetched per pack, not listed."""
    tokens = row.get("tokens") or {}
    return {
        "pack_id": row["pack_id"],
        "pack_key": row["pack_key"],
        "version": row["version"],
        "source_file": row["source_file"],
        "slide_count": row["slide_count"],
        "status": row["status"],
        "created_at": row["created_at"],
        "created_by": row["created_by"],
        # Where a pack came from, in one sentence, when it was learned from something other than an
        # uploaded deck (a generated report). None for an upload.
        "notes": row.get("notes"),
        "approved_by": row.get("approved_by"),
        "format": row.get("format") or {},
        "locale_contested": bool((tokens.get("writing") or {}).get("locale_contested")),
    }


def _layout_summary(row: dict) -> dict:
    return {
        "layout_id": row["layout_id"],
        "pack_id": row["pack_id"],
        "layout_key": row["layout_key"],
        "proposed_name": row["proposed_name"],
        "name": row.get("name"),
        # 'template' (a reusable shape, offered in the Layout picker) or 'page' (one arranged
        # slide, offered as a starting page). Passed through, never inferred here.
        "kind": row.get("kind") or "template",
        "status": row["status"],
        "slide_refs": list(row.get("slide_refs") or []),
        "regions": row.get("regions") or [],
        "slots": row.get("slots") or {},
        "example_fill": row.get("example_fill") or {},
        "example_source": row.get("example_source") or {},
        "problems": row.get("problems") or [],
    }


def _import_response(result) -> dict:
    """What an import answers with. Shared by the upload route and the from-report route, so a pack
    looks the same to the review screen whichever way it was made."""
    return {
        "pack_id": result.pack_id,
        "pack_key": result.pack_key,
        "version": result.version,
        "layouts": [{"layout_key": l["id"], "proposed_name": l["proposed_name"],
                     "slides": l["slide_refs"], "problems": len(l["problems"])}
                    for l in result.layouts],
        # The arranged pages, which §6a deferred until step 2 could place their boxes. Named
        # rather than described, so this list reads as a table of contents.
        "pages": [{"layout_key": p["id"], "name": p["name"],
                   "slides": p["slide_refs"], "problems": len(p["problems"])}
                  for p in result.pages],
        "single_use": result.single_use,
        "problems": result.problems,
        "diff": result.diff,
        "locale_contested": bool(result.pack["writing"]["locale_contested"]),
        "locale_suggested": result.pack["writing"]["locale_suggested"],
    }


@router.post("/import")
async def post_import(file: UploadFile = File(...), principal=Depends(require_user)):
    gate("style_pack.write", principal, agent=_AGENT,
         context={"filename": file.filename or ""})
    name = file.filename or ""
    if not name.lower().endswith(_PPTX_SUFFIX):
        raise HTTPException(status_code=415,
                            detail="only a .pptx can be read for its style")
    if key_for(name).startswith(_FROM_REPORT_PREFIX):
        # The pack key comes from the file name, and a pack keeps its names and rejections from
        # one version to the next. An upload must not be able to become a version of a pack that
        # was learned from a report, or to claim its key first.
        raise HTTPException(status_code=400,
                            detail=f"a deck cannot be uploaded under a name beginning "
                                   f"'{_FROM_REPORT_PREFIX}': that is reserved for styles learned "
                                   "from a generated report")
    data = await file.read()
    if len(data) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413,
                            detail=f"the file is larger than {_MAX_UPLOAD_BYTES // (1024 * 1024)}MB")
    try:
        with get_conn() as conn:
            result = await run_in_threadpool(import_pack, data, name, _subject(principal), conn)
    except ImportRefused as exc:
        # The importer's own reason, passed on rather than paraphrased. Nothing was stored.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _import_response(result)


_FROM_REPORT_PREFIX = "from-report-"
_FROM_REPORT_NOTE = ("Learned from the generated report {job_id} ({report_type}). This pack measures "
                     "how that report looked, which is not necessarily how its source style pack "
                     "was defined.")


@router.post("/import/from-report/{job_id}")
async def post_import_from_report(job_id: str, principal=Depends(require_user)):
    gate("style_pack.write", principal, agent=_AGENT,
         context={"job_id": job_id, "source": "report"})
    # Reading the deck asks the REPORT'S rules (sign-off hold, refusal, report.read, released-only,
    # the signed hash) -- the same function the download calls, so the two cannot drift apart.
    job, content, _media_type, _filename = await run_in_threadpool(
        readable_deck, job_id, principal, allow_review=False)
    try:
        with get_conn() as conn:
            result = await run_in_threadpool(
                import_pack, content, f"{_FROM_REPORT_PREFIX}{job_id}.pptx", _subject(principal), conn)
            try:
                await run_in_threadpool(
                    store.set_pack_notes, conn, result.pack_id,
                    _FROM_REPORT_NOTE.format(job_id=job_id,
                                             report_type=job.get("report_type") or "report"))
            except Exception:
                # The pack is already stored as a candidate, so a 500 would hide a pack that exists
                # and a retry would make a second version. Where it came from is still on record in
                # its source file (from-report-<job_id>.pptx); only the sentence is missing.
                logger.exception("style pack %s: the provenance note was not written", result.pack_id)
    except ImportRefused as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _import_response(result)


@router.get("/packs")
def get_packs(principal=Depends(require_user)):
    gate("style_pack.read", principal, agent=_AGENT)
    with get_conn() as conn:
        return {"packs": [_pack_summary(row) for row in store.packs(conn)]}


@router.get("/packs/{pack_id}")
def get_pack(pack_id: str, principal=Depends(require_user)):
    gate("style_pack.read", principal, agent=_AGENT, context={"pack_id": pack_id})
    with get_conn() as conn:
        row = store.pack(conn, pack_id)
        if not row:
            raise HTTPException(status_code=404, detail="no such pack")
        summary = _pack_summary(row)
        summary["tokens"] = row.get("tokens") or {}
        summary["layouts"] = [_layout_summary(l) for l in store.layouts(conn, pack_id=pack_id)]
        return summary


@router.get("/packs/{pack_id}/evidence")
def get_evidence(pack_id: str, principal=Depends(require_user)):
    gate("style_pack.read", principal, agent=_AGENT, context={"pack_id": pack_id})
    with get_conn() as conn:
        row = store.pack(conn, pack_id)
        if not row:
            raise HTTPException(status_code=404, detail="no such pack")
        return {"pack_id": pack_id, "evidence": row.get("evidence") or {}}


@router.get("/layouts")
def get_layouts(status: Optional[str] = None, pack_id: Optional[str] = None,
                kind: Optional[Literal["template", "page"]] = None,
                principal=Depends(require_user)):
    """`kind` is typed rather than validated by hand: an unknown value is a 422, and silently
    returning everything for kind=quadrant would fill a picker with the wrong thing."""
    gate("style_pack.read", principal, agent=_AGENT)
    with get_conn() as conn:
        found: List[dict] = store.layouts(conn, pack_id=pack_id, status=status, kind=kind)
        return {"layouts": [_layout_summary(row) for row in found]}


@router.post("/layouts/{layout_id}")
def post_rename(layout_id: str, body: RenameBody, principal=Depends(require_user)):
    gate("style_pack.write", principal, agent=_AGENT, context={"layout_id": layout_id})
    if not body.name.strip():
        raise HTTPException(status_code=400, detail="a layout needs a name")
    with get_conn() as conn:
        if not store.rename_layout(conn, layout_id, body.name.strip(), _subject(principal)):
            raise HTTPException(status_code=404, detail="no such layout")
        return {"layout_id": layout_id, "name": body.name.strip()}


@router.post("/layouts/{layout_id}/approve")
def post_approve_layout(layout_id: str, principal=Depends(require_user)):
    gate("style_pack.approve", principal, agent=_AGENT, context={"layout_id": layout_id})
    with get_conn() as conn:
        if not store.set_layout_status(conn, layout_id, "approved", _subject(principal)):
            raise HTTPException(status_code=404,
                                detail="no such layout, or its pack is still importing")
        return {"layout_id": layout_id, "status": "approved"}


@router.post("/layouts/{layout_id}/reject")
def post_reject_layout(layout_id: str, principal=Depends(require_user)):
    """Rejecting a CANDIDATE is triage; rejecting something APPROVED withdraws an approval.

    They are not the same authority. Discarding a layout the importer proposed is ordinary
    `write` work — whoever imported a deck should be able to throw away the arrangements that
    were never templates. But a layout that is already approved is in every report built on it,
    and taking it back out changes what those reports look like — which is precisely why
    approving is `configure`. Left as a blanket `write`, any Buyer could withdraw an Admin's
    approval; the asymmetry was real and is closed here rather than by making triage
    Admin-only.
    """
    with get_conn() as conn:
        current = store.layout_status(conn, layout_id)
        action = "style_pack.approve" if current == "approved" else "style_pack.write"
        gate(action, principal, agent=_AGENT,
             context={"layout_id": layout_id, "was": current or "unknown"})
        if not store.set_layout_status(conn, layout_id, "rejected", _subject(principal)):
            raise HTTPException(status_code=404,
                                detail="no such layout, or its pack is still importing")
        return {"layout_id": layout_id, "status": "rejected"}


@router.post("/packs/{pack_id}/approve")
def post_approve_pack(pack_id: str, principal=Depends(require_user)):
    gate("style_pack.approve", principal, agent=_AGENT, context={"pack_id": pack_id})
    with get_conn() as conn:
        if not store.set_pack_status(conn, pack_id, "approved", _subject(principal)):
            raise HTTPException(status_code=404,
                                detail="no such pack, or it is still importing")
        return {"pack_id": pack_id, "status": "approved"}


@router.post("/packs/{pack_id}/rating-scales")
def post_rating_scale(pack_id: str, body: RatingScaleBody, principal=Depends(require_user)):
    gate("style_pack.write", principal, agent=_AGENT,
         context={"pack_id": pack_id, "scale": body.name})
    errors = validate_rating_scale(body.name, body.chips)
    if errors:
        # Unchecked, a chip of "navy-ish" went into the pack, could be approved, and then failed
        # the browser's validator at render time.
        raise HTTPException(status_code=400, detail="; ".join(errors))
    promoted = None
    with get_conn() as conn:
        if body.promote:
            # A column may only be promoted to a scale that defines EVERY label its cells use. A
            # half-defined scale renders a blank chip on a real page.
            missing = [label for label in (body.promote.get("labels") or [])
                       if label not in body.chips]
            if missing:
                raise HTTPException(
                    status_code=400,
                    detail=f'the scale does not define {", ".join(sorted(missing))}')
        store.define_rating_scale(conn, pack_id, body.name.strip(), body.chips,
                                  _subject(principal))
        if body.promote:
            layout_key = body.promote.get("layout_key")
            column = body.promote.get("column")
            if not layout_key or not column:
                raise HTTPException(status_code=400,
                                    detail="a promotion names a layout_key and a column")
            # The promotion is applied, not merely checked: returning 200 for a promotion that
            # never happened is the shape of bug this whole build is about.
            if not store.promote_column(conn, pack_id=pack_id, layout_key=layout_key,
                                        column=column, scale=body.name.strip()):
                raise HTTPException(status_code=404,
                                    detail="no such layout, or it has no such table column")
            promoted = {"layout_key": layout_key, "column": column}
    return {"pack_id": pack_id, "scale": body.name.strip(),
            "labels": sorted(body.chips), "promoted": promoted}
