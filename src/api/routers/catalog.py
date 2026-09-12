"""Distributor catalog API: column maps, feed import, SKU <-> history matches.

A feed is asserted data read by a column map -- no model runs here. A match is a
claim, so it is proposed and a person decides; who decided comes from the token.
"""
from __future__ import annotations

import datetime as dt
import os
import tempfile
from dataclasses import asdict
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from src.services import catalog_import, catalog_match
# Not src.services.db.get_conn: that opens autocommit connections, under which the
# sell-side services' rollback-before-raise is a no-op and FOR UPDATE locks don't hold.
from src.services.sell_side._db import transactional_conn as get_conn

from api.auth import require_user
from api.endpoint_gate import require as gate
from api.sell_side_http import http_errors, money_json

router = APIRouter(prefix="/catalog", tags=["Catalog"])
_AGENT = "CatalogRouter"
_FEED_SUFFIXES = (".csv", ".xlsx", ".xls")


def _subject(principal) -> Optional[str]:
    return getattr(principal, "subject", None) or None


def _max_upload_bytes() -> int:
    """The same per-file cap document intake enforces, from the same policy row.
    Imported lazily: the documents router pulls in the RAG stack."""
    from api.routers.documents import _intake_limits

    return _intake_limits()[1]


class MappingEntry(BaseModel):
    target_column: str
    source_header: str
    transform: Optional[str] = None
    is_required: bool = False


class MappingBody(BaseModel):
    distributor_id: str
    entries: List[MappingEntry]


class ProposeBody(BaseModel):
    distributor_id: str


class HumanMatchBody(BaseModel):
    distributor_id: str
    distributor_sku: str
    item_id: str


@router.get("/mappings/{mapping_profile}")
def get_mapping(mapping_profile: str):
    with get_conn() as c:
        return money_json({"mapping_profile": mapping_profile,
                            "entries": catalog_import.get_mapping(c, mapping_profile)})


@router.put("/mappings/{mapping_profile}")
def put_mapping(mapping_profile: str, body: MappingBody, principal=Depends(require_user)):
    gate("catalog.write", principal, agent=_AGENT,
         context={"mapping_profile": mapping_profile, "distributor_id": body.distributor_id})
    with http_errors(), get_conn() as c:
        return money_json({"mapping_profile": mapping_profile, "entries": catalog_import.save_mapping(
            c, mapping_profile=mapping_profile, distributor_id=body.distributor_id,
            entries=[e.model_dump() for e in body.entries])})


@router.post("/import")
async def import_feed(
    file: UploadFile = File(...), distributor_id: str = Form(...),
    feed_name: str = Form(...), mapping_profile: str = Form(...),
    price_effective: dt.date = Form(...), principal=Depends(require_user),
):
    gate("catalog.write", principal, agent=_AGENT,
         context={"distributor_id": distributor_id, "feed_name": feed_name})
    suffix = os.path.splitext(file.filename or "")[1].lower()
    if suffix not in _FEED_SUFFIXES:
        raise HTTPException(status_code=415,
                            detail=f"a catalog feed is one of {_FEED_SUFFIXES}, not {suffix or 'unnamed'}")
    max_bytes = _max_upload_bytes()
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail=f"feed exceeds {max_bytes} bytes")
    with http_errors(), tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(data)
        tmp.flush()
        result = await run_in_threadpool(
            catalog_import.import_catalog, distributor_id=distributor_id,
            feed_name=feed_name, mapping_profile=mapping_profile,
            price_effective=price_effective, imported_by=_subject(principal),
            file_bytes=data, file_name=file.filename, file_path=tmp.name)
    return money_json(asdict(result))


@router.post("/matches/propose")
def propose(body: ProposeBody, principal=Depends(require_user)):
    gate("catalog.write", principal, agent=_AGENT, context={"distributor_id": body.distributor_id})
    with http_errors(), get_conn() as c:
        return money_json(catalog_match.propose_matches(c, body.distributor_id))


@router.get("/matches")
def list_matches(distributor_id: Optional[str] = None, status: str = "proposed", limit: int = 100):
    with get_conn() as c:
        rows = catalog_match.list_matches(c, distributor_id=distributor_id, status=status, limit=limit)
    return money_json({"count": len(rows), "matches": rows})


@router.post("/matches/{match_id}/confirm")
def confirm(match_id: int, principal=Depends(require_user)):
    gate("catalog.write", principal, agent=_AGENT, context={"match_id": match_id})
    with http_errors(), get_conn() as c:
        return money_json(catalog_match.confirm_match(c, match_id, _subject(principal)))


@router.post("/matches/{match_id}/reject")
def reject(match_id: int, principal=Depends(require_user)):
    gate("catalog.write", principal, agent=_AGENT, context={"match_id": match_id})
    with http_errors(), get_conn() as c:
        return money_json(catalog_match.reject_match(c, match_id, _subject(principal)))


@router.post("/matches/human")
def human_match(body: HumanMatchBody, principal=Depends(require_user)):
    gate("catalog.write", principal, agent=_AGENT, context=body.model_dump())
    with http_errors(), get_conn() as c:
        return money_json(catalog_match.record_human_match(
            c, distributor_id=body.distributor_id, distributor_sku=body.distributor_sku,
            item_id=body.item_id, reviewer=_subject(principal)))
