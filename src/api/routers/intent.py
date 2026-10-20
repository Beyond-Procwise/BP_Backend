"""POST /intent/route -- what a free-text request is asking for.

Reads a request and proposes the report type or workflow it names, with the inputs it could find and
the ones still missing (``services/intent_router``). It PROPOSES: nothing is run or filed here, and a
person confirms. The only data read is a supplier's name or id, to resolve one the request mentions.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from api.auth import require_user
from api.endpoint_gate import require as gate
from src.services import intent_router

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/intent", tags=["Intent"])


class RouteBody(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


def _supplier_lookup(text: str) -> List[Tuple[str, str]]:
    """Suppliers whose name or id appears in ``text``. The router then keeps whole-word, longest matches."""
    from src.services.db import get_conn

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT supplier_id, supplier_name FROM proc.bp_deal_overview "
            "WHERE supplier_id IS NOT NULL "
            "  AND (strpos(lower(%s), lower(supplier_name)) > 0 OR strpos(lower(%s), lower(supplier_id)) > 0) "
            "LIMIT 50", (text, text))
        return [(r[0], r[1] or r[0]) for r in cur.fetchall()]


@router.post("/route", summary="What a free-text request is asking for")
def route_intent(body: RouteBody, principal=Depends(require_user)) -> Dict[str, Any]:
    # It proposes only, but it looks supplier names up to do so: asked as reading a report, first.
    gate("report.read", principal, agent="IntentRouter")
    return intent_router.route(body.text, supplier_lookup=_supplier_lookup)
