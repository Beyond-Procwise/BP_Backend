# src/api/routers/fx.py
"""GET /fx/rates — live USD-quoted exchange rates for the dashboard currency
selector.

Serves from the ``proc.bp_fx_rates`` cache when the newest batch is younger
than 12h; otherwise re-fetches from the live open.er-api.com source (via the
existing fetcher in ``derivation_rules/fx.py`` — not duplicated here),
persists a new batch, and serves that. If the live fetch fails, falls back
to a stale cached batch (labelled honestly with its real ``fetched_at``)
rather than ever inventing a rate. If there is no cache at all and the fetch
fails, this fails closed with 503 — never a hardcoded rate.

The refresh/fallback chain itself lives in ``repositories.fx_rate_repo`` so
the opportunity-miner's GBP conversion (which needs the same honest
USD->GBP rate) can share it instead of duplicating this logic.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from repositories import fx_rate_repo

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/fx", tags=["FX"])


def _iso(ts: datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.isoformat()


@router.get("/rates", summary="Live USD-quoted FX rates for the dashboard currency selector")
def get_rates() -> Dict[str, Any]:
    try:
        result = fx_rate_repo.get_or_refresh_rates()
    except Exception:
        logger.exception("fx: get_or_refresh_rates raised unexpectedly")
        result = None

    if result is None:
        raise HTTPException(status_code=503, detail="exchange rates unavailable")

    return {
        "base": result["base_currency"],
        "rates": result["rates"],
        "fetched_at": _iso(result["fetched_at"]),
        "stale": result["stale"],
    }
