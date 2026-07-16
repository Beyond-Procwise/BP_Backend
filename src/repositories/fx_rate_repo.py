# src/repositories/fx_rate_repo.py
"""proc.bp_fx_rates — cached USD-quoted FX rate snapshots for the dashboard
currency selector (GET /fx/rates).

Distinct from the per-document ``exchange_rate_to_usd`` / ``converted_amount_usd``
columns already persisted on every ``bp_*_trgt`` row (that's the rate baked
in at extraction time, one row at a time). This table is a periodically
refreshed snapshot of the live open.er-api.com rates table, used only so the
UI can convert dashboard aggregates between currencies on demand — no
hardcoded rates, ever.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from services.db import get_conn
from services.structural_extractor.derivation_rules.fx import fetch_usd_quoted_rates

logger = logging.getLogger(__name__)

_STALE_AFTER = timedelta(hours=12)

DDL = """
CREATE SCHEMA IF NOT EXISTS proc;

CREATE TABLE IF NOT EXISTS proc.bp_fx_rates (
    id             BIGSERIAL PRIMARY KEY,
    base_currency  TEXT NOT NULL DEFAULT 'USD',
    currency       TEXT NOT NULL,
    rate           NUMERIC NOT NULL,
    fetched_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_bp_fx_rates_fetched_at
    ON proc.bp_fx_rates (fetched_at DESC);

CREATE INDEX IF NOT EXISTS ix_bp_fx_rates_currency_fetched
    ON proc.bp_fx_rates (currency, fetched_at DESC);
"""


def ensure_schema() -> None:
    """Ensure the ``proc.bp_fx_rates`` table exists."""

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(DDL)
        cur.close()


def get_latest_batch() -> Optional[Dict[str, Any]]:
    """Return the newest fetched_at batch, or ``None`` if the table is empty.

    Shape: ``{"fetched_at": datetime, "base_currency": str, "rates": {ccy: rate}}``.
    All rows sharing the newest ``fetched_at`` timestamp make up one batch
    (they're inserted together by :func:`insert_batch`).
    """

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT MAX(fetched_at) FROM proc.bp_fx_rates")
        row = cur.fetchone()
        latest_ts = row[0] if row else None
        if latest_ts is None:
            cur.close()
            return None
        cur.execute(
            "SELECT currency, rate, base_currency FROM proc.bp_fx_rates WHERE fetched_at = %s",
            (latest_ts,),
        )
        rows = cur.fetchall() or []
        cur.close()
        if not rows:
            return None
        rates = {r[0]: float(r[1]) for r in rows}
        base = rows[0][2] or "USD"
        return {"fetched_at": latest_ts, "base_currency": base, "rates": rates}


def insert_batch(rates: Dict[str, float], base_currency: str = "USD") -> datetime:
    """Persist one new batch (every row sharing the same ``fetched_at``)."""

    fetched_at = datetime.now(timezone.utc)
    with get_conn() as conn:
        cur = conn.cursor()
        for currency, rate in rates.items():
            cur.execute(
                "INSERT INTO proc.bp_fx_rates (base_currency, currency, rate, fetched_at) "
                "VALUES (%s, %s, %s, %s)",
                (base_currency, str(currency), float(rate), fetched_at),
            )
        cur.close()
    return fetched_at


def get_or_refresh_rates(max_age: timedelta = _STALE_AFTER) -> Optional[Dict[str, Any]]:
    """Serve the freshest possible USD-quoted rates batch, fail-closed.

    Order: fresh cache -> live fetch (persisted as a new batch) -> stale
    cache (returned as-is, honestly labelled ``stale: True`` with its real
    ``fetched_at``) -> ``None``. Never fabricates a rate — callers (GET
    /fx/rates, the opportunity-miner GBP conversion) must treat ``None`` as
    "cannot honestly convert" rather than assuming any 1:1 fallback.
    """

    batch = None
    try:
        batch = get_latest_batch()
    except Exception:
        logger.exception("fx_rate_repo: failed to read bp_fx_rates cache")
        batch = None

    now = datetime.now(timezone.utc)
    if batch is not None:
        fetched_at = batch["fetched_at"]
        if fetched_at.tzinfo is None:
            fetched_at = fetched_at.replace(tzinfo=timezone.utc)
        if (now - fetched_at) < max_age:
            return {**batch, "stale": False}

    live_rates = None
    try:
        live_rates = fetch_usd_quoted_rates()
    except Exception:
        logger.exception("fx_rate_repo: live rate fetch raised")
        live_rates = None

    if live_rates:
        try:
            fetched_at = insert_batch(live_rates, base_currency="USD")
        except Exception:
            logger.exception("fx_rate_repo: failed to persist new bp_fx_rates batch")
            fetched_at = now
        return {
            "fetched_at": fetched_at,
            "base_currency": "USD",
            "rates": live_rates,
            "stale": False,
        }

    if batch is not None:
        logger.warning(
            "fx_rate_repo: live fetch failed, serving stale cache from %s", batch["fetched_at"]
        )
        return {**batch, "stale": True}

    return None


def get_gbp_per_usd_rate() -> Optional[float]:
    """Return the current GBP-per-1-USD rate, or ``None`` if unavailable.

    Used by callers that need to chain native -> USD (via a row's own
    persisted ``exchange_rate_to_usd``) -> GBP without ever assuming 1:1.
    """

    result = get_or_refresh_rates()
    if not result:
        return None
    rate = result.get("rates", {}).get("GBP")
    return float(rate) if rate is not None else None
