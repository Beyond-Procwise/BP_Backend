"""Persistence for advice sessions and buyer-stated facts.

Stated facts live in their own table: a value the buyer asserts must never be
mistaken for measured data, and must be withdrawable so the advice reverts
cleanly. Withdrawal stamps withdrawn_at rather than deleting, keeping the trail.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Optional


def _rows(cur, sql: str, params: tuple = ()) -> list[dict]:
    cur.execute(sql, params)
    cols = [d[0] for d in (cur.description or [])]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def save_advice(conn, *, deal_id: str, supplier_id: Optional[str],
                quadrant: Optional[str], quadrant_source: str,
                quadrant_confidence: Optional[float], style: Optional[str],
                style_source: str, signals: dict, plays: list,
                created_by: Optional[str]) -> dict:
    advice_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    cur = conn.cursor()
    # One advice row per deal, refreshed in place. Every dashboard view rebuilds
    # advice, so an unconditional INSERT grew the table by a row per page
    # refresh — and worse, moved the advice_id that a buyer's stated facts hang
    # off, orphaning them on the next turn. created_at and created_by belong to
    # the first build and are left alone.
    cur.execute(
        "INSERT INTO proc.bp_negotiation_advice "
        "(advice_id, deal_id, supplier_id, quadrant, quadrant_source, "
        " quadrant_confidence, style, style_source, signals, plays, "
        " created_by, created_at, updated_at) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
        "ON CONFLICT (deal_id) DO UPDATE SET "
        " supplier_id = EXCLUDED.supplier_id, "
        " quadrant = EXCLUDED.quadrant, "
        " quadrant_source = EXCLUDED.quadrant_source, "
        " quadrant_confidence = EXCLUDED.quadrant_confidence, "
        " style = EXCLUDED.style, "
        " style_source = EXCLUDED.style_source, "
        " signals = EXCLUDED.signals, "
        " plays = EXCLUDED.plays, "
        " created_by = COALESCE(proc.bp_negotiation_advice.created_by, "
        "                       EXCLUDED.created_by), "
        " updated_at = EXCLUDED.updated_at "
        "RETURNING advice_id",
        (advice_id, deal_id, supplier_id, quadrant, quadrant_source,
         quadrant_confidence, style, style_source,
         json.dumps(signals, default=str), json.dumps(plays, default=str),
         created_by, now, now),
    )
    row = cur.fetchone()
    if row:
        advice_id = row[0]
    conn.commit()
    return {"advice_id": advice_id, "deal_id": deal_id,
            "supplier_id": supplier_id, "quadrant": quadrant,
            "quadrant_source": quadrant_source,
            "quadrant_confidence": quadrant_confidence, "style": style,
            "style_source": style_source, "signals": signals, "plays": plays,
            "created_by": created_by, "created_at": now.isoformat()}


def load_advice(conn, deal_id: str) -> Optional[dict]:
    rows = _rows(conn.cursor(),
                 "SELECT * FROM proc.bp_negotiation_advice WHERE deal_id=%s "
                 "ORDER BY created_at DESC LIMIT 1", (deal_id,))
    return rows[0] if rows else None


def state_fact(conn, *, advice_id: str, fact_key: str, fact_value: Any,
               stated_by: Optional[str]) -> None:
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO proc.bp_negotiation_advice_fact "
        "(advice_id, fact_key, fact_value, stated_by, stated_at, withdrawn_at) "
        "VALUES (%s,%s,%s,%s,%s,NULL) "
        "ON CONFLICT (advice_id, fact_key) DO UPDATE SET "
        "fact_value = EXCLUDED.fact_value, stated_by = EXCLUDED.stated_by, "
        "stated_at = EXCLUDED.stated_at, withdrawn_at = NULL",
        (advice_id, fact_key, str(fact_value), stated_by,
         datetime.now(timezone.utc)),
    )
    conn.commit()


def withdraw_fact(conn, *, advice_id: str, fact_key: str) -> None:
    cur = conn.cursor()
    cur.execute(
        "UPDATE proc.bp_negotiation_advice_fact SET withdrawn_at=%s "
        "WHERE advice_id=%s AND fact_key=%s AND withdrawn_at IS NULL",
        (datetime.now(timezone.utc), advice_id, fact_key),
    )
    conn.commit()


def active_facts(conn, advice_id: str) -> dict:
    rows = _rows(conn.cursor(),
                 "SELECT fact_key, fact_value FROM "
                 "proc.bp_negotiation_advice_fact "
                 "WHERE advice_id=%s AND withdrawn_at IS NULL", (advice_id,))
    return {r["fact_key"]: r["fact_value"] for r in rows}
