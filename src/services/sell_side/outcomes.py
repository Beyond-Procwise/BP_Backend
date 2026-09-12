"""Won or lost, and why -- the only thing that can ever calibrate win_probability."""
from __future__ import annotations

from typing import Any, Dict, Optional

import psycopg2.errors

from src.services.sell_side._db import NotFound, StateConflict, dict_cursor

QUOTE_OUTCOMES = frozenset({"won", "lost", "expired", "withdrawn"})
LOST_REASONS = frozenset({"price", "lead_time", "incumbent", "no_budget", "spec", "other"})
_OPPORTUNITY_OUTCOME = {"won": "won", "lost": "lost", "withdrawn": "withdrawn"}


def record_outcome(conn: Any, *, sales_quote_id: int, outcome: str, outcome_date: Any,
                   recorded_by: Optional[str], lost_reason: Optional[str] = None,
                   competitor_name: Optional[str] = None) -> Dict[str, Any]:
    if outcome not in QUOTE_OUTCOMES:
        raise ValueError(f"outcome must be one of {sorted(QUOTE_OUTCOMES)}")
    if lost_reason is not None:
        if outcome != "lost":
            raise ValueError("lost_reason is only recorded for a lost quote")
        if lost_reason not in LOST_REASONS:
            raise ValueError(f"lost_reason must be one of {sorted(LOST_REASONS)}")
    cur = dict_cursor(conn)
    try:
        cur.execute("SELECT status, total_ex_tax, total_margin FROM proc.bp_sales_quote "
                    "WHERE sales_quote_id = %s FOR UPDATE", (sales_quote_id,))
        q = cur.fetchone()
        if q is None:
            raise NotFound(f"quote {sales_quote_id} does not exist")
        if q["status"] != "issued":
            raise StateConflict(f"quote {sales_quote_id} is {q['status']}; only an issued "
                                "quote has an outcome")
        won = outcome == "won"
        try:
            cur.execute(
                "INSERT INTO proc.bp_sales_quote_outcome (sales_quote_id, outcome, outcome_date, "
                "lost_reason, competitor_name, won_value, won_margin, recorded_by) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING *",
                (sales_quote_id, outcome, outcome_date, lost_reason, competitor_name,
                 q["total_ex_tax"] if won else None, q["total_margin"] if won else None,
                 recorded_by))
        except psycopg2.errors.UniqueViolation:
            raise StateConflict(f"quote {sales_quote_id} already has an outcome") from None
        row = dict(cur.fetchone())
        if outcome == "expired":
            cur.execute("UPDATE proc.bp_sales_quote SET status = 'expired', "
                        "last_modified_date = now() WHERE sales_quote_id = %s", (sales_quote_id,))
        if outcome in _OPPORTUNITY_OUTCOME:
            cur.execute(
                "UPDATE proc.bp_sales_opportunity SET outcome = %s, last_modified_date = now() "
                "WHERE outcome = 'open' AND sales_opportunity_id IN (SELECT sales_opportunity_id "
                "FROM proc.bp_sales_quote_line WHERE sales_quote_id = %s "
                "AND sales_opportunity_id IS NOT NULL)",
                (_OPPORTUNITY_OUTCOME[outcome], sales_quote_id))
    except Exception:
        conn.rollback()
        raise
    conn.commit()
    return row
