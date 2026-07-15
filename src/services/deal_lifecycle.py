"""Draft <-> tracked lifecycle writes for proc.bp_deal (see 2026-07-15 spec)."""
from __future__ import annotations
from typing import Any
from src.services.db import get_conn


def _promote(c: Any, deal_id: str) -> None:
    cur = c.cursor()
    cur.execute(
        "insert into proc.bp_deal (deal_id, is_tracked, is_saved_reference, tracked_at) "
        "values (%s, true, false, now()) "
        "on conflict (deal_id) do update set "
        "is_tracked=true, is_saved_reference=false, tracked_at=now(), updated_at=now()",
        (str(deal_id),))
    # Advance any linked opportunities out of 'identified' when the deal is committed.
    cur.execute(
        "update proc.bp_opportunity set stage='negotiation', stage_updated_at=now(), "
        "updated_at=now() where deal_id=%s and stage='identified'",
        (str(deal_id),))


def _save_ref(c: Any, deal_id: str) -> None:
    cur = c.cursor()
    cur.execute(
        "insert into proc.bp_deal (deal_id, is_tracked, is_saved_reference) "
        "values (%s, false, true) "
        "on conflict (deal_id) do update set is_saved_reference=true, updated_at=now()",
        (str(deal_id),))


def _with_txn(fn, deal_id: str, conn: Any) -> None:
    if conn is None:
        with get_conn() as own:
            own.autocommit = False
            try:
                fn(own, deal_id); own.commit()
            except Exception:
                own.rollback(); raise
    else:
        fn(conn, deal_id)


def promote_deal(deal_id: str, conn: Any = None) -> None:
    _with_txn(_promote, deal_id, conn)


def save_reference(deal_id: str, conn: Any = None) -> None:
    _with_txn(_save_ref, deal_id, conn)
