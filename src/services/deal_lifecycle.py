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


# Where a deal's name lives. It is not a column of proc.bp_deal: it is stamped onto
# each document (_persist_deal writes stg/trgt headers and lines), the deal's document
# map and the upload rows that tagged the deal. bp_deal_overview reads max(deal_name),
# so a rename that missed any of these would show the old name or a mix of both. _raw
# is not written here: the assignment pass mirrors deal_name down from _trgt.
_NAME_TABLES = (
    "proc.bp_quote_stg", "proc.bp_quote_trgt",
    "proc.bp_quote_line_items_stg", "proc.bp_quote_line_items_trgt",
    "proc.bp_purchase_order_stg", "proc.bp_purchase_order_trgt",
    "proc.bp_po_line_items_stg", "proc.bp_po_line_items_trgt",
    "proc.bp_invoice_stg", "proc.bp_invoice_trgt",
    "proc.bp_invoice_line_items_stg", "proc.bp_invoice_line_items_trgt",
    "proc.bp_deal_document_map", "proc.process_monitor",
)
_NAME_MAX = 200


def _save(c: Any, deal_id: str, name: str, actor: str | None) -> None:
    from src.services import agent_actions
    cur = c.cursor()
    cur.execute("select 1 from proc.bp_deal_document_map where deal_id=%s limit 1", (deal_id,))
    if not cur.fetchall():
        raise LookupError(f"no documents on deal {deal_id}")
    cur.execute("select max(deal_name) from proc.bp_deal_document_map where deal_id=%s",
                (deal_id,))
    before = (cur.fetchall() or [[None]])[0][0]
    renamed = {}
    for table in _NAME_TABLES:
        cur.execute(f"update {table} set deal_name=%s where deal_id=%s "
                    f"and deal_name is distinct from %s", (name, deal_id, name))
        renamed[table] = cur.rowcount or 0
    _promote(c, deal_id)
    agent_actions.record_action_or_fail(
        phase="consolidation", action_type="deal.save", agent="user", conn=c,
        deal_id=deal_id, status="saved",
        summary=f"Deal saved as '{name}' and confirmed (was '{before}')",
        details={"principal": actor, "name_before": before, "name_after": name,
                 "rows_renamed": renamed},
    )


def save_deal(deal_id: str, name: str, *, actor: str | None, conn: Any = None) -> None:
    """Name a deal and confirm it (is_tracked) in one transaction.

    Confirming is what makes the deal authoritative for deal assignment
    (deal_assignment_service.is_established_deal): new documents uploaded to it are
    then stamped onto it instead of waiting in the grouping proposals.
    """
    clean = " ".join(str(name or "").split())
    if not clean:
        raise ValueError("a deal needs a name")
    if len(clean) > _NAME_MAX:
        raise ValueError(f"a deal name is at most {_NAME_MAX} characters")
    deal_id = str(deal_id)
    if conn is None:
        with get_conn() as own:
            own.autocommit = False
            try:
                _save(own, deal_id, clean, actor); own.commit()
            except Exception:
                own.rollback(); raise
    else:
        _save(conn, deal_id, clean, actor)
