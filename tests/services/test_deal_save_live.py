"""Saving a deal names it and confirms it, in one step.

    PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest tests/services/test_deal_save_live.py

A deal's name is not stored on proc.bp_deal -- it is stamped onto every document
of the deal (stg/trgt headers and lines), the deal's document map and the upload
rows that tagged it. A rename that missed one would show two names for one deal
(bp_deal_overview takes max(deal_name)). Confirming is proc.bp_deal.is_tracked.

Runs on Test Deal TESTDEAL2026072901 inside a transaction that is rolled back.
"""

from __future__ import annotations

import os

import pytest

from src.services.db import get_conn
from src.services.deal_lifecycle import save_deal

_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in (
    "1", "true", "yes", "on")
pytestmark = pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")

_DEAL = "TESTDEAL2026072901"
_NAME = "Test Deal renamed by test_deal_save_live"


def _scalar(cur, sql, params):
    cur.execute(sql, params)
    return cur.fetchall()


def test_saving_renames_every_copy_of_the_name_and_confirms_the_deal():
    with get_conn() as conn:
        conn.autocommit = False
        try:
            save_deal(_DEAL, _NAME, actor="test-user", conn=conn)
            cur = conn.cursor()
            for table in ("proc.bp_quote_trgt", "proc.bp_quote_stg",
                          "proc.bp_quote_line_items_trgt", "proc.bp_purchase_order_trgt",
                          "proc.bp_invoice_trgt", "proc.bp_deal_document_map",
                          "proc.process_monitor"):
                names = {r[0] for r in _scalar(
                    cur, f"select distinct deal_name from {table} where deal_id=%s", (_DEAL,))}
                assert names == {_NAME}, (table, names)
            assert _scalar(cur, "select is_tracked from proc.bp_deal where deal_id=%s",
                           (_DEAL,)) == [(True,)]
            audit = _scalar(cur, "select summary, details->>'principal' from proc.bp_agent_actions "
                                 "where deal_id=%s and action_type='deal.save' "
                                 "order by created_at desc limit 1", (_DEAL,))
            assert audit and audit[0][1] == "test-user" and _NAME in audit[0][0], audit
        finally:
            conn.rollback()


def test_a_blank_name_is_refused_and_nothing_changes():
    with get_conn() as conn:
        conn.autocommit = False
        try:
            with pytest.raises(ValueError):
                save_deal(_DEAL, "   ", actor="test-user", conn=conn)
            cur = conn.cursor()
            assert _scalar(cur, "select is_tracked from proc.bp_deal where deal_id=%s",
                           (_DEAL,)) == [(False,)]
        finally:
            conn.rollback()


def test_a_deal_with_no_documents_is_not_invented():
    with get_conn() as conn:
        conn.autocommit = False
        try:
            with pytest.raises(LookupError):
                save_deal("NO-SUCH-DEAL-test_deal_save_live", _NAME, actor="t", conn=conn)
        finally:
            conn.rollback()
