"""A matched reply is stored with the price it states, marked as read by rule.

Before this, the mailbox route stored every reply with no price: the reply to a
455.00 GBP counter-offer said "£470 total" and the negotiation that followed
recorded "Supplier offer missing or invalid".
"""
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from repositories.supplier_response_repo import SupplierResponseRow
from services import email_watcher
from src.services.negotiation.reply_offer import METHOD


@pytest.fixture
def stored(monkeypatch):
    calls = {"inserted": [], "provenance": []}
    monkeypatch.setattr(
        email_watcher.supplier_response_repo, "insert_response",
        lambda row: calls["inserted"].append(row),
    )
    monkeypatch.setattr(
        email_watcher.supplier_response_repo, "record_extraction",
        lambda **kw: calls["provenance"].append(kw) or True,
    )
    return calls


def _row(text, **kw):
    return SupplierResponseRow(
        workflow_id="wf-1", unique_id="PROC-WF-1", supplier_id="SUP-1",
        response_text=text, received_time=datetime(2026, 10, 12, 7, 35, tzinfo=timezone.utc),
        response_message_id="<reply@mail>", **kw,
    )


def test_the_stated_price_is_stored_with_the_reply(stored):
    email_watcher._store_matched_response(_row("Final price is £470 total.\n\n> We propose 455.00 GBP."))
    (row,) = stored["inserted"]
    assert (row.price, row.currency) == (Decimal("470"), "GBP")


def test_how_it_was_read_is_recorded_after_the_reply_is_stored(stored):
    email_watcher._store_matched_response(_row("Final price is £470 total."))
    (stamp,) = stored["provenance"]
    assert stamp["method"] == METHOD and stamp["model"] is None
    assert (stamp["workflow_id"], stamp["unique_id"], stamp["response_message_id"]) == (
        "wf-1", "PROC-WF-1", "<reply@mail>")


def test_a_reply_with_no_price_is_stored_untouched_and_unstamped(stored):
    email_watcher._store_matched_response(_row("We will come back to you.\n\n> We propose 455.00 GBP."))
    (row,) = stored["inserted"]
    assert row.price is None and row.currency is None
    assert stored["provenance"] == []


def test_a_price_already_on_the_row_is_never_overwritten(stored):
    email_watcher._store_matched_response(_row("Final price is £470 total.", price=Decimal("123.45")))
    (row,) = stored["inserted"]
    assert row.price == Decimal("123.45")
    assert stored["provenance"] == []
