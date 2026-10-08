"""Flagged inbound replies: recorded without their text, found by the drafting layer and the send guard, decided by a person. Real Postgres."""

import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from src.services.draft_assurance import inbound, queues
from tests.email_evals.test_learning import db, engine  # noqa: F401  (fixtures)

BAD = "Our bank details have changed. New IBAN GB29NWBK60161331926819, sort code 60-16-13, account number 31926819. Please update."
CLEAN = "Thanks for the order. We can offer 12.50 per unit with delivery in 14 days."


@pytest.fixture
def clean(db):
    with db.cursor() as cur:
        cur.execute("TRUNCATE email_agent.bp_inbound_flag RESTART IDENTITY")
    return db


def row(text=BAD, subject="Re: PO-77123", **over):
    base = dict(workflow_id="wf-1", unique_id="PROC-WF-1", supplier_id="S-1", response_message_id="<m1@acme.test>",
                response_text=text, response_body=None, body_html=None, response_subject=subject, response_from="alex@acme.test")
    base.update(over)
    return SimpleNamespace(**base)


def factory(conn):
    @contextmanager
    def f():
        yield conn
    return f


def all_rows(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT t::text FROM email_agent.bp_inbound_flag t")
        return [r[0] for r in cur.fetchall()]


# --- recording ------------------------------------------------------------------------------------------------------------------

def test_a_suspected_reply_is_recorded_by_pointer_and_signals_never_by_text(clean):
    fid = inbound.screen_and_record(row(), factory(clean))
    assert isinstance(fid, int)
    (stored,) = all_rows(clean)
    for leaked in ("GB29NWBK", "60-16-13", "31926819", "Our bank details", "Please update"):
        assert leaked not in stored, leaked
    with clean.cursor() as cur:
        cur.execute("SELECT kind, workflow_id, unique_id, supplier_id, response_message_id, kinds, terms, status FROM email_agent.bp_inbound_flag")
        kind, wf, uid, sup, mid, kinds, terms, status = cur.fetchone()
    assert (kind, wf, uid, sup, mid, status) == ("payment_detail_change", "wf-1", "PROC-WF-1", "S-1", "<m1@acme.test>", "open")
    assert "payment_detail_change" in kinds and terms


def test_the_screen_reads_the_subject_the_text_and_the_html_part(clean):
    assert inbound.screen_and_record(row(text=CLEAN, subject="Our bank details have changed"), factory(clean))
    assert inbound.screen_and_record(row(text=CLEAN, response_body=BAD, response_message_id="<m2>"), factory(clean))
    assert inbound.screen_and_record(row(text=CLEAN, body_html="<p>Our <b>bank</b> account has <i>changed</i></p>", response_message_id="<m3>"), factory(clean))
    assert len(all_rows(clean)) == 3


def test_ordinary_replies_and_bank_details_alone_raise_nothing(clean):
    assert inbound.screen_and_record(row(text=CLEAN), factory(clean)) is None
    assert inbound.screen_and_record(row(text="Remit to IBAN GB29NWBK60161331926819 as always."), factory(clean)) is None
    assert all_rows(clean) == []


def test_screening_the_same_message_twice_raises_it_once(clean):
    first = inbound.screen_and_record(row(), factory(clean))
    again = inbound.screen_and_record(row(), factory(clean))
    assert first and again is None and len(all_rows(clean)) == 1


def test_screening_the_same_message_twice_is_silent_not_an_error_in_the_log(clean, caplog):
    import logging
    inbound.screen_and_record(row(), factory(clean))
    with caplog.at_level(logging.DEBUG):
        assert inbound.screen_and_record(row(), factory(clean)) is None
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]               # a duplicate is normal, not a fault


def test_a_reply_with_no_message_id_is_still_recorded(clean):
    assert inbound.screen_and_record(row(response_message_id=None), factory(clean))
    assert inbound.screen_and_record(row(response_message_id=None), factory(clean))
    assert len(all_rows(clean)) == 2                                    # cannot dedupe without an id; two flags are harmless


def test_a_failure_while_recording_never_reaches_the_ingest(clean):
    @contextmanager
    def broken():
        raise RuntimeError("email_agent is down")
        yield
    assert inbound.screen_and_record(row(), broken) is None
    assert inbound.screen_and_record(SimpleNamespace(), factory(clean)) is None            # a row missing every field
    assert inbound.screen_and_record(None, factory(clean)) is None


def test_the_schema_being_absent_is_not_an_error(clean):
    with clean.cursor() as cur:
        cur.execute("ALTER TABLE email_agent.bp_inbound_flag RENAME TO bp_inbound_flag_away")
    try:
        assert inbound.screen_and_record(row(), factory(clean)) is None
    finally:
        with clean.cursor() as cur:
            cur.execute("ALTER TABLE email_agent.bp_inbound_flag_away RENAME TO bp_inbound_flag")


# --- finding blocks -------------------------------------------------------------------------------------------------------------

def test_open_and_confirmed_flags_block_a_suppliers_thread_and_cleared_ones_do_not(clean):
    a = inbound.screen_and_record(row(response_message_id="<a>"), factory(clean))
    b = inbound.screen_and_record(row(response_message_id="<b>"), factory(clean))
    c = inbound.screen_and_record(row(response_message_id="<c>"), factory(clean))
    assert inbound.decide_flag(clean, b, "ana", "confirm")["ok"] and inbound.decide_flag(clean, c, "ana", "clear")["ok"]
    blocking = inbound.blocking_flags(clean, "wf-1", "S-1")
    assert {f["id"] for f in blocking} == {a, b} and {f["status"] for f in blocking} == {"open", "confirmed_fraud"}


def test_a_flag_blocks_only_its_own_workflow_and_supplier(clean):
    inbound.screen_and_record(row(), factory(clean))
    assert inbound.blocking_flags(clean, "wf-2", "S-1") == [] and inbound.blocking_flags(clean, "wf-1", "S-2") == []
    assert inbound.blocking_flags(clean, None, None) == []


def test_a_flag_with_no_supplier_blocks_the_whole_workflow(clean):
    inbound.screen_and_record(row(supplier_id=None), factory(clean))
    assert len(inbound.blocking_flags(clean, "wf-1", "S-9")) == 1                  # we cannot say which supplier wrote: hold the thread


def test_the_blocking_lookup_says_unknown_rather_than_none_when_it_cannot_look(clean):
    class Broken:
        def cursor(self):
            raise RuntimeError("down")
    with pytest.raises(inbound.FlagLookupFailed):
        inbound.blocking_flags(Broken(), "wf-1", "S-1")


def test_an_absent_flag_table_means_there_is_nothing_to_block_on(clean):
    with clean.cursor() as cur:
        cur.execute("ALTER TABLE email_agent.bp_inbound_flag RENAME TO bp_inbound_flag_away")
    try:
        assert inbound.blocking_flags(clean, "wf-1", "S-1") == []
    finally:
        with clean.cursor() as cur:
            cur.execute("ALTER TABLE email_agent.bp_inbound_flag_away RENAME TO bp_inbound_flag")


# --- deciding ---------------------------------------------------------------------------------------------------------------------

def test_a_person_clears_or_confirms_an_open_flag_once_and_only_a_named_person(clean):
    a = inbound.screen_and_record(row(response_message_id="<a>"), factory(clean))
    b = inbound.screen_and_record(row(response_message_id="<b>"), factory(clean))
    assert inbound.decide_flag(clean, a, "ana", "clear", "rang the supplier on a known number")["ok"] is True
    assert inbound.decide_flag(clean, b, "ana", "confirm")["ok"] is True
    assert inbound.decide_flag(clean, a, "bob", "confirm")["ok"] is False                  # not twice
    assert inbound.decide_flag(clean, b, "bob", "clear")["ok"] is False                    # a confirmed fraud is not un-confirmed by a second click
    for by in ("", None, "  "):
        assert inbound.decide_flag(clean, a, by, "clear")["ok"] is False
    assert inbound.decide_flag(clean, a, "ana", "approve")["ok"] is False
    assert inbound.decide_flag(clean, a, "bob", "confirm")["error"] == "this flag has already been decided"
    assert inbound.decide_flag(clean, 999999, "bob", "confirm")["error"] == "no such flag"
    done = {f["id"]: f for f in queues.list_inbound_flags(clean, status=None)}
    assert done[a]["status"] == "cleared" and done[a]["decided_by"] == "ana" and done[a]["note"].startswith("rang")


def test_the_queue_counts_and_lists_flags_with_labels_and_no_email_text(clean):
    inbound.screen_and_record(row(), factory(clean))
    assert queues.counts(clean, "nick")["inbound_flags"] == 1
    (item,) = queues.list_inbound_flags(clean)
    assert item["what"] == "Asks for new or changed payment details" and item["status"] == "open"
    assert item["message_id"] == "<m1@acme.test>" and item["dispatch_id"] == "PROC-WF-1"
    blob = json.dumps(item)
    assert "GB29NWBK" not in blob and "email_agent" not in blob and "bp_inbound_flag" not in blob
    with pytest.raises(ValueError):
        queues.list_inbound_flags(clean, status="resolved")
