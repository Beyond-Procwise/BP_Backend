"""Sent text and diffs: stored, restricted, masked, and aged out. Against a real Postgres.

The learning job needs the words people changed, so the text that went out is kept (decision 2026-10-08), but only
as long as the governed retention period, only for the writer role, and never with bank details in it.
"""

import json
from datetime import datetime, timedelta, timezone

import pytest

from src.services.draft_assurance import capture, retention
from tests.email_evals.test_learning import DRAFT, db, engine, sent, count  # noqa: F401  (db, engine are fixtures)

NOW = datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc)
EDITED = DRAFT.replace("44.80", "40.00")


def stored(db, uid):
    with db.cursor() as cur:
        cur.execute("SELECT t.sent_text, t.diff, t.redactions, t.text_hash FROM email_agent.bp_draft_sent_text t "
                    "JOIN email_agent.bp_draft_capture c ON c.capture_id = t.capture_id WHERE c.unique_id = %s", (uid,))
        return cur.fetchone()


def send(db, uid, body, days=90):
    return capture.record_sent(db, uid, body, reviewed_by="boss", sent_by="u1", retention_days=days)


# --- what is stored -----------------------------------------------------------------------------------------------

def test_the_sent_text_is_stored_with_a_diff_that_rebuilds_it_from_the_draft(db):
    uid = sent(db, record=False)
    assert send(db, uid, EDITED)
    text, diff, redactions, _ = stored(db, uid)
    assert text == capture.plain(EDITED)
    assert capture.apply_diff(capture.plain(DRAFT), diff) == text          # the diff is a true, complete diff
    assert any(op[0] == "ins" and "40.00" in op[1] for op in diff) and any(op[0] == "del" and "44.80" in op[1] for op in diff)
    assert redactions == {}


def test_an_unedited_send_stores_the_text_and_a_diff_of_pure_equality(db):
    uid = sent(db, record=False)
    send(db, uid, DRAFT)
    text, diff, _, _ = stored(db, uid)
    assert [op[0] for op in diff] == ["eq"] and text == capture.plain(DRAFT)


def test_the_derived_measures_are_still_recorded_exactly_as_before(db):
    uid = sent(db, record=False)
    send(db, uid, EDITED)
    with db.cursor() as cur:
        cur.execute("SELECT o.edit_distance, o.edit_class FROM email_agent.bp_draft_outcome o JOIN email_agent.bp_draft_capture c "
                    "ON c.capture_id = o.capture_id WHERE c.unique_id = %s", (uid,))
        distance, klass = cur.fetchone()
    assert klass == "reasoned" and float(distance) > 0


def test_without_a_retention_period_no_text_is_stored_but_the_outcome_still_is(db):
    uid = sent(db, record=False)
    assert capture.record_sent(db, uid, EDITED, reviewed_by="boss", sent_by="u1")            # no retention_days
    assert stored(db, uid) is None and count(db, "bp_draft_outcome") == 1


@pytest.mark.parametrize("days", [0, -5])
def test_a_non_positive_period_stores_no_text(db, days):
    uid = sent(db, record=False)
    send(db, uid, EDITED, days=days)
    assert stored(db, uid) is None


def test_sending_twice_stores_one_text(db):
    uid = sent(db, record=False)
    send(db, uid, EDITED)
    send(db, uid, EDITED)
    assert count(db, "bp_draft_sent_text") == 1


# --- bank details are never stored --------------------------------------------------------------------------------

SECRETS = ("GB29NWBK60161331926819", "60-16-13", "31926819")


def test_bank_details_are_masked_in_the_stored_text_the_diff_and_everywhere_else(db):
    body = DRAFT.replace("</p><p>Kind", " Pay to IBAN GB29NWBK60161331926819, sort code 60-16-13, account number 31926819.</p><p>Kind")
    uid = sent(db, record=False)
    send(db, uid, body)
    text, diff, redactions, _ = stored(db, uid)
    assert "[BANK DETAILS REMOVED]" in text and sum(redactions.values()) >= 2
    blob = json.dumps(diff)
    for secret in SECRETS:
        assert secret not in text and secret not in blob
    with db.cursor() as cur:                                     # and in NO table of the schema
        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'email_agent'")
        for (table,) in cur.fetchall():
            cur.execute(f"SELECT t::text FROM email_agent.{table} t")
            for (row,) in cur.fetchall():
                for secret in SECRETS:
                    assert secret not in row, (table, secret)


def test_a_draft_that_itself_carried_bank_details_is_masked_in_the_diff_too(db):
    uid = sent(db, record=False)
    with db.cursor() as cur:
        cur.execute("UPDATE email_agent.bp_draft_capture SET draft_text = draft_text || ' IBAN GB29NWBK60161331926819' WHERE unique_id = %s", (uid,))
    send(db, uid, DRAFT)
    text, diff, _, _ = stored(db, uid)
    assert "GB29NWBK60161331926819" not in json.dumps(diff) and "GB29NWBK60161331926819" not in text


def test_a_model_draft_that_contains_bank_details_is_masked_before_it_is_captured(db):
    a = {"family_id": "negotiation_counter", "family_version": 1, "mode": "shadow", "status": "needs_review", "facts": {},
         "conflicts": [], "reasoned": {}, "assumptions": [], "violations": [], "repaired": False, "carried_unverified": {},
         "unverified_figures": [], "assumption_items": [], "family_source": "declared", "ready": False,
         "accountability": {"initiated_by": "NegotiationAgent", "kind": "agent"}, "clarification": {}}
    cid = capture.record_draft(db, {"unique_id": "U-BANK", "workflow_id": "wf", "supplier_id": "S-1",
                                    "body": "<p>Please use IBAN GB29NWBK60161331926819 and sort code 60-16-13.</p>",
                                    "metadata": {"intent": "NEGOTIATION_COUNTER"}, "assurance": a})
    assert cid
    with db.cursor() as cur:
        cur.execute("SELECT draft_text FROM email_agent.bp_draft_capture WHERE capture_id = %s", (cid,))
        text = cur.fetchone()[0]
    assert "[BANK DETAILS REMOVED]" in text and "GB29NWBK" not in text and "60-16-13" not in text


# --- the reviewer's screen never carries raw text ------------------------------------------------------------------

def test_the_reviewer_view_carries_neither_the_sent_text_nor_the_models_draft(db):
    uid = sent(db, record=False)
    send(db, uid, EDITED)
    view = json.dumps(capture.to_view(capture.load_raw(db, uid)), default=str)
    assert "40.00" not in view and "Thank you for your latest offer" not in view


# --- retention ----------------------------------------------------------------------------------------------------

def _age(db, uid, days, now=NOW):
    with db.cursor() as cur:
        cur.execute("UPDATE email_agent.bp_draft_sent_text SET stored_at = %s WHERE capture_id IN "
                    "(SELECT capture_id FROM email_agent.bp_draft_capture WHERE unique_id = %s)", (now - timedelta(days=days), uid))
        cur.execute("UPDATE email_agent.bp_draft_capture SET captured_at = %s WHERE unique_id = %s", (now - timedelta(days=days), uid))


def test_the_purge_deletes_text_older_than_the_period_and_keeps_newer(db):
    old, new = sent(db, record=False), sent(db, record=False)
    send(db, old, EDITED)
    send(db, new, EDITED)
    _age(db, old, 100)
    _age(db, new, 10)
    report = retention.purge_expired(db, 90, now=NOW)
    assert report["sent_text_deleted"] == 1 and stored(db, old) is None and stored(db, new) is not None


def test_the_purge_blanks_the_models_draft_text_but_keeps_every_derived_value(db):
    uid = sent(db, record=False)
    send(db, uid, EDITED)
    _age(db, uid, 100)
    with db.cursor() as cur:
        cur.execute("SELECT facts, reasoned, family_id, draft_hash FROM email_agent.bp_draft_capture WHERE unique_id = %s", (uid,))
        before = cur.fetchone()
    assert retention.purge_expired(db, 90, now=NOW)["draft_text_blanked"] == 1
    with db.cursor() as cur:
        cur.execute("SELECT draft_text, text_expired_at, facts, reasoned, family_id, draft_hash FROM email_agent.bp_draft_capture WHERE unique_id = %s", (uid,))
        text, expired, *after = cur.fetchone()
        cur.execute("SELECT edit_distance, edit_class FROM email_agent.bp_draft_outcome o JOIN email_agent.bp_draft_capture c ON c.capture_id = o.capture_id WHERE c.unique_id = %s", (uid,))
        distance, klass = cur.fetchone()
    assert text == "" and expired is not None and tuple(after) == before
    assert klass == "reasoned" and float(distance) > 0                    # the derived measures survive


def test_purging_twice_changes_nothing_the_second_time(db):
    uid = sent(db, record=False)
    send(db, uid, EDITED)
    _age(db, uid, 100)
    retention.purge_expired(db, 90, now=NOW)
    assert retention.purge_expired(db, 90, now=NOW) == {"sent_text_deleted": 0, "draft_text_blanked": 0}


def test_a_draft_sent_after_its_text_expired_records_no_false_distance(db):
    uid = sent(db, record=False)
    _age(db, uid, 100)
    with db.cursor() as cur:
        cur.execute("UPDATE email_agent.bp_draft_sent_text SET stored_at = stored_at")      # no-op: nothing stored yet
    retention.purge_expired(db, 90, now=NOW)
    assert send(db, uid, EDITED)
    with db.cursor() as cur:
        cur.execute("SELECT o.edit_distance, o.edit_class FROM email_agent.bp_draft_outcome o JOIN email_agent.bp_draft_capture c ON c.capture_id = o.capture_id WHERE c.unique_id = %s", (uid,))
        assert cur.fetchone() == (None, None)                              # NULL = not captured, never a made-up 1.0
    text, diff, _, _ = stored(db, uid)
    assert text == capture.plain(EDITED) and diff is None                  # the sent text is kept; no draft to diff against


@pytest.mark.parametrize("days", [0, -1, None, "90", True])
def test_the_purge_refuses_a_period_that_is_not_a_positive_number(db, days):
    with pytest.raises(ValueError):
        retention.purge_expired(db, days, now=NOW)


# --- the period is governed ---------------------------------------------------------------------------------------

def test_the_period_is_read_from_the_policy_row(db, engine):
    assert retention.load_rules(engine) == {"raw_text_days": 90}
    assert retention.raw_text_days(engine) == 90


def test_a_missing_or_bad_period_makes_the_reader_refuse_and_the_helper_return_none():
    none = type("E", (), {"get_policy": staticmethod(lambda slug: None)})()
    bad = type("E", (), {"get_policy": staticmethod(lambda slug: {"details": {"rules": {"raw_text_days": 0}}})})()
    boom = type("E", (), {"get_policy": staticmethod(lambda slug: (_ for _ in ()).throw(RuntimeError("down")))})()
    for engine_ in (None, none, bad, boom):
        with pytest.raises(retention.RetentionRulesUnavailable):
            retention.load_rules(engine_)
        assert retention.raw_text_days(engine_) is None


# --- the scheduler -------------------------------------------------------------------------------------------------

def test_the_retention_job_is_registered_unless_switched_off(monkeypatch):
    from src.services.backend_scheduler import BackendScheduler
    for flag, expected in ((None, True), ("1", True), ("0", False), ("false", False)):
        sched = BackendScheduler.__new__(BackendScheduler)
        sched._jobs = {}
        registered = []
        sched.register_job = lambda name, fn, **kw: registered.append(name)
        if flag is None:
            monkeypatch.delenv("EMAIL_TEXT_RETENTION_ENABLED", raising=False)
        else:
            monkeypatch.setenv("EMAIL_TEXT_RETENTION_ENABLED", flag)
        sched._register_email_text_retention_job()
        assert bool(registered) is expected, flag
