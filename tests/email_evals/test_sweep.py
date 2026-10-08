"""The abandoned-draft sweep: a draft nobody sent and nobody abandoned is closed after a governed period, and ONLY when
the product tables confirm it was not sent. Real Postgres.

The costly mistake is the false abandon: marking a draft abandoned that was in fact sent (because recording the send is
best-effort and can fail). Every test about skipping exists to stop that.
"""

from datetime import datetime, timedelta, timezone

import pytest

from src.services.draft_assurance import capture, metrics, sweep
from tests.email_evals.test_learning import DRAFT, db, engine, sent, count  # noqa: F401  (fixtures)

NOW = datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc)
RULES = {"abandon_after_days": 14, "batch_size": 500}


@pytest.fixture
def clean(db):
    with db.cursor() as cur:
        cur.execute("TRUNCATE email_agent.bp_draft_outcome, email_agent.bp_draft_capture RESTART IDENTITY CASCADE")
        cur.execute("TRUNCATE proc.draft_rfq_emails, proc.workflow_email_tracking")
    return db


def draft(db, age_days=30, product="unsent", tracking=False):
    """A captured draft, ``age_days`` old, with a product-table row saying what happened to it."""
    uid = sent(db, record=False)
    with db.cursor() as cur:
        cur.execute("UPDATE email_agent.bp_draft_capture SET captured_at = %s WHERE unique_id = %s", (NOW - timedelta(days=age_days), uid))
        if product == "unsent":
            cur.execute("INSERT INTO proc.draft_rfq_emails (rfq_id, subject, body, unique_id, sent) VALUES ('R', 's', 'b', %s, false)", (uid,))
        elif product == "sent":
            cur.execute("INSERT INTO proc.draft_rfq_emails (rfq_id, subject, body, unique_id, sent, sent_on) VALUES ('R', 's', 'b', %s, true, now())", (uid,))
        elif product == "sent_on_only":
            cur.execute("INSERT INTO proc.draft_rfq_emails (rfq_id, subject, body, unique_id, sent, sent_on) VALUES ('R', 's', 'b', %s, false, now())", (uid,))
        if tracking:
            cur.execute("INSERT INTO proc.workflow_email_tracking (workflow_id, unique_id, dispatch_key, dispatched_at) VALUES ('wf', %s, 'k', now())", (uid,))
    return uid


def outcomes(db, uid):
    with db.cursor() as cur:
        cur.execute("SELECT o.outcome, o.abandoned_by, o.abandon_reason FROM email_agent.bp_draft_outcome o JOIN email_agent.bp_draft_capture c "
                    "ON c.capture_id = o.capture_id WHERE c.unique_id = %s ORDER BY o.outcome_id", (uid,))
        return cur.fetchall()


def run(db, rules=RULES):
    return sweep.sweep(db, db, rules, now=NOW)


# --- what it closes ---------------------------------------------------------------------------------------------------------

def test_a_stale_draft_the_product_confirms_was_not_sent_is_abandoned_by_the_system(clean):
    uid = draft(clean)
    assert run(clean) == {"examined": 1, "abandoned": 1, "skipped_sent": 0, "skipped_unverifiable": 0}
    ((outcome, by, reason),) = outcomes(clean, uid)
    assert outcome == "abandoned" and by == "system:draft-sweep" and "14 days" in reason


def test_a_young_draft_is_left_alone(clean):
    uid = draft(clean, age_days=3)
    assert run(clean)["examined"] == 0 and outcomes(clean, uid) == []


def test_a_draft_exactly_at_the_limit_is_not_yet_stale(clean):
    uid = draft(clean, age_days=14)
    assert run(clean)["abandoned"] == 0 and outcomes(clean, uid) == []


@pytest.mark.parametrize("product,tracking", [("sent", False), ("sent_on_only", False), ("unsent", True)])
def test_a_draft_that_was_in_fact_sent_is_never_abandoned(clean, product, tracking):
    uid = draft(clean, product=product, tracking=tracking)
    r = run(clean)
    assert (r["abandoned"], r["skipped_sent"]) == (0, 1) and outcomes(clean, uid) == []


def test_a_draft_the_product_has_no_record_of_is_not_abandoned_because_nothing_is_confirmed(clean):
    uid = draft(clean, product="none")
    r = run(clean)
    assert (r["abandoned"], r["skipped_unverifiable"]) == (0, 1) and outcomes(clean, uid) == []


def test_a_draft_with_any_outcome_is_not_examined(clean):
    sent_uid, gone_uid = draft(clean), draft(clean)
    capture.record_sent(clean, sent_uid, DRAFT, sent_by="u1")
    capture.record_abandoned(clean, gone_uid, "nick", "changed mind")
    assert run(clean)["examined"] == 0
    assert [o[0] for o in outcomes(clean, sent_uid)] == ["sent"] and [o[1] for o in outcomes(clean, gone_uid)] == ["nick"]


def test_only_the_latest_regeneration_is_considered_and_the_replaced_one_is_not_called_abandoned(clean):
    uid = draft(clean)
    with clean.cursor() as cur:
        cur.execute("INSERT INTO email_agent.bp_draft_capture (unique_id, family_id, assurance_status, draft_text, draft_hash, captured_at) "
                    "VALUES (%s, 'negotiation_counter', 'verified', 'v2', 'h2', %s)", (uid, NOW - timedelta(days=20)))
    r = run(clean)
    assert r["examined"] == 1 and r["abandoned"] == 1
    assert len(outcomes(clean, uid)) == 1                                           # one outcome for the one live draft


def test_running_twice_closes_nothing_the_second_time(clean):
    draft(clean); draft(clean)
    assert run(clean)["abandoned"] == 2
    assert run(clean) == {"examined": 0, "abandoned": 0, "skipped_sent": 0, "skipped_unverifiable": 0}
    assert count(clean, "bp_draft_outcome") == 2


def test_the_batch_limit_is_respected_and_the_rest_waits_for_the_next_run(clean):
    for _ in range(5):
        draft(clean)
    assert run(clean, {"abandon_after_days": 14, "batch_size": 2})["abandoned"] == 2
    assert run(clean, {"abandon_after_days": 14, "batch_size": 2})["abandoned"] == 2
    assert run(clean, {"abandon_after_days": 14, "batch_size": 2})["abandoned"] == 1


# --- failing safe -----------------------------------------------------------------------------------------------------------

def test_if_the_product_check_cannot_be_made_nothing_is_abandoned(clean):
    uid = draft(clean)

    class Broken:
        def cursor(self):
            raise RuntimeError("product tables unreachable")

    r = sweep.sweep(clean, Broken(), RULES, now=NOW)
    assert r["abandoned"] == 0 and "unreachable" in r["error"] and outcomes(clean, uid) == []


def test_missing_or_bad_rules_mean_the_sweep_refuses_and_abandons_nothing(clean):
    uid = draft(clean)
    for bad in ({}, {"abandon_after_days": 0, "batch_size": 5}, {"abandon_after_days": "14", "batch_size": 5}, {"abandon_after_days": 14}):
        with pytest.raises(sweep.SweepRulesUnavailable):
            sweep.check_rules(bad)
    assert outcomes(clean, uid) == []


def test_the_period_is_read_from_the_policy_row(clean, engine):
    assert sweep.load_rules(engine) == {"abandon_after_days": 14, "batch_size": 500}
    for e in (None, type("E", (), {"get_policy": staticmethod(lambda s: None)})()):
        with pytest.raises(sweep.SweepRulesUnavailable):
            sweep.load_rules(e)


# --- what it means afterwards --------------------------------------------------------------------------------------------------

def test_a_swept_draft_counts_as_abandoned_in_the_metrics(clean):
    draft(clean)
    run(clean)
    (row,) = metrics.by_family(clean, bucket="month")
    assert (row["drafts"], row["sent"], row["abandoned"]) == (1, 0, 1)


def test_a_draft_sent_after_it_was_swept_is_counted_as_sent_not_abandoned(clean):
    uid = draft(clean)
    run(clean)
    assert capture.record_sent(clean, uid, DRAFT, reviewed_by="boss", sent_by="u1")
    (row,) = metrics.by_family(clean, bucket="month")
    assert (row["sent"], row["abandoned"]) == (1, 0)


# --- the scheduler ----------------------------------------------------------------------------------------------------------------

def test_the_sweep_job_is_registered_unless_switched_off(monkeypatch):
    from src.services.backend_scheduler import BackendScheduler
    for flag, expected in ((None, True), ("1", True), ("0", False), ("false", False)):
        sched = BackendScheduler.__new__(BackendScheduler)
        sched._jobs = {}
        registered = []
        sched.register_job = lambda name, fn, **kw: registered.append(name)
        monkeypatch.delenv("EMAIL_DRAFT_SWEEP_ENABLED", raising=False) if flag is None else monkeypatch.setenv("EMAIL_DRAFT_SWEEP_ENABLED", flag)
        sched._register_email_draft_sweep_job()
        assert bool(registered) is expected, flag


def test_closing_the_same_capture_twice_is_refused_even_when_called_directly(clean):
    uid = draft(clean)
    cid = sweep.find_stale(clean, 14, NOW, 10)[0]["capture_id"]
    assert sweep.abandon(clean, cid, 14) is True and sweep.abandon(clean, cid, 14) is False
    assert len(outcomes(clean, uid)) == 1


def test_a_batch_of_mixed_drafts_is_sorted_correctly_in_one_pass(clean):
    unsent, went, ghost = draft(clean), draft(clean, product="sent"), draft(clean, product="none")
    r = run(clean)
    assert r == {"examined": 3, "abandoned": 1, "skipped_sent": 1, "skipped_unverifiable": 1}
    assert [o[0] for o in outcomes(clean, unsent)] == ["abandoned"] and outcomes(clean, went) == [] and outcomes(clean, ghost) == []
