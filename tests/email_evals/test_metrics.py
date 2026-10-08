"""Metrics by family over time, against a real Postgres."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from src.services.draft_assurance import capture, metrics
from tests.email_evals.test_learning import DRAFT, db, sent  # noqa: F401  (db is a fixture)


def _by(rows, family):
    return [r for r in rows if r["family_id"] == family]


def test_edit_distance_and_fact_edit_rate_per_family(db):
    sent(db, family="negotiation_counter")                                           # untouched
    sent(db, family="negotiation_counter", edit=lambda t: t.replace("44.80", "40.00"))
    sent(db, family="email_family_free_prompt")
    rows = metrics.by_family(db)
    counter = _by(rows, "negotiation_counter")[0]
    assert counter["drafts"] == 2 and counter["sent"] == 2
    assert counter["mean_edit_distance"] > 0
    assert counter["fact_edit_rate"] in (0.0, 0.5)
    free = _by(rows, "email_family_free_prompt")[0]
    assert free["mean_edit_distance"] == 0.0 and free["fact_edit_rate"] == 0.0


def test_conflict_rate_ignores_drafts_whose_conflicts_were_never_captured(db):
    a = sent(db, record=False)
    b = sent(db, record=False)
    c = sent(db, record=False)
    with db.cursor() as cur:
        cur.execute("UPDATE email_agent.bp_draft_capture SET conflicts = %s::jsonb WHERE unique_id = %s",
                    (json.dumps([{"field": "price"}]), a))
        cur.execute("UPDATE email_agent.bp_draft_capture SET conflicts = '[]'::jsonb WHERE unique_id = %s", (b,))
        cur.execute("UPDATE email_agent.bp_draft_capture SET conflicts = NULL WHERE unique_id = %s", (c,))
    row = _by(metrics.by_family(db), "negotiation_counter")[0]
    assert row["conflicts_captured"] == 2 and row["fact_conflict_rate"] == 0.5   # NULL is not "no conflict"


def test_abandoned_and_unsent_are_counted_apart(db):
    sent(db)
    unsent = sent(db, record=False)
    gone = sent(db, record=False)
    assert capture.record_abandoned(db, gone, "u1", "changed mind")
    row = _by(metrics.by_family(db), "negotiation_counter")[0]
    assert (row["drafts"], row["sent"], row["abandoned"]) == (3, 1, 1)
    assert unsent


def test_periods_split_and_since_filters(db):
    old = sent(db)
    sent(db)
    with db.cursor() as cur:
        cur.execute("UPDATE email_agent.bp_draft_capture SET captured_at = now() - interval '40 days' WHERE unique_id = %s", (old,))
    assert len(metrics.by_family(db, bucket="month")) == 2
    recent = metrics.by_family(db, since=datetime.now(timezone.utc) - timedelta(days=7))
    assert sum(r["drafts"] for r in recent) == 1
    assert metrics.by_family(db, family="nope") == []


def test_unknown_bucket_is_refused_before_it_reaches_sql(db):
    with pytest.raises(ValueError):
        metrics.by_family(db, bucket="week'); DROP TABLE x;--")
