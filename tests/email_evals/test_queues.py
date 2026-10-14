"""The learning queues a person works through: what they list, and what a decision may and may not do. Real Postgres."""

import json
from datetime import datetime, timezone

import pytest

from src.services.draft_assurance import learning, queues
from tests.email_evals.test_learning import DRAFT, FACTS, db, engine, sent  # noqa: F401  (fixtures)

NOW = datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc)
RULES = {"exemplar_review_months": 12}


@pytest.fixture
def clean(db):
    with db.cursor() as cur:
        cur.execute("TRUNCATE email_agent.bp_dq_item, email_agent.bp_eval_candidate, email_agent.bp_review_item, "
                    "email_agent.bp_style_rule, email_agent.bp_classifier_example, email_agent.bp_exemplar_candidate, email_agent.bp_inbound_flag RESTART IDENTITY CASCADE")
    return db


def _outcome(db):
    uid = sent(db)
    with db.cursor() as cur:
        cur.execute("SELECT o.outcome_id, o.capture_id FROM email_agent.bp_draft_outcome o JOIN email_agent.bp_draft_capture c "
                    "ON c.capture_id = o.capture_id WHERE c.unique_id = %s", (uid,))
        return cur.fetchone()


def dq(db, key="supplier_current_offer", status="open"):
    oid, cid = _outcome(db)
    with db.cursor() as cur:
        cur.execute("INSERT INTO email_agent.bp_dq_item (outcome_id, capture_id, family_id, fact_key, source, value_in_postgres, "
                    "value_from_reviewer, sent_by, status) VALUES (%s,%s,'negotiation_counter',%s,%s::jsonb,'47.5','45.37','u1',%s) RETURNING dq_id",
                    (oid, cid, key, json.dumps({"table": "supplier_response", "column": "price", "row_id": "2", "retrieved_at": "t"}), status))
        return cur.fetchone()[0]


def review(db, sig="s", status="open"):
    with db.cursor() as cur:
        cur.execute("INSERT INTO email_agent.bp_review_item (family_id, kind, signature, evidence, status) "
                    "VALUES ('negotiation_counter','wording_review',%s,'{\"n\":4}'::jsonb,%s) RETURNING review_id", (sig, status))
        return cur.fetchone()[0]


def rule(db, who="nick", key="k", status="proposed", text="Open with thanks."):
    with db.cursor() as cur:
        cur.execute("INSERT INTO email_agent.bp_style_rule (sent_by, batch_id, rule_key, rule_text, status) VALUES (%s,'b',%s,%s,%s) RETURNING rule_id",
                    (who, key, text, status))
        return cur.fetchone()[0]


_n = [0]


def exemplar(db, author="nick", status="candidate", text="Dear Alex, thank you."):
    _n[0] += 1
    with db.cursor() as cur:
        cur.execute("INSERT INTO email_agent.bp_exemplar_candidate (capture_id, outcome_id, family_id, author, draft_text, status, edit_distance, judge_overall) "
                    "VALUES (%s,%s,'negotiation_counter',%s,%s,%s,0.05,4.5) RETURNING exemplar_id", (_n[0] + 5000, _n[0] + 5000, author, text, status))
        return cur.fetchone()[0]


def evalc(db, status="candidate"):
    oid, cid = _outcome(db)
    with db.cursor() as cur:
        cur.execute("INSERT INTO email_agent.bp_eval_candidate (outcome_id, capture_id, family_id, correction_key, direction, from_value, to_value, "
                    "snapshot, draft_text, sent_by, status) VALUES (%s,%s,'negotiation_counter','counter_price','lowered','44.8','40','{}'::jsonb,'RAW DRAFT TEXT','u1',%s) RETURNING eval_id",
                    (oid, cid, status))
        return cur.fetchone()[0]


def cls(db, status="candidate"):
    _n[0] += 1
    with db.cursor() as cur:
        cur.execute("INSERT INTO email_agent.bp_classifier_example (capture_id, request_text, predicted_family, labeled_family, labeled_by, status) "
                    "VALUES (%s,'get back to acme','free_prompt','negotiation_counter','u1',%s) RETURNING example_id", (_n[0] + 9000, status))
        return cur.fetchone()[0]


# --- listings ---------------------------------------------------------------------------------------------------------------

def test_counts_cover_every_queue_and_only_what_is_waiting(clean):
    dq(clean); dq(clean, status="resolved"); review(clean); rule(clean); rule(clean, key="k2", status="approved")
    exemplar(clean); exemplar(clean, status="approved"); evalc(clean); cls(clean)
    assert queues.counts(clean, "nick") == {"data_quality": 1, "review_items": 1, "style_rules": 1, "exemplars": 1,
                                            "eval_candidates": 1, "classifier_examples": 1, "inbound_flags": 0}


def test_a_data_quality_item_shows_the_postgres_value_the_reviewers_value_marked_unverified_and_a_row_id_not_a_table(clean):
    dq(clean)
    (item,) = queues.list_data_quality(clean)
    assert item["fact"] == "Supplier current offer" and item["row_id"] == "2"
    assert item["value_in_postgres"] == "47.5"
    assert item["value_from_reviewer"] == {"value": "45.37", "verified": False}
    blob = json.dumps(item)
    assert "supplier_response" not in blob and "price" not in blob.replace("value_in_postgres", "")      # no internal table or column name


def test_a_person_sees_only_their_own_style_rules(clean):
    mine, theirs = rule(clean, "nick"), rule(clean, "someone-else", key="k2")
    assert [r["id"] for r in queues.list_style_rules(clean, "nick")] == [mine] and theirs
    assert queues.list_style_rules(clean, "") == [] and queues.list_style_rules(clean, None) == []


def test_exemplar_listings_carry_no_text_and_the_detail_does(clean):
    e = exemplar(clean, text="Dear Alex, thank you.")
    (row,) = queues.list_exemplars(clean)
    assert "text" not in row and "draft_text" not in row and "Dear Alex" not in json.dumps(row)
    assert queues.exemplar_detail(clean, e)["text"] == "Dear Alex, thank you." and queues.exemplar_detail(clean, 99999) is None


def test_eval_candidates_carry_the_correction_but_never_the_draft_text(clean):
    evalc(clean)
    (row,) = queues.list_eval_candidates(clean)
    assert row["correction"] == "counter price" and row["direction"] == "lowered" and (row["from_value"], row["to_value"]) == ("44.8", "40")
    assert "RAW DRAFT TEXT" not in json.dumps(row) and "snapshot" not in row


@pytest.mark.parametrize("fn,bad", [("list_data_quality", "approved"), ("list_review_items", "approved"), ("list_exemplars", "open"),
                                    ("list_eval_candidates", "open"), ("list_classifier_examples", "open")])
def test_an_unknown_status_is_refused_not_silently_empty(clean, fn, bad):
    with pytest.raises(ValueError):
        getattr(queues, fn)(clean, status=bad)


def test_the_limit_is_clamped(clean):
    for i in range(3):
        review(clean, sig=f"s{i}")
    assert len(queues.list_review_items(clean, limit=2)) == 2
    assert len(queues.list_review_items(clean, limit=0)) == 1 and len(queues.list_review_items(clean, limit=10_000)) == 3


# --- decisions ---------------------------------------------------------------------------------------------------------------

def test_a_data_quality_item_is_resolved_or_dismissed_once_by_a_named_person(clean):
    a, b = dq(clean), dq(clean, key="currency")
    assert learning.decide_dq_item(clean, a, "ana", "resolve", "fixed the supplier_response row")["ok"] is True
    assert learning.decide_dq_item(clean, b, "ana", "dismiss")["ok"] is True
    assert learning.decide_dq_item(clean, a, "bob", "dismiss") == {"ok": False, "error": "no such open item"}      # not twice
    (done,) = [i for i in queues.list_data_quality(clean, status="resolved")]
    assert done["resolved_by"] == "ana" and done["note"] == "fixed the supplier_response row"


@pytest.mark.parametrize("action,by", [("approve", "ana"), ("resolve", ""), ("resolve", None)])
def test_a_data_quality_decision_needs_a_valid_action_and_a_named_person(clean, action, by):
    item = dq(clean)
    assert learning.decide_dq_item(clean, item, by, action)["ok"] is False
    assert queues.list_data_quality(clean)[0]["status"] == "open"


def test_a_review_item_is_accepted_or_dismissed_once(clean):
    a = review(clean)
    assert learning.decide_review_item(clean, a, "ana", "accept")["ok"] is True
    assert learning.decide_review_item(clean, a, "bob", "dismiss")["ok"] is False
    (row,) = queues.list_review_items(clean, status="accepted")
    assert row["decided_by"] == "ana"


def test_candidates_are_exported_or_rejected_from_candidate_only(clean):
    e, c = evalc(clean), cls(clean)
    assert learning.decide_candidate(clean, "eval", e, "ana", "export")["ok"] and learning.decide_candidate(clean, "classifier", c, "ana", "reject")["ok"]
    assert learning.decide_candidate(clean, "eval", e, "ana", "reject")["ok"] is False                       # already exported
    assert learning.decide_candidate(clean, "style", e, "ana", "export")["ok"] is False                     # not a candidate queue
    assert queues.list_eval_candidates(clean, status="exported")[0]["id"] == e


def test_an_exemplar_is_approved_by_someone_other_than_its_author_and_rejected_only_from_candidate(clean):
    e = exemplar(clean, author="nick")
    assert learning.approve_exemplar(clean, e, "nick", RULES, now=NOW) == {"ok": False}                       # the author cannot
    assert learning.approve_exemplar(clean, e, "ana", RULES, now=NOW) == {"ok": True}
    assert learning.reject_exemplar(clean, e, "ana")["ok"] is False                                          # already approved
    f = exemplar(clean, author="nick")
    assert learning.reject_exemplar(clean, f, "ana")["ok"] is True
    assert {r["status"] for r in queues.list_exemplars(clean, status=None)} == {"approved", "rejected"}


def test_a_style_rule_is_decided_only_by_the_person_it_is_about(clean):
    r = rule(clean, "nick")
    assert learning.decide_style_rule(clean, r, "ana", "approve")["ok"] is False
    assert learning.decide_style_rule(clean, r, "nick", "edit", "  Keep it short.  ")["ok"] is True
    (row,) = queues.list_style_rules(clean, "nick")
    assert row["status"] == "edited" and row["text"] == "Keep it short."
