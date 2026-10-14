"""Stage 6: the learning job, against a real Postgres, driven by real captures and real sent outcomes."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from evals.email import runner
from src.services.draft_assurance import capture, learning

DRAFT = ("<p>Thank you for your latest offer of 47.50 GBP. We would like to propose 44.80 GBP for this order. "
         "Could you please confirm whether you can agree this price by 30 October 2026?</p>"
         "<p>Kind regards, Procurement Team</p>")
FACTS = {"supplier_current_offer": {"value": "47.5000", "label": "Supplier's latest offer", "table": "supplier_response",
                                    "column": "price", "row_id": "2", "retrieved_at": "t"}}
REASONED = {"counter_price": {"value": "44.8", "basis": ["supplier_current_offer"]}}
NOW = datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def db(eval_db):
    with eval_db.cursor() as cur:
        cur.execute("TRUNCATE email_agent.bp_exemplar_candidate, email_agent.bp_classifier_example, email_agent.bp_style_rule, "
                    "email_agent.bp_review_item, email_agent.bp_eval_candidate, email_agent.bp_dq_item, "
                    "email_agent.bp_draft_outcome, email_agent.bp_draft_capture RESTART IDENTITY CASCADE")
    return eval_db


@pytest.fixture
def engine(db):
    return runner.make_engine(db)


_n = [0]


def sent(db, *, user="u1", family="negotiation_counter", edit=lambda t: t, facts=FACTS, reasoned=REASONED, judge=4.5,
         items=(), resolution=None, carried=None, unverified=(), status="verified", request=None, record=True,
         clarification=None, violations=()):
    """A draft that was captured and then sent (with ``edit`` applied). Returns the unique id."""
    _n[0] += 1
    uid = f"U-{_n[0]}"
    a = {"family_id": family, "family_version": 1, "mode": "shadow", "status": status, "facts": facts, "conflicts": [],
         "reasoned": reasoned, "assumptions": [], "violations": list(violations), "repaired": False, "carried_unverified": carried or {},
         "unverified_figures": list(unverified), "assumption_items": list(items), "request_text": request,
         "judge": {"status": "scored", "overall": judge} if judge is not None else None,
         "tone": {"variables": {"escalation_level": 2}, "sources": {"escalation_level": {"source": "postgres"}}},
         "exemplars": {"ids": [], "scope": "none"}, "family_source": "declared", "ready": True,
         "accountability": {"initiated_by": "NegotiationAgent", "kind": "agent"}, "clarification": clarification or {}}
    cid = capture.record_draft(db, {"unique_id": uid, "workflow_id": "wf-1", "supplier_id": "S-1", "body": DRAFT,
                                    "metadata": {"intent": "NEGOTIATION_COUNTER"}, "assurance": a})
    assert cid, "capture failed"
    if resolution:
        with db.cursor() as cur:
            cur.execute("UPDATE email_agent.bp_draft_capture SET assumptions_resolution = %s::jsonb WHERE capture_id = %s",
                        (json.dumps(resolution), cid))
    if record:
        assert capture.record_sent(db, uid, edit(DRAFT), reviewed_by="boss", sent_by=user)
    return uid


def count(db, table, where="true"):
    with db.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM email_agent.{table} WHERE {where}")
        return cur.fetchone()[0]


def run(db, engine, now=NOW):
    return learning.run_learning(db, engine, now=now)


# --- the rules ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("key", list(learning.INT_KEYS + learning.NUM_KEYS))
def test_a_missing_threshold_stops_the_job_and_writes_nothing(db, key):
    sent(db)

    class Eng:
        def get_policy(self, slug):
            rules = json.loads(open("deploy/sql/2026-10-09_email_agent_learning.sql").read().split("$json$")[1])["rules"]
            rules.pop(key)
            return {"details": {"rules": rules}}
    with pytest.raises(learning.LearningRulesUnavailable):
        learning.run_learning(db, Eng(), now=NOW)
    assert count(db, "bp_draft_outcome", "learning_processed_at IS NOT NULL") == 0


def test_no_policy_or_a_broken_store_stops_the_job(db):
    class Gone:
        def get_policy(self, slug): return None
    class Boom:
        def get_policy(self, slug): raise RuntimeError("down")
    for eng in (Gone(), Boom(), None):
        with pytest.raises(learning.LearningRulesUnavailable):
            learning.run_learning(db, eng, now=NOW)


def test_the_real_row_loads(engine):
    r = learning.load_rules(engine)
    assert (r["min_distinct_users"], r["style_window_edits"], r["max_style_rules"], r["exemplar_max_distance"],
            r["exemplar_min_judge"], r["exemplar_review_months"]) == (3, 50, 15, 0.15, 4.0, 12)


# --- route 1: a fact changed ------------------------------------------------------------------------------------

def test_a_changed_fact_goes_to_the_data_quality_queue_with_the_row_it_came_from(db, engine):
    sent(db, edit=lambda t: t.replace("47.50", "45.37"))
    rep = run(db, engine)
    assert rep["dq_items"] == 1
    with db.cursor() as cur:
        cur.execute("SELECT fact_key, source, value_in_postgres, value_from_reviewer, sent_by, status FROM email_agent.bp_dq_item")
        key, source, was, now, by, status = cur.fetchone()
    assert key == "supplier_current_offer" and source["row_id"] == "2" and source["table"] == "supplier_response"
    assert float(was) == 47.5 and now == "45.37" and by == "u1" and status == "open"


def test_a_changed_fact_is_never_learned_from_anywhere_else(db, engine):
    sent(db, edit=lambda t: t.replace("47.50", "45.37"))
    run(db, engine)
    for table in ("bp_eval_candidate", "bp_exemplar_candidate", "bp_style_rule", "bp_classifier_example", "bp_review_item"):
        assert count(db, table) == 0, table
        with db.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM email_agent.{table} t WHERE t::text LIKE '%45.37%'")
            assert cur.fetchone()[0] == 0, f"the reviewer's value leaked into {table}"
    assert count(db, "bp_dq_item", "value_from_reviewer = '45.37'") == 1


def test_a_changed_fact_does_not_count_toward_the_reviewers_style(db, engine):
    for _ in range(12):
        sent(db, user="u1", edit=lambda t: t.replace("47.50", "45.37").replace("Kind regards, Procurement Team", "Cheers"))
    run(db, engine)
    assert count(db, "bp_style_rule") == 0


def test_a_changed_figure_that_is_no_known_fact_is_not_a_fact_edit(db, engine):
    sent(db, edit=lambda t: t.replace("47.50", "45.37"), facts={"other": {"value": "9", "row_id": "1"}})
    run(db, engine)
    assert count(db, "bp_dq_item") == 0
    assert _one(db, "SELECT learning_routes FROM email_agent.bp_draft_outcome") == []


def test_cutting_the_sentence_that_held_a_fact_is_an_omission_not_a_correction(db, engine):
    sent(db, edit=lambda t: t.replace("Thank you for your latest offer of 47.50 GBP. ", ""))
    run(db, engine)
    assert count(db, "bp_dq_item") == 0
    assert "fact" not in _one(db, "SELECT learning_routes FROM email_agent.bp_draft_outcome")


# --- route 2: a judgement of ours changed -------------------------------------------------------------------------

def test_a_corrected_price_becomes_an_eval_candidate_with_its_direction(db, engine):
    sent(db, edit=lambda t: t.replace("44.80", "46.00"))
    sent(db, edit=lambda t: t.replace("44.80", "43.00"), user="u2")
    rep = run(db, engine)
    assert rep["eval_candidates"] == 2 and rep["dq_items"] == 0
    with db.cursor() as cur:
        cur.execute("SELECT correction_key, direction, from_value, to_value, snapshot->'facts' FROM email_agent.bp_eval_candidate ORDER BY eval_id")
        rows = cur.fetchall()
    assert [(r[0], r[1]) for r in rows] == [("counter_price", "raised"), ("counter_price", "lowered")]
    assert rows[0][2] == "44.8" and rows[0][3] == "46" and rows[0][4] == {"supplier_current_offer": "47.5000"}


def test_three_different_reviewers_correcting_the_same_way_opens_a_review_item(db, engine):
    for u in ("u1", "u2", "u3"):
        sent(db, user=u, edit=lambda t: t.replace("44.80", "46.00"))
    rep = run(db, engine)
    assert rep["review_items"] == 1
    with db.cursor() as cur:
        cur.execute("SELECT family_id, kind, signature, status, evidence->>'distinct_reviewers' FROM email_agent.bp_review_item")
        assert cur.fetchone() == ("negotiation_counter", "reasoning_guidance", "counter_price:raised", "open", "3")


def test_two_reviewers_is_not_enough(db, engine):
    for u in ("u1", "u2"):
        sent(db, user=u, edit=lambda t: t.replace("44.80", "46.00"))
    assert run(db, engine)["review_items"] == 0


def test_one_reviewer_three_times_is_one_voice_not_three(db, engine):
    for _ in range(3):
        sent(db, user="u1", edit=lambda t: t.replace("44.80", "46.00"))
    assert run(db, engine)["review_items"] == 0


def test_opposite_corrections_do_not_add_up(db, engine):
    sent(db, user="u1", edit=lambda t: t.replace("44.80", "46.00"))
    sent(db, user="u2", edit=lambda t: t.replace("44.80", "43.00"))
    sent(db, user="u3", edit=lambda t: t.replace("44.80", "46.00"))
    assert run(db, engine)["review_items"] == 0


def test_corrections_outside_the_window_do_not_count(db, engine):
    for u in ("u1", "u2", "u3"):
        sent(db, user=u, edit=lambda t: t.replace("44.80", "46.00"))
    assert run(db, engine, now=NOW + timedelta(days=400))["review_items"] == 0


def test_an_open_review_item_is_not_opened_twice(db, engine):
    for u in ("u1", "u2", "u3"):
        sent(db, user=u, edit=lambda t: t.replace("44.80", "46.00"))
    run(db, engine)
    sent(db, user="u4", edit=lambda t: t.replace("44.80", "46.00"))
    assert run(db, engine)["review_items"] == 0 and count(db, "bp_review_item") == 1


# --- route 3 and 4: wording ----------------------------------------------------------------------------------------

REWRITE = lambda t: ("<p>Hi, quick one. Can you do 44.80 GBP on this? Shout if not and we can chat it through properly. Cheers</p>")


def test_a_wording_edit_is_routed_as_wording_and_a_figure_only_edit_is_not(db, engine):
    sent(db, edit=REWRITE)
    sent(db, edit=lambda t: t.replace("44.80", "46.00"))
    run(db, engine)
    with db.cursor() as cur:
        cur.execute("SELECT learning_routes FROM email_agent.bp_draft_outcome ORDER BY outcome_id")
        assert [r[0] for r in cur.fetchall()] == [["wording"], ["reasoned"]]


def test_three_reviewers_heavily_rewriting_one_family_opens_a_wording_review_that_is_honest_about_its_limits(db, engine):
    for u in ("u1", "u2", "u3"):
        sent(db, user=u, edit=REWRITE)
    run(db, engine)
    with db.cursor() as cur:
        cur.execute("SELECT signature, evidence->>'meaning' FROM email_agent.bp_review_item WHERE kind = 'wording_review'")
        sig, meaning = cur.fetchone()
    assert sig == "heavy_rewrite" and "NOT 'the same correction'" in meaning


def test_two_reviewers_rewriting_is_not_a_pattern(db, engine):
    for u in ("u1", "u2"):
        sent(db, user=u, edit=REWRITE)
    assert run(db, engine)["review_items"] == 0


def test_light_wording_changes_do_not_open_a_review(db, engine):
    for u in ("u1", "u2", "u3"):
        sent(db, user=u, edit=lambda t: t.replace("Thank you", "Thanks"))
    assert run(db, engine)["review_items"] == 0


def test_a_reviewer_who_shortens_every_draft_gets_a_proposed_rule_with_the_numbers_behind_it(db, engine):
    short = lambda t: "<p>Offer noted at 47.50 GBP. Can you do 44.80 GBP by 30 October 2026?</p>"
    for _ in range(10):
        sent(db, user="u1", edit=short)
    assert run(db, engine)["style_rule_batches"] == 1
    with db.cursor() as cur:
        cur.execute("SELECT rule_key, rule_text, status, evidence FROM email_agent.bp_style_rule WHERE sent_by = 'u1'")
        rows = cur.fetchall()
    keys = {r[0] for r in rows}
    assert "shorter" in keys and all(r[2] == "proposed" for r in rows)
    shorter = next(r for r in rows if r[0] == "shorter")
    assert "shorter than drafted" in shorter[1] and shorter[3]["edits_considered"] == 10 and shorter[3]["median_length_ratio"] < 0.85


def test_too_few_edits_propose_no_rule(db, engine):
    for _ in range(9):
        sent(db, user="u1", edit=lambda t: "<p>Short. 44.80 GBP by 30 October 2026?</p>")
    assert run(db, engine)["style_rule_batches"] == 0 and count(db, "bp_style_rule") == 0


def test_unedited_drafts_propose_nothing(db, engine):
    for _ in range(12):
        sent(db, user="u1")
    run(db, engine)
    assert count(db, "bp_style_rule") == 0


def test_rules_are_per_reviewer_and_never_mixed(db, engine):
    for _ in range(10):
        sent(db, user="u1", edit=lambda t: "<p>Short. 44.80 GBP by 30 October 2026?</p>")
        sent(db, user="u2")
    run(db, engine)
    assert count(db, "bp_style_rule", "sent_by = 'u2'") == 0 and count(db, "bp_style_rule", "sent_by = 'u1'") >= 1


def test_the_same_signal_again_does_not_churn_the_rules_and_a_changed_one_supersedes(db, engine):
    short = lambda t: "<p>Short. 44.80 GBP by 30 October 2026?</p>"
    for _ in range(10):
        sent(db, user="u1", edit=short)
    run(db, engine)
    first = count(db, "bp_style_rule", "status = 'proposed'")
    sent(db, user="u1", edit=short)
    assert run(db, engine)["style_rule_batches"] in (0, 1)
    longer = lambda t: t.replace("Kind regards", "Kind regards and many thanks for all of your help with this. " * 6)
    for _ in range(50):
        sent(db, user="u1", edit=longer)
    run(db, engine)
    keys = {r for r in _keys(db, "proposed")}
    assert "shorter" not in keys and "longer" in keys
    assert count(db, "bp_style_rule", "status = 'superseded'") >= first


def _keys(db, status):
    with db.cursor() as cur:
        cur.execute("SELECT rule_key FROM email_agent.bp_style_rule WHERE sent_by = 'u1' AND status = %s", (status,))
        return [r[0] for r in cur.fetchall()]


def test_no_more_than_the_maximum_number_of_rules(db, engine):
    for _ in range(10):
        sent(db, user="u1", edit=lambda t: "<p>x</p>")
    run(db, engine)
    assert 1 <= count(db, "bp_style_rule", "status = 'proposed'") <= 15


def test_only_the_person_a_rule_is_about_can_decide_it(db, engine):
    for _ in range(10):
        sent(db, user="u1", edit=lambda t: "<p>Short. 44.80 GBP by 30 October 2026?</p>")
    run(db, engine)
    with db.cursor() as cur:
        cur.execute("SELECT rule_id FROM email_agent.bp_style_rule WHERE sent_by = 'u1' LIMIT 1")
        rid = cur.fetchone()[0]
    assert learning.decide_style_rule(db, rid, "someone-else", "approve")["ok"] is False
    assert learning.decide_style_rule(db, rid, "u1", "edit")["ok"] is False                  # an edit needs words
    assert learning.decide_style_rule(db, rid, "u1", "frobnicate")["ok"] is False
    assert learning.decide_style_rule(db, rid, "u1", "edit", "Keep it to four sentences.")["ok"] is True
    with db.cursor() as cur:
        cur.execute("SELECT status, edited_text, decided_by FROM email_agent.bp_style_rule WHERE rule_id = %s", (rid,))
        assert cur.fetchone() == ("edited", "Keep it to four sentences.", "u1")


# --- route 5: the family was switched ---------------------------------------------------------------------------------

CLAR = {"id": "clarification", "key": None, "text": "Is this A, or B?", "options": ["free_prompt", "negotiation_counter"], "resolution": None}


def test_choosing_the_other_family_is_a_labelled_classifier_example(db, engine):
    sent(db, family="free_prompt", request="Counter their offer", items=[CLAR], record=False, clarification={"question": "?"},
         resolution={"clarification": {"action": "edit", "value": "negotiation_counter", "by": "u1", "at": "t"}})
    assert run(db, engine)["classifier_examples"] == 1
    with db.cursor() as cur:
        cur.execute("SELECT request_text, predicted_family, labeled_family, labeled_by FROM email_agent.bp_classifier_example")
        assert cur.fetchone() == ("Counter their offer", "free_prompt", "negotiation_counter", "u1")


def test_confirming_the_chosen_family_is_not_a_switch(db, engine):
    sent(db, family="free_prompt", request="x", items=[CLAR], record=False,
         resolution={"clarification": {"action": "confirm", "by": "u1", "at": "t"}})
    assert run(db, engine)["classifier_examples"] == 0


def test_a_label_that_was_not_one_of_the_offered_families_is_not_a_label(db, engine):
    sent(db, family="free_prompt", request="x", items=[CLAR], record=False,
         resolution={"clarification": {"action": "edit", "value": "wire_transfer", "by": "u1", "at": "t"}})
    assert run(db, engine)["classifier_examples"] == 0


def test_a_switch_is_learned_even_if_the_draft_was_never_sent(db, engine):
    sent(db, family="free_prompt", request="x", items=[CLAR], record=False,
         resolution={"clarification": {"action": "edit", "value": "negotiation_counter", "by": "u1", "at": "t"}})
    assert run(db, engine)["outcomes"] == 0 and count(db, "bp_classifier_example") == 1


# --- exemplars ----------------------------------------------------------------------------------------------------------

def test_a_barely_touched_well_judged_clean_draft_is_an_exemplar_candidate(db, engine):
    sent(db, edit=lambda t: t.replace("Thank you", "Thanks"), judge=4.5)
    assert run(db, engine)["exemplar_candidates"] == 1
    with db.cursor() as cur:
        cur.execute("SELECT family_id, author, reviewed_by, status, tone_variables FROM email_agent.bp_exemplar_candidate")
        assert cur.fetchone() == ("negotiation_counter", "u1", "boss", "candidate", {"escalation_level": 2})


@pytest.mark.parametrize("why,kw", [
    ("edited too much", dict(edit=lambda t: "<p>Totally different email about something else entirely, thanks. 44.80 GBP?</p>")),
    ("judge too low", dict(judge=3.9)),
    ("never judged", dict(judge=None)),
    ("a figure rests on the request alone", dict(unverified=["25"])),
    ("a fact was carried in unverified", dict(carried={"lead_time": "7"})),
    ("a date rests on the request alone", dict(violations=[{"kind": "unverified_date", "detail": "12 March 2027", "severity": "warn"}])),
    ("an assumption was left unconfirmed", dict(items=[{"id": "a", "key": "a", "text": "t", "resolution": None}])),
    ("an assumption was edited, not confirmed", dict(items=[{"id": "a", "key": "a", "text": "t", "resolution": None}],
                                                     resolution={"a": {"action": "edit", "value": "x"}})),
    ("the draft was unassured", dict(status="unassured")),
    ("a fact was edited", dict(edit=lambda t: t.replace("47.50", "45.37"))),
])
def test_each_exemplar_condition_stands_alone(db, engine, why, kw):
    sent(db, **{"edit": lambda t: t.replace("Thank you", "Thanks"), **kw})
    assert run(db, engine)["exemplar_candidates"] == 0, why


def test_confirmed_assumptions_do_not_disqualify(db, engine):
    sent(db, items=[{"id": "a", "key": "a", "text": "t", "resolution": None}], resolution={"a": {"action": "confirm", "by": "x"}})
    assert run(db, engine)["exemplar_candidates"] == 1


def test_the_distance_threshold_is_exclusive(db, engine, monkeypatch):
    sent(db, edit=lambda t: t.replace("Thank you", "Thanks"))
    with db.cursor() as cur:
        cur.execute("SELECT edit_distance FROM email_agent.bp_draft_outcome")
        d = float(cur.fetchone()[0])
    rules = learning.load_rules(engine)
    assert learning.classify_edit({"edit_distance": d, "drafted_words": 40, "sent_words": 40}, {"judge": {"status": "scored", "overall": 5}, "assumption_items": []}, {**rules, "exemplar_max_distance": d})["exemplar"] is False
    assert learning.classify_edit({"edit_distance": d, "drafted_words": 40, "sent_words": 40}, {"judge": {"status": "scored", "overall": 5}, "assumption_items": []}, {**rules, "exemplar_max_distance": d + 0.001})["exemplar"] is True


def test_promotion_needs_a_second_person_and_sets_the_review_date(db, engine):
    sent(db, edit=lambda t: t.replace("Thank you", "Thanks"))
    run(db, engine)
    eid = _one(db, "SELECT exemplar_id FROM email_agent.bp_exemplar_candidate")
    rules = learning.load_rules(engine)
    assert learning.approve_exemplar(db, eid, "u1", rules, NOW)["ok"] is False               # the author cannot approve their own
    assert learning.approve_exemplar(db, eid, "boss", rules, NOW)["ok"] is True
    assert str(_one(db, "SELECT review_after FROM email_agent.bp_exemplar_candidate")) == "2027-10-09"
    assert learning.approve_exemplar(db, eid, "boss2", rules, NOW)["ok"] is False             # only a candidate can be approved


def test_an_approved_exemplar_is_re_reviewed_after_twelve_months(db, engine):
    sent(db, edit=lambda t: t.replace("Thank you", "Thanks"))
    run(db, engine)
    learning.approve_exemplar(db, _one(db, "SELECT exemplar_id FROM email_agent.bp_exemplar_candidate"), "boss", learning.load_rules(engine), NOW)
    assert run(db, engine, now=NOW + timedelta(days=300))["exemplars_expired"] == 0
    assert run(db, engine, now=NOW + timedelta(days=400))["exemplars_expired"] == 1
    assert _one(db, "SELECT status FROM email_agent.bp_exemplar_candidate") == "expired"


def _one(db, sql):
    with db.cursor() as cur:
        cur.execute(sql)
        return cur.fetchone()[0]


# --- the job itself --------------------------------------------------------------------------------------------------------

def test_a_second_run_finds_nothing_to_do(db, engine):
    sent(db, edit=lambda t: t.replace("44.80", "46.00"))
    sent(db, edit=lambda t: t.replace("47.50", "45.37"), user="u2")
    first = run(db, engine)
    assert first["outcomes"] == 2
    again = run(db, engine)
    assert again["outcomes"] == 0 and sum(again.values()) == 0
    assert count(db, "bp_draft_outcome", "learning_processed_at IS NULL") == 0


def test_an_edit_with_nothing_to_learn_is_still_marked_processed(db, engine):
    sent(db)
    run(db, engine)
    assert _one(db, "SELECT learning_routes FROM email_agent.bp_draft_outcome") == []


def test_abandoned_drafts_are_not_learned_from(db, engine):
    uid = sent(db, record=False)
    capture.record_abandoned(db, uid, "u1", "wrong supplier")
    assert run(db, engine)["outcomes"] == 0


def test_the_batch_size_bounds_one_run_and_the_next_continues(db, engine, monkeypatch):
    real = learning.load_rules
    monkeypatch.setattr(learning, "load_rules", lambda e: {**real(e), "batch_size": 2})
    for _ in range(5):
        sent(db)
    assert run(db, engine)["outcomes"] == 2 and run(db, engine)["outcomes"] == 2 and run(db, engine)["outcomes"] == 1


# --- classify_edit, no database ---------------------------------------------------------------------------------------------

RULES = {"wording_edit_min_words": 3, "exemplar_max_distance": 0.15, "exemplar_min_judge": 4.0}


def _o(**k):
    return {"edit_distance": 0.2, "drafted_words": 30, "sent_words": 30, "removed_figures": [], "added_figures": [], **k}


def test_classify_maps_a_changed_figure_to_every_fact_with_that_value():
    cap = {"facts": {"offer": {"value": "47.5000"}, "total": {"value": "47.5"}, "other": {"value": "9"}}}
    v = learning.classify_edit(_o(removed_figures=[{"value": "47.5", "class": "fact"}], added_figures=[{"value": "45", "class": "other"}]), cap, RULES)
    assert sorted(f["fact_key"] for f in v["fact"]) == ["offer", "total"] and v["fact"][0]["now"] == "45"


def test_classify_reasoned_direction():
    cap = {"reasoned": {"counter_price": {"value": "44.8"}}}
    up = learning.classify_edit(_o(removed_figures=[{"value": "44.8", "class": "reasoned"}], added_figures=[{"value": "46", "class": "other"}]), cap, RULES)
    dn = learning.classify_edit(_o(removed_figures=[{"value": "44.8", "class": "reasoned"}], added_figures=[{"value": "43", "class": "other"}]), cap, RULES)
    none = learning.classify_edit(_o(removed_figures=[{"value": "44.8", "class": "reasoned"}], added_figures=[]), cap, RULES)
    assert (up["reasoned"][0]["direction"], dn["reasoned"][0]["direction"]) == ("raised", "lowered")
    assert none["reasoned"] == []                      # no replacement figure: an omission, not a correction
    same = learning.classify_edit(_o(removed_figures=[{"value": "44.8", "class": "reasoned"}], added_figures=[{"value": "44.80", "class": "other"}]), cap, RULES)
    assert same["reasoned"][0]["direction"] == "changed"


def test_classify_wording_counts_words_beyond_the_figures():
    base = dict(removed_figures=[{"value": "1", "class": "other"}], added_figures=[{"value": "2", "class": "other"}])
    assert learning.classify_edit(_o(edit_distance=0.2, **base), {}, RULES)["words_changed"] == 5          # 6 words - 1 figure
    assert learning.classify_edit(_o(edit_distance=0.1, **base), {}, RULES)["wording"] is False            # 3 - 1 = 2 < 3
    assert learning.classify_edit(_o(edit_distance=0.0), {}, RULES)["routes"] == []


def test_classify_a_removed_figure_with_nothing_in_its_place_is_not_a_change():
    cap = {"facts": {"offer": {"value": "47.5"}}, "reasoned": {"price": {"value": "44.8"}}}
    v = learning.classify_edit(_o(removed_figures=[{"value": "47.5", "class": "fact"}, {"value": "44.8", "class": "reasoned"}], added_figures=[]), cap, RULES)
    assert v["fact"] == [] and v["reasoned"] == []
    two = learning.classify_edit(_o(removed_figures=[{"value": "47.5", "class": "fact"}, {"value": "44.8", "class": "reasoned"}],
                                   added_figures=[{"value": "45", "class": "other"}]), cap, RULES)
    assert [f["fact_key"] for f in two["fact"]] == ["offer"] and two["reasoned"] == []            # paired in order; the extra removal is an omission


# --- scheduling ---------------------------------------------------------------------------------------------------------------

def test_the_job_is_registered_only_when_switched_on(monkeypatch):
    from src.services.backend_scheduler import BackendScheduler
    for flag, expected in (("", False), ("0", False), ("1", True)):
        sched = BackendScheduler.__new__(BackendScheduler)
        sched._jobs = {}
        registered = []
        sched.register_job = lambda name, fn, **kw: registered.append(name)
        monkeypatch.setenv("EMAIL_LEARNING_ENABLED", flag)
        sched._register_email_learning_job()
        assert bool(registered) is expected


# --- gaps found by breaking the job on purpose --------------------------------------------------------------------------

def test_an_edit_that_changes_a_fact_AND_our_price_still_teaches_nothing_but_the_data_issue(db, engine):
    """Both figures change in one send. The fact rule wins outright: the data owner hears about it, and the
    price correction is NOT mined for an eval case, an exemplar or a review item."""
    sent(db, edit=lambda t: t.replace("47.50", "45.37").replace("44.80", "46.00"))
    rep = run(db, engine)
    assert rep["dq_items"] == 1 and rep["eval_candidates"] == 0 and rep["exemplar_candidates"] == 0
    for table in ("bp_eval_candidate", "bp_exemplar_candidate", "bp_review_item"):
        assert count(db, table) == 0, table


def test_the_review_evidence_counts_people_not_edits(db, engine):
    for u in ("u1", "u1", "u2", "u3"):
        sent(db, user=u, edit=lambda t: t.replace("44.80", "46.00"))
    run(db, engine)
    with db.cursor() as cur:
        cur.execute("SELECT evidence->>'distinct_reviewers', jsonb_array_length(evidence->'outcome_ids') FROM email_agent.bp_review_item")
        assert cur.fetchone() == ("3", 4)


def test_a_reviewer_repeated_does_not_reach_the_threshold_even_with_other_edits(db, engine):
    for u in ("u1", "u1", "u1", "u2"):
        sent(db, user=u, edit=lambda t: t.replace("44.80", "46.00"))
    assert run(db, engine)["review_items"] == 0                         # 4 edits, 2 people


def test_a_judge_score_exactly_on_the_floor_qualifies_and_one_hair_under_does_not(db, engine):
    sent(db, edit=lambda t: t.replace("Thank you", "Thanks"), judge=4.0, user="u1")
    sent(db, edit=lambda t: t.replace("Thank you", "Thanks"), judge=3.99, user="u2")
    run(db, engine)
    assert _one(db, "SELECT author FROM email_agent.bp_exemplar_candidate") == "u1" and count(db, "bp_exemplar_candidate") == 1


def test_a_fact_edit_is_never_an_exemplar_whatever_else_is_perfect():
    cap = {"judge": {"status": "scored", "overall": 5}, "assumption_items": [], "facts": {"offer": {"value": "47.5"}},
           "assurance_status": "verified"}
    clean = learning.classify_edit(_o(edit_distance=0.05, removed_figures=[], added_figures=[]), cap, RULES)
    fact = learning.classify_edit(_o(edit_distance=0.05, removed_figures=[{"value": "47.5", "class": "fact"}],
                                     added_figures=[{"value": "45", "class": "other"}]), cap, RULES)
    assert clean["exemplar"] is True and fact["exemplar"] is False


def test_content_corrections_do_not_distort_a_reviewers_style_signal(db, engine):
    """Ten clean, shortened edits make a 'shorter' rule. Twenty LATER fact corrections that happen to lengthen
    the email must not flip it: they are about the data, not the person's voice."""
    short = lambda t: "<p>Short. 44.80 GBP by 30 October 2026?</p>"
    for _ in range(10):
        sent(db, user="u1", edit=short)
    run(db, engine)
    assert "shorter" in _keys(db, "proposed")
    pad = "Many thanks for all of your continued help and support with this and with everything else too. " * 4
    for _ in range(20):
        sent(db, user="u1", edit=lambda t: t.replace("47.50", "45.37").replace("Kind regards", pad + "Kind regards"))
    sent(db, user="u1", edit=short)                                     # one clean edit, so the user's style is recomputed
    run(db, engine)
    assert "shorter" in _keys(db, "proposed") and "longer" not in _keys(db, "proposed")


def test_a_moderate_rewrite_is_wording_but_not_a_heavy_rewrite(db, engine):
    moderate = lambda t: t.replace("We would like to propose", "We would be keen to suggest").replace("this order", "the whole order")
    for u in ("u1", "u2", "u3"):
        sent(db, user=u, edit=moderate)
    run(db, engine)
    assert _one(db, "SELECT learning_routes FROM email_agent.bp_draft_outcome LIMIT 1") == ["wording"]
    assert _one(db, "SELECT max(edit_distance) FROM email_agent.bp_draft_outcome") < learning.load_rules(engine)["high_divergence"]
    assert count(db, "bp_review_item") == 0


def test_the_scheduler_job_is_off_when_the_switch_is_not_set_at_all(monkeypatch):
    from src.services.backend_scheduler import BackendScheduler
    monkeypatch.delenv("EMAIL_LEARNING_ENABLED", raising=False)
    sched = BackendScheduler.__new__(BackendScheduler)
    sched._jobs, registered = {}, []
    sched.register_job = lambda name, fn, **kw: registered.append(name)
    sched._register_email_learning_job()
    assert registered == []


def test_a_draft_whose_price_a_person_overruled_is_not_an_exemplar_however_few_words_changed(db, engine):
    sent(db, edit=lambda t: t.replace("44.80", "46.00"))
    rep = run(db, engine)
    assert rep["eval_candidates"] == 1 and rep["exemplar_candidates"] == 0
    cap = {"judge": {"status": "scored", "overall": 5}, "assumption_items": [], "reasoned": {"price": {"value": "44.8"}}, "assurance_status": "verified"}
    v = learning.classify_edit(_o(edit_distance=0.04, removed_figures=[{"value": "44.8", "class": "reasoned"}],
                                  added_figures=[{"value": "46", "class": "other"}]), cap, RULES)
    assert v["reasoned"] and v["exemplar"] is False
