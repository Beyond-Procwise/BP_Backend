"""Capture: what a draft rested on, and an edit score with the changed figures. Never the sent text."""

import json
from datetime import datetime, timezone
from types import SimpleNamespace

from src.services.draft_assurance import capture

ASSURANCE = {
    "family_id": "negotiation_counter", "family_version": 1, "mode": "shadow", "status": "needs_review",
    "facts": {"supplier_current_offer": {"value": "47.50", "table": "supplier_response",
                                         "column": "price", "row_id": "2"}},
    "reasoned": {"counter_price": {"value": "44.8", "basis": ["supplier_current_offer"]}},
    "conflicts": [], "assumptions": [], "violations": [], "unverified_figures": [], "repaired": False,
}
DRAFT = "<p>Thank you for your offer of 47.50 GBP.</p><p>We propose 44.80 GBP. Please confirm by 30 October 2026?</p>"


# --- measuring an edit -------------------------------------------------------

def test_an_unedited_draft_scores_zero_even_through_markup_and_markers():
    sent = "<!-- tracking abc123 -->" + DRAFT
    m = capture.measure_edit(DRAFT, sent, ASSURANCE)
    assert (m["edit_distance"], m["edit_class"], m["removed"], m["added"]) == (0, "none", [], [])


def test_a_wording_only_edit_changes_no_figures():
    m = capture.measure_edit(DRAFT, DRAFT.replace("Thank you for", "Thanks for"), ASSURANCE)
    assert m["edit_class"] == "wording" and 0 < m["edit_distance"] < 0.5
    assert m["removed"] == [] and m["added"] == []


def test_changing_a_postgres_fact_is_classed_as_a_fact_edit():
    m = capture.measure_edit(DRAFT, DRAFT.replace("47.50", "45"), ASSURANCE)
    assert m["edit_class"] == "fact"
    assert {"value": "47.5", "kind": "figure", "class": "fact"} in m["removed"]
    assert any(a["value"] == "45" for a in m["added"])


def test_changing_the_counter_price_is_a_reasoned_edit_not_a_fact_edit():
    m = capture.measure_edit(DRAFT, DRAFT.replace("44.80", "46.00"), ASSURANCE)
    assert m["edit_class"] == "reasoned"


def test_changing_an_unrelated_figure_is_figure_other():
    m = capture.measure_edit(DRAFT + " Order 25 units.", DRAFT + " Order 30 units.", ASSURANCE)
    assert m["edit_class"] == "figure_other"


def test_a_fact_edit_outranks_a_reasoned_edit_made_in_the_same_send():
    m = capture.measure_edit(DRAFT, DRAFT.replace("47.50", "45").replace("44.80", "46"), ASSURANCE)
    assert m["edit_class"] == "fact"


def test_a_changed_date_is_reported():
    m = capture.measure_edit(DRAFT, DRAFT.replace("30 October 2026", "6 November 2026"), ASSURANCE)
    assert any(r["kind"] == "date" for r in m["removed"]) and any(a["kind"] == "date" for a in m["added"])


def test_the_measurement_carries_no_prose():
    sent = DRAFT.replace("Thank you for your offer", "Cheers mate regarding the number")
    blob = json.dumps(capture.measure_edit(DRAFT, sent, ASSURANCE))
    assert "Cheers" not in blob and "mate" not in blob


# --- the two writes ----------------------------------------------------------------

class Cur:
    def __init__(self, select_row=None, boom=False):
        self.calls, self.select_row, self.boom, self._last = [], select_row, boom, None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        if self.boom:
            raise RuntimeError("db down")
        self.calls.append((sql, params))
        self._last = sql

    def fetchone(self):
        return self.select_row if "SELECT capture_id" in self._last else (99,)


class Conn:
    def __init__(self, cur):
        self.cur = cur

    def cursor(self):
        return self.cur


def _draft(**over):
    return {"unique_id": "U-1", "workflow_id": "wf", "supplier_id": "S-1", "body": DRAFT,
            "metadata": {"intent": "NEGOTIATION_COUNTER"}, "assurance": dict(ASSURANCE),
            "requested_by": "nick", **over}


def _row(cur):
    """The INSERT's parameters keyed by column name, so a test never counts positions."""
    import re
    sql, params = cur.calls[0]
    cols = [c.strip() for c in re.search(r"\(([^)]*)\)\s*VALUES", sql, re.S).group(1).split(",")]
    assert len(cols) == len(params), (len(cols), len(params))
    return dict(zip(cols, params))


def test_record_draft_stores_the_model_text_the_hash_and_the_assurance():
    cur = Cur()
    assert capture.record_draft(Conn(cur), _draft()) == 99
    assert "email_agent.bp_draft_capture" in cur.calls[0][0]
    r = _row(cur)
    assert r["unique_id"] == "U-1" and r["path"] == "NEGOTIATION_COUNTER" and r["family_id"] == "negotiation_counter"
    assert "47.50 GBP" in r["draft_text"] and "<p>" not in r["draft_text"]
    assert len(r["draft_hash"]) == 64
    assert json.loads(r["facts"])["supplier_current_offer"]["row_id"] == "2"


def test_a_stage_that_never_ran_is_null_and_one_that_ran_empty_is_an_empty_value():
    # not captured -> SQL NULL
    r = _row_for(_draft())
    assert r["tone_variables"] is None and r["exemplar_ids"] is None and r["brief"] is None and r["judge"] is None
    # captured and empty -> '[]' / '{}'
    a = {**ASSURANCE, "exemplars": {"ids": [], "scope": "none"}, "clarification": {}, "assumption_items": []}
    r = _row_for(_draft(assurance=a))
    assert r["exemplar_ids"] == "[]" and r["exemplar_scope"] == "none"
    assert r["clarification"] == "{}" and r["assumption_items"] == "[]"


def _row_for(draft):
    cur = Cur()
    capture.record_draft(Conn(cur), draft)
    return _row(cur)


def test_the_initiator_is_recorded_as_the_initiator_not_as_the_user():
    a = {**ASSURANCE, "accountability": {"initiated_by": "NegotiationAgent", "kind": "agent"}}
    r = _row_for(_draft(assurance=a))
    assert r["initiated_by"] == "NegotiationAgent" and r["initiated_by_kind"] == "agent"
    assert "user_id" not in r


def test_a_draft_without_an_assurance_record_is_not_captured():
    cur = Cur()
    assert capture.record_draft(Conn(cur), _draft(assurance=None)) is None and cur.calls == []


def test_a_draft_without_a_unique_id_is_not_captured():
    cur = Cur()
    assert capture.record_draft(Conn(cur), _draft(unique_id=None)) is None and cur.calls == []


def test_record_draft_swallows_a_database_failure():
    assert capture.record_draft(Conn(Cur(boom=True)), _draft()) is None


def _captured_row(text=DRAFT, captures=2):
    return (7, capture.plain(text), datetime.now(timezone.utc), captures,
            json.dumps(ASSURANCE["facts"]), json.dumps(ASSURANCE["reasoned"]), None)     # text_expired_at


def test_record_sent_without_a_retention_period_stores_a_score_and_changed_figures_but_not_the_sent_text():
    cur = Cur(select_row=_captured_row())
    sent = DRAFT.replace("Thank you for your offer", "Cheers regarding").replace("47.50", "45")
    assert capture.record_sent(Conn(cur), "U-1", sent) == 99
    sql, p = cur.calls[1]
    assert "email_agent.bp_draft_outcome" in sql
    assert p[0] == 7 and p[1] > 0 and p[2] == "fact"
    assert p[7] == 1                                   # two captures -> one regeneration
    everything = json.dumps([str(x) for x in p])
    assert "Cheers" not in everything and "regarding" not in everything


def test_record_sent_with_no_capture_is_a_quiet_no_op():
    cur = Cur(select_row=None)
    assert capture.record_sent(Conn(cur), "U-1", DRAFT) is None and len(cur.calls) == 1


def test_record_sent_swallows_a_database_failure():
    assert capture.record_sent(Conn(Cur(boom=True)), "U-1", DRAFT) is None
    assert capture.record_sent(Conn(Cur()), None, DRAFT) is None


# --- the agent hook ------------------------------------------------------------------

def test_the_agent_captures_assured_drafts_and_skips_the_rest(monkeypatch):
    from agents import email_drafting_agent as module
    agent = module.EmailDraftingAgent()
    got = []
    monkeypatch.setattr(capture, "record_draft", lambda conn, d: got.append(d["unique_id"]))
    agent._capture_draft(_draft())
    agent._capture_draft(_draft(assurance=None, unique_id="U-2"))
    assert got == ["U-1"]


def test_the_agent_never_lets_a_capture_failure_escape(monkeypatch):
    from agents import email_drafting_agent as module
    agent = module.EmailDraftingAgent()

    def boom(conn, d):
        raise RuntimeError("x")

    monkeypatch.setattr(capture, "record_draft", boom)
    agent._capture_draft(_draft())                      # must not raise


def test_the_record_carries_the_persons_request_for_prompt_drafts():
    from src.services import draft_assurance as da
    from tests.services.test_draft_assurance import FakeConn, _free_prompt_rules
    fam = da.parse_family(_free_prompt_rules(), 1)
    inp = da.prepare_inputs(FakeConn({}), fam, {"prompt": "Ask Acme to quote for 25 chairs"},
                            lookup_keys={})
    assert inp.finalize("Please quote for 25 chairs.", [], [])["request_text"] == "Ask Acme to quote for 25 chairs"
    assert inp.finalize("x", [], [])["request_text"] == "Ask Acme to quote for 25 chairs"


def test_a_hidden_marker_containing_an_angle_bracket_is_removed_whole():
    marked = "<!-- ProcWise token a>b secret-xyz -->" + DRAFT
    assert "secret-xyz" not in capture.plain(marked)
    assert capture.measure_edit(DRAFT, marked, ASSURANCE)["edit_distance"] == 0


def test_storing_a_draft_triggers_its_capture(monkeypatch):
    from agents import email_drafting_agent as module
    agent = module.EmailDraftingAgent()
    got = []
    monkeypatch.setattr(agent, "_capture_draft", lambda d: got.append(d["unique_id"]))
    agent._store_draft({"supplier_id": "S-1", "unique_id": "U-9", "workflow_id": "wf",
                        "subject": "s", "body": DRAFT, "recipients": ["a@x.test"],
                        "assurance": dict(ASSURANCE), "metadata": {}})
    assert got == ["U-9"]


def test_the_distance_matches_the_style_packages_measure():
    from src.services.style.feedback import divergence_score
    pairs = [("", ""), ("a b c", ""), ("one two three four", "one two three four"),
             ("thank you for your offer", "thanks for the offer"), ("x", "y z w v u")]
    for a, b in pairs:
        assert capture.word_distance(a, b) == divergence_score(a, b), (a, b)


def test_large_figures_are_stored_as_plain_numbers_not_scientific_notation():
    a = {**ASSURANCE, "facts": {"supplier_current_offer": {"value": "132500.0000"}}}
    m = capture.measure_edit("We note 132,500.00 GBP.", "We note 130,000.00 GBP.", a)
    assert m["removed"] == [{"value": "132500", "kind": "figure", "class": "fact"}]
    assert m["added"] == [{"value": "130000", "kind": "figure", "class": "other"}]
