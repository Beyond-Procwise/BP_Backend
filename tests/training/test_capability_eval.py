"""Tests for the AgentNick capability evaluation harness (deterministic scorers).

These tests exercise the pure scoring logic with synthetic model answers — no
Ollama / GPU needed. The model backend is covered separately as an integration
check.
"""
from __future__ import annotations

import json

import pytest

from src.training import capability_eval as ce


# ---------------------------------------------------------------------------
# score_concept
# ---------------------------------------------------------------------------
def test_concept_all_groups_present_scores_one():
    text = "A three-way match reconciles the purchase order, the invoice and the goods receipt."
    groups = [["purchase order", "po"], ["invoice"], ["goods receipt", "grn", "receipt"]]
    assert ce.score_concept(text, groups) == pytest.approx(1.0)


def test_concept_partial_groups():
    text = "It matches the purchase order and the invoice."
    groups = [["purchase order", "po"], ["invoice"], ["goods receipt", "grn"]]
    # 2 of 3 groups matched
    assert ce.score_concept(text, groups) == pytest.approx(2 / 3)


def test_concept_synonym_counts_as_match():
    text = "Compare the PO against the bill and the receipt."
    groups = [["purchase order", "po"], ["invoice", "bill"], ["goods receipt", "receipt"]]
    assert ce.score_concept(text, groups) == pytest.approx(1.0)


def test_concept_forbidden_term_penalises():
    text = "The tax amount is larger than the subtotal."  # WRONG: forbidden phrasing
    groups = [["tax"]]
    forbid = ["larger than the subtotal", "greater than the subtotal"]
    # group matched (1.0) but forbidden present -> penalised to 0
    assert ce.score_concept(text, groups, forbid=forbid) == pytest.approx(0.0)


def test_concept_case_insensitive():
    assert ce.score_concept("NET 30 means payment due in 30 days", [["net 30"]]) == 1.0


# ---------------------------------------------------------------------------
# score_orchestration
# ---------------------------------------------------------------------------
VALID = {"data_extraction", "discrepancy_detection", "opportunity_miner",
         "supplier_ranking", "quote_evaluation", "negotiation",
         "email_drafting", "email_dispatch", "rag", "approvals"}


def _plan(agents):
    return json.dumps({"goal": "g", "steps": [
        {"agent": a, "parallel_group": i, "required": True} for i, a in enumerate(agents)
    ], "escalation_policy": {}})


def test_orchestration_perfect_plan():
    text = _plan(["opportunity_miner", "supplier_ranking", "quote_evaluation"])
    r = ce.score_orchestration(text, expected={"opportunity_miner", "supplier_ranking"},
                               forbidden=set(), valid_ids=VALID)
    assert r["valid_json"] is True
    assert r["score"] == pytest.approx(1.0)


def test_orchestration_missing_expected_agent_lowers_recall():
    text = _plan(["supplier_ranking"])
    r = ce.score_orchestration(text, expected={"opportunity_miner", "supplier_ranking"},
                               forbidden=set(), valid_ids=VALID)
    # recall 1/2, no bad agents
    assert r["score"] == pytest.approx(0.5)


def test_orchestration_hallucinated_agent_penalised():
    text = _plan(["supplier_ranking", "make_coffee"])
    r = ce.score_orchestration(text, expected={"supplier_ranking"},
                               forbidden=set(), valid_ids=VALID)
    # recall 1.0, but 1 of 2 predicted is hallucinated -> clean_frac 0.5
    assert r["score"] == pytest.approx(0.5)
    assert "make_coffee" in r["hallucinated"]


def test_orchestration_forbidden_agent_penalised():
    text = _plan(["data_extraction"])
    r = ce.score_orchestration(text, expected={"data_extraction"},
                               forbidden={"data_extraction"}, valid_ids=VALID)
    assert r["score"] == pytest.approx(0.0)


def test_orchestration_invalid_json_scores_zero():
    r = ce.score_orchestration("I would run supplier_ranking then quote_evaluation.",
                               expected={"supplier_ranking"}, forbidden=set(), valid_ids=VALID)
    assert r["valid_json"] is False
    assert r["score"] == 0.0


# ---------------------------------------------------------------------------
# score_item dispatch + item loading
# ---------------------------------------------------------------------------
def test_score_item_dispatches_concept():
    item = ce.CapabilityItem(id="c1", section="concepts", prompt="?",
                             scoring={"include": [["invoice"]]})
    assert ce.score_item(item, "the invoice total") == pytest.approx(1.0)


def test_score_item_dispatches_orchestration():
    item = ce.CapabilityItem(id="o1", section="orchestration", prompt="?",
                             scoring={"expected": ["supplier_ranking"], "forbidden": []})
    txt = _plan(["supplier_ranking"])
    assert ce.score_item(item, txt, valid_ids=VALID) == pytest.approx(1.0)


def test_real_eval_set_loads_and_is_wellformed():
    items = ce.load_items()
    assert len(items) >= 12
    sections = {i.section for i in items}
    assert {"concepts", "product", "orchestration"} <= sections
    for it in items:
        assert it.id and it.prompt and it.section in ce.SECTIONS
        if it.section == "orchestration":
            assert "expected" in it.scoring
        else:
            assert "include" in it.scoring
