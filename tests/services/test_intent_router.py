"""The free-text intent router: a request in, a proposed report or workflow out.

It only PROPOSES: nothing here runs anything, and a person confirms. Rules decide (exact and
explainable). Below the bar the answer is "none" with the nearest options, so a router mistake costs a
question and never a wrong action. Today is injected so periods resolve the same in every run.
"""
from datetime import date

import pytest

from src.services import intent_router as ir

TODAY = date(2026, 10, 20)
SUPPLIERS = [("SUP-RedstoneWorks9", "Redstone Works 9"), ("SUP-RedstoneWorks91", "Redstone Works 91"),
             ("SUP-AcmeLtd", "Acme Ltd")]


def lookup(text):
    t = text.lower()
    return [(i, n) for i, n in SUPPLIERS if n.lower() in t or i.lower() in t]


def route(text, **kw):
    return ir.route(text, today=TODAY, supplier_lookup=lookup, **kw)


def ids(result):
    return [(s["kind"], s["id"]) for s in result["steps"]]


# --- one request, one target ----------------------------------------------------------------

@pytest.mark.parametrize("text,kind,target", [
    ("Prepare a QBR for Acme Ltd for Q3 2026", "report", "supplier_qbr"),
    ("I need the quarterly business review for Acme Ltd", "report", "supplier_qbr"),
    ("Give me the executive procurement summary for last quarter", "report", "exec_procurement_summary"),
    ("Write a board paper for DEALV2-001696", "report", "board_paper"),
    ("How dependent are we on our biggest suppliers? supplier criticality review please", "report", "supplier_criticality_review"),
    ("Rank the suppliers on this deal", "workflow", "supplier_ranking"),
    ("Compare the quotes we received", "workflow", "quote_evaluation"),
    ("Find savings opportunities in our spend", "workflow", "opportunity_mining"),
    ("Email the suppliers and ask for a quote", "workflow", "supplier_interaction"),
    ("Extract the data from the uploaded invoices", "workflow", "document_extraction"),
])
def test_a_clear_request_routes_to_its_target(text, kind, target):
    r = route(text)
    assert r["decision"] in ("route", "clarify")
    assert (kind, target) in ids(r), r


def test_a_decision_carries_its_score_matches_and_reason():
    s = route("Prepare a QBR for Acme Ltd for Q3 2026")["steps"][0]
    assert s["score"] >= ir.MIN_SCORE and s["matched"] and s["label"]
    assert route("Prepare a QBR")["reason"]


# --- never a guess ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["", "   ", "hello", "what is the weather", "approval", "review",
                                  "tell me about suppliers"])
def test_below_the_bar_is_none_with_the_nearest_options(text):
    r = route(text)
    assert r["decision"] == "none" and r["steps"] == []
    assert r["options"], "the person is shown what the router can do"


def test_a_lone_weak_word_never_routes():
    assert route("approvals")["decision"] == "none"


def test_a_tie_asks_rather_than_picks(monkeypatch):
    monkeypatch.setattr(ir, "TARGETS", [
        ir.Target("report", "a", "A", [(r"\bfoo\b", 5)], []),
        ir.Target("report", "b", "B", [(r"\bfoo\b", 5)], [])])
    r = route("foo")
    assert r["decision"] == "clarify"
    assert {a["id"] for a in r["alternatives"]} == {"a", "b"}
    assert r["steps"] == []


def test_non_text_input_is_none_not_an_error():
    assert route(None)["decision"] == "none"
    assert route(123)["decision"] == "none"


# --- several requests in one -------------------------------------------------------------------

def test_two_actions_become_an_ordered_plan():
    r = route("Rank these suppliers and then email them")
    assert r["decision"] == "plan"
    assert ids(r) == [("workflow", "supplier_ranking"), ("workflow", "supplier_interaction")]


def test_order_follows_the_request():
    r = route("Email the suppliers for a quote, then compare the quotes")
    assert ids(r) == [("workflow", "supplier_interaction"), ("workflow", "quote_evaluation")]


def test_a_part_that_routes_nowhere_is_reported_not_dropped():
    r = route("Rank the suppliers and make me a coffee")
    assert ids(r) == [("workflow", "supplier_ranking")]
    assert any("coffee" in u for u in r["unrouted"])


def test_a_trailing_name_is_part_of_the_request_not_a_stray_clause():
    r = route("Email Acme Ltd and Redstone Works 9 for a quote")
    assert ids(r) == [("workflow", "supplier_interaction")]
    assert r["unrouted"] == []


def test_the_same_target_twice_is_one_step():
    assert len(route("Rank the suppliers, then rank the suppliers again")["steps"]) == 1


# --- inputs ------------------------------------------------------------------------------------

def test_a_period_and_supplier_are_resolved_for_a_qbr():
    s = route("QBR for Acme Ltd for Q3 2026")["steps"][0]
    assert s["inputs"]["supplier_id"] == "SUP-AcmeLtd"
    assert (s["inputs"]["period_start"], s["inputs"]["period_end"]) == ("2026-07-01", "2026-09-30")
    assert s["inputs"]["period_label"] == "Q3 2026"
    assert s["missing"] == [] and s["ready"] is True


def test_what_is_missing_is_listed_and_the_step_is_not_ready():
    s = route("Prepare a QBR")["steps"][0]
    assert set(s["missing"]) == {"supplier_id", "period"}
    assert s["ready"] is False


def test_the_longest_supplier_name_wins_on_a_word_boundary():
    s = route("QBR for Redstone Works 91 for Q3 2026")["steps"][0]
    assert s["inputs"]["supplier_id"] == "SUP-RedstoneWorks91"
    s = route("QBR for Redstone Works 9 for Q3 2026")["steps"][0]
    assert s["inputs"]["supplier_id"] == "SUP-RedstoneWorks9"


def test_a_name_inside_a_longer_name_does_not_make_it_ambiguous():
    both = [("S-A", "Acme"), ("S-AH", "Acme Holdings")]
    look = lambda text: [(i, n) for i, n in both if n.lower() in text.lower()]  # noqa: E731
    s = ir.route("QBR for Acme Holdings Q3 2026", today=TODAY, supplier_lookup=look)["steps"][0]
    assert s["inputs"]["supplier_id"] == "S-AH"                       # not S-A, and not "ambiguous"
    s = ir.route("QBR for Acme Q3 2026", today=TODAY, supplier_lookup=look)["steps"][0]
    assert s["inputs"]["supplier_id"] == "S-A"


def test_a_second_supplier_and_a_period_are_not_a_separate_request():
    r = route("QBR for Acme Ltd and Redstone Works 9 for Q3 2026")
    assert r["unrouted"] == [] and len(r["steps"]) == 1
    r = route("Executive procurement summary and Q3 2026")
    assert r["unrouted"] == []
    r = route("Rank the suppliers on DEALV2-001696 and DEALV2-001697")
    assert r["unrouted"] == []


def test_two_suppliers_for_a_one_supplier_report_is_ambiguous_not_a_pick():
    s = route("QBR for Acme Ltd and Redstone Works 9 for Q3 2026")["steps"][0]
    assert "supplier_id" in s["missing"]
    assert {c["id"] for c in s["candidates"]["supplier_id"]} == {"SUP-AcmeLtd", "SUP-RedstoneWorks9"}


def test_a_supplier_given_by_id_resolves():
    s = route("QBR for SUP-AcmeLtd Q1 2026")["steps"][0]
    assert s["inputs"]["supplier_id"] == "SUP-AcmeLtd"


def test_a_deal_id_is_found_for_a_board_paper():
    s = route("Board paper for DEALV2-001696")["steps"][0]
    assert s["inputs"]["deal_id"] == "DEALV2-001696" and s["missing"] == []
    assert route("Write a board paper")["steps"][0]["missing"] == ["deal_id"]


@pytest.mark.parametrize("text,start,end,label", [
    ("summary for Q3 2026", "2026-07-01", "2026-09-30", "Q3 2026"),
    ("summary for Q1 2025", "2025-01-01", "2025-03-31", "Q1 2025"),
    ("summary for last quarter", "2026-07-01", "2026-09-30", "Q3 2026"),
    ("summary for this quarter", "2026-10-01", "2026-12-31", "Q4 2026"),
    ("summary for last month", "2026-09-01", "2026-09-30", "September 2026"),
    ("summary for 2025", "2025-01-01", "2025-12-31", "2025"),
    ("summary for H1 2026", "2026-01-01", "2026-06-30", "H1 2026"),
    ("summary from 2026-01-15 to 2026-02-20", "2026-01-15", "2026-02-20", "2026-01-15 to 2026-02-20"),
])
def test_periods_resolve_against_today(text, start, end, label):
    s = route("Executive procurement " + text)["steps"][0]
    assert (s["inputs"]["period_start"], s["inputs"]["period_end"], s["inputs"]["period_label"]) == (start, end, label)


def test_a_quarter_without_a_year_is_assumed_and_says_so():
    s = route("Executive procurement summary for Q2")["steps"][0]
    assert s["inputs"]["period_label"] == "Q2 2026"
    assert any("Q2" in a for a in s["assumptions"])


def test_an_impossible_period_is_missing_not_guessed():
    s = route("Executive procurement summary from 2026-05-01 to 2026-01-01")["steps"][0]
    assert "period" in s["missing"]


def test_a_currency_is_picked_up_when_stated():
    assert route("Executive procurement summary for Q3 2026 in USD")["steps"][0]["inputs"]["currency"] == "USD"
    assert "currency" not in route("Executive procurement summary for Q3 2026")["steps"][0]["inputs"]


# --- the router never runs anything and never breaks --------------------------------------------

def test_it_proposes_and_never_executes():
    r = route("Prepare a QBR for Acme Ltd for Q3 2026")
    assert r["proposal_only"] is True


def test_a_failing_lookup_is_a_missing_input_not_a_crash():
    def boom(_):
        raise RuntimeError("db down")
    s = ir.route("QBR for Acme Ltd Q3 2026", today=TODAY, supplier_lookup=boom)["steps"][0]
    assert "supplier_id" in s["missing"]


def test_no_lookup_means_the_supplier_is_missing():
    s = ir.route("QBR for Acme Ltd Q3 2026", today=TODAY)["steps"][0]
    assert "supplier_id" in s["missing"]


# --- the router cannot drift from what the platform offers ------------------------------------------

def test_every_registered_report_type_is_reachable_or_explicitly_excluded():
    import src.services.rga.builders  # noqa: F401
    from src.services.rga.factpack import registered_types
    reachable = {t.id for t in ir.TARGETS if t.kind == "report"}
    missing = set(registered_types()) - reachable - ir.NOT_ROUTABLE_REPORTS
    assert missing == set(), f"report types the router cannot reach: {missing}"


def test_every_named_workflow_is_reachable_or_explicitly_excluded():
    from orchestration.workflow_definitions import WORKFLOW_REGISTRY
    reachable = {t.id for t in ir.TARGETS if t.kind == "workflow"}
    missing = set(WORKFLOW_REGISTRY) - reachable - ir.NOT_ROUTABLE_WORKFLOWS
    assert missing == set(), f"workflows the router cannot reach: {missing}"


def test_every_target_names_a_report_or_workflow_that_exists():
    import src.services.rga.builders  # noqa: F401
    from orchestration.workflow_definitions import WORKFLOW_REGISTRY
    from src.services.rga.factpack import registered_types
    for t in ir.TARGETS:
        assert t.id in (registered_types() if t.kind == "report" else WORKFLOW_REGISTRY), t.id


def test_the_examples_of_each_target_route_to_it():
    for t in ir.TARGETS:
        for example in t.examples:
            assert (t.kind, t.id) in ids(route(example)), f"{t.id}: {example!r}"
