"""What a created agent's run puts on its result card.

A reasoning run now returns a run id, whether it was recorded, its token usage
and its named outputs. Left to the generic summariser, the run's bookkeeping
would fill the card's few fact slots ("Run id: 3f9c...", "Recorded: yes") ahead
of anything the agent was actually defined to return.
"""
from __future__ import annotations

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from api.routers.agent_workflows import _summarise_node_results

GRAPH = {"nodes": [{"id": "n1", "agent_slug": "overcharge-hunter"}]}
RUN = {
    "run_id": "3f9c0a", "recorded": True, "usage": {"prompt_tokens": 2300},
    "tools_used": ["get_corpus_facts"], "rounds": 2,
    "answer": "Acme Ltd billed GBP 812.50 above its PO.",
    "outputs": {
        "supplier": {"value": "Acme Ltd", "status": "evidence_found",
                     "evidence": '"supplier": "Acme Ltd"'},
        "overbilled_amount": {"value": "GBP 812.50", "status": "evidence_not_found",
                              "evidence": "GBP 812.50"},
        "contract_end": {"value": None, "status": "missing", "evidence": None},
    },
    "supplier": "Acme Ltd", "overbilled_amount": "GBP 812.50",
}


def _card(data):
    return _summarise_node_results({"n1": data}, GRAPH)["n1"]


def test_the_card_names_the_record_of_the_run():
    assert _card(RUN)["run_id"] == "3f9c0a"


def test_a_run_that_was_not_recorded_names_no_record():
    """A link to a record that does not exist is worse than no link."""
    assert "run_id" not in _card({**RUN, "recorded": False})


def test_the_named_outputs_are_on_the_card_in_the_order_they_were_defined():
    assert _card(RUN)["outputs"] == [
        {"name": "supplier", "label": "Supplier", "value": "Acme Ltd",
         "status": "evidence_found"},
        {"name": "overbilled_amount", "label": "Overbilled amount", "value": "GBP 812.50",
         "status": "evidence_not_found"},
        {"name": "contract_end", "label": "Contract end", "value": None, "status": "missing"},
    ]


def test_the_evidence_quote_stays_in_the_record_and_off_the_card():
    for output in _card(RUN)["outputs"]:
        assert set(output) == {"name", "label", "value", "status"}


def test_bookkeeping_and_the_outputs_are_not_repeated_as_facts():
    labels = [f["label"] for f in _card(RUN)["facts"]]
    assert "Run id" not in labels and "Recorded" not in labels
    assert "Supplier" not in labels and "Overbilled amount" not in labels
    assert "Tools used" in labels          # what it consulted is still a fact


def test_an_output_value_is_output_safety_gated_like_everything_else_that_leaves():
    leaked = {**RUN, "outputs": {"supplier": {
        "value": "see proc.bp_supplier", "status": "evidence_found", "evidence": "x"}}}
    assert "proc.bp_supplier" not in str(_card(leaked)["outputs"])


def test_a_built_in_agents_card_is_unchanged():
    card = _card({"summary": "3 suppliers ranked.", "ranked": [1, 2, 3]})
    assert "run_id" not in card and "outputs" not in card
    assert card["facts"] == [{"label": "Ranked", "value": "3 items"}]
