"""A supplier's reply is read for what it says, and the counter answers it.

Live, 2026-10-14: the supplier replied that our counter was too short, gave no
reason and set a deadline, and that the tone was rude. It stated no price. The
counter the agent drafted ignored all of that and asked for "tiered price @
100/250/500" and a "Net45/60" payment trade, figures from nowhere in the deal:
- the model's reading of the reply called the complaint a final offer, and that
  alone switched the plan to a stock "package trade";
- with no price in the reply the agent had no offer at all, though the supplier's
  quote of 490.00 GBP was on file, so it fell back to stock clarifying asks;
- the email writer was never shown the reply, so it could not answer it.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import agents.negotiation.analysis as analysis
from agents.email_drafting_agent import EmailDraftingAgent
from agents.negotiation_agent import NegotiationAgent, decide_strategy

COMPLAINT = (
    "Hi,\r\n\r\nThis email is too short and you are asking for a reduction with no reason\r\n"
    "and a date in which I need to confirm by,  I find your tone rude and not\r\nacceptable.\r\n\r\n"
    "Regards\r\n\r\nOn Wed, Oct 14, 2026 at 7:42 AM <supplierconnect@procwise.co.uk> wrote:\r\n\r\n"
    "> Dear Nicholas Geelen,\r\n>\r\n> We formally propose a revised total of 455.00 GBP for the same items.\r\n"
    "> This is our final offer.\r\n"
)


def _signals(message, monkeypatch, model_says=None):
    monkeypatch.setattr(analysis, "LLM_ENABLED", model_says is not None)
    if model_says is not None:
        import services.ollama_client as client

        monkeypatch.setattr(client, "ollama_generate", lambda *a, **k: model_says)
    return analysis.extract_negotiation_signals(supplier_message=message, snippets=[], market={}, performance={})


def test_the_model_alone_cannot_call_a_complaint_a_final_offer(monkeypatch):
    signals = _signals(COMPLAINT, monkeypatch, model_says='{"tone": "firm", "finality_hint": true}')

    assert signals["finality_hint"] is False
    assert signals["tone"] == "firm"       # the model's other readings still count


def test_our_own_quoted_words_are_not_the_supplier_s(monkeypatch):
    """The quoted copy of our email says "final offer"; the supplier did not."""

    assert _signals(COMPLAINT, monkeypatch)["finality_hint"] is False


def test_a_governed_final_offer_phrase_still_counts(monkeypatch):
    assert _signals("Thanks. 470 GBP is our final price for these items.", monkeypatch)["finality_hint"] is True


def test_a_clarifying_ask_states_no_figures_of_its_own():
    plan = decide_strategy({"current_offer": None, "target_price": 455.0})

    assert plan["strategy"] == "clarify"
    assert plan["asks"]
    assert not any(ch.isdigit() for ask in plan["asks"] for ch in ask)


def test_the_email_writer_is_shown_what_the_supplier_wrote():
    position = EmailDraftingAgent._counter_position(
        {"current_offer": 490.0, "counter_price": 455.0, "currency": "GBP", "supplier_message": COMPLAINT})

    assert "I find your tone rude" in position
    assert "This is our final offer" not in position      # our own quoted email is not their words


def test_no_reply_no_reply_section():
    position = EmailDraftingAgent._counter_position({"current_offer": 490.0, "currency": "GBP"})

    assert "supplier wrote" not in position.lower()


class _Cursor:
    def __init__(self, row):
        self.row, self.params = row, None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.sql, self.params = sql, params

    def fetchone(self):
        return self.row


def _negotiator(row):
    agent = NegotiationAgent.__new__(NegotiationAgent)
    cursor = _Cursor(row)

    @contextmanager
    def _connect():
        yield SimpleNamespace(cursor=lambda: cursor)

    agent.agent_nick = SimpleNamespace(get_db_connection=_connect)
    return agent, cursor


def test_a_reply_with_no_price_keeps_the_offer_on_file():
    agent, cursor = _negotiator((490.0, "GBP"))

    assert agent._last_stated_offer("wf-1", "SUP-1") == (490.0, "GBP")
    assert cursor.params == ("wf-1", "SUP-1")


def test_no_offer_on_file_is_none():
    agent, _ = _negotiator(None)

    assert agent._last_stated_offer("wf-1", "SUP-1") == (None, None)
