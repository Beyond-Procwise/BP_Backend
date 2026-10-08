"""The inbound analyser says HOW it read a price, and stamping that on the stored row can never break storing the reply.

What it does is unchanged: the first number in the email is the price unless a model returns one. What is new is that the choice is
recorded, because the first number in an email being stored as a price is exactly the thing a reader needs to know happened.
"""

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from agents.supplier_interaction_agent import SupplierInteractionAgent
from repositories import supplier_response_repo, workflow_email_tracking_repo


class Qdrant:
    def __getattr__(self, name):
        return lambda *a, **k: None


class Nick:
    def __init__(self):
        self.settings = SimpleNamespace(qdrant_collection_name="t", script_user="t", email_response_poll_seconds=1,
                                        email_response_timeout_seconds=5, email_response_batch_limit=3)
        self.qdrant_client = Qdrant()
        self.embedding_model = SimpleNamespace(encode=lambda *a, **k: [0.0])
        self.agents, self.dispatch_service_started = {}, True

    def get_db_connection(self):
        raise RuntimeError("no database in this test")


@pytest.fixture
def agent(monkeypatch):
    return SupplierInteractionAgent(Nick())


def parse(agent, monkeypatch, text, llm=None, model="agentnick:unified"):
    if llm is None:
        monkeypatch.setattr(agent, "call_ollama", lambda **kw: (_ for _ in ()).throw(RuntimeError("model unavailable")))
    else:
        monkeypatch.setattr(agent, "resolve_agent_model", lambda *a, **k: model, raising=False)
        monkeypatch.setattr(agent, "call_ollama", lambda **kw: {"response": json.dumps(llm)})
        monkeypatch.setattr(agent.agent_nick, "get_agent_model", lambda *a, **k: model, raising=False)
    return agent._parse_response(text, subject="Re: RFQ", rfq_id="RFQ-1", supplier_id="S-1")


def test_the_first_number_in_the_email_is_labelled_as_exactly_that(agent, monkeypatch):
    out = parse(agent, monkeypatch, "Hello, order 4471 ships in 7 days. Price 12.50 each.")
    assert out["price"] == 4471.0                                         # unchanged behaviour: the FIRST number
    assert out["extraction"]["price_method"] == "regex_first_number" and out["extraction"]["model"] is None


def test_a_price_the_model_returned_is_labelled_as_the_models_with_its_name(agent, monkeypatch):
    out = parse(agent, monkeypatch, "Order 4471: we can do twelve pounds fifty.", llm={"price": 12.5, "lead_time_days": 7, "summary": "ok"})
    assert out["price"] == 12.5 and out["extraction"]["price_method"] == "llm"
    assert out["extraction"]["model"] and out["extraction"]["prompt_version"].startswith("inline:")


def test_a_model_that_returns_no_price_leaves_the_regex_label(agent, monkeypatch):
    out = parse(agent, monkeypatch, "Order 4471.", llm={"price": None, "lead_time_days": None, "summary": "no figures"})
    assert out["extraction"]["price_method"] == "regex_first_number"


def test_an_email_with_no_number_has_no_price_method(agent, monkeypatch):
    out = parse(agent, monkeypatch, "Thanks very much, we will get back to you.")
    assert out["price"] is None and out["extraction"]["price_method"] is None


def test_the_lead_time_method_is_labelled_too(agent, monkeypatch):
    assert parse(agent, monkeypatch, "Price 9.99, delivery in 14 days.")["extraction"]["lead_time_method"] == "regex_days"
    out = parse(agent, monkeypatch, "Price 9.99.", llm={"price": 9.99, "lead_time_days": 21, "summary": "x"})
    assert out["extraction"]["lead_time_method"] == "llm"


def test_the_values_themselves_are_exactly_what_they_were_before_this_change(agent, monkeypatch):
    out = parse(agent, monkeypatch, "Price 12.50, delivery in 7 days.")
    assert (out["price"], out["lead_time"]) == (12.5, "7") and out["response_text"].startswith("Price")


# --- storing the reply --------------------------------------------------------------------------------------------------------

def store(agent, monkeypatch, parsed, stamp):
    inserted = []
    monkeypatch.setattr(supplier_response_repo, "init_schema", lambda: None)
    monkeypatch.setattr(workflow_email_tracking_repo, "lookup_workflow_for_unique", lambda **_: None)
    monkeypatch.setattr(workflow_email_tracking_repo, "lookup_dispatch_row", lambda **_: SimpleNamespace(
        dispatched_at=datetime(2026, 1, 1, tzinfo=timezone.utc), message_id="orig", subject="s"))
    monkeypatch.setattr(supplier_response_repo, "lookup_workflow_for_unique", lambda **_: None, raising=False)
    monkeypatch.setattr(supplier_response_repo, "insert_response", lambda row: inserted.append(row))
    monkeypatch.setattr(supplier_response_repo, "record_extraction", stamp)
    agent._store_response("wf-1", "S-1", "Price 12.50", parsed, unique_id="PROC-WF-1", message_id="<m1>")
    return inserted


PARSED = {"price": 12.5, "lead_time": "7", "response_text": "Price 12.50",
          "extraction": {"price_method": "llm", "lead_time_method": "regex_days", "model": "agentnick:unified", "prompt_version": "inline:x"}}


def test_storing_a_reply_stamps_its_provenance_after_the_insert(agent, monkeypatch):
    order, stamped = [], {}
    monkeypatch.setattr(supplier_response_repo, "insert_response", lambda row: order.append("insert"), raising=False)

    def stamp(**kw):
        order.append("stamp")
        stamped.update(kw)
        return True
    inserted = store(agent, monkeypatch, PARSED, stamp)
    assert len(inserted) == 1
    assert stamped["response_message_id"] == "<m1>" and stamped["unique_id"] == "PROC-WF-1" and stamped["method"] == "llm"
    assert stamped["model"] == "agentnick:unified" and stamped["prompt_version"] == "inline:x"


def test_a_failing_stamp_never_stops_the_reply_being_stored(agent, monkeypatch):
    def boom(**kw):
        raise RuntimeError("provenance columns missing")
    assert len(store(agent, monkeypatch, PARSED, boom)) == 1


def test_a_reply_with_no_extracted_value_is_not_stamped(agent, monkeypatch):
    calls = []
    store(agent, monkeypatch, {"price": None, "lead_time": None, "response_text": "thanks", "extraction": {"price_method": None, "lead_time_method": None}},
          lambda **kw: calls.append(kw))
    assert calls == []


def test_a_reply_parsed_before_this_change_has_no_extraction_block_and_is_stored_as_before(agent, monkeypatch):
    calls = []
    inserted = store(agent, monkeypatch, {"price": 12.5, "lead_time": "7", "response_text": "x"}, lambda **kw: calls.append(kw))
    assert len(inserted) == 1 and calls == []
