"""The RFQ batch under assurance (shadow): every supplier's draft is checked, none is changed.

An RFQ states what the BUYER asks for, so its figures are carried and reported unverified; a figure that
nothing carried is a violation. The templated body is never rewritten and no model is called for it.
"""

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.base_agent import AgentContext
from agents.email_drafting_agent import EmailDraftingAgent
from src.services import supplier_contact
from tests.services.test_draft_assurance import FakeConn, TABLES

MIGRATION = Path(__file__).resolve().parents[2] / "deploy/sql/2026-10-08_email_family_rfq_batch.sql"
RULES = json.loads(MIGRATION.read_text().split("$json$")[1])["rules"]

TABLES_BOTH = {**TABLES, "bp_supplier": [{"supplier_id": "S-1", "contact_name_1": "Alex Morgan"},
                                          {"supplier_id": "S-2", "contact_name_1": "Priya Nair"}]}
MARKER = re.compile(r"<!-- PROCWISE_MARKER:.*?-->", re.S)

RANKING = [{"supplier_id": "S-1", "supplier_name": "Acme"}, {"supplier_id": "S-2", "supplier_name": "Brightline"}]
PROFILES = {"S-1": {"items": ["steel brackets"]}, "S-2": {"items": ["steel brackets"]}}


def _engine(with_family=True):
    rows = {"email_family_rfq_batch": {"details": {"rules": RULES}, "version": 1, "policy_desc": "RFQ batch"}} \
        if with_family else {}
    return SimpleNamespace(get_policy=lambda slug: rows.get(slug), list_policies=lambda: [])


def _agent(monkeypatch, with_family=True, tables=None):
    agent = EmailDraftingAgent()
    monkeypatch.setattr(agent, "_master_contact",
                        lambda sid: supplier_contact.SupplierContact(emails=[f"{sid}@x.test"], name="Alex Morgan"))
    agent.agent_nick.get_db_connection = lambda: FakeConn(tables or TABLES_BOTH)
    agent.agent_nick.policy_engine = _engine(with_family)
    agent._model_calls = []
    monkeypatch.setattr(agent, "_store_draft", lambda draft: None)
    monkeypatch.setattr(agent, "call_ollama", lambda **kw: agent._model_calls.append(kw) or {"message": {"content": ""}})
    return agent


def _run(agent, **extra):
    out = agent.run(AgentContext(workflow_id="wf-1", agent_id="email_drafting", user_id="u", input_data={
        "ranking": RANKING, "supplier_profiles": PROFILES, "policies": [],
        "deadline": "30 October 2026", **extra}))
    return out.data["drafts"]


def test_every_supplier_draft_carries_an_assurance_record_for_the_rfq_family(monkeypatch):
    drafts = _run(_agent(monkeypatch))
    assert len(drafts) == 2
    for d in drafts:
        a = d["assurance"]
        assert a["family_id"] == "rfq_batch" and a["mode"] == "shadow"
        assert a["status"] != "unassured"
    names = {d["supplier_id"]: d["assurance"]["facts"]["supplier_contact_name"] for d in drafts}
    assert names["S-1"]["value"] == "Alex Morgan" and names["S-2"]["value"] == "Priya Nair"
    assert names["S-1"]["row_id"] == "S-1" and names["S-1"]["source"] == "postgres"    # traced to a row


def test_the_buyers_own_deadline_is_accepted_not_failed(monkeypatch):
    # KNOWN GAP (shared checker, all families): a carried DATE is accepted but not listed under
    # unverified_figures; only numeric figures are. See the RFQ note in the pending-verification spec.
    for d in _run(_agent(monkeypatch)):
        assert "30 October 2026" in d["body"]
        assert not [v for v in d["assurance"]["violations"] if v["severity"] == "fail"], d["assurance"]["violations"]


def test_a_quantity_the_buyer_carried_is_reported_unverified_not_failed(monkeypatch):
    agent = _agent(monkeypatch)
    real = agent._render_template_string
    monkeypatch.setattr(agent, "_render_template_string", lambda t, a: real(t, a) + "<p>Quantity required: 500 units.</p>")
    for d in _run(agent, line_items=[{"description": "steel brackets", "quantity": 500}]):
        a = d["assurance"]
        assert "500" in a["unverified_figures"]
        assert not [v for v in a["violations"] if v["severity"] == "fail"]
        assert a["status"] == "needs_review"          # unverified figures always ask a reviewer to look


def test_shadow_never_changes_the_body_and_calls_no_model(monkeypatch):
    wrapped = _agent(monkeypatch)
    plain = _agent(monkeypatch, with_family=False)
    bodies_wrapped = {d["supplier_id"]: MARKER.sub("", d["body"]) for d in _run(wrapped)}
    bodies_plain = {d["supplier_id"]: MARKER.sub("", d["body"]) for d in _run(plain)}
    assert bodies_wrapped == bodies_plain                    # behaviour-preserving
    assert wrapped._model_calls == []                        # a templated RFQ is neither judged nor repaired


def test_a_missing_family_row_leaves_the_drafts_going_out_marked_unassured(monkeypatch):
    drafts = _run(_agent(monkeypatch, with_family=False))
    assert len(drafts) == 2
    assert all(d["assurance"]["status"] == "unassured" for d in drafts)


def test_a_price_nothing_carried_is_a_violation(monkeypatch):
    agent = _agent(monkeypatch)
    real = agent._render_template_string
    monkeypatch.setattr(agent, "_render_template_string",
                        lambda t, a: real(t, a) + "<p>We can offer a unit price of 12.75 GBP.</p>")
    for d in _run(agent):
        assert any(v["severity"] == "fail" for v in d["assurance"]["violations"])


def test_an_assurance_fault_on_one_supplier_does_not_lose_the_batch(monkeypatch):
    agent = _agent(monkeypatch)
    real = agent._assurance_prepare
    calls = []

    def flaky(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("database fell over")
        return real(*a, **k)

    monkeypatch.setattr(agent, "_assurance_prepare", flaky)
    drafts = _run(agent)
    assert len(drafts) == 2 and all(d["body"] for d in drafts)
    assert sum(1 for d in drafts if d["assurance"]["status"] == "unassured") == 1


def test_the_rfq_assurance_is_captured_with_its_own_path(monkeypatch):
    for d in _run(_agent(monkeypatch)):
        assert d["metadata"]["intent"] == "RFQ_BATCH"
