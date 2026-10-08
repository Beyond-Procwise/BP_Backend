"""Emails a PERSON typed (reply panel, report panel, manual passthrough) under assurance, in shadow.

The person is the author, so their figures are carried and listed as unverified unless they equal what
Postgres holds. Nothing is repaired or judged (no model writes these) and the text is never altered.
"""

import importlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.base_agent import AgentContext
from agents.email_drafting_agent import EmailDraftingAgent
from src.services import supplier_contact
from tests.services.test_draft_assurance import FakeConn, FakeCursor



def W():
    """The workflows router, imported when a test runs, not when this file is collected.

    tests/test_email_dispatch_service.py stubs sys.modules['services.backend_scheduler'] at ITS import; a
    module-level router import here (as in tests/api/test_email_prepare_endpoint.py) loads the real one first
    and breaks that file's three workflow-notification tests, in whichever order CI lists them.
    """
    return importlib.import_module("src.api.routers.workflows")


base_agent_mod = importlib.import_module("agents.base_agent")

MIGRATION = Path(__file__).resolve().parents[2] / "deploy/sql/2026-10-08_email_family_human_written.sql"
RULES = json.loads(MIGRATION.read_text().split("$json$")[1])["rules"]
MARKER = re.compile(r"<!-- PROCWISE_MARKER:.*?-->", re.S)

SUPPLIER = "PeopleFirst HR Solutions Ltd"
TABLES = {
    "supplier_response": [
        {"id": 1, "workflow_id": "WF-THREAD-1", "supplier_id": SUPPLIER, "round_number": 1, "price": 50, "currency": "GBP"},
        {"id": 2, "workflow_id": "WF-THREAD-1", "supplier_id": SUPPLIER, "round_number": 2, "price": 47.5, "currency": "GBP"},
    ],
    "bp_supplier": [{"supplier_id": SUPPLIER, "contact_name_1": "Alex Morgan"}],
}
SOURCE = {"unique_id": "PROC-WF-SRC", "workflow_id": "WF-THREAD-1", "supplier_id": SUPPLIER,
          "supplier_name": SUPPLIER, "payload": {}}


def _engine(with_family=True):
    rows = {"email_family_human_written": {"details": {"rules": RULES}, "version": 1, "policy_desc": "human"}} if with_family else {}
    return SimpleNamespace(get_policy=lambda slug: rows.get(slug), list_policies=lambda: [])


class Cur(FakeCursor):
    """Answers the assurance reads AND the draft INSERT the endpoint makes."""
    def execute(self, sql, params=None):
        n = " ".join(sql.split())
        self.conn.log.append(n)
        if n.startswith("INSERT INTO proc.draft_rfq_emails"):
            self.conn.inserts.append((n, params))
            self.last = (101,)
        elif "COALESCE(MAX(thread_index)" in n:
            self.last = (0,)
        else:
            self.last = None
            super().execute(sql, params)

    def fetchone(self):
        return getattr(self, "last", None)


class Conn(FakeConn):
    def __init__(self, tables):
        super().__init__(tables)
        self.inserts, self.committed = [], False

    def cursor(self):
        return Cur(self)

    def commit(self):
        self.committed = True


@pytest.fixture(autouse=True)
def _stubs(monkeypatch):
    monkeypatch.setattr(base_agent_mod, "configure_gpu", lambda *_, **__: "cpu")
    monkeypatch.setattr(W(), "gate", lambda *a, **k: None)
    monkeypatch.setattr(W().draft_rfq_emails_repo, "load_by_unique_id",
                        lambda uid: dict(SOURCE) if uid == "PROC-WF-SRC" else None)
    monkeypatch.setattr(W().workflow_email_tracking_repo, "lookup_dispatch_row", lambda **kw: None)
    monkeypatch.setattr(W(), "_load_draft_attachments", lambda d: [])


def _nick(conn, with_family=True):
    return SimpleNamespace(
        settings=SimpleNamespace(ses_default_sender="buyer@example.com"),
        prompt_engine=SimpleNamespace(get_prompt=lambda *a, **k: None), learning_repository=None,
        process_routing_service=SimpleNamespace(log_process=lambda **k: None, log_run_detail=lambda **k: None,
                                                log_action=lambda **k: None),
        _context_dataset_writer=SimpleNamespace(), workflow_memory=None,
        get_db_connection=lambda: conn, policy_engine=_engine(with_family))


def _agent(monkeypatch, with_family=True):
    conn = Conn(TABLES)
    agent = EmailDraftingAgent(_nick(conn, with_family))
    monkeypatch.setattr(agent, "_master_contact",
                        lambda sid: supplier_contact.SupplierContact(emails=["billing@peoplefirst.example.com"] if sid == SUPPLIER else [], name="Alex"))
    agent.model_calls = []
    monkeypatch.setattr(agent, "call_ollama", lambda **kw: agent.model_calls.append(kw) or {"message": {"content": ""}})
    return agent


def _assure(agent, text, recipients=("billing@peoplefirst.example.com",), supplier=SUPPLIER, user="sub-buyer-1"):
    return agent.assure_human_written(text=text, recipients=list(recipients), supplier_id=supplier,
                                      workflow_id="WF-THREAD-1", requested_by=user)


def test_a_figure_equal_to_the_suppliers_postgres_offer_is_verified_and_the_row_is_cited(monkeypatch):
    a = _assure(_agent(monkeypatch), "Thanks for your offer of 47.50 GBP. We accept.")
    f = a["facts"]["supplier_current_offer"]
    assert (f["value"], f["source"], f["table"], f["column"]) == ("47.5", "postgres", "supplier_response", "price")
    assert f["row_id"] == "2"                                    # the latest round, not round 1
    assert "47.50" not in a["unverified_figures"] and a["family_id"] == "human_written"


def test_a_figure_the_person_typed_that_postgres_does_not_hold_is_listed_unverified_not_failed(monkeypatch):
    a = _assure(_agent(monkeypatch), "Thanks for your offer of 47.50 GBP. We can do 41.20 GBP.")
    assert "41.20" in a["unverified_figures"] and "47.50" not in a["unverified_figures"]
    assert not [v for v in a["violations"] if v["severity"] == "fail"]
    assert a["status"] == "needs_review"


def test_bank_details_typed_by_a_person_fail_the_check(monkeypatch):
    a = _assure(_agent(monkeypatch), "Please pay to IBAN GB29NWBK60161331926819 using the new account.")
    assert any(v["severity"] == "fail" and "bank" in json.dumps(v).lower() for v in a["violations"])


def test_a_recipient_not_on_the_supplier_master_is_flagged(monkeypatch):
    a = _assure(_agent(monkeypatch), "Thanks, agreed.", recipients=("someone@elsewhere.example",))
    assert any(v["kind"] == "recipient_not_on_master" for v in a["violations"])


def test_the_person_is_recorded_as_the_initiator_and_no_model_is_called(monkeypatch):
    agent = _agent(monkeypatch)
    a = _assure(agent, "Thanks, agreed.")
    assert a["accountability"] == {"initiated_by": "sub-buyer-1", "kind": "user"}
    assert agent.model_calls == []
    assert a["judge"]["status"] == "unavailable"                 # nothing judged, and it says so


def test_no_family_row_gives_an_unassured_record_not_an_error(monkeypatch):
    a = _assure(_agent(monkeypatch, with_family=False), "Thanks, agreed.")
    assert a["status"] == "unassured"


def test_a_fault_while_assuring_never_reaches_the_caller(monkeypatch):
    agent = _agent(monkeypatch)
    monkeypatch.setattr(agent, "_assurance_prepare", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db down")))
    a = _assure(agent, "Thanks, agreed.")
    assert a["status"] == "unassured" and "db down" in a["reason"]


# ---- the reply panel / report panel endpoint ---------------------------------------------------------------------

def _prepare(agent_nick, **kw):
    return W().prepare_email_draft(
        W().EmailPrepareRequest(to=["billing@peoplefirst.example.com"], subject="RE: Negotiation",
                                body=kw.pop("body", "Thanks for 47.50 GBP. We can do 41.20 GBP."), **kw),
        agent_nick=agent_nick, principal=SimpleNamespace(subject="sub-buyer-1"))


def _persisted(conn):
    assert len(conn.inserts) == 1
    sql, params = conn.inserts[0]
    return json.loads(params[11]), params


def test_a_reply_draft_is_stored_with_its_assurance_and_its_intent():
    conn = Conn(TABLES)
    _prepare(_nick(conn), reply_to_unique_id="PROC-WF-SRC")
    payload, _ = _persisted(conn)
    a = payload["assurance"]
    assert a["family_id"] == "human_written" and a["facts"]["supplier_current_offer"]["row_id"] == "2"
    assert "41.20" in a["unverified_figures"]
    assert payload["metadata"]["intent"] == "REPLY_PANEL"


def test_a_report_panel_email_with_no_thread_has_its_own_intent():
    conn = Conn(TABLES)
    _prepare(_nick(conn), deal_id="DEAL-9")
    payload, _ = _persisted(conn)
    assert payload["metadata"]["intent"] == "REPORT_PANEL" and payload["assurance"]["family_id"] == "human_written"


def test_the_text_the_person_wrote_is_stored_exactly_as_before():
    with_wrap, without = Conn(TABLES), Conn(TABLES)
    _prepare(_nick(with_wrap), reply_to_unique_id="PROC-WF-SRC")
    _prepare(_nick(without, with_family=False), reply_to_unique_id="PROC-WF-SRC")
    strip = lambda s: re.sub(r"<!--.*?-->", "", s, flags=re.S)       # the random tracking marker is not the person's text
    assert strip(_persisted(with_wrap)[1][5]) == strip(_persisted(without)[1][5])


def test_an_assurance_fault_does_not_stop_the_reply_being_prepared(monkeypatch):
    monkeypatch.setattr(EmailDraftingAgent, "assure_human_written",
                        lambda self, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    conn = Conn(TABLES)
    resp = _prepare(_nick(conn), reply_to_unique_id="PROC-WF-SRC")
    assert resp.unique_id and conn.committed
    assert "assurance" not in _persisted(conn)[0] or _persisted(conn)[0]["assurance"]["status"] == "unassured"


# ---- the manual passthrough --------------------------------------------------------------------------------------

def test_a_manual_passthrough_draft_is_assured_with_its_own_intent_and_its_body_untouched(monkeypatch):
    def run(with_family):
        agent = _agent(monkeypatch, with_family)
        monkeypatch.setattr(agent, "_store_draft", lambda d: None)
        out = agent.run(AgentContext(workflow_id="WF-M", agent_id="email_drafting", user_id="u", input_data={
            "recipients": ["someone@elsewhere.example"], "subject": "Hello", "body": "<p>We can do 41.20 GBP.</p>",
            "ranking": [], "supplier_profiles": {}, "policies": []}))
        return out.data["drafts"][0]

    wrapped, plain = run(True), run(False)
    assert MARKER.sub("", wrapped["body"]).replace(wrapped["unique_id"], "") == MARKER.sub("", plain["body"]).replace(plain["unique_id"], "")
    assert wrapped["metadata"]["intent"] == "MANUAL_PASSTHROUGH"
    a = wrapped["assurance"]
    assert a["family_id"] == "human_written" and "41.20" in a["unverified_figures"]
    assert any(v["kind"] == "recipient_not_on_master" for v in a["violations"])   # no supplier => nothing on the master
