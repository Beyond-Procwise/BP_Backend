"""P8 phase 2, deal / governance routers: the caller reaches the handler.

The deal and governance half of the 49 write endpoints that were authenticated
at the router and never told WHO inside the handler.

Four of them took a name from the caller and wrote it: deal proposals'
confirmed_by and rejected_by, and extraction feedback's approver -- which
defaulted to the literal "api" and was written into proc.bp_prompt, the
governance table whose prompts override the code. Negotiation facts were
stated_by the literal "buyer".

As in the agent/workflow file: actor tests assert on what reaches the store,
and every endpoint is shown to take the principal by overriding require_user
with a 418 that only a handler which declares it can produce.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from api.auth import require_user

CALLER = "sub-real-caller"
IMPERSONATED = "sub-someone-else"


class _Principal:
    def __init__(self, subject):
        self.subject = subject
        self.email = f"{subject}@ourcompany.com"


class _Tripwire:
    def __getattr__(self, name):
        raise RuntimeError(f"handler ran without resolving the caller (touched .{name})")

    def __call__(self, *a, **k):
        raise RuntimeError("handler ran without resolving the caller")


def _app(router, *, subject=CALLER):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[require_user] = (
        (lambda: _Principal(subject)) if subject else (lambda: None)
    )
    return TestClient(app, raise_server_exceptions=False)


def _refusing_app(router):
    def _refuse():
        raise HTTPException(status_code=418, detail="principal resolved")

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[require_user] = _refuse
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# every endpoint in this group takes the principal
# ---------------------------------------------------------------------------
_ENDPOINTS = [
    # (router module, method, path, json body, attributes to trip on the module)
    ("deal_proposals", "POST", "/deals/proposals/generate", {"batch_deal_id": "B"}, ["get_conn"]),
    ("deal_proposals", "POST", "/deals/proposals/1/confirm", {"confirmed_by": "x"}, ["get_conn"]),
    ("deal_proposals", "POST", "/deals/proposals/1/reject", {"rejected_by": "x"}, ["get_conn"]),
    ("deal_proposals", "PATCH", "/deals/proposals/1/members", {}, ["get_conn"]),
    ("deal_summary", "POST", "/deals/analysis-summary/sync", None, []),
    ("deal_summary", "POST", "/deals/D-1/reconcile", None, ["reconcile_deal"]),
    ("deal_summary", "POST", "/deals/D-1/promote", None, ["promote_deal"]),
    ("deal_summary", "POST", "/deals/D-1/save-reference", None, ["save_reference"]),
    ("extraction_feedback", "POST", "/extraction/proposals/1/approve", {}, ["apply"]),
    ("extraction_feedback", "POST", "/extraction/proposals/1/reject", {}, ["apply"]),
    ("extraction_feedback", "POST", "/extraction/proposals/run", None, ["proposer"]),
    ("governance", "POST", "/agents/govern", {"task": "x"}, []),
    ("negotiate", "POST", "/deals/D-1/advice/message", {}, ["apply_turn"]),
    ("negotiate", "DELETE", "/deals/D-1/advice/fact/k", None, ["apply_turn"]),
    ("opportunities", "POST", "/opportunities/link-deals", None, ["link_opportunities_to_deals"]),
    ("opportunities", "POST", "/opportunities/sync", None, ["sync_findings_from_json"]),
    ("opportunities", "POST", "/opportunities/O-1/stage", {"stage": "negotiation"}, ["set_stage"]),
    ("promotion", "POST", "/promotion/canonicalize-po", None, ["canonicalize_po_references"]),
    ("summary", "POST", "/summary", {"persona": "cfo"}, ["summary_agent"]),
    ("summary", "POST", "/summary/precompute", {}, ["summary_agent"]),
    ("benchmark", "POST", "/benchmark/preview", {"quote": {}, "points": []}, ["evaluate", "ensure_registered"]),
]


@pytest.mark.parametrize("module_name,method,path,body,trip", _ENDPOINTS,
                         ids=[f"{m} {p}" for _, m, p, _, _ in _ENDPOINTS])
def test_the_handler_resolves_the_caller(monkeypatch, module_name, method, path, body, trip):
    module = __import__(f"api.routers.{module_name}", fromlist=["*"])
    for attr in trip:
        monkeypatch.setattr(module, attr, _Tripwire())
    if module_name == "deal_summary":
        import src.services.deal_analysis_service as das
        monkeypatch.setattr(das, "sync_deal_summaries", _Tripwire())
    if module_name == "governance":
        import src.services.governance_tools.governed_reasoning as gr
        monkeypatch.setattr(gr, "govern", _Tripwire())

    kwargs = {} if body is None else {"json": body}
    response = _refusing_app(module.router).request(method, path, **kwargs)

    assert response.status_code == 418, (
        f"{method} {path} never asked who the caller is "
        f"({response.status_code}: {response.text[:160]})")


# ---------------------------------------------------------------------------
# deal_proposals -- confirming mints a deal; the caller typed who confirmed it
# ---------------------------------------------------------------------------
class _Conn:
    autocommit = True

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def cursor(self):
        return object()

    def commit(self):
        pass

    def rollback(self):
        pass


def test_confirming_a_deal_proposal_is_attributed_to_the_token(monkeypatch):
    from api.routers import deal_proposals

    seen = {}
    monkeypatch.setattr(deal_proposals, "_confirm",
                        lambda pid, confirmed_by, expected:
                        seen.update(confirmed_by=confirmed_by) or {"status": "confirmed"})

    r = _app(deal_proposals.router).post(
        "/deals/proposals/1/confirm", json={"confirmed_by": IMPERSONATED})

    assert r.status_code == 200, r.text
    assert seen.get("confirmed_by") == CALLER, (
        f"a deal was minted on a typed name's authority: {seen}")


def test_rejecting_a_deal_proposal_is_attributed_to_the_token(monkeypatch):
    from api.routers import deal_proposals

    seen = {}
    monkeypatch.setattr(deal_proposals, "get_conn", lambda: _Conn())
    monkeypatch.setattr(deal_proposals.proposal_store, "reject_proposal",
                        lambda cur, pid, rejected_by: seen.update(rejected_by=rejected_by))

    r = _app(deal_proposals.router).post(
        "/deals/proposals/1/reject", json={"rejected_by": IMPERSONATED})

    assert r.status_code == 200, r.text
    assert seen.get("rejected_by") == CALLER, seen


def test_without_a_principal_a_deal_is_confirmed_by_nobody(monkeypatch):
    from api.routers import deal_proposals

    seen = {}
    monkeypatch.setattr(deal_proposals, "_confirm",
                        lambda pid, confirmed_by, expected:
                        seen.update(confirmed_by=confirmed_by) or {"status": "confirmed"})

    _app(deal_proposals.router, subject=None).post(
        "/deals/proposals/1/confirm", json={"confirmed_by": IMPERSONATED})

    assert "confirmed_by" in seen and seen["confirmed_by"] is None, seen


# ---------------------------------------------------------------------------
# extraction_feedback -- the approver is written into proc.bp_prompt
# ---------------------------------------------------------------------------
def test_approving_a_hint_proposal_is_attributed_to_the_token(monkeypatch):
    from api.routers import extraction_feedback

    seen = {}
    monkeypatch.setattr(extraction_feedback.apply, "approve",
                        lambda pid, approver: seen.update(approver=approver) or {"prompt_id": 1, "version": 1})

    r = _app(extraction_feedback.router).post(
        "/extraction/proposals/1/approve", json={"approver": IMPERSONATED})

    assert r.status_code == 200, r.text
    assert seen.get("approver") == CALLER, (
        f"a governance prompt was approved in a typed name: {seen}")


def test_rejecting_a_hint_proposal_is_attributed_to_the_token(monkeypatch):
    from api.routers import extraction_feedback

    seen = {}
    monkeypatch.setattr(extraction_feedback.apply, "reject",
                        lambda pid, approver, reason=None: seen.update(approver=approver))

    r = _app(extraction_feedback.router).post(
        "/extraction/proposals/1/reject", json={"approver": IMPERSONATED, "reason": "noise"})

    assert r.status_code == 200, r.text
    assert seen.get("approver") == CALLER, seen


def test_without_a_principal_a_hint_is_not_approved_as_api(monkeypatch):
    """The model's default was the literal "api", written to bp_prompt.created_by."""
    from api.routers import extraction_feedback

    seen = {}
    monkeypatch.setattr(extraction_feedback.apply, "approve",
                        lambda pid, approver: seen.update(approver=approver) or {"prompt_id": 1, "version": 1})

    _app(extraction_feedback.router, subject=None).post("/extraction/proposals/1/approve", json={})

    assert "approver" in seen and seen["approver"] is None, seen


class _RecordingCursor:
    def __init__(self, log):
        self.log = log
        self.rowcount = 1
        self._next = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=()):
        self.log.append((" ".join(sql.split()), params))
        if sql.lstrip().upper().startswith("SELECT DOC_TYPE"):
            self._next = ("invoice", "ACME", "tax_amount", "hint", "pending")
        elif "MAX(version)" in sql:
            self._next = (0,)
        elif "RETURNING prompt_id" in sql:
            self._next = (42,)

    def fetchone(self):
        return self._next


class _RecordingConn:
    def __init__(self, log):
        self.log = log

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def cursor(self):
        return _RecordingCursor(self.log)

    def commit(self):
        pass


def test_an_approval_by_nobody_writes_the_prompt_tables_own_default(monkeypatch):
    """proc.bp_prompt.created_by / last_modified_by are NOT NULL DEFAULT 'system'.

    With no principal the approval names nobody where it can (the proposal's
    reviewed_by is NULL) and, where the column refuses NULL, writes that
    column's own default rather than failing the approval -- and never "api"."""
    from src.services.extraction_feedback import apply

    log = []
    monkeypatch.setattr(apply, "get_conn", lambda: _RecordingConn(log))
    monkeypatch.setattr(apply.HINT_STORE, "refresh", lambda: None)
    monkeypatch.setattr(apply, "record_action", lambda **k: None)

    apply.approve(7, None)

    insert = next(p for s, p in log if s.startswith("INSERT INTO proc.bp_prompt"))
    supersede = next(p for s, p in log if s.startswith("UPDATE proc.bp_prompt"))
    reviewed = next(p for s, p in log if "reviewed_by" in s)
    assert insert[-2:] == ("system", "system"), insert
    assert supersede[0] == "system", supersede
    assert reviewed[0] is None, reviewed


# ---------------------------------------------------------------------------
# negotiate -- a buyer-stated fact was stated_by the literal "buyer"
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method,path,body", [
    ("POST", "/deals/D-1/advice/message",
     {"action": "state_fact", "fact_key": "volume", "fact_value": "100"}),
    ("DELETE", "/deals/D-1/advice/fact/volume", None),
])
def test_an_advice_turn_is_taken_as_the_token(monkeypatch, method, path, body):
    from api.routers import negotiate

    seen = {}
    monkeypatch.setattr(negotiate, "apply_turn",
                        lambda deal_id, message, **k: seen.update(k) or {"ok": True})

    kwargs = {} if body is None else {"json": body}
    r = _app(negotiate.router).request(method, path, **kwargs)

    assert r.status_code == 200, r.text
    assert seen.get("created_by") == CALLER, seen


def test_without_a_principal_a_stated_fact_is_stated_by_nobody(monkeypatch):
    """Not "buyer": the role was written into stated_by as though it named one."""
    from src.services.negotiation_advice import advisor

    stated = {}
    monkeypatch.setattr(advisor, "load_advice", lambda conn, deal_id: {"advice_id": "A-1"})
    monkeypatch.setattr(advisor, "state_fact", lambda conn, **k: stated.update(k))
    monkeypatch.setattr(advisor, "active_facts", lambda conn, advice_id: {})
    monkeypatch.setattr(advisor, "build_advice", lambda *a, **k: None)

    advisor.apply_turn("D-1", {"action": "state_fact", "fact_key": "volume",
                               "fact_value": "100"}, conn=object(), created_by=None)

    assert "stated_by" in stated and stated["stated_by"] is None, stated
