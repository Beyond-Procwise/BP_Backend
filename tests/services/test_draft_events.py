"""Abandon, assumption confirmation, readiness and the reviewer view -- and the HTTP surface over them."""

import json
import re
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.services.draft_assurance import capture
import api.routers.draft_assurance as router_mod


class MemDb:
    """The two email_agent tables, answering exactly the statements capture.py issues."""

    def __init__(self):
        self.captures, self.outcomes, self.autocommit = [], [], False

    def add_capture(self, unique_id="U-1", **over):
        row = {"capture_id": len(self.captures) + 1, "unique_id": unique_id, "family_id": "negotiation_counter",
               "family_version": 1, "mode": "shadow", "assurance_status": "needs_review", "family_source": "declared",
               "facts": {"supplier_current_offer": {"value": "47.50", "label": "Supplier's latest offer", "table": "supplier_response",
                                                      "column": "price", "row_id": "2", "retrieved_at": "t"}},
               "carried_unverified": {"lead_time": "7 days"}, "conflicts": [{"fact": "supplier_current_offer", "postgres": "47.50",
                                                                           "supplied": 49.0, "resolution": "postgres_wins"}],
               "reasoned": {}, "unverified_figures": ["25"], "violations": [{"kind": "x", "detail": "d", "severity": "warn"}],
               "brief": {"status": "ready", "goal": "g"}, "assumption_items": [
                   {"id": "response_deadline", "key": "response_deadline", "text": "deadline has no basis", "resolution": None},
                   {"id": "a2", "key": None, "text": "second", "resolution": None}],
               "assumptions_resolution": None, "judge": None, "authority": None, "tone_variables": {"escalation_level": 2},
               "tone_sources": {"escalation_level": {"source": "postgres"}}, "exemplar_ids": [], "exemplar_scope": "none",
               "clarification": None, "classification": None, "stage_status": {}, "ready": False, "needs_redraft": False,
               "initiated_by": "NegotiationAgent", "initiated_by_kind": "agent", "repaired": False}
        row.update(over)
        self.captures.append(row)
        return row

    def cursor(self):
        return MemCur(self)


class MemCur:
    def __init__(self, db):
        self.db, self._rows = db, []

    def __enter__(self): return self
    def __exit__(self, *a): return False

    def _mine(self, uid):
        return sorted([c for c in self.db.captures if c["unique_id"] == uid], key=lambda c: -c["capture_id"])

    def execute(self, sql, params=None):
        p = list(params or [])
        m = re.match(r"SELECT (.+?) FROM email_agent\.bp_draft_capture\s+WHERE unique_id = %s", " ".join(sql.split()), re.S)
        if m:
            cols = [c.strip() for c in m.group(1).split(",")]
            mine = self._mine(p[0])[:1]
            self._rows = [tuple(json.dumps(c[k]) if isinstance(c.get(k), (dict, list)) and k not in ("brief",) and False else c.get(k) for k in cols) for c in mine]
        elif "FROM email_agent.bp_draft_outcome WHERE capture_id = %s AND outcome = 'sent'" in sql:
            self._rows = [(1,)] if any(o["capture_id"] == p[0] and o["outcome"] == "sent" for o in self.db.outcomes) else []
        elif sql.lstrip().startswith("INSERT INTO email_agent.bp_draft_outcome"):
            self.db.outcomes.append({"capture_id": p[0], "outcome": "abandoned", "abandoned_by": p[1], "abandon_reason": p[2]})
            self._rows = [(len(self.db.outcomes),)]
        elif sql.lstrip().startswith("UPDATE email_agent.bp_draft_capture SET assumptions_resolution"):
            res, ready, redraft, _, cid = p
            row = next(c for c in self.db.captures if c["capture_id"] == cid)
            row.update(assumptions_resolution=json.loads(res), ready=ready, needs_redraft=redraft)
        else:
            raise AssertionError("unexpected SQL: " + sql)

    def fetchone(self): return self._rows[0] if self._rows else None
    def fetchall(self): return self._rows


# --- confirm ------------------------------------------------------------------------------------------

def test_confirming_every_assumption_makes_the_draft_ready():
    db = MemDb(); db.add_capture()
    r = capture.confirm_assumptions(db, "U-1", [{"id": "response_deadline", "action": "confirm"}, {"id": "a2", "action": "confirm"}], "nick")
    assert r == {"ok": True, "ready": True, "unresolved": [], "needs_redraft": False}
    assert db.captures[0]["assumptions_resolution"]["a2"]["by"] == "nick" and db.captures[0]["ready"] is True


def test_a_partial_confirmation_leaves_the_draft_not_ready_and_says_what_is_left():
    db = MemDb(); db.add_capture()
    r = capture.confirm_assumptions(db, "U-1", [{"id": "a2", "action": "confirm"}], "nick")
    assert r["ready"] is False and r["unresolved"] == ["response_deadline"]


def test_a_rejection_blocks_readiness_even_when_everything_is_answered():
    db = MemDb(); db.add_capture()
    r = capture.confirm_assumptions(db, "U-1", [{"id": "response_deadline", "action": "reject"}, {"id": "a2", "action": "confirm"}], "nick")
    assert r["ready"] is False and r["needs_redraft"] is True


def test_an_edit_means_the_text_is_stale_so_it_needs_a_redraft():
    db = MemDb(); db.add_capture()
    r = capture.confirm_assumptions(db, "U-1", [{"id": "response_deadline", "action": "edit", "value": "6 Nov"}, {"id": "a2", "action": "confirm"}], "nick")
    assert r["ready"] is False and r["needs_redraft"] is True
    assert db.captures[0]["assumptions_resolution"]["response_deadline"]["value"] == "6 Nov"


@pytest.mark.parametrize("bad,why", [
    ([{"id": "nope", "action": "confirm"}], "unknown assumption"),
    ([{"id": "a2", "action": "approve"}], "action must be one of"),
    ([{"id": "a2", "action": "edit"}], "needs a value"),
    ([{"id": "a2", "action": "edit", "value": ""}], "needs a value"),
])
def test_a_bad_confirmation_is_refused_whole_and_changes_nothing(bad, why):
    db = MemDb(); db.add_capture()
    r = capture.confirm_assumptions(db, "U-1", [{"id": "response_deadline", "action": "confirm"}] + bad, "nick")
    assert r["ok"] is False and why in r["error"]
    assert db.captures[0]["assumptions_resolution"] is None and db.captures[0]["ready"] is False


def test_confirmation_needs_a_named_person_and_an_existing_draft():
    db = MemDb(); db.add_capture()
    assert capture.confirm_assumptions(db, "U-1", [], "")["ok"] is False
    assert capture.confirm_assumptions(db, "U-404", [], "nick") == {"ok": False, "error": "no such draft"}


def test_the_classifiers_question_is_answered_like_any_assumption():
    clar = {"id": "clarification", "key": None, "text": "Is this A, or B?", "options": ["a", "b"], "resolution": None}
    db = MemDb(); db.add_capture(clarification={"question": "Is this A, or B?", "options": ["a", "b"]},
                                 assumption_items=[clar])
    r = capture.confirm_assumptions(db, "U-1", [], "nick")
    assert r["ready"] is False and r["unresolved"] == ["clarification"]
    assert capture.confirm_assumptions(db, "U-1", [{"id": "clarification", "action": "confirm"}], "nick")["ready"] is True


def test_choosing_the_other_family_means_a_redraft():
    clar = {"id": "clarification", "key": None, "text": "Is this A, or B?", "options": ["a", "b"], "resolution": None}
    db = MemDb(); db.add_capture(assumption_items=[clar])
    r = capture.confirm_assumptions(db, "U-1", [{"id": "clarification", "action": "edit", "value": "b"}], "nick")
    assert r["ready"] is False and r["needs_redraft"] is True


def test_confirmation_applies_to_the_latest_capture_only():
    db = MemDb(); db.add_capture(); db.add_capture()
    capture.confirm_assumptions(db, "U-1", [{"id": "a2", "action": "confirm"}], "nick")
    assert db.captures[0]["assumptions_resolution"] is None and db.captures[1]["assumptions_resolution"] is not None


# --- abandon ------------------------------------------------------------------------------------------

def test_abandoning_records_who_and_why():
    db = MemDb(); db.add_capture()
    assert capture.record_abandoned(db, "U-1", "nick", "wrong supplier") == 1
    assert db.outcomes[0]["abandoned_by"] == "nick" and db.outcomes[0]["abandon_reason"] == "wrong supplier"


def test_a_sent_draft_cannot_be_abandoned():
    db = MemDb(); db.add_capture(); db.outcomes.append({"capture_id": 1, "outcome": "sent"})
    assert capture.record_abandoned(db, "U-1", "nick") is None


def test_abandon_needs_a_person_and_a_known_draft():
    db = MemDb(); db.add_capture()
    assert capture.record_abandoned(db, "U-1", "") is None and capture.record_abandoned(db, "U-9", "nick") is None


# --- readiness ----------------------------------------------------------------------------------------------

def test_readiness_is_true_false_or_honestly_unknown():
    db = MemDb(); db.add_capture(ready=True)
    assert capture.readiness(db, "U-1") == {"ready": True, "needs_redraft": False}
    db.add_capture("U-2", ready=False, needs_redraft=True)
    assert capture.readiness(db, "U-2") == {"ready": False, "needs_redraft": True}
    assert capture.readiness(db, "U-9")["ready"] is None and capture.readiness(db, None)["ready"] is None


def test_readiness_that_cannot_be_read_is_unknown_not_ready():
    class Boom:
        def cursor(self): raise RuntimeError("down")
    r = capture.readiness(Boom(), "U-1")
    assert r["ready"] is None and "unreadable" in r["reason"]


# --- the view ---------------------------------------------------------------------------------------------------

def test_the_view_shows_labels_and_row_ids_and_never_internal_names():
    db = MemDb(); db.add_capture()
    v = capture.to_view(capture.load_raw(db, "U-1"), reviewed_by="buyer@acme.test")
    blob = json.dumps(v)
    assert "supplier_response" not in blob and '"table"' not in blob and '"column"' not in blob
    assert v["facts"]["supplier_current_offer"] == {"value": "47.50", "label": "Supplier's latest offer", "source": "postgres",
                                                    "row_id": "2", "retrieved_at": "t"}
    assert v["facts"]["lead_time"]["source"] == "carried_unverified"        # the unverified flag
    assert v["conflicts"][0]["label"] == "Supplier's latest offer" and v["conflicts"][0]["resolution"] == "postgres_wins"
    assert v["accountability"] == {"initiated_by": "NegotiationAgent", "reviewed_by": "buyer@acme.test"}
    assert v["unverified_figures"] == ["25"] and v["ready"] is False


def test_confidence_reaches_the_client_as_a_band_not_a_number_to_format():
    brief = {"status": "ready", "reasoned": {"a": {"value": 1, "basis": [], "confidence": 0.9},
                                              "b": {"value": 1, "basis": [], "confidence": 0.5},
                                              "c": {"value": 1, "basis": [], "confidence": 0.49},
                                              "d": {"value": 1, "basis": [], "confidence": None}}}
    db = MemDb(); db.add_capture(brief=brief)
    r = capture.to_view(capture.load_raw(db, "U-1"))["brief"]["reasoned"]
    assert [r[k]["confidence_label"] for k in "abcd"] == ["high", "medium", "low", None]


def test_a_resolution_shows_up_on_its_assumption_in_the_view():
    db = MemDb(); db.add_capture(assumptions_resolution={"a2": {"action": "confirm", "by": "nick", "at": "t"}})
    items = {a["id"]: a for a in capture.to_view(capture.load_raw(db, "U-1"))["assumptions"]}
    assert items["a2"]["resolution"]["action"] == "confirm" and items["response_deadline"]["resolution"] is None


# --- HTTP ---------------------------------------------------------------------------------------------------------

class P:
    subject = "sub-nick"


@pytest.fixture
def api(monkeypatch):
    db = MemDb(); db.add_capture()
    from contextlib import contextmanager

    @contextmanager
    def conn():
        yield db
    monkeypatch.setattr(router_mod, "_conn", conn)
    monkeypatch.setattr(router_mod, "_reviewer", lambda c, u: "buyer@acme.test")
    monkeypatch.setattr(router_mod.guardrail, "authorize", lambda *a, **k: SimpleNamespace(allowed=True))
    monkeypatch.setattr(router_mod.rbac, "policy_engine", lambda: None)
    app = FastAPI(); app.include_router(router_mod.router)
    app.dependency_overrides[router_mod.require_user] = lambda: P()
    client = TestClient(app)
    client.db, client.app_ = db, app
    return client


def test_get_assurance(api):
    r = api.get("/drafts/U-1/assurance")
    assert r.status_code == 200 and r.json()["accountability"]["reviewed_by"] == "buyer@acme.test"
    assert api.get("/drafts/U-404/assurance").status_code == 404


def test_confirm_over_http_returns_the_new_state(api):
    r = api.post("/drafts/U-1/assumptions/confirm", json={"confirmations": [
        {"id": "response_deadline", "action": "confirm"}, {"id": "a2", "action": "confirm"}]})
    assert r.status_code == 200 and r.json()["ready"] is True
    assert api.db.captures[0]["assumptions_resolution"]["a2"]["by"] == "sub-nick"     # the principal, not the body


def test_the_confirming_person_cannot_be_named_in_the_body(api):
    api.post("/drafts/U-1/assumptions/confirm", json={"confirmations": [{"id": "a2", "action": "confirm", "by": "someone-else"}],
                                                       "by": "someone-else"})
    assert api.db.captures[0]["assumptions_resolution"]["a2"]["by"] == "sub-nick"


def test_a_bad_confirmation_is_a_422_and_an_unknown_draft_a_404(api):
    assert api.post("/drafts/U-1/assumptions/confirm", json={"confirmations": [{"id": "zzz", "action": "confirm"}]}).status_code == 422
    assert api.post("/drafts/U-404/assumptions/confirm", json={"confirmations": []}).status_code == 404


def test_an_unidentified_caller_cannot_confirm_or_abandon(api):
    api.app_.dependency_overrides[router_mod.require_user] = lambda: SimpleNamespace(subject="")
    assert api.post("/drafts/U-1/assumptions/confirm", json={"confirmations": []}).status_code == 401
    assert api.post("/drafts/U-1/abandon", json={}).status_code == 401


def test_a_caller_the_policy_refuses_gets_a_403(api, monkeypatch):
    monkeypatch.setattr(router_mod.guardrail, "authorize", lambda *a, **k: SimpleNamespace(allowed=False))
    assert api.post("/drafts/U-1/assumptions/confirm", json={"confirmations": []}).status_code == 403


def test_abandon_over_http(api):
    assert api.post("/drafts/U-1/abandon", json={"reason": "wrong supplier"}).json() == {"ok": True}
    assert api.db.outcomes[0]["abandoned_by"] == "sub-nick"
    api.db.outcomes.append({"capture_id": 1, "outcome": "sent"})
    assert api.post("/drafts/U-1/abandon", json={}).status_code == 409


def test_preflight_reports_a_moved_fact_by_label(api, monkeypatch):
    monkeypatch.setattr(router_mod, "recheck_for_send", lambda conn, draft, eng: {
        "checked": True, "mode": "shadow", "changed": [{"fact": "supplier_current_offer", "was": "47.50", "now": "45.00"}]})
    r = api.post("/drafts/U-1/preflight").json()
    assert r["ok"] is False and r["changed"] == [{"fact": "supplier_current_offer", "label": "Supplier's latest offer",
                                                    "was": "47.50", "now": "45.00"}]


def test_preflight_is_ok_only_when_checked_unchanged_and_ready(api, monkeypatch):
    monkeypatch.setattr(router_mod, "recheck_for_send", lambda c, d, e: {"checked": True, "mode": "shadow", "changed": []})
    assert api.post("/drafts/U-1/preflight").json()["ok"] is False        # not ready: assumptions open
    api.db.captures[0]["ready"] = True
    assert api.post("/drafts/U-1/preflight").json()["ok"] is True
    monkeypatch.setattr(router_mod, "recheck_for_send", lambda c, d, e: {"checked": False, "mode": "shadow", "changed": [], "reason": "family unreadable"})
    r = api.post("/drafts/U-1/preflight").json()
    assert r["ok"] is False and r["checked"] is False and r["reason"] == "family unreadable"


def test_the_router_is_registered_in_the_authenticated_list():
    """Read main.py as text rather than importing the whole app: importing api.main from a test
    shifts module resolution for every test that runs after it (it broke test_serve_sell_side_demo)."""
    import ast
    from pathlib import Path
    tree = ast.parse((Path(__file__).resolve().parents[2] / "src/api/main.py").read_text())
    listed = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
              and any(getattr(t, "id", "") == "_AUTHENTICATED_ROUTERS" for t in n.targets)]
    assert listed, "_AUTHENTICATED_ROUTERS not found"
    names = {ast.unparse(e) for e in listed[0].value.elts}
    assert "draft_assurance_router.router" in names
    assert len([r for r in router_mod.router.routes]) == 4
