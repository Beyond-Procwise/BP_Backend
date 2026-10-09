"""The agent-policy screens through the WHOLE app -- output-safety middleware included.

User ruling 2026-10-08: the screens' reads and the upload-link answer carry the customer's own
policy text and file names, so their 2xx bodies pass the scrubber untouched. Exactly these
method+path pairs; a trailing slash, another method, an error answer or an unrelated route is
still scrubbed. Every test here goes through `api.main.app`, never a bare router.
"""
import copy
import json

import pytest
from fastapi.testclient import TestClient

from api.routers import agent_policies as R
from tests.agent_policy.fixtures import FORM_EXAMPLE, REGISTRY, SETTINGS

EXCERPT = "From 01/04/2026, refunds or credits above $500 need approval from the Finance Manager."
REASON = "The orchestrator does not receive the refund total at this checkpoint."
SAFE_NAMES = ["IT_SEC_POLICY_V3.pdf", "bp_rules.md"]
HDR = {"X-Gateway-Key": "k1", "X-User-Sub": "u1", "X-User-Groups": json.dumps(["PROCWISE_ADMIN"])}


class _FakeConnCtx:
    def __enter__(self): return object()
    def __exit__(self, *a): return False


def _policy():
    return {"policyKey": "FIN-0012", "title": "Refunds", "status": "live",
            "versions": [{"version": 1, "form": {"source": {"excerpt": EXCERPT}},
                          "problems": [{"code": "not_enforceable", "message": REASON}]}]}


def _run():
    return {"runId": 7, "status": "done", "events": [],
            "items": [{"excerpt": EXCERPT, "notEnforceable": [{"text": EXCERPT, "reason": REASON}]}]}


@pytest.fixture
def client(monkeypatch):
    from api.main import app
    monkeypatch.setenv("AGENT_POLICY_GATEWAY_KEY", "k1")
    monkeypatch.setenv("AGENT_POLICY_ORCHESTRATOR_KEY", "o1")
    monkeypatch.setattr(R, "load_registry", lambda conn=None: REGISTRY)
    monkeypatch.setattr(R, "load_settings", lambda conn=None: SETTINGS)
    monkeypatch.setattr(R, "_conn", _FakeConnCtx)
    monkeypatch.setattr(R, "_role_of", lambda principal: "Admin")
    monkeypatch.setattr(R.agent_actions, "record_action_or_fail", lambda **kw: None)
    monkeypatch.setattr(R.repo, "never_suggest_for", lambda conn, area: False)
    monkeypatch.setattr(R.repo, "list_policies", lambda conn: [_policy()])
    monkeypatch.setattr(R.repo, "get_policy", lambda conn, key, **_kw: _policy())
    monkeypatch.setattr(R.conflict_history, "viewer", lambda conn, p, *, is_admin: None)   # fake conn: no decider map
    monkeypatch.setattr(R.documents, "list_documents",
                        lambda conn: [{"documentId": 3, "title": SAFE_NAMES[0], "excerpt": EXCERPT}])
    monkeypatch.setattr(R.documents, "document_text", lambda conn, doc, ver: f"1. Refunds\n{EXCERPT} v{ver}")
    monkeypatch.setattr(R.sections, "diff_sections",
                        lambda old, new: [{"heading": "1. Refunds", "old": old, "new": new}])
    monkeypatch.setattr(R.run_store, "list_recent", lambda conn: [_run()])
    monkeypatch.setattr(R.run_store, "get", lambda conn, run_id, after_seq=0: _run())
    monkeypatch.setattr(R.documents, "issue_uploads",
                        lambda files, actor: [{"uploadId": f"u{i}", "safeName": f["name"]}
                                              for i, f in enumerate(files)])
    return TestClient(app)


def _withheld(body) -> bool:
    return "[withheld]" in json.dumps(body)


def test_get_one_policy_keeps_the_excerpt_and_reason(client):
    r = client.get("/agent-policies/FIN-0012", headers=HDR)
    assert r.status_code == 200
    v = r.json()["versions"][0]
    assert v["form"]["source"]["excerpt"] == EXCERPT
    assert v["problems"][0]["message"] == REASON


def test_list_policies_is_untouched(client):
    r = client.get("/agent-policies", headers=HDR)
    assert r.status_code == 200 and r.json() == {"policies": [_policy()]}


def test_compare_keeps_the_section_text(client):
    r = client.get("/agent-policies/documents/3/compare?from=1&to=2", headers=HDR)
    assert r.status_code == 200
    s = r.json()["sections"][0]
    assert s["old"] == f"1. Refunds\n{EXCERPT} v1" and s["new"] == f"1. Refunds\n{EXCERPT} v2"


def test_documents_list_is_untouched(client):
    r = client.get("/agent-policies/documents", headers=HDR)
    assert r.status_code == 200 and not _withheld(r.json())
    assert r.json()["documents"][0]["title"] == SAFE_NAMES[0]


def test_run_items_keep_excerpt_and_reason(client):
    r = client.get("/agent-policies/extraction-runs/7", headers=HDR)
    assert r.status_code == 200 and r.json() == _run()
    r = client.get("/agent-policies/extraction-runs", headers=HDR)
    assert r.status_code == 200 and r.json() == {"runs": [_run()]}


def test_preview_keeps_the_excerpt(client):
    form = copy.deepcopy(FORM_EXAMPLE)
    form["source"]["excerpt"] = EXCERPT
    r = client.post("/agent-policies/preview", json={"form": form}, headers=HDR)
    assert r.status_code == 200
    assert r.json()["compiled"]["source"]["excerpt"] == EXCERPT


def test_upload_urls_keep_safe_names(client):
    r = client.post("/agent-policies/documents/upload-urls",
                    json={"files": [{"name": n, "size": 10} for n in SAFE_NAMES]}, headers=HDR)
    assert r.status_code == 200
    assert [u["safeName"] for u in r.json()["uploads"]] == SAFE_NAMES


# ------------------------------------------------------------------ the negative cases


def test_trailing_slash_is_still_scrubbed(client):
    """`/agent-policies/` is routed (FastAPI redirects or 404s); either way it is not exempt."""
    from api import main as M
    assert not M._agent_policy_screen_exempt("GET", "/agent-policies/", 200)
    assert not M._agent_policy_screen_exempt("GET", "/agent-policies/FIN-0012/", 200)
    assert not M._agent_policy_screen_exempt("GET", "/agent-policies/fin-0012", 200)
    assert not M._agent_policy_screen_exempt("GET", "/agent-policies/FIN-0012/versions", 200)
    assert not M._agent_policy_screen_exempt("GET", "/agent-policies/documents/3/compare/x", 200)
    assert not M._agent_policy_screen_exempt("GET", "/agent-policies/extraction-runs/7\n", 200)


@pytest.fixture
def probe():
    """Registers a 2xx route on the real app for one test, then removes it."""
    from api.main import app
    added = []

    def _add(path):
        app.get(path)(lambda: {"excerpt": EXCERPT})
        added.append(path)
    yield _add
    app.router.routes[:] = [rt for rt in app.router.routes if getattr(rt, "path", None) not in added]


def test_trailing_slash_request_body_is_scrubbed(client, probe):
    """Through the app: a 2xx served on the slashed path still has its excerpt withheld."""
    probe("/agent-policies/")
    r = client.get("/agent-policies/", headers=HDR, follow_redirects=False)
    assert r.status_code == 200 and r.json()["excerpt"] == "[withheld]"


def test_another_method_on_a_read_path_is_scrubbed(client, monkeypatch):
    """POST /agent-policies/{key}/versions is not exempt, nor is a POST to a GET-only path."""
    from api import main as M
    assert not M._agent_policy_screen_exempt("POST", "/agent-policies/FIN-0012", 200)
    assert not M._agent_policy_screen_exempt("POST", "/agent-policies/documents", 200)
    assert not M._agent_policy_screen_exempt("GET", "/agent-policies/preview", 200)
    # Through the app: register (POST /documents) answers 2xx and is scrubbed.
    monkeypatch.setattr(R.documents, "register_uploads",
                        lambda conn, uploads, actor: [{"documentId": 3, "title": SAFE_NAMES[0],
                                                       "excerpt": EXCERPT}])
    r = client.post("/agent-policies/documents", json={"uploads": [{"uploadId": "u0", "name": "x"}]},
                    headers=HDR)
    assert r.status_code == 200
    d = r.json()["documents"][0]
    assert d["title"] == "[withheld]" and d["excerpt"] == "[withheld]"


def test_error_answers_are_not_exempt(client, monkeypatch):
    from api import main as M
    assert not M._agent_policy_screen_exempt("GET", "/agent-policies/FIN-0012", 404)
    assert not M._agent_policy_screen_exempt("POST", "/agent-policies/documents/upload-urls", 422)

    def _refuse(files, actor):
        raise ValueError(f"{SAFE_NAMES[0]} is not a type we accept")
    monkeypatch.setattr(R.documents, "issue_uploads", _refuse)
    r = client.post("/agent-policies/documents/upload-urls", json={"files": [{"name": "a"}]}, headers=HDR)
    assert r.status_code == 422
    assert r.json()["problems"][0]["message"] != f"{SAFE_NAMES[0]} is not a type we accept"


def test_an_unrelated_route_is_still_scrubbed(client, probe):
    from api import main as M
    assert not M._agent_policy_screen_exempt("GET", "/reports/jobs", 200)
    assert not M._agent_policy_screen_exempt("GET", "/agent-policiesX", 200)
    assert not M._agent_policy_screen_exempt("GET", "/x/agent-policies", 200)
    probe("/zz-unrelated-probe")
    r = client.get("/zz-unrelated-probe")
    assert r.status_code == 200 and r.json()["excerpt"] == "[withheld]"
