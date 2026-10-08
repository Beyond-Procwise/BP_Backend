import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers import agent_policies as R
from tests.agent_policy.fixtures import FORM_EXAMPLE, REGISTRY, SETTINGS

GOOD = {"X-Gateway-Key": "k1", "X-User-Sub": "u1", "X-User-Email": "u1@x", "X-User-Groups": json.dumps(["PROCWISE_ADMIN"])}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("AGENT_POLICY_GATEWAY_KEY", "k1")
    monkeypatch.setenv("AGENT_POLICY_ORCHESTRATOR_KEY", "o1")
    roles = {"PROCWISE_ADMIN": "Admin", "PROCWISE_VIEWER": "Viewer", "PROCWISE_PROCUMENT_BUYER_ANALYST": "Buyer"}
    monkeypatch.setattr(R, "_role_of", lambda principal: roles.get((principal.claims or {}).get("cognito:groups", [""])[0], "Viewer"))
    audits = []
    monkeypatch.setattr(R.agent_actions, "record_action_or_fail", lambda **kw: audits.append(kw))
    monkeypatch.setattr(R, "load_registry", lambda conn=None: REGISTRY)
    monkeypatch.setattr(R, "load_settings", lambda conn=None: SETTINGS)
    monkeypatch.setattr(R, "_conn", _FakeConnCtx)
    monkeypatch.setattr(R.repo, "never_suggest_for", lambda conn, area: False)
    app = FastAPI(); app.include_router(R.router); app.include_router(R.orchestrator_router)
    c = TestClient(app); c.audits = audits
    return c


class _FakeConnCtx:
    def __enter__(self): return object()
    def __exit__(self, *a): return False


def test_missing_or_wrong_gateway_key_is_refused(client):
    assert client.get("/agent-policies").status_code == 401
    assert client.get("/agent-policies", headers={**GOOD, "X-Gateway-Key": "nope"}).status_code == 401


def test_unset_key_env_refuses_everything(client, monkeypatch):
    monkeypatch.delenv("AGENT_POLICY_GATEWAY_KEY")
    assert client.get("/agent-policies", headers=GOOD).status_code == 503


def test_viewer_cannot_create(client, monkeypatch):
    hdr = {**GOOD, "X-User-Groups": json.dumps(["PROCWISE_VIEWER"])}
    assert client.post("/agent-policies", json={"form": {"name": "x"}}, headers=hdr).status_code == 403
    assert client.audits and client.audits[-1]["status"] == "denied"


def test_buyer_creates_and_write_is_audited(client, monkeypatch):
    monkeypatch.setattr(R.repo, "create_draft", lambda conn, form, actor: {"policyKey": "GEN-0001", "version": 1})
    hdr = {**GOOD, "X-User-Groups": json.dumps(["PROCWISE_PROCUMENT_BUYER_ANALYST"])}
    r = client.post("/agent-policies", json={"form": {"name": "x"}}, headers=hdr)
    assert r.status_code == 200 and r.json() == {"policyKey": "GEN-0001", "version": 1}
    assert client.audits[-1]["action_type"] == "agent_policy.write" and client.audits[-1]["status"] == "allowed"


def test_buyer_cannot_activate(client):
    hdr = {**GOOD, "X-User-Groups": json.dumps(["PROCWISE_PROCUMENT_BUYER_ANALYST"])}
    body = {"form": FORM_EXAMPLE, "baseVersion": 1, "intent": "activate", "changeNote": ""}
    assert client.post("/agent-policies/GEN-0001/versions", json=body, headers=hdr).status_code == 403


def test_not_ready_returns_every_problem(client, monkeypatch):
    def refuse(*a, **k): raise R.repo.NotReady([{"field": "owner", "message": "Owner is required."}])
    monkeypatch.setattr(R.repo, "save_version", refuse)
    body = {"form": {"name": "x"}, "baseVersion": 1, "intent": "activate", "changeNote": ""}
    r = client.post("/agent-policies/GEN-0001/versions", json=body, headers=GOOD)
    assert r.status_code == 422 and r.json()["problems"][0]["field"] == "owner"


def test_stale_save_is_409(client, monkeypatch):
    def stale(*a, **k): raise R.repo.StaleVersion("latest is 3")
    monkeypatch.setattr(R.repo, "save_version", stale)
    body = {"form": {"name": "x"}, "baseVersion": 1, "intent": "draft", "changeNote": ""}
    assert client.post("/agent-policies/GEN-0001/versions", json=body, headers=GOOD).status_code == 409


def test_retiring_an_already_retired_policy_is_409(client, monkeypatch):
    def again(*a, **k): raise R.repo.InvalidTransition("already retired")
    monkeypatch.setattr(R.repo, "retire", again)
    r = client.post("/agent-policies/GEN-0001/retire", json={"baseVersion": 2, "changeNote": ""}, headers=GOOD)
    assert r.status_code == 409 and r.json()["detail"] == "This policy is already retired."


def test_preview_hides_json_from_non_admins(client):
    hdr = {**GOOD, "X-User-Groups": json.dumps(["PROCWISE_VIEWER"])}
    r = client.post("/agent-policies/preview", json={"form": FORM_EXAMPLE}, headers=hdr).json()
    assert "compiled" not in r and r["examples"][0]["label"] == "A person decides"
    r = client.post("/agent-policies/preview", json={"form": FORM_EXAMPLE}, headers=GOOD).json()
    assert r["compiled"]["schema"] == "hard-policy/2"


def test_feed_needs_its_own_key(client):
    assert client.get("/orchestrator/agent-policies/v2/live").status_code == 401
    assert client.get("/orchestrator/agent-policies/v2/live", headers={"X-Orchestrator-Key": "k1"}).status_code == 401


def _docs():
    from services.agent_policy.compiler import compile_policy
    good = dict(FORM_EXAMPLE, checked={"by": "u", "at": "2026-10-08T00:00:00Z"})
    bad = json.loads(json.dumps(good)); bad["hidden"]["condition"]["all"][0]["value"] = ["gone.tool"]
    return [compile_policy(good, policy_key="FIN-0001", version=1, status="live", settings=SETTINGS, never_suggest=False),
            compile_policy(bad, policy_key="FIN-0002", version=1, status="live", settings=SETTINGS, never_suggest=False)]


def test_feed_refuses_policy_whose_tool_left_registry(client, monkeypatch):
    docs = _docs()
    monkeypatch.setattr(R.repo, "live_documents", lambda conn: docs)
    r = client.get("/orchestrator/agent-policies/v2/live", headers={"X-Orchestrator-Key": "o1"}).json()
    assert [p["id"] for p in r["policies"]] == ["FIN-0001"]
    assert r["refused"][0]["id"] == "FIN-0002" and r["feed"] == "hard-policy-feed/2"


def test_one_document_that_cannot_be_validated_does_not_sink_the_feed(client, monkeypatch):
    docs = _docs()
    monkeypatch.setattr(R.repo, "live_documents", lambda conn: docs)
    real = R.contract.validate

    def flaky(doc, registry):
        if doc.get("id") == "FIN-0002":
            raise KeyError("boom")
        return real(doc, registry)
    monkeypatch.setattr(R.contract, "validate", flaky)
    r = client.get("/orchestrator/agent-policies/v2/live", headers={"X-Orchestrator-Key": "o1"})
    assert r.status_code == 200
    body = r.json()
    assert [p["id"] for p in body["policies"]] == ["FIN-0001"]
    assert body["refused"] == [{"id": "FIN-0002", "problems": ["could not be validated: KeyError"]}]


# ---- fix round 1 -------------------------------------------------------------------------
BUYER = {**GOOD, "X-User-Groups": json.dumps(["PROCWISE_PROCUMENT_BUYER_ANALYST"])}
VIEWER = {**GOOD, "X-User-Groups": json.dumps(["PROCWISE_VIEWER"])}
OKBODY = {"form": {"name": "x"}}


@pytest.mark.parametrize("sub", [None, "", "   "])
def test_blank_or_missing_user_sub_is_401(client, sub):
    hdr = {k: v for k, v in GOOD.items() if k != "X-User-Sub"}
    if sub is not None:
        hdr["X-User-Sub"] = sub
    assert client.get("/agent-policies", headers=hdr).status_code == 401


@pytest.mark.parametrize("groups", ["not json", json.dumps({"a": 1}), json.dumps("PROCWISE_ADMIN"), json.dumps([1, None])])
def test_malformed_or_non_list_groups_mean_no_groups(client, monkeypatch, groups):
    seen = []
    monkeypatch.setattr(R, "_role_of", lambda p: seen.append(p.claims["cognito:groups"]) or "Viewer")
    hdr = {**GOOD, "X-User-Groups": groups}
    assert client.post("/agent-policies", json=OKBODY, headers=hdr).status_code == 403
    assert seen == [[]]


def _policy():
    return {"policyKey": "FIN-0001", "versions": [{"version": 1, "compiled": {"a": 1}}, {"version": 2, "compiled": {"a": 2}}]}


def test_get_policy_strips_compiled_for_non_admin_only(client, monkeypatch):
    monkeypatch.setattr(R.repo, "get_policy", lambda conn, key: _policy())
    for hdr in (VIEWER, BUYER):
        assert all("compiled" not in v for v in client.get("/agent-policies/FIN-0001", headers=hdr).json()["versions"])
    assert all("compiled" in v for v in client.get("/agent-policies/FIN-0001", headers=GOOD).json()["versions"])


def test_feed_is_503_when_its_key_is_unset(client, monkeypatch):
    monkeypatch.delenv("AGENT_POLICY_ORCHESTRATOR_KEY")
    assert client.get("/orchestrator/agent-policies/v2/live", headers={"X-Orchestrator-Key": "o1"}).status_code == 503


class _Cur:
    def execute(self, *a, **k): pass
    def fetchone(self): return ("Finance",)


class _CurConn:
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def cursor(self): return _Cur()


def test_taxonomy_update_needs_admin(client, monkeypatch):
    monkeypatch.setattr(R, "_conn", _CurConn)
    body = {"subAreas": ["Refunds"], "neverSuggest": False, "secondReviewer": False}
    assert client.put("/agent-policies/taxonomy/Finance", json=body, headers=BUYER).status_code == 403
    r = client.put("/agent-policies/taxonomy/Finance", json=body, headers=GOOD)
    assert r.status_code == 200 and r.json()["subAreas"] == ["General", "Refunds"]


def test_retire_needs_approver(client, monkeypatch):
    monkeypatch.setattr(R.repo, "retire", lambda *a, **k: {"policyKey": "FIN-0001", "version": 3})
    body = {"baseVersion": 2, "changeNote": ""}
    assert client.post("/agent-policies/FIN-0001/retire", json=body, headers=BUYER).status_code == 403
    assert client.post("/agent-policies/FIN-0001/retire", json=body, headers=GOOD).status_code == 200


def test_roles_resolve_through_the_real_rbac_table(client, monkeypatch):
    """_role_of is left REAL: groups -> role goes through rbac with a stand-in policy engine."""
    from services import rbac
    from tests.guardrails.test_rbac import FakePolicyEngine, ROLE_ASSIGNMENT, ROLE_DEFINITION
    import copy
    assign = copy.deepcopy(ROLE_ASSIGNMENT)
    assign["details"]["rules"]["group_to_role"] = {"PROCWISE_ADMIN": "Admin", "PROCWISE_VIEWER": "Viewer"}
    engine = FakePolicyEngine({"role_definition": ROLE_DEFINITION, "role_assignment": assign})
    monkeypatch.setattr(R, "_role_of", lambda principal: rbac.effective_role(principal))
    monkeypatch.setattr(rbac, "_build_engine", lambda: engine)
    monkeypatch.setattr(rbac, "_load_role_assignments", lambda: {})
    rbac.reset_policy_cache()
    monkeypatch.setattr(R.repo, "create_draft", lambda conn, form, actor: {"policyKey": "GEN-0001", "version": 1})
    try:
        assert client.post("/agent-policies", json=OKBODY, headers=VIEWER).status_code == 403
        assert client.post("/agent-policies", json=OKBODY, headers=GOOD).status_code == 200
    finally:
        rbac.reset_policy_cache()


def test_a_non_dict_stored_document_is_refused_not_a_500(client, monkeypatch):
    good = _docs()[0]
    monkeypatch.setattr(R.repo, "live_documents", lambda conn: [None, good])
    r = client.get("/orchestrator/agent-policies/v2/live", headers={"X-Orchestrator-Key": "o1"})
    assert r.status_code == 200
    body = r.json()
    assert [p["id"] for p in body["policies"]] == ["FIN-0001"] and len(body["refused"]) == 1
    assert body["refused"][0]["id"] is None


# ---- final review fixes ------------------------------------------------------------------
def test_preview_returns_the_company_response_time(client):
    r = client.post("/agent-policies/preview", json={"form": FORM_EXAMPLE}, headers=VIEWER).json()
    assert r["companyResponseTime"] == SETTINGS["response_time"] == "PT4H"
    r = client.post("/agent-policies/preview", json={"form": FORM_EXAMPLE}, headers=GOOD).json()
    assert r["companyResponseTime"] == "PT4H"


def test_preview_uses_the_business_areas_real_never_suggest(client, monkeypatch):
    asked = []
    monkeypatch.setattr(R.repo, "never_suggest_for", lambda conn, area: asked.append(area) or area == "Finance")
    r = client.post("/agent-policies/preview", json={"form": FORM_EXAMPLE}, headers=GOOD).json()
    assert asked == ["Finance"] and r["compiled"]["learning"]["eligible"] is False
    other = dict(FORM_EXAMPLE, businessArea="Operations")
    r = client.post("/agent-policies/preview", json={"form": other}, headers=GOOD).json()
    assert r["compiled"]["learning"]["eligible"] is True


def test_create_with_withheld_text_is_a_422(client, monkeypatch):
    def refuse(conn, form, actor): raise R.repo.NotReady([{"field": "form", "code": "withheld_text", "message": "m"}])
    monkeypatch.setattr(R.repo, "create_draft", refuse)
    r = client.post("/agent-policies", json={"form": {"name": "[withheld]"}}, headers=BUYER)
    assert r.status_code == 422 and r.json()["problems"][0]["code"] == "withheld_text"


# ---- stage 2: documents, extraction runs, agent-fix ----------------------------------------
UPKEY = "agent-policy-documents/uploads/3d0b8f2e-1111-4a4a-8888-123456789abc/policy.pdf"


def _audited(client, intent):
    return [a for a in client.audits if a["details"].get("intent") == intent]


@pytest.mark.parametrize("method,path,body", [
    ("post", "/agent-policies/documents/upload-urls", {"files": [{"name": "a.pdf", "size": 1}]}),
    ("post", "/agent-policies/documents", {"uploads": [{"key": UPKEY, "name": "policy.pdf"}]}),
    ("post", "/agent-policies/extraction-runs", {"documents": [{"documentId": 1, "version": 1}]}),
    ("post", "/agent-policies/FIN-0001/agent-fix", {"baseVersion": 1, "flipped": [{"input": {}}]}),
])
def test_stage2_writes_need_buyer_and_the_refusal_is_audited(client, monkeypatch, method, path, body):
    called = []
    for name in ("presign_uploads", "register_uploads"):
        monkeypatch.setattr(R.documents, name, lambda *a, **k: called.append(1) or [])
    monkeypatch.setattr(R.run_store, "create", lambda *a, **k: called.append(1) or {"run_id": 1})
    r = client.request(method, path, json=body, headers=VIEWER)
    assert r.status_code == 403 and not called
    assert client.audits[-1]["status"] == "denied" and client.audits[-1]["action_type"] == "agent_policy.write"


@pytest.mark.parametrize("path", ["/agent-policies/documents", "/agent-policies/documents/1/compare?from=1&to=2",
                                  "/agent-policies/extraction-runs", "/agent-policies/extraction-runs/1"])
def test_stage2_reads_need_the_gateway_key(client, path):
    assert client.get(path).status_code == 401


def test_upload_urls_returns_uploads_and_is_audited_first(client, monkeypatch):
    order = []
    monkeypatch.setattr(R.agent_actions, "record_action_or_fail", lambda **kw: order.append(("audit", kw)))
    out = [{"uploadId": "u", "key": UPKEY, "url": "https://s3/x", "headers": {}}]
    monkeypatch.setattr(R.documents, "presign_uploads", lambda files, actor: order.append(("presign", files, actor)) or out)
    r = client.post("/agent-policies/documents/upload-urls", json={"files": [{"name": "a.pdf", "size": 3}]}, headers=BUYER)
    assert r.status_code == 200 and r.json() == {"uploads": out}
    assert [o[0] for o in order] == ["audit", "presign"]
    assert order[0][1]["status"] == "allowed" and order[0][1]["details"]["intent"] == "upload_urls"
    assert order[1][1] == [{"name": "a.pdf", "size": 3}] and order[1][2] == "u1"


@pytest.mark.parametrize("files", [[], [{"name": "a.exe", "size": 1}], [{"name": "a.pdf"}]])
def test_upload_refusal_is_a_422_problems_body(client, monkeypatch, files):
    def refuse(files, actor): raise ValueError("a.exe: only .docx, .md, .pdf, .txt files are accepted.")
    monkeypatch.setattr(R.documents, "presign_uploads", refuse)
    r = client.post("/agent-policies/documents/upload-urls", json={"files": files}, headers=BUYER)
    assert r.status_code == 422
    assert r.json() == {"problems": [{"field": "files", "code": "upload_refused",
                                      "message": "a.exe: only .docx, .md, .pdf, .txt files are accepted."}]}


def test_intake_limits_503_passes_through(client, monkeypatch):
    def unavailable(files, actor): raise R.HTTPException(status_code=503, detail="intake limits are not set")
    monkeypatch.setattr(R.documents, "presign_uploads", unavailable)
    r = client.post("/agent-policies/documents/upload-urls", json={"files": [{"name": "a.pdf", "size": 1}]}, headers=BUYER)
    assert r.status_code == 503


def test_presign_key_outside_uploads_is_refused(client, monkeypatch):
    monkeypatch.setattr(R.documents, "presign_uploads",
                        lambda files, actor: [{"uploadId": "u", "key": "agent-policy-documents/other/x.pdf", "url": "u"}])
    r = client.post("/agent-policies/documents/upload-urls", json={"files": [{"name": "x.pdf", "size": 1}]}, headers=BUYER)
    assert r.status_code == 422 and r.json()["problems"][0]["code"] == "upload_refused"


@pytest.mark.parametrize("key", ["agent-policy-documents/other/x.pdf", "contracts/x.pdf", "", None,
                                 "x/agent-policy-documents/uploads/a/b.pdf"])
def test_register_refuses_a_key_outside_uploads_without_touching_s3(client, monkeypatch, key):
    called = []
    monkeypatch.setattr(R.documents, "register_uploads", lambda *a, **k: called.append(1) or [])
    r = client.post("/agent-policies/documents", json={"uploads": [{"key": key, "name": "x.pdf"}]}, headers=BUYER)
    assert r.status_code == 422 and r.json()["problems"][0]["field"] == "files" and not called
    assert _audited(client, "register_documents")[-1]["status"] == "allowed"


def test_register_returns_documents_and_maps_valueerror(client, monkeypatch):
    seen = []
    monkeypatch.setattr(R.documents, "register_uploads",
                        lambda conn, uploads, actor: seen.append((uploads, actor)) or [{"documentId": 4, "version": 2}])
    body = {"uploads": [{"key": UPKEY, "name": "policy.pdf", "revisionOf": 4}]}
    r = client.post("/agent-policies/documents", json=body, headers=BUYER)
    assert r.status_code == 200 and r.json() == {"documents": [{"documentId": 4, "version": 2}]}
    assert seen == [(body["uploads"], "u1")]

    def refuse(conn, uploads, actor): raise ValueError("Document 4 does not exist.")
    monkeypatch.setattr(R.documents, "register_uploads", refuse)
    r = client.post("/agent-policies/documents", json=body, headers=BUYER)
    assert r.status_code == 422 and r.json()["problems"] == [
        {"field": "files", "code": "upload_refused", "message": "Document 4 does not exist."}]


def test_register_with_no_uploads_is_refused(client):
    r = client.post("/agent-policies/documents", json={"uploads": []}, headers=BUYER)
    assert r.status_code == 422 and r.json()["problems"][0]["code"] == "upload_refused"


def test_viewer_lists_documents_unaudited(client, monkeypatch):
    monkeypatch.setattr(R.documents, "list_documents", lambda conn: [{"documentId": 1}])
    r = client.get("/agent-policies/documents", headers=VIEWER)
    assert r.status_code == 200 and r.json() == {"documents": [{"documentId": 1}]} and client.audits == []


def test_compare_diffs_the_two_versions(client, monkeypatch):
    texts = {1: "1.1 Old words here.\n", 2: "1.1 New words here.\n"}
    monkeypatch.setattr(R.documents, "document_text", lambda conn, d, v: texts[v])
    r = client.get("/agent-policies/documents/7/compare?from=1&to=2", headers=VIEWER)
    assert r.status_code == 200
    assert r.json() == {"sections": R.sections.diff_sections(texts[1], texts[2])}


def test_compare_missing_version_is_404_and_unreadable_is_422(client, monkeypatch):
    def missing(conn, d, v): raise ValueError("Document 7 version 9 does not exist.")
    monkeypatch.setattr(R.documents, "document_text", missing)
    assert client.get("/agent-policies/documents/7/compare?from=1&to=9", headers=VIEWER).status_code == 404

    def unreadable(conn, d, v): raise R.documents.DocumentUnreadable("scan.pdf: no text could be read from the document")
    monkeypatch.setattr(R.documents, "document_text", unreadable)
    r = client.get("/agent-policies/documents/7/compare?from=1&to=2", headers=VIEWER)
    assert r.status_code == 422 and r.json()["problems"][0]["code"] == "unreadable"


@pytest.mark.parametrize("path", ["/agent-policies/documents/x/compare?from=1&to=2",
                                  "/agent-policies/documents/7/compare?from=a&to=2",
                                  "/agent-policies/documents/7/compare?to=2",
                                  "/agent-policies/extraction-runs/abc",
                                  "/agent-policies/extraction-runs/1?afterSeq=-1"])
def test_bad_ids_are_refused(client, path):
    assert client.get(path, headers=VIEWER).status_code == 422


def test_start_extraction_is_202_files_the_run_and_submits_it(client, monkeypatch):
    created, submitted = [], []
    monkeypatch.setattr(R, "_missing_versions", lambda conn, refs: [])
    monkeypatch.setattr(R.run_store, "create",
                        lambda conn, kind, request, actor: created.append((kind, request, actor)) or {"run_id": 31})
    monkeypatch.setattr(R.run_runner, "submit", lambda run_id, work: submitted.append((run_id, work)))
    body = {"documents": [{"documentId": 4, "version": 2}]}
    r = client.post("/agent-policies/extraction-runs", json=body, headers=BUYER)
    assert r.status_code == 202 and r.json() == {"runId": 31}
    assert created == [("extract", body, "u1")] and submitted == [(31, R._work)]
    assert _audited(client, "extract")[-1]["status"] == "allowed"


def test_start_extraction_refuses_unknown_versions_and_empty_lists(client, monkeypatch):
    monkeypatch.setattr(R, "_missing_versions", lambda conn, refs: ["Document 4 version 9 does not exist."])
    monkeypatch.setattr(R.run_store, "create", lambda *a, **k: pytest.fail("no run for a missing version"))
    r = client.post("/agent-policies/extraction-runs", json={"documents": [{"documentId": 4, "version": 9}]}, headers=BUYER)
    assert r.status_code == 422 and r.json()["problems"][0]["field"] == "documents"
    assert client.post("/agent-policies/extraction-runs", json={"documents": []}, headers=BUYER).status_code == 422


def test_runs_list_and_one_run_with_after_seq(client, monkeypatch):
    monkeypatch.setattr(R.run_store, "list_recent", lambda conn: [{"run_id": 2}, {"run_id": 1}])
    asked = []
    monkeypatch.setattr(R.run_store, "get", lambda conn, run_id, after_seq=0: asked.append((run_id, after_seq))
                        or ({"run_id": run_id, "items": []} if run_id == 2 else None))
    assert client.get("/agent-policies/extraction-runs", headers=VIEWER).json() == {"runs": [{"run_id": 2}, {"run_id": 1}]}
    assert client.get("/agent-policies/extraction-runs/2?afterSeq=5", headers=VIEWER).json() == {"run_id": 2, "items": []}
    assert client.get("/agent-policies/extraction-runs/2", headers=VIEWER).status_code == 200
    assert client.get("/agent-policies/extraction-runs/3", headers=VIEWER).status_code == 404
    assert asked == [(2, 5), (2, 0), (3, 0)] and client.audits == []


def test_documents_and_runs_are_not_read_as_policy_ids(client, monkeypatch):
    monkeypatch.setattr(R.repo, "get_policy", lambda conn, key: pytest.fail(f"read {key} as a policy"))
    monkeypatch.setattr(R.documents, "list_documents", lambda conn: [])
    monkeypatch.setattr(R.run_store, "list_recent", lambda conn: [])
    assert client.get("/agent-policies/documents", headers=VIEWER).status_code == 200
    assert client.get("/agent-policies/extraction-runs", headers=VIEWER).status_code == 200


def test_agent_fix_is_202_with_the_request_stored(client, monkeypatch):
    created, submitted = [], []
    monkeypatch.setattr(R.repo, "get_policy", lambda conn, key: _policy())
    monkeypatch.setattr(R.run_store, "create",
                        lambda conn, kind, request, actor: created.append((kind, request, actor)) or {"run_id": 8})
    monkeypatch.setattr(R.run_runner, "submit", lambda run_id, work: submitted.append(run_id))
    flipped = [{"input": {"amount": 600}, "agentExpected": "allow", "flipped": True}]
    r = client.post("/agent-policies/FIN-0001/agent-fix", json={"baseVersion": 2, "flipped": flipped}, headers=BUYER)
    assert r.status_code == 202 and r.json() == {"runId": 8}
    assert created == [("fix", {"policyKey": "FIN-0001", "baseVersion": 2, "flipped": flipped}, "u1")]
    assert submitted == [8] and _audited(client, "agent_fix")[-1]["status"] == "allowed"


def test_agent_fix_refuses_unknown_policy_version_and_bad_flips(client, monkeypatch):
    monkeypatch.setattr(R.run_store, "create", lambda *a, **k: pytest.fail("no run"))
    monkeypatch.setattr(R.repo, "get_policy", lambda conn, key: _policy())
    ok = [{"input": {"a": 1}}]
    assert client.post("/agent-policies/FIN-0001/agent-fix", json={"baseVersion": 9, "flipped": ok}, headers=BUYER).status_code == 404
    assert client.post("/agent-policies/FIN-0001/agent-fix", json={"baseVersion": 1, "flipped": []}, headers=BUYER).status_code == 422
    assert client.post("/agent-policies/FIN-0001/agent-fix", json={"baseVersion": 1, "flipped": [{"input": 3}]},
                       headers=BUYER).status_code == 422

    def gone(conn, key): raise R.repo.NotFound(key)
    monkeypatch.setattr(R.repo, "get_policy", gone)
    assert client.post("/agent-policies/FIN-0009/agent-fix", json={"baseVersion": 1, "flipped": ok}, headers=BUYER).status_code == 404


def test_work_dispatches_on_the_run_kind(monkeypatch):
    import services.agent_policy.extraction_run as ER
    monkeypatch.setattr(ER, "run_extract", lambda conn, run, emit: {"did": "extract"})
    monkeypatch.setattr(ER, "run_fix", lambda conn, run, emit: {"did": "fix"})
    assert R._work(object(), {"kind": "extract"}, None) == {"did": "extract"}
    assert R._work(object(), {"kind": "fix"}, None) == {"did": "fix"}
    with pytest.raises(ValueError):
        R._work(object(), {"kind": "other"}, None)
