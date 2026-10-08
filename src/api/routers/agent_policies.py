"""Agent policies: screen endpoints (via the gateway only) and the orchestrator feed.

WHY A GATEWAY KEY (design §3.2, ruling C 2026-10-08): this service's own sign-in check
(ASK_AUTH_MODE) is off in development and unconfirmed in production, so a direct browser
call could not be shown to carry the gateway's checks. These routes therefore trust ONLY
the gateway: it verifies the Cognito token and the role, then forwards the verified
identity with a shared key. No key, wrong key, or unset key -> refused. The role is
re-derived here from the forwarded groups through the same role table every other gate
uses, and every write is audited before it returns.
"""
from __future__ import annotations

import hmac
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from api.auth import Principal
from repositories import agent_policy_repo as repo
from services import agent_actions, rbac
from services.agent_policy import conditions, contract, documents, readiness, run_runner, run_store, sections
from services.agent_policy.compiler import compile_policy
from services.agent_policy.registry import load_registry
from services.agent_policy.settings import load_settings
from services.db import get_conn

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/agent-policies", tags=["Agent policies"])
orchestrator_router = APIRouter(prefix="/orchestrator/agent-policies", tags=["Agent policies"])

_RANK = {"Viewer": 1, "Buyer": 2, "Approver": 3, "Admin": 4}


def _conn():
    return get_conn()


def _key_ok(given: Optional[str], env: str) -> None:
    expected = os.getenv(env)
    if not expected:
        raise HTTPException(status_code=503, detail="agent policies are not available right now")
    if not given or not hmac.compare_digest(given.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="not accepted")


def _role_of(principal: Principal) -> str:
    return rbac.effective_role(principal)


def gateway_principal(request: Request) -> Principal:
    _key_ok(request.headers.get("x-gateway-key"), "AGENT_POLICY_GATEWAY_KEY")
    sub = (request.headers.get("x-user-sub") or "").strip()
    if not sub:
        raise HTTPException(status_code=401, detail="not accepted")
    try:
        groups = json.loads(request.headers.get("x-user-groups") or "[]")
    except ValueError:
        groups = []
    if not isinstance(groups, list):
        groups = []
    return Principal(subject=sub, email=request.headers.get("x-user-email"),
                     claims={"cognito:groups": [str(g) for g in groups if isinstance(g, str)]})


def _require(principal: Principal, minimum: str, action: str, details: Dict[str, Any]) -> str:
    role = _role_of(principal)
    allowed = _RANK.get(role, 0) >= _RANK[minimum]
    if action != "agent_policy.read":
        agent_actions.record_action_or_fail(
            phase="authorize", action_type=action, agent="agent_policy_api",
            status="allowed" if allowed else "denied",
            summary=f"{role} {'may' if allowed else 'may not'} {action}",
            details={**details, "principal": principal.subject, "role": role, "minimum": minimum})
    if not allowed:
        raise HTTPException(status_code=403, detail=f"{action} needs the {minimum} role or higher")
    return role


class CreateBody(BaseModel):
    form: Dict[str, Any]


class VersionBody(BaseModel):
    form: Dict[str, Any]
    baseVersion: int
    intent: str = Field(pattern="^(draft|activate)$")
    changeNote: str = ""


class RetireBody(BaseModel):
    baseVersion: int
    changeNote: str = ""


class PreviewBody(BaseModel):
    form: Dict[str, Any]
    policyKey: Optional[str] = None
    version: Optional[int] = None


class AreaBody(BaseModel):
    subAreas: List[str]
    neverSuggest: bool
    secondReviewer: bool


@router.get("")
def list_policies(p: Principal = Depends(gateway_principal)):
    _require(p, "Viewer", "agent_policy.read", {})
    with _conn() as conn:
        return {"policies": repo.list_policies(conn)}


@router.get("/taxonomy")
def taxonomy(p: Principal = Depends(gateway_principal)):
    _require(p, "Viewer", "agent_policy.read", {})
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT area_name, id_prefix, sub_areas, never_suggest, second_reviewer, is_unassigned"
                    " FROM proc.bp_business_area ORDER BY is_unassigned, area_name")
        return {"areas": [{"areaName": r[0], "prefix": r[1], "subAreas": list(r[2]), "neverSuggest": r[3],
                           "secondReviewer": r[4], "unassigned": r[5]} for r in cur.fetchall()]}


@router.put("/taxonomy/{area}")
def update_area(area: str, body: AreaBody, p: Principal = Depends(gateway_principal)):
    _require(p, "Admin", "agent_policy.admin", {"area": area})
    subs = [s.strip() for s in body.subAreas if s.strip()]
    if "General" not in subs:
        subs.insert(0, "General")
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute("UPDATE proc.bp_business_area SET sub_areas=%s, never_suggest=%s, second_reviewer=%s,"
                    " last_modified_by=%s, last_modified_at=now() WHERE area_name=%s RETURNING area_name",
                    (subs, body.neverSuggest, body.secondReviewer, p.subject, area))
        if not cur.fetchone():
            raise HTTPException(status_code=404, detail="no such business area")
    return {"areaName": area, "subAreas": subs, "neverSuggest": body.neverSuggest, "secondReviewer": body.secondReviewer}


@router.post("/preview")
def preview(body: PreviewBody, p: Principal = Depends(gateway_principal)):
    role = _require(p, "Viewer", "agent_policy.read", {})
    registry, settings = load_registry(), load_settings()
    out: Dict[str, Any] = {"examples": conditions.reviewer_view(body.form, settings),
                           "howEnforced": readiness.how_enforced(body.form, registry, settings),
                           "problems": readiness.activation_problems(body.form, registry, settings),
                           "companyResponseTime": settings["response_time"]}
    if role == "Admin":
        # The business area's real never_suggest, looked up as a save looks it up, so the
        # technical view's learning.eligible is what the saved policy will say.
        with _conn() as conn:
            never_suggest = repo.never_suggest_for(conn, body.form.get("businessArea"))
        doc = compile_policy(body.form, policy_key=body.policyKey or "GEN-0000", version=body.version or 1,
                             status="draft", settings=settings, never_suggest=never_suggest)
        out["compiled"] = doc
        out["compiledProblems"] = contract.validate(doc, registry)
    return out


# ---------------------------------------------------------------- documents and extraction runs
# Declared before "/{key}" so "documents" and "extraction-runs" are never read as policy ids.

# Files and uploads stay plain dicts: documents.presign_uploads/register_uploads judge each one and
# say what is wrong in words, which arrives as the 422 {problems} shape rather than a schema error.
class UploadUrlsBody(BaseModel):
    files: List[Dict[str, Any]]


class RegisterBody(BaseModel):
    uploads: List[Dict[str, Any]]


class DocumentRef(BaseModel):
    documentId: int
    version: int


class ExtractionRunBody(BaseModel):
    documents: List[DocumentRef] = Field(min_length=1, max_length=20)


class FixBody(BaseModel):
    baseVersion: int
    flipped: List[Dict[str, Any]] = Field(min_length=1)


def _refused(message: str) -> JSONResponse:
    return JSONResponse(status_code=422, content={"problems": [
        {"field": "files", "code": "upload_refused", "message": message}]})


def _outside_uploads(key: str) -> bool:
    return not str(key or "").startswith(documents.UPLOAD_PREFIX)


def _missing_versions(conn, refs: List[DocumentRef]) -> List[str]:
    cur = conn.cursor()
    missing = []
    for ref in refs:
        cur.execute("SELECT 1 FROM proc.bp_policy_document_version WHERE document_id = %s AND version = %s",
                    (ref.documentId, ref.version))
        if cur.fetchone() is None:
            missing.append(f"Document {ref.documentId} version {ref.version} does not exist.")
    return missing


def _work(conn, run: Dict[str, Any], emit) -> Dict[str, Any]:
    """The runner's work: it opens the connection, claims the run, and hands both here."""
    from services.agent_policy import extraction_run  # local: pulls in the model client

    kind = (run or {}).get("kind")
    if kind == "extract":
        return extraction_run.run_extract(conn, run, emit)
    if kind == "fix":
        return extraction_run.run_fix(conn, run, emit)
    raise ValueError(f"unknown run kind {kind!r}")


def _start(kind: str, request_body: Dict[str, Any], actor: str) -> JSONResponse:
    with _conn() as conn:
        run = run_store.create(conn, kind=kind, request=request_body, actor=actor)
    run_runner.submit(run["run_id"], _work)
    return JSONResponse(status_code=202, content={"runId": run["run_id"]})


@router.post("/documents/upload-urls")
def upload_urls(body: UploadUrlsBody, p: Principal = Depends(gateway_principal)):
    _require(p, "Buyer", "agent_policy.write", {"intent": "upload_urls", "files": len(body.files)})
    try:
        uploads = documents.presign_uploads(body.files, actor=p.subject)
    except ValueError as exc:
        return _refused(str(exc))
    if any(_outside_uploads(u.get("key")) for u in uploads):
        logger.error("agent-policy presign issued a key outside %s", documents.UPLOAD_PREFIX)
        return _refused("The upload could not be prepared.")
    return {"uploads": uploads}


@router.post("/documents")
def register_documents(body: RegisterBody, p: Principal = Depends(gateway_principal)):
    _require(p, "Buyer", "agent_policy.write", {"intent": "register_documents",
                                                "keys": [str(u.get("key") or "") for u in body.uploads]})
    if not body.uploads:
        return _refused("No uploads were sent.")
    for u in body.uploads:
        if _outside_uploads(u.get("key")):
            return _refused(f"{u.get('name') or 'An upload'} is not an agent-policy upload.")
    with _conn() as conn:
        try:
            return {"documents": documents.register_uploads(conn, body.uploads, actor=p.subject)}
        except ValueError as exc:
            return _refused(str(exc))


@router.get("/documents")
def list_documents(p: Principal = Depends(gateway_principal)):
    _require(p, "Viewer", "agent_policy.read", {})
    with _conn() as conn:
        return {"documents": documents.list_documents(conn)}


@router.get("/documents/{document_id}/compare")
def compare_document(document_id: int, from_: int = Query(alias="from"), to: int = Query(),
                     p: Principal = Depends(gateway_principal)):
    _require(p, "Viewer", "agent_policy.read", {"document": document_id})
    with _conn() as conn:
        try:
            old = documents.document_text(conn, document_id, from_)
            new = documents.document_text(conn, document_id, to)
        except documents.DocumentUnreadable as exc:
            return JSONResponse(status_code=422, content={"problems": [
                {"field": "document", "code": "unreadable", "message": exc.reason}]})
        except ValueError:
            raise HTTPException(status_code=404, detail="no such document version")
    return {"sections": sections.diff_sections(old, new)}


@router.post("/extraction-runs")
def start_extraction(body: ExtractionRunBody, p: Principal = Depends(gateway_principal)):
    refs = [{"documentId": d.documentId, "version": d.version} for d in body.documents]
    _require(p, "Buyer", "agent_policy.write", {"intent": "extract", "documents": refs})
    with _conn() as conn:
        missing = _missing_versions(conn, body.documents)
    if missing:
        return JSONResponse(status_code=422, content={"problems": [
            {"field": "documents", "code": "not_found", "message": m} for m in missing]})
    return _start("extract", {"documents": refs}, p.subject)


@router.get("/extraction-runs")
def list_extraction_runs(p: Principal = Depends(gateway_principal)):
    _require(p, "Viewer", "agent_policy.read", {})
    with _conn() as conn:
        return {"runs": run_store.list_recent(conn)}


@router.get("/extraction-runs/{run_id}")
def get_extraction_run(run_id: int, afterSeq: int = Query(default=0, ge=0),
                       p: Principal = Depends(gateway_principal)):
    _require(p, "Viewer", "agent_policy.read", {"run": run_id})
    with _conn() as conn:
        run = run_store.get(conn, run_id, after_seq=afterSeq)
    if run is None:
        raise HTTPException(status_code=404, detail="no such extraction run")
    return run


@router.get("/{key}")
def get_one(key: str, p: Principal = Depends(gateway_principal)):
    role = _require(p, "Viewer", "agent_policy.read", {"policy": key})
    with _conn() as conn:
        try:
            got = repo.get_policy(conn, key)
        except repo.NotFound:
            raise HTTPException(status_code=404, detail="no such policy")
    if role != "Admin":
        for v in got["versions"]:
            v.pop("compiled", None)
    return got


@router.post("")
def create(body: CreateBody, p: Principal = Depends(gateway_principal)):
    _require(p, "Buyer", "agent_policy.write", {"intent": "create"})
    with _conn() as conn:
        try:
            return repo.create_draft(conn, body.form, actor=p.subject)
        except repo.NotReady as exc:
            return JSONResponse(status_code=422, content={"problems": exc.problems})


@router.post("/{key}/versions")
def save(key: str, body: VersionBody, p: Principal = Depends(gateway_principal)):
    minimum, action = ("Approver", "agent_policy.activate") if body.intent == "activate" else ("Buyer", "agent_policy.write")
    _require(p, minimum, action, {"policy": key, "intent": body.intent, "baseVersion": body.baseVersion})
    with _conn() as conn:
        try:
            return repo.save_version(conn, key, body.form, base_version=body.baseVersion, intent=body.intent,
                                     actor=p.subject, change_note=body.changeNote)
        except repo.StaleVersion as exc:
            raise HTTPException(status_code=409, detail=f"Someone saved a newer version ({exc}). Reload and try again.")
        except repo.NotReady as exc:
            return JSONResponse(status_code=422, content={"problems": exc.problems})
        except repo.NotFound:
            raise HTTPException(status_code=404, detail="no such policy")


@router.post("/{key}/agent-fix")
def agent_fix(key: str, body: FixBody, p: Principal = Depends(gateway_principal)):
    _require(p, "Buyer", "agent_policy.write", {"policy": key, "intent": "agent_fix",
                                                "baseVersion": body.baseVersion, "flipped": len(body.flipped)})
    if any(not isinstance(f.get("input"), dict) for f in body.flipped):
        return JSONResponse(status_code=422, content={"problems": [
            {"field": "flipped", "code": "invalid", "message": "Every flipped example needs its inputs."}]})
    with _conn() as conn:
        try:
            got = repo.get_policy(conn, key)
        except repo.NotFound:
            raise HTTPException(status_code=404, detail="no such policy")
    if not any(v.get("version") == body.baseVersion for v in got.get("versions") or []):
        raise HTTPException(status_code=404, detail="no such policy version")
    return _start("fix", {"policyKey": key, "baseVersion": body.baseVersion, "flipped": body.flipped}, p.subject)


@router.post("/{key}/retire")
def retire(key: str, body: RetireBody, p: Principal = Depends(gateway_principal)):
    _require(p, "Approver", "agent_policy.activate", {"policy": key, "intent": "retire"})
    with _conn() as conn:
        try:
            return repo.retire(conn, key, base_version=body.baseVersion, actor=p.subject, change_note=body.changeNote)
        except repo.StaleVersion as exc:
            raise HTTPException(status_code=409, detail=f"Someone saved a newer version ({exc}). Reload and try again.")
        except repo.InvalidTransition:
            raise HTTPException(status_code=409, detail="This policy is already retired.")
        except repo.NotFound:
            raise HTTPException(status_code=404, detail="no such policy")


@orchestrator_router.get("/v2/live")
def live_feed(request: Request):
    _key_ok(request.headers.get("x-orchestrator-key"), "AGENT_POLICY_ORCHESTRATOR_KEY")
    registry = load_registry()
    with _conn() as conn:
        docs = repo.live_documents(conn)
    good, refused = [], []
    for doc in docs:
        # One bad stored policy (even a non-dict row) must never take the whole feed down.
        doc_id = doc.get("id") if isinstance(doc, dict) else None
        try:
            problems = contract.validate(doc, registry)
        except Exception as exc:  # noqa: BLE001
            logger.exception("agent-policy feed could not validate %s", doc_id)
            problems = [f"could not be validated: {type(exc).__name__}"]
        if problems:
            refused.append({"id": doc_id, "problems": problems})
        else:
            good.append(doc)
    if refused:
        logger.warning("agent-policy feed refused %d live policies: %s", len(refused), [r["id"] for r in refused])
    return {"feed": "hard-policy-feed/2", "generatedAt": datetime.now(timezone.utc).isoformat(),
            "policies": good, "refused": refused}
