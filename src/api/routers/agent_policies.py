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
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from api.auth import Principal
from repositories import agent_policy_repo as repo
from services import agent_actions, rbac
from services.agent_policy import (approval_views, approvals, conditions, conflict_cases, contract, documents,
                                   live_policies, readiness, run_runner, run_store, sections)
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

# Files and uploads stay plain dicts: documents.issue_uploads/register_uploads judge each one and
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
    flipped: List[Dict[str, Any]] = Field(min_length=1, max_length=20)


def _refused(message: str) -> JSONResponse:
    return JSONResponse(status_code=422, content={"problems": [
        {"field": "files", "code": "upload_refused", "message": message}]})


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
    """Validates the request and issues an id per file. The GATEWAY signs the S3 PUT (fix round 1):
    a URL or key in this answer would be withheld by the output scrubber, so neither is ever sent."""
    _require(p, "Buyer", "agent_policy.write", {"intent": "upload_urls", "files": len(body.files)})
    try:
        uploads = documents.issue_uploads(body.files, actor=p.subject)
    except ValueError as exc:
        return _refused(str(exc))
    # The ids are only known once issued; recorded before they are returned, so every key a later
    # register can name traces back to who was given it.
    agent_actions.record_action_or_fail(
        phase="issue", action_type="agent_policy.write", agent="agent_policy_api", status="issued",
        summary=f"{len(uploads)} upload id(s) issued to {p.subject}",
        details={"intent": "upload_urls", "principal": p.subject,
                 "uploads": [{"uploadId": u["uploadId"], "safeName": u["safeName"]} for u in uploads]})
    return {"uploads": uploads}


@router.post("/documents")
def register_documents(body: RegisterBody, p: Principal = Depends(gateway_principal)):
    """uploads: [{uploadId, name, revisionOf?}]; the key is rebuilt from the id and the name."""
    _require(p, "Buyer", "agent_policy.write", {"intent": "register_documents",
                                                "uploadIds": [str(u.get("uploadId") or "") for u in body.uploads]})
    if not body.uploads:
        return _refused("No uploads were sent.")
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


# ---------------------------------------------------------------- approvals, notifications, deciders
# Declared before "/{key}" so "approvals", "notifications" and "deciders" are never read as policy ids.
# Who may DECIDE is computed here per caller from the decider map at the case's current level;
# the Admin role reads every case but decides only when linked (separation of duties).

class DecideBody(BaseModel):
    verb: str = Field(max_length=16)
    reason: Optional[str] = Field(default=None, max_length=2000)


class DeciderBody(BaseModel):
    groups: List[Any] = Field(default_factory=list, max_length=50)
    emails: List[Any] = Field(default_factory=list, max_length=200)
    notes: Optional[Any] = None


def _decide_problem(exc: approvals.ApprovalRefused) -> JSONResponse:
    field = "reason" if exc.code == "reason_required" else "verb"
    return JSONResponse(status_code=422, content={"problems": [
        {"field": field, "code": exc.code, "message": exc.message}]})


_replay_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="agent-policy-replay")


def _replay_later(decision_id: int) -> None:
    """The approved action runs off the request thread: a tool call may outlast the gateway's
    timeout, and the person's decision is already committed (act calls this after commit)."""
    from services.agent_policy import replay  # local: pulls in the orchestrator tools

    # Resolved now, on the request thread, and handed to the worker. (The scheduler singleton is
    # process-wide, so the worker could resolve it too; this keeps the worker free of lookups.)
    # None -> replay.run claims nothing and the sweeper's retry runs it later.
    nick = replay._resolve_agent_nick(None)

    def _go():
        try:
            replay.run(decision_id, agent_nick=nick)
        except Exception:  # noqa: BLE001
            logger.exception("replay after approval %s failed", decision_id)

    _replay_pool.submit(_go)


@router.get("/approvals")
def list_approvals(status: str = Query(default="open", pattern="^(open|closed|all)$"),
                   p: Principal = Depends(gateway_principal)):
    role = _require(p, "Viewer", "agent_policy.read", {})
    with _conn() as conn:
        return {"approvals": approval_views.list_cases(conn, p, is_admin=role == "Admin", status=status)}


@router.get("/approvals/{decision_id}")
def get_approval(decision_id: int, p: Principal = Depends(gateway_principal)):
    role = _require(p, "Viewer", "agent_policy.read", {"decision": decision_id})
    with _conn() as conn:
        got = approval_views.get_case(conn, decision_id, p, is_admin=role == "Admin")
    if got is None:
        raise HTTPException(status_code=404, detail="No such approval request.")
    return got


@router.post("/approvals/{decision_id}/decide")
def decide(decision_id: int, body: DecideBody, p: Principal = Depends(gateway_principal)):
    # Viewer is the floor; eligibility (the decider map at the current level) decides inside act().
    role = _require(p, "Viewer", "agent_policy.decide", {"decision": decision_id, "verb": body.verb})

    def _audit(status: str, summary: str, **details):
        agent_actions.record_action_or_fail(
            phase="decide", action_type="agent_policy.decide", agent="agent_policy_api", status=status,
            summary=summary, details={"decision": decision_id, "verb": body.verb, "principal": p.subject, **details})

    with _conn() as conn:
        # Someone who may not read the case is told it does not exist, as GET answers them.
        if not approval_views.readable(conn, decision_id, p, is_admin=role == "Admin"):
            _audit("refused", f"{p.subject} could not {body.verb} approval {decision_id}: not_found", code="not_found")
            raise HTTPException(status_code=404, detail="No such approval request.")
        try:
            out = approvals.act(conn, decision_id, principal=p, verb=body.verb, reason=body.reason,
                                now=datetime.now(timezone.utc), replay=_replay_later)
        except approvals.ApprovalRefused as exc:
            _audit("refused", f"{p.subject} could not {body.verb} approval {decision_id}: {exc.code}", code=exc.code)
            if exc.status == 422:
                return _decide_problem(exc)
            raise HTTPException(status_code=exc.status, detail=exc.message)
        except Exception as exc:  # noqa: BLE001
            logger.exception("deciding approval %s failed", decision_id)
            _audit("error", f"{p.subject} could not {body.verb} approval {decision_id}: {type(exc).__name__}",
                   code="error")
            raise HTTPException(status_code=500, detail="The decision could not be recorded. Please try again.")
    # The decision is committed: a failed audit record must not turn it into an error answer.
    try:
        _audit("done", f"{p.subject} {out['result']} approval {decision_id}", level=out["level"],
               levelName=out["levelName"], actionId=out["actionId"])
    except Exception:  # noqa: BLE001
        logger.exception("decision %s was saved but its audit record failed", decision_id)
    return out


@router.get("/notifications")
def my_notifications(mine: int = Query(default=1, ge=1, le=1), limit: int = Query(default=50, ge=1, le=200),
                     p: Principal = Depends(gateway_principal)):
    role = _require(p, "Viewer", "agent_policy.read", {})
    with _conn() as conn:
        return {"notifications": approval_views.my_notifications(conn, p, limit, is_admin=role == "Admin")}


@router.post("/notifications/{notification_id}/read")
def read_notification(notification_id: int, p: Principal = Depends(gateway_principal)):
    role = _require(p, "Viewer", "agent_policy.notification_read", {"notification": notification_id})
    with _conn() as conn:
        got = approval_views.mark_read(conn, notification_id, p, is_admin=role == "Admin")
    if got is None:
        raise HTTPException(status_code=404, detail="No such notification.")
    return got


@router.get("/deciders")
def list_deciders(p: Principal = Depends(gateway_principal)):
    _require(p, "Viewer", "agent_policy.read", {})
    with _conn() as conn:
        return {"deciders": approval_views.list_deciders(conn)}


@router.put("/deciders/{name}")
def put_decider(name: str, body: DeciderBody, p: Principal = Depends(gateway_principal)):
    _require(p, "Admin", "agent_policy.admin", {"decider": name, "groups": len(body.groups),
                                                "emails": len(body.emails)})
    problems, groups, emails = approval_views.decider_problems(name, body.groups, body.emails, body.notes)
    if problems:
        agent_actions.record_action_or_fail(
            phase="decider", action_type="agent_policy.admin", agent="agent_policy_api", status="refused",
            summary=f"{p.subject} could not save decider {name}",
            details={"decider": name, "principal": p.subject, "problems": [x["code"] for x in problems]})
        return JSONResponse(status_code=422, content={"problems": problems})
    notes = body.notes.strip() if isinstance(body.notes, str) and body.notes.strip() else None
    with _conn() as conn:
        saved = approval_views.upsert_decider(conn, name, groups, emails, notes, actor=p.subject)
    agent_actions.record_action_or_fail(
        phase="decider", action_type="agent_policy.admin", agent="agent_policy_api", status="saved",
        summary=f"{p.subject} saved decider {name}",
        details={"decider": name, "principal": p.subject, "groups": groups, "emails": len(emails)})
    # Only the name and the time: group and email values in a 2xx body are withheld by
    # OutputSafety (no exemption for this write), and the screen re-reads GET /deciders.
    return {"name": saved["name"], "savedAt": saved["lastModifiedAt"]}


@router.get("/{key}/firings")
def policy_firings(key: str, limit: int = Query(default=50, ge=1, le=200),
                   p: Principal = Depends(gateway_principal)):
    _require(p, "Viewer", "agent_policy.read", {"policy": key})
    if not approval_views.KEY_RE.match(key):
        raise HTTPException(status_code=404, detail="no such policy")
    with _conn() as conn:
        return {"firings": approval_views.policy_firings(conn, key, limit)}


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
            out = repo.create_draft(conn, body.form, actor=p.subject)
        except repo.NotReady as exc:
            return JSONResponse(status_code=422, content={"problems": exc.problems})
    _detect_conflicts(out["policyKey"])
    return out


@router.post("/{key}/versions")
def save(key: str, body: VersionBody, p: Principal = Depends(gateway_principal)):
    minimum, action = ("Approver", "agent_policy.activate") if body.intent == "activate" else ("Buyer", "agent_policy.write")
    _require(p, minimum, action, {"policy": key, "intent": body.intent, "baseVersion": body.baseVersion})
    try:
        with _conn() as conn:
            try:
                out = repo.save_version(conn, key, body.form, base_version=body.baseVersion, intent=body.intent,
                                        actor=p.subject, change_note=body.changeNote)
            except repo.StaleVersion as exc:
                raise HTTPException(status_code=409, detail=f"Someone saved a newer version ({exc}). Reload and try again.")
            except repo.NotReady as exc:
                return JSONResponse(status_code=422, content={"problems": exc.problems})
            except repo.NotFound:
                raise HTTPException(status_code=404, detail="no such policy")
    finally:
        live_policies.invalidate()   # after the connection closes, so the next check sees this save
    _detect_conflicts(out["policyKey"])
    return out


def _detect_conflicts(key: str) -> None:
    """Conflict detection after a successful save, on a fresh connection. Best effort: the save is
    never undone, and the hourly scan catches what this missed."""
    try:
        with _conn() as conn:
            conflict_cases.after_save(conn, key)
    except Exception as exc:  # noqa: BLE001
        logger.error("conflict detection failed for %s: %s", key, type(exc).__name__)


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
    try:
        with _conn() as conn:
            try:
                return repo.retire(conn, key, base_version=body.baseVersion, actor=p.subject, change_note=body.changeNote)
            except repo.StaleVersion as exc:
                raise HTTPException(status_code=409, detail=f"Someone saved a newer version ({exc}). Reload and try again.")
            except repo.InvalidTransition:
                raise HTTPException(status_code=409, detail="This policy is already retired.")
            except repo.NotFound:
                raise HTTPException(status_code=404, detail="no such policy")
    finally:
        live_policies.invalidate()


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
