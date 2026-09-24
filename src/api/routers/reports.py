"""Report Generation Agent: the way in from outside the process.

Composing a report takes the local model minutes, too long to hold a request
open, so the door files a job and answers at once:

    POST /reports/generate           -> 202 {job_id, status: queued}
    GET  /reports/jobs               -> the newest jobs, without their decks
    GET  /reports/jobs/{job_id}      -> queued | running | released | blocked | failed
    GET  /reports/jobs/{job_id}/deck -> the .pptx, for a released job only

  * The gate is asked before a job is filed, as ``report.generate``, by whoever
    the token says is calling. Downloading a deck asks ``report.read``. A status
    poll does not ask the gate: it runs every few seconds, would write an audit
    row each time, and reveals no figure -- the router-level auth still applies.
  * A BLOCKED report returns its reasons and never its deck. The worker does not
    even store one, and the table refuses a deck on a job that was not released.
  * Asking again for the same report while it is queued or running returns the
    same job; the GPU runs it once.
"""
from __future__ import annotations

import datetime as dt
import re
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field, field_validator, model_validator

import src.services.rga  # noqa: F401  registers the Fact Pack builders
from src.services.rga import audit, job_runner, job_store, signoff
from src.services.rga.factpack import registered_types

from src.services.agent_actions import record_action_or_fail

from api.auth import require_user
from api.endpoint_gate import require as gate

router = APIRouter(prefix="/reports", tags=["Reports"])
_AGENT = "ReportsRouter"
_CURRENCY = re.compile(r"^[A-Z]{3}$")
_PUBLIC = ("job_id", "report_type", "scope", "as_of", "status", "requested_by",
           "requested_at", "started_at", "finished_at", "run_id", "stage_reached",
           "blocking", "error", "dismissed_at", "dismissed_by", "dismiss_reason", "has_page")


class GenerateBody(BaseModel):
    report_type: str
    period_start: dt.date
    period_end: dt.date
    period_label: Optional[str] = None
    currency: str = "GBP"

    @field_validator("currency")
    @classmethod
    def _iso_currency(cls, v: str) -> str:
        if not _CURRENCY.match(v):
            raise ValueError("currency is a three-letter ISO code, e.g. GBP")
        return v

    @model_validator(mode="after")
    def _ordered(self) -> "GenerateBody":
        if self.period_start > self.period_end:
            raise ValueError("period_start is after period_end")
        return self

    def scope(self) -> dict:
        start, end = self.period_start.isoformat(), self.period_end.isoformat()
        return {"period_start": start, "period_end": end,
                "period_label": self.period_label or f"{start} to {end}",
                "currency": self.currency}


def _view(job: Dict[str, Any], s: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    out = {k: job.get(k) for k in _PUBLIC}
    # Check codes go out lower-case. The output-safety boundary in api/main.py
    # reads some upper-case codes (REPORT_UNTRACED_FIGURE) as environment-variable
    # names and withholds them, which left the screens with no reason to show.
    if out.get("blocking"):
        out["blocking"] = [{**b, "code": str(b.get("code") or "").lower()}
                           for b in out["blocking"]]
    # Sign-off (ruled 2026-09-24): a released deck leaves only once signed off, if the
    # policy says its report type needs it. The hash it is bound to stays server-side.
    # A list passes the state it read in one batch (signoff.states_for); a single job reads it.
    s = s if s is not None else signoff.state(job)
    out["signoff"] = {k: v for k, v in s.items() if k != "deck_sha256"}
    released = job.get("status") == "released"
    # Flags, not links: the output-safety boundary in api/main.py withholds any field that
    # names an internal route. The caller builds /reports/jobs/{job_id}/deck itself.
    out["deck_ready"] = released and s["state"] in ("not_required", "signed_off")
    out["review_available"] = released and s["state"] == "awaiting"
    # The printable page follows the deck exactly; older jobs have none.
    out["has_page"] = bool(job.get("has_page"))
    out["page_ready"] = out["deck_ready"] and out["has_page"]
    out["page_review_available"] = out["review_available"] and out["has_page"]
    return out


def _job_or_404(job_id: str) -> Dict[str, Any]:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"no report job {job_id!r}")
    return job


@router.get("/types")
def list_types():
    return {"report_types": registered_types()}


@router.post("/generate", status_code=202)
def generate(body: GenerateBody, principal=Depends(require_user)):
    if body.report_type not in registered_types():
        raise HTTPException(status_code=404,
                            detail=f"no report type {body.report_type!r}; "
                                   f"available: {registered_types()}")
    scope = body.scope()
    decision = gate("report.generate", principal, agent=_AGENT,
                    context={"report_type": body.report_type, "scope": scope})
    subject = getattr(principal, "subject", None) or None
    evidence = getattr(decision, "evidence", None) or {}
    # The gate's own audit row has no trace id, so the job keeps the decision and
    # the run repeats it on report.scope_resolved. `shadowed` is read from the
    # evidence, never inferred from `allowed`: under shadow mode a refusal is
    # recorded as allowed (docs/rga/build-report.md §6).
    entitlement = {"action": "report.generate", "principal": subject,
                   "allowed": bool(getattr(decision, "allowed", False)),
                   "role": evidence.get("role"),
                   "policy_name": getattr(decision, "policy_name", None),
                   "policy_version": getattr(decision, "policy_version", None),
                   "resolution": getattr(decision, "resolution", None),
                   "shadowed": bool(evidence.get("shadowed"))}

    job, created = job_store.create(
        body.report_type, scope=scope, as_of=dt.date.today().isoformat(),
        requested_by=subject, entitlement=entitlement)
    if created:
        job_runner.submit(job["job_id"])
    return {"job_id": job["job_id"], "status": job["status"],
            "already_requested": not created}


@router.get("/jobs")
def list_jobs(limit: int = 20):
    """Recent jobs, newest first, without their decks. Not gated, like a single
    status poll: the Reports screen reloads it while a job runs."""
    jobs = job_store.recent(max(1, min(limit, 50)))
    states = signoff.states_for(jobs)
    return {"jobs": [_view(j, states[j["job_id"]]) for j in jobs]}


@router.get("/attention")
def attention(limit: int = 50):
    """Blocked or failed reports nobody has dealt with -- the SpendIQ Action
    Centre's Reports items. Not gated, like the other job reads."""
    items = []
    jobs = job_store.needs_attention(max(1, min(limit, 100)))
    states = signoff.states_for(jobs)
    for job in jobs:
        item = _view(job, states[job["job_id"]])
        # The store lists every released deck without a sign-off; the policy decides
        # which of those actually need one.
        if job.get("status") == "released" and item["signoff"]["state"] not in ("awaiting", "refused"):
            continue
        item["rerun_status"] = job.get("rerun_status")
        items.append(item)
    return {"items": items}


class DismissBody(BaseModel):
    reason: Optional[str] = Field(default=None, max_length=500)


@router.post("/jobs/{job_id}/dismiss")
def dismiss(job_id: str, body: DismissBody, principal=Depends(require_user)):
    """A person decides a blocked or failed report needs no further action.

    Gated as finding.resolve -- the same act as closing any other Action Centre
    finding -- so a Viewer can see the item but not clear it."""
    job = _job_or_404(job_id)
    gate("finding.resolve", principal, agent=_AGENT,
         context={"job_id": job_id, "run_id": job.get("run_id")})
    subject = getattr(principal, "subject", None) or None
    reason = (body.reason or "").strip() or None
    refused = job["status"] == "released" and signoff.state(job)["state"] == "refused"
    if not job_store.dismiss(job_id, by=subject, reason=reason, allow_released=refused):
        raise HTTPException(status_code=409,
                            detail=f"report job {job_id} is {job['status']}; only a blocked "
                                   "or failed report can be dismissed, and only once")
    with audit.run_context(job_id=job_id, requested_by=job.get("requested_by")):
        audit.emit(audit.DISMISSED, run_id=job.get("run_id") or job_id, agent=_AGENT,
                   summary=f"{job['status']} report dismissed",
                   details={"dismissed_by": subject, "reason": reason,
                            "status": job["status"]})
    return {"job_id": job_id, "dismissed": True}


class SignoffBody(BaseModel):
    reason: Optional[str] = Field(default=None, max_length=500)


class RefuseBody(BaseModel):
    reason: str = Field(max_length=500)

    @field_validator("reason")
    @classmethod
    def _stated(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("a refusal must say why")
        return v.strip()


def _decide(job_id: str, verdict: str, reason: Optional[str], principal: Any) -> Dict[str, Any]:
    """Sign a report off, or refuse to. Who may is policy (report.signoff's permit, Approver
    and above); whether a requester may sign off their own is policy too."""
    job = _job_or_404(job_id)
    gate(signoff.ACTION, principal, agent=_AGENT,
         context={"job_id": job_id, "run_id": job.get("run_id"), "verdict": verdict})
    subject = getattr(principal, "subject", None) or None
    requester = job.get("requested_by")
    # A job with no recorded requester cannot match anyone -- the email approvals' rule.
    if signoff.self_approval_denied() and requester and subject and requester == subject:
        # On the record, like the email approvals' self-approval refusal: the gate above has
        # already logged report.signoff as allowed, and this refusal must not leave no trace.
        # The raising writer -- a refusal that cannot be recorded is still a refusal.
        record_action_or_fail(
            phase="authorize", action_type=signoff.ACTION, agent=_AGENT, status="denied",
            summary="a report cannot be signed off by the person who asked for it",
            details={"job_id": job_id, "run_id": job.get("run_id"), "principal": subject,
                     "verdict": verdict, "policy_name": "ReportSignoffPolicy",
                     "evidence": {"rule": "self_approval", "requested_by": requester,
                                  "actioned_by": subject}})
        raise HTTPException(status_code=403,
                            detail="you asked for this report, so someone else must sign it off")
    try:
        decided = signoff.decide(job_id, verdict=verdict, by=subject or "", reason=reason,
                                 policy_name="ReportSignoffAuthorityPolicy")
    except signoff.NotDecidable as exc:
        raise HTTPException(status_code=409,
                            detail=f"report job {job_id} is {exc.state}; only a report "
                                   "awaiting sign-off can be signed off or refused")
    except ValueError as exc:   # no signed-in person to record
        raise HTTPException(status_code=403, detail=str(exc))
    # Irreversible events: the raising writer. The decision is already committed, so a
    # failed write surfaces as an error rather than passing silently.
    granted = verdict == "sign_off"
    with audit.run_context(job_id=job_id, requested_by=requester):
        audit.emit(audit.APPROVAL_GRANTED if granted else audit.APPROVAL_DENIED,
                   run_id=job.get("run_id") or job_id, agent=_AGENT,
                   summary="report signed off" if granted else "report sign-off refused",
                   details={("signed_off_by" if granted else "refused_by"): subject,
                            "reason": reason, "approval_id": decided.get("approval_id"),
                            "policy": "ReportSignoffAuthorityPolicy"})
    return _view(job_store.get(job_id) or job)


@router.post("/jobs/{job_id}/signoff")
def sign_off(job_id: str, body: SignoffBody, principal=Depends(require_user)):
    return _decide(job_id, "sign_off", (body.reason or "").strip() or None, principal)


@router.post("/jobs/{job_id}/refuse")
def refuse(job_id: str, body: RefuseBody, principal=Depends(require_user)):
    return _decide(job_id, "refuse", body.reason, principal)


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    return _view(_job_or_404(job_id))


# The printable page is served inline, to be read and printed in a browser tab. It is
# self-contained and every text in it is escaped; the policy below makes sure that even a
# mistake in that escaping could not run anything: no scripts, no fetches, inline style only.
_PAGE_CSP = "default-src 'none'; style-src 'unsafe-inline'"


def _serve(job_id: str, principal: Any, fmt: str) -> Response:
    """The deck or the page -- one set of rules for both, so the two cannot drift apart."""
    job = _job_or_404(job_id)
    s = signoff.state(job)
    # Held until signed off. Before then only someone the policy lets sign it off may open
    # it -- to review it -- and that download is audited as a review. A refused report is
    # held from everyone.
    if s["state"] == "awaiting" and not signoff.may_sign_off(principal):
        raise HTTPException(status_code=409,
                            detail=f"report job {job_id} is awaiting sign-off; only a person "
                                   "who may sign it off can open it before then")
    if s["state"] == "refused":
        raise HTTPException(status_code=409,
                            detail=f"report job {job_id} was refused sign-off: "
                                   f"{s.get('reason') or 'no reason recorded'}")
    gate("report.read", principal, agent=_AGENT,
         context={"job_id": job_id, "run_id": job.get("run_id"),
                  "review": s["state"] == "awaiting", "format": fmt})
    if job["status"] != "released":
        # The status decides, not the presence of bytes.
        raise HTTPException(status_code=409,
                            detail=f"report job {job_id} is {job['status']}; only a "
                                   "released report has a deck")
    if fmt == "deck":
        found = job_store.deck(job_id)
        signed_hash = s.get("deck_sha256")
    else:
        found = job_store.page(job_id)
        signed_hash = s.get("page_sha256")
    if found is None:
        if fmt == "page":
            raise HTTPException(status_code=404,
                                detail=f"report job {job_id} has no printable page -- it was "
                                       "made before pages existed")
        raise HTTPException(status_code=409, detail=f"report job {job_id} has no deck")
    content = found[0]
    # A sign-off is for the files that were reviewed. Both are immutable, so a mismatch --
    # or a sign-off that never saw this file -- means this is not what was signed off.
    if s["state"] == "signed_off" and (not signed_hash
                                       or signoff.deck_hash(content) != signed_hash):
        raise HTTPException(status_code=409,
                            detail=f"report job {job_id}: the stored {fmt} does not match "
                                   "the one that was signed off")
    if fmt == "deck":
        _, media_type, filename = found
        return Response(content=content, media_type=media_type,
                        headers={"Content-Disposition": f'attachment; filename="{filename}"',
                                 "X-Report-Run-Id": job.get("run_id") or ""})
    name = f"{job.get('report_type')}_{job.get('run_id') or job_id}.html"
    return Response(content=content, media_type=found[1],
                    headers={"Content-Disposition": f'inline; filename="{name}"',
                             "Content-Security-Policy": _PAGE_CSP,
                             "X-Content-Type-Options": "nosniff",
                             "X-Report-Run-Id": job.get("run_id") or ""})


@router.get("/jobs/{job_id}/deck")
def get_deck(job_id: str, principal=Depends(require_user)):
    return _serve(job_id, principal, "deck")


@router.get("/jobs/{job_id}/page")
def get_page(job_id: str, principal=Depends(require_user)):
    """The printable A4 page, inline, for a browser tab to show and print."""
    return _serve(job_id, principal, "page")
