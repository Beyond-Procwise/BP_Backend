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

from api.auth import require_user
from api.endpoint_gate import require as gate

router = APIRouter(prefix="/reports", tags=["Reports"])
_AGENT = "ReportsRouter"
_CURRENCY = re.compile(r"^[A-Z]{3}$")
_PUBLIC = ("job_id", "report_type", "scope", "as_of", "status", "requested_by",
           "requested_at", "started_at", "finished_at", "run_id", "stage_reached",
           "blocking", "error", "dismissed_at", "dismissed_by", "dismiss_reason")


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


def _view(job: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: job.get(k) for k in _PUBLIC}
    # Check codes go out lower-case. The output-safety boundary in api/main.py
    # reads some upper-case codes (REPORT_UNTRACED_FIGURE) as environment-variable
    # names and withholds them, which left the screens with no reason to show.
    if out.get("blocking"):
        out["blocking"] = [{**b, "code": str(b.get("code") or "").lower()}
                           for b in out["blocking"]]
    # Sign-off (ruled 2026-09-24): a released deck leaves only once signed off, if the
    # policy says its report type needs it. The hash it is bound to stays server-side.
    s = signoff.state(job)
    out["signoff"] = {k: v for k, v in s.items() if k != "deck_sha256"}
    released = job.get("status") == "released"
    # Flags, not links: the output-safety boundary in api/main.py withholds any field that
    # names an internal route. The caller builds /reports/jobs/{job_id}/deck itself.
    out["deck_ready"] = released and s["state"] in ("not_required", "signed_off")
    out["review_available"] = released and s["state"] == "awaiting"
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
    return {"jobs": [_view(j) for j in job_store.recent(max(1, min(limit, 50)))]}


@router.get("/attention")
def attention(limit: int = 50):
    """Blocked or failed reports nobody has dealt with -- the SpendIQ Action
    Centre's Reports items. Not gated, like the other job reads."""
    items = []
    for job in job_store.needs_attention(max(1, min(limit, 100))):
        item = _view(job)
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
    if not job_store.dismiss(job_id, by=subject, reason=reason):
        raise HTTPException(status_code=409,
                            detail=f"report job {job_id} is {job['status']}; only a blocked "
                                   "or failed report can be dismissed, and only once")
    with audit.run_context(job_id=job_id, requested_by=job.get("requested_by")):
        audit.emit(audit.DISMISSED, run_id=job.get("run_id") or job_id, agent=_AGENT,
                   summary=f"{job['status']} report dismissed",
                   details={"dismissed_by": subject, "reason": reason,
                            "status": job["status"]})
    return {"job_id": job_id, "dismissed": True}


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    return _view(_job_or_404(job_id))


@router.get("/jobs/{job_id}/deck")
def get_deck(job_id: str, principal=Depends(require_user)):
    job = _job_or_404(job_id)
    s = signoff.state(job)
    # Held until signed off. Before then only someone the policy lets sign it off may open
    # it -- to review it -- and that download is audited as a review. A refused deck is
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
                  "review": s["state"] == "awaiting"})
    # The status decides, not the presence of bytes.
    found = job_store.deck(job_id) if job["status"] == "released" else None
    if found is None:
        raise HTTPException(status_code=409,
                            detail=f"report job {job_id} is {job['status']}; only a "
                                   "released report has a deck")
    content, media_type, filename = found
    # A sign-off is for the file that was reviewed. The deck is immutable, so a mismatch
    # means tampering or a bug -- either way this is not the deck that was signed off.
    if s["state"] == "signed_off" and signoff.deck_hash(content) != s.get("deck_sha256"):
        raise HTTPException(status_code=409,
                            detail=f"report job {job_id}: the stored deck does not match the "
                                   "one that was signed off")
    return Response(content=content, media_type=media_type,
                    headers={"Content-Disposition": f'attachment; filename="{filename}"',
                             "X-Report-Run-Id": job.get("run_id") or ""})
