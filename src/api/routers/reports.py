"""Report Generation Agent: the way in from outside the process.

Composing a report takes the local model minutes, too long to hold a request
open, so the door files a job and answers at once:

    POST /reports/generate           -> 202 {job_id, status: queued}
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
from pydantic import BaseModel, field_validator, model_validator

import src.services.rga  # noqa: F401  registers the Fact Pack builders
from src.services.rga import job_runner, job_store
from src.services.rga.factpack import registered_types

from api.auth import require_user
from api.endpoint_gate import require as gate

router = APIRouter(prefix="/reports", tags=["Reports"])
_AGENT = "ReportsRouter"
_CURRENCY = re.compile(r"^[A-Z]{3}$")
_PUBLIC = ("job_id", "report_type", "scope", "as_of", "status", "requested_by",
           "requested_at", "started_at", "finished_at", "run_id", "stage_reached",
           "blocking", "error")


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
    # A flag, not a link: the output-safety boundary in api/main.py withholds any
    # field that names an internal route, so a URL here arrives as "[withheld]".
    # The caller builds /reports/jobs/{job_id}/deck from the id it already has.
    out["deck_ready"] = job.get("status") == "released"
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
    gate("report.generate", principal, agent=_AGENT,
         context={"report_type": body.report_type, "scope": scope})

    job, created = job_store.create(
        body.report_type, scope=scope, as_of=dt.date.today().isoformat(),
        requested_by=getattr(principal, "subject", None) or None)
    if created:
        job_runner.submit(job["job_id"])
    return {"job_id": job["job_id"], "status": job["status"],
            "already_requested": not created}


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    return _view(_job_or_404(job_id))


@router.get("/jobs/{job_id}/deck")
def get_deck(job_id: str, principal=Depends(require_user)):
    job = _job_or_404(job_id)
    gate("report.read", principal, agent=_AGENT,
         context={"job_id": job_id, "run_id": job.get("run_id")})
    # The status decides, not the presence of bytes.
    found = job_store.deck(job_id) if job["status"] == "released" else None
    if found is None:
        raise HTTPException(status_code=409,
                            detail=f"report job {job_id} is {job['status']}; only a "
                                   "released report has a deck")
    content, media_type, filename = found
    return Response(content=content, media_type=media_type,
                    headers={"Content-Disposition": f'attachment; filename="{filename}"',
                             "X-Report-Run-Id": job.get("run_id") or ""})
