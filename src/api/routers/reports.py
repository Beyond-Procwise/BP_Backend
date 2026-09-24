"""Report Generation Agent: the way in from outside the process.

Until this router nothing could reach ``services/rga`` -- it ran only when a test
or a script called it. This is the door, and it is deliberately narrow:

  * The gate is asked before any figure is computed, as ``report.generate``, by
    whoever the token says is calling.
  * A released report comes back as the deck itself, with its run id in a header
    so the file can be traced to its Fact Pack in the audit spine.
  * A BLOCKED report returns its reasons and never its deck. The pipeline still
    renders one, so a person investigating can rebuild it, but the gate failed
    closed and a download link would be the gate failing open.

Nothing is stored. Each request is a fresh run; the audit events are the record.
"""
from __future__ import annotations

import datetime as dt
import re
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, field_validator, model_validator
from starlette.concurrency import run_in_threadpool

import src.services.rga  # noqa: F401  registers the Fact Pack builders
from src.services.rga.factpack import registered_types
from src.services.rga.pipeline import generate_report

from api.auth import require_user
from api.endpoint_gate import require as gate

router = APIRouter(prefix="/reports", tags=["Reports"])
_AGENT = "ReportsRouter"
_CURRENCY = re.compile(r"^[A-Z]{3}$")


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


@router.get("/types")
def list_types():
    return {"report_types": registered_types()}


@router.post("/generate")
async def generate(body: GenerateBody, principal=Depends(require_user)):
    if body.report_type not in registered_types():
        raise HTTPException(status_code=404,
                            detail=f"no report type {body.report_type!r}; "
                                   f"available: {registered_types()}")
    scope = body.scope()
    gate("report.generate", principal, agent=_AGENT,
         context={"report_type": body.report_type, "scope": scope})

    # COMPOSE calls the local model; seconds to minutes, never on the event loop.
    run = await run_in_threadpool(generate_report, body.report_type, scope=scope)

    if not run.released or run.artefact is None:
        return JSONResponse(status_code=422, content={
            "released": False,
            "run_id": run.run_id,
            "report_type": run.report_type_id,
            "stage_reached": run.stage_reached,
            "blocking": [{"finding_id": f.finding_id, "code": f.code.value,
                          "severity": f.severity.value, "detail": f.detail}
                         for f in run.findings if f.blocks_release],
        })

    filename = f"{run.report_type_id}_{run.run_id}.pptx"
    return Response(content=run.artefact.content, media_type=run.artefact.media_type,
                    headers={"Content-Disposition": f'attachment; filename="{filename}"',
                             "X-Report-Run-Id": run.run_id})
