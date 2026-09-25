"""GET /spendiq/value-summary — the Value Found headline (W1).
One read-model over discrepancies + opportunities + benchmark deltas.
Spec: docs/superpowers/specs/2026-07-30-value-found-design.md"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from api.auth import require_user
from src.services import value_query_service, value_summary_service
from src.services.email_dispatch_guard import DispatchDenied

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/spendiq", tags=["Value Found"])


class QuerySend(BaseModel):
    """The draft as the human left it. Edits are expected — the review IS the approval, so
    what gets sent is what they last saw, not what the service first proposed."""
    to: str
    subject: str
    body: str


def trim_findings_for_response(findings: list[dict], limit: int) -> list[dict]:
    """HTTP-response-only trim (Ruling R15). Scrubbing 3,600+ findings through
    OutputSafetyMiddleware stalls the whole event loop for ~28s, so the response
    ships a bounded list while every summary total keeps counting every finding.

    Order: every finding whose ledger_state == 'claimed' first (the Being-claimed
    list needs them all, whatever the limit); then the remaining live findings
    (superseded_by is None) sorted by amount_gbp descending, None last; then
    superseded findings, only while room remains. A claimed finding is never
    dropped, even if claimed findings alone exceed ``limit``."""
    claimed = [f for f in findings if f.get("ledger_state") == "claimed"]
    live = [f for f in findings
            if f.get("ledger_state") != "claimed" and f.get("superseded_by") is None]
    superseded = [f for f in findings
                  if f.get("ledger_state") != "claimed" and f.get("superseded_by") is not None]

    live_sorted = sorted(
        live,
        key=lambda f: (f.get("amount_gbp") is None, -(f.get("amount_gbp") or 0)),
    )

    cap = max(limit, len(claimed))
    return (claimed + live_sorted + superseded)[:cap]


@router.get("/value-summary", summary="Evidence-backed value found / recovered / potential")
def get_value_summary(limit: int = 200) -> dict:
    limit = min(max(limit, 1), 1000)
    try:
        result = value_summary_service.build_value_summary()
    except Exception as exc:                     # the service isolates per-source failures;
        logger.exception("value-summary failed")  # reaching here means something structural
        raise HTTPException(status_code=500, detail=str(exc))
    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    findings = result.get("findings", [])
    result["findings_total"] = len(findings)
    result["findings"] = trim_findings_for_response(findings, limit)
    result["findings_shown"] = len(result["findings"])
    return result


@router.get("/value-summary/findings/{finding_id}/query-draft",
            summary="Draft a supplier query for one finding (never sends)")
def get_query_draft(finding_id: str, request: Request, tone: str = "formal") -> dict:
    # The agent is optional HERE, unlike on the send path: `formal` needs no model at all,
    # and a re-tone with no agent to ask degrades to the grounded template with a note
    # rather than failing a buyer's draft outright.
    agent_nick = getattr(request.app.state, "agent_nick", None)
    try:
        return value_query_service.build_draft(finding_id, tone=tone, agent_nick=agent_nick)
    except ValueError as exc:
        # Not queryable: resolved, wrong issue type, an opportunity id, or unknown. 409
        # rather than 404 — the finding may well exist, it just cannot be queried.
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:
        logger.exception("query draft failed for %s", finding_id)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/value-summary/findings/{finding_id}/query-send",
             summary="Send the reviewed query and stamp the finding")
def post_query_send(
    finding_id: str,
    payload: QuerySend,
    request: Request,
    principal=Depends(require_user),
) -> dict:
    agent_nick = getattr(request.app.state, "agent_nick", None)
    if agent_nick is None:
        raise HTTPException(status_code=503, detail="AgentNick not available")
    try:
        return value_query_service.send_query(
            finding_id, to=payload.to, subject=payload.subject, body=payload.body,
            agent_nick=agent_nick, principal=principal)
    except DispatchDenied as exc:
        # The guard refused: no approval/allow-list/sensitivity/policy match.
        # A 403 tells the caller this was a permission refusal, not a
        # malformed or stale request (409) or a delivery failure (502).
        raise HTTPException(status_code=403, detail=exc.decision.reason)
    except ValueError as exc:
        # No recipient, already queried, no longer open — the caller's request is the
        # problem, and re-sending unchanged will not help.
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:
        # Delivery failed. The finding is deliberately NOT stamped, so retrying is safe.
        logger.exception("query send failed for %s", finding_id)
        raise HTTPException(status_code=502, detail=str(exc))
