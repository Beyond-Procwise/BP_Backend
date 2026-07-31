"""GET /spendiq/value-summary — the Value Found headline (W1).
One read-model over discrepancies + opportunities + benchmark deltas.
Spec: docs/superpowers/specs/2026-07-30-value-found-design.md"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from src.services import value_query_service, value_summary_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/spendiq", tags=["Value Found"])


class QuerySend(BaseModel):
    """The draft as the human left it. Edits are expected — the review IS the approval, so
    what gets sent is what they last saw, not what the service first proposed."""
    to: str
    subject: str
    body: str


@router.get("/value-summary", summary="Evidence-backed value found / recovered / potential")
def get_value_summary() -> dict:
    try:
        result = value_summary_service.build_value_summary()
    except Exception as exc:                     # the service isolates per-source failures;
        logger.exception("value-summary failed")  # reaching here means something structural
        raise HTTPException(status_code=500, detail=str(exc))
    result["generated_at"] = datetime.now(timezone.utc).isoformat()
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
def post_query_send(finding_id: str, payload: QuerySend, request: Request) -> dict:
    agent_nick = getattr(request.app.state, "agent_nick", None)
    if agent_nick is None:
        raise HTTPException(status_code=503, detail="AgentNick not available")
    try:
        return value_query_service.send_query(
            finding_id, to=payload.to, subject=payload.subject, body=payload.body,
            agent_nick=agent_nick)
    except ValueError as exc:
        # No recipient, already queried, no longer open — the caller's request is the
        # problem, and re-sending unchanged will not help.
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:
        # Delivery failed. The finding is deliberately NOT stamped, so retrying is safe.
        logger.exception("query send failed for %s", finding_id)
        raise HTTPException(status_code=502, detail=str(exc))
