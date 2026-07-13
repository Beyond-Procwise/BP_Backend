"""Contact support — the AI support agent."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from starlette.responses import StreamingResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/support", tags=["Support"])


def get_agent_nick(request: Request):
    nick = getattr(request.app.state, "agent_nick", None)
    if not nick:
        raise HTTPException(status_code=503, detail="AgentNick not available")
    return nick


class SupportRequest(BaseModel):
    message: str
    user_name: Optional[str] = None
    user_email: Optional[str] = None
    session_id: Optional[str] = None


@router.post("/contact")
async def contact_support(
    req: SupportRequest,
    agent_nick=Depends(get_agent_nick),
) -> Dict[str, Any]:
    """Greet the user, try to solve their problem, and escalate if it can't.

    Returns the agent's reply plus whether it resolved the issue or raised a ticket
    (with the reference the user can quote).
    """
    from services.support_agent import SupportAgent

    if not (req.message or "").strip():
        raise HTTPException(status_code=400, detail="message is required")

    agent = SupportAgent(agent_nick)
    # The agent calls Ollama and SMTP — both blocking. Off the event loop.
    return await run_in_threadpool(
        agent.handle,
        req.message,
        user_name=req.user_name,
        user_email=req.user_email,
        session_id=req.session_id,
    )


@router.post("/contact/stream")
async def contact_support_stream(
    req: SupportRequest,
    agent_nick=Depends(get_agent_nick),
):
    """The same support agent, streamed as SSE.

    Events: stage (thinking | looking_up, with the topic), delta (reply prose), done
    (reference, whether it was escalated, whether the admin was actually emailed, and
    whether we are waiting on the user to confirm the fix worked), error.
    """
    import asyncio
    import json
    import queue as _queue

    from services.support_agent import SupportAgent

    if not (req.message or "").strip():
        raise HTTPException(status_code=400, detail="message is required")

    events: _queue.Queue = _queue.Queue()
    _DONE = object()

    def _emit(kind: str, payload: Dict[str, Any]) -> None:
        events.put({"type": kind, **payload})

    def _work() -> None:
        try:
            result = SupportAgent(agent_nick).handle_stream(
                req.message,
                user_name=req.user_name,
                user_email=req.user_email,
                session_id=req.session_id,
                emit=_emit,
            )
            events.put({"type": "done", **result})
        except Exception as exc:  # noqa: BLE001
            logger.exception("support stream failed")
            events.put({"type": "error", "message": str(exc)[:300]})
        finally:
            events.put(_DONE)

    async def _publish():
        loop = asyncio.get_running_loop()
        task = loop.run_in_executor(None, _work)
        try:
            while True:
                event = await loop.run_in_executor(None, events.get)
                if event is _DONE:
                    break
                yield f"data: {json.dumps(event, default=str)}\n\n"
        finally:
            await task

    return StreamingResponse(
        _publish(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


class ConfirmRequest(BaseModel):
    resolved: bool
    note: Optional[str] = None


@router.post("/{reference}/confirm")
def confirm_support(
    reference: str,
    body: ConfirmRequest,
    agent_nick=Depends(get_agent_nick),
) -> Dict[str, Any]:
    """Did the guidance actually fix it?

    Guidance is not a fix. If the user says it worked, the ticket closes. If they say it
    did not, THIS is where it becomes a real escalation and the admin gets emailed —
    which is the case that matters most, because an assistant that sounds convincing is
    exactly the one whose advice needs checking.
    """
    from services.support_agent import SupportAgent

    result = SupportAgent(agent_nick).confirm(
        reference, resolved=body.resolved, note=body.note
    )
    if result.get("error"):
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.get("/tickets")
def list_tickets(agent_nick=Depends(get_agent_nick), limit: int = 25) -> Dict[str, Any]:
    """Open support tickets, so escalations are visible rather than only emailed."""
    rows: List[Dict[str, Any]] = []
    try:
        with agent_nick.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT reference, user_name, user_email, message, outcome,
                           email_status, status, created_at
                    FROM proc.bp_support_ticket
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (max(1, min(limit, 200)),),
                )
                cols = [d[0] for d in cur.description]
                rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    except Exception as exc:  # noqa: BLE001
        logger.exception("failed to list support tickets")
        raise HTTPException(status_code=500, detail=str(exc))
    return {"count": len(rows), "tickets": rows}
