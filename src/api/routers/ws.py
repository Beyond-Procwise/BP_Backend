"""WebSocket endpoint for real-time session-status notifications.

The frontend connects once per upload session:

    WS /ws/session/{session_id}[?token=<api_key>]

The connection stays open.  When every document in the session reaches a
terminal outcome (target / discrepancy / failed) the server sends a single
JSON message and the connection can be closed by either side.

Message shape (mirrors the pg_notify payload):
    {
        "session_id":    "SESSION-001",
        "action_status": "partially_completed",   // completed | partially_completed | failed
        "total":         4,
        "target":        3,
        "discrepancy":   0,
        "failed":        1,
        "resolved_at":   "2026-06-30T12:34:56Z"
    }

Authentication
--------------
The global `verify_api_key` Depends runs on the WebSocket upgrade request and
reads the `X-API-Key` header (default: auth disabled when PROCWISE_API_KEY is
not set).  For browser clients that cannot set custom headers on a native
WebSocket, pass the key as the `token` query parameter — this endpoint handles
that case directly before any other logic.
"""
from __future__ import annotations

import logging
import os

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status

from src.services.ws_manager import ws_manager

log = logging.getLogger(__name__)

router = APIRouter(tags=["WebSocket"])

_API_KEY = os.getenv("PROCWISE_API_KEY", "")


def _auth_ok(token: str) -> bool:
    """Return True when auth is disabled or the supplied token matches."""
    if not _API_KEY:
        return True  # auth not configured — open access
    return token == _API_KEY


@router.websocket("/ws/session/{session_id}")
async def session_status_ws(
    session_id: str,
    websocket: WebSocket,
    token: str = "",
) -> None:
    """Stream session outcome events to the connected frontend.

    ``token`` query parameter is the fallback authentication path for browser
    clients.  When ``PROCWISE_API_KEY`` is set and neither the header
    (handled by the global Depends) nor the query parameter matches, the
    handshake is rejected with policy-violation close code 1008.
    """
    if not _auth_ok(token):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        log.warning("WS rejected: bad token for session=%s", session_id)
        return

    if not session_id or not session_id.strip():
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        log.warning("WS rejected: blank session_id")
        return

    await ws_manager.connect(session_id, websocket)
    try:
        # Keep the connection alive until the client disconnects or the server
        # closes it.  We only receive pings / text frames from the client here;
        # all server-initiated messages are sent via ws_manager.broadcast_to_session.
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        log.debug("WS receive error session=%s: %s", session_id, exc)
    finally:
        await ws_manager.disconnect(session_id, websocket)
