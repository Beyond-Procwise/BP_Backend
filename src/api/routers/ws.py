"""WebSocket endpoint for real-time session-status notifications.

The frontend connects once per upload session:

    WS /ws/session/{session_id}

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
"""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, status

from api.auth import AuthError, principal_from_token
from src.services.ws_manager import ws_manager

log = logging.getLogger(__name__)

router = APIRouter(tags=["WebSocket"])


def _get_resolved_session(session_id: str) -> dict | None:
    """Check if session is already resolved in process_monitor.

    Returns the payload dict if resolved, None if still in progress.
    Runs in a thread pool to avoid blocking the event loop.
    """
    try:
        from services.db import get_conn
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        pm.action_status,
                        COUNT(*) AS total,
                        COUNT(*) FILTER (WHERE sdo.outcome = 'target')      AS target,
                        COUNT(*) FILTER (WHERE sdo.outcome = 'discrepancy') AS discrepancy,
                        COUNT(*) FILTER (WHERE sdo.outcome = 'failed')      AS failed,
                        MAX(sdo.created_at) AS resolved_at
                    FROM proc.process_monitor pm
                    LEFT JOIN proc.session_document_outcome sdo
                        ON sdo.session_id = pm.session_id
                    WHERE pm.session_id = %s
                      AND pm.action_status IS NOT NULL
                    GROUP BY pm.action_status
                    LIMIT 1
                    """,
                    (session_id,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                action_status, total, target, discrepancy, failed, resolved_at = row

                # Fetch category and deal_name
                cur.execute(
                    """
                    SELECT
                        array_agg(DISTINCT pm.category)  AS category,
                        array_agg(DISTINCT ddm.deal_name) AS deal_name
                    FROM proc.session_document_outcome sdo
                    JOIN proc.process_monitor pm
                        ON pm.file_path  = sdo.file_path
                       AND pm.session_id = sdo.session_id
                    LEFT JOIN proc.bp_deal_document_map ddm
                        ON ddm.source_file = pm.file_path
                    WHERE sdo.session_id = %s
                    """,
                    (session_id,),
                )
                extra = cur.fetchone()
                category  = extra[0] if extra else None
                deal_name = extra[1] if extra else None

                # Per-document quality-action breakdown (additive)
                cur.execute(
                    """
                    SELECT
                        COUNT(*) FILTER (WHERE doc_action = 'duplicate'),
                        COUNT(*) FILTER (WHERE doc_action = 'updated'),
                        COUNT(*) FILTER (WHERE doc_action = 'needs_review'),
                        COUNT(*) FILTER (WHERE doc_action = 'unsupported'),
                        COALESCE(
                            json_agg(json_build_object(
                                'file_path', file_path, 'doc_action', doc_action))
                                FILTER (WHERE doc_action IS NOT NULL),
                            '[]'::json)
                    FROM proc.process_monitor
                    WHERE session_id = %s
                    """,
                    (session_id,),
                )
                da = cur.fetchone() or (0, 0, 0, 0, [])

                return {
                    "session_id":    session_id,
                    "action_status": action_status,
                    "total":         total,
                    "target":        target,
                    "discrepancy":   discrepancy,
                    "failed":        failed,
                    "duplicate":     da[0] or 0,
                    "updated":       da[1] or 0,
                    "needs_review":  da[2] or 0,
                    "unsupported":   da[3] or 0,
                    "documents":     da[4] or [],
                    "resolved_at":   resolved_at.strftime("%Y-%m-%dT%H:%M:%SZ") if resolved_at else None,
                    "category":      category,
                    "deal_name":     deal_name,
                }
    except Exception as exc:
        log.warning("WS catch-up query failed for session=%s: %s", session_id, exc)
        return None


@router.websocket("/ws/session/{session_id}")
async def session_status_ws(
    session_id: str,
    websocket: WebSocket,
    token: str | None = Query(default=None),
) -> None:
    """Stream session outcome events to the connected frontend.

    The caller is identified before the handshake is accepted. A browser cannot
    set ``Authorization`` on a WebSocket upgrade, so the same Cognito ID token
    the UI already sends on HTTP arrives here as ``?token=`` and is verified by
    ``api.auth`` — the same verifier, honouring ASK_AUTH_MODE the same way. A
    query parameter is the only channel the platform offers; it is worth knowing
    that this puts the token in URLs, and therefore in access logs.

    On connect, immediately checks if the session is already resolved. If so,
    sends the result right away — fixes the race condition where processing
    finishes before the client connects.
    """
    try:
        principal = principal_from_token(token)
    except AuthError as exc:
        # Closed without accepting first: an unauthenticated caller never holds
        # an open socket, and never reaches the catch-up query below.
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        log.warning("WS rejected session=%s: %s", session_id, exc)
        return

    if not session_id or not session_id.strip():
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        log.warning("WS rejected: blank session_id")
        return

    await ws_manager.connect(session_id, websocket)
    log.info(
        "WS authenticated session=%s subject=%s",
        session_id,
        principal.subject if principal else "auth-off",
    )
    try:
        # Catch-up check: if session already resolved before this connection
        # was opened, send the result immediately so the client never waits forever.
        payload = await asyncio.get_event_loop().run_in_executor(
            None, _get_resolved_session, session_id
        )
        if payload:
            log.info("WS catch-up: session=%s already resolved, sending immediately", session_id)
            await websocket.send_json(payload)
            return

        # Session still in progress — keep connection alive and wait for
        # ws_manager.broadcast_to_session() to push the result when pg_notify fires.
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        log.debug("WS receive error session=%s: %s", session_id, exc)
    finally:
        await ws_manager.disconnect(session_id, websocket)
