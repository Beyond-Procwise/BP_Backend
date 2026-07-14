"""WebSocket connection manager for session-status notifications.

Maintains a registry of active WebSocket connections keyed by session_id.
All public methods are async-safe; concurrent access is serialised with an
asyncio.Lock so the registry is never mutated while a broadcast is in flight.
"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Dict, List

from fastapi import WebSocket

log = logging.getLogger(__name__)


class WebSocketManager:
    """Tracks open WebSocket connections per session_id.

    Instantiated once at module level (``ws_manager``); imported by both the
    WS router and the SessionNotifyListener.
    """

    def __init__(self) -> None:
        self._connections: Dict[str, List[WebSocket]] = defaultdict(list)
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self, session_id: str, ws: WebSocket) -> None:
        """Accept the handshake and register the connection."""
        await ws.accept()
        async with self._lock:
            self._connections[session_id].append(ws)
        log.info(
            "WS connected   session=%s  total_clients=%d",
            session_id,
            len(self._connections[session_id]),
        )

    async def disconnect(self, session_id: str, ws: WebSocket) -> None:
        """Deregister a connection; prune the session key when the list empties."""
        async with self._lock:
            conns = self._connections.get(session_id, [])
            try:
                conns.remove(ws)
            except ValueError:
                pass  # already removed by a failed broadcast
            if not conns:
                self._connections.pop(session_id, None)
        log.info("WS disconnected session=%s", session_id)

    # ------------------------------------------------------------------
    # Broadcast
    # ------------------------------------------------------------------

    async def broadcast_to_session(self, session_id: str, payload: dict) -> None:
        """Send *payload* as JSON to every client watching *session_id*.

        Dead sockets (any send error) are collected and pruned after the
        broadcast round so the iteration does not modify the list mid-loop.
        """
        async with self._lock:
            sockets = list(self._connections.get(session_id, []))

        if not sockets:
            log.debug("WS broadcast: no clients for session=%s", session_id)
            return

        # The HTTP middleware in api/main.py cannot see a WebSocket frame — a socket is not
        # a response. This is the same contract, applied at the only other way out of the
        # process. It matters: these payloads carry a `file_path` per document.
        from services import output_safety as osafe

        payload = osafe.scrub_payload(payload, where=f"ws {session_id}")

        dead: list[WebSocket] = []
        for ws in sockets:
            try:
                await ws.send_json(payload)
            except Exception as exc:
                log.warning(
                    "WS send failed session=%s — removing dead socket: %s",
                    session_id, exc,
                )
                dead.append(ws)

        if dead:
            async with self._lock:
                conns = self._connections.get(session_id, [])
                for ws in dead:
                    try:
                        conns.remove(ws)
                    except ValueError:
                        pass
                if not self._connections.get(session_id):
                    self._connections.pop(session_id, None)

        log.info(
            "WS broadcast   session=%s  action_status=%s  sent=%d  failed=%d",
            session_id,
            payload.get("action_status"),
            len(sockets) - len(dead),
            len(dead),
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def active_sessions(self) -> Dict[str, int]:
        """Return a snapshot of {session_id: client_count} for health checks."""
        return {sid: len(conns) for sid, conns in self._connections.items()}


# Module-level singleton — shared by ws.py router and session_notify_listener.py
ws_manager = WebSocketManager()
