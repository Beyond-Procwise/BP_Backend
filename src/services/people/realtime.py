"""Per-user live signals. A signal carries a type and ids only, never message text: the client
reacts by fetching over HTTP, where authentication and participation are checked again.

Uses its OWN connection manager (not the session one): the session socket accepts any session id
from any signed-in caller, so sharing it would let one user subscribe to another's channel.
Single-process only (the service runs one uvicorn worker); a missed signal is harmless because the
client also polls GET /notifications/unread-count and GET /messages/conversations.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Iterable, Optional

from services.ws_manager import WebSocketManager

log = logging.getLogger(__name__)
manager = WebSocketManager()
_loop: Optional[asyncio.AbstractEventLoop] = None


def set_loop(loop: asyncio.AbstractEventLoop) -> None:
    global _loop
    _loop = loop


def push(subjects: Iterable[str], payload: dict) -> None:
    """Fire-and-forget from any thread. Never raises."""
    loop = _loop
    if loop is None or loop.is_closed():
        return
    for s in set(subjects):
        try:
            asyncio.run_coroutine_threadsafe(manager.broadcast_to_session(str(s), dict(payload)), loop)
        except Exception:
            log.debug("live signal to a user failed", exc_info=True)
