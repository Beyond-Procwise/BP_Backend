"""The Mode C2 cache: fetched email bodies, in memory, briefly.

Mode C2 reads the customer's mailbox at drafting time. Those bodies must never reach a
table — that is the difference between C2 and every other mode, and it is the reason a
customer would choose it. So they live here instead: a process-local dictionary with a
five-minute expiry, flushed the moment a binding is unbound or revoked.

Deliberately not Redis, and deliberately not the semantic cache this platform already
runs. Both are shared, both persist, and both would turn "we never store your
correspondence" into a sentence with an asterisk. A dictionary that dies with the process
is the point, not a limitation.

Keyed by ``(binding_id, intent)`` so a flush can target one customer's binding without
disturbing anyone else's.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = 300  # five minutes

_lock = threading.Lock()
_entries: Dict[Tuple[int, Optional[str]], "_Entry"] = {}


@dataclass
class _Entry:
    expires_at: float
    payload: List[Any]


def _now() -> float:
    return time.monotonic()


def get(binding_id: int, intent: Optional[str]) -> Optional[List[Any]]:
    """Cached exemplars for this binding and intent, or None."""

    key = (binding_id, intent)
    with _lock:
        entry = _entries.get(key)
        if entry is None:
            return None
        if entry.expires_at <= _now():
            # Expired entries are removed on read rather than left for a sweeper. A body
            # that is past its TTL should not survive in memory until something happens to
            # look for it.
            _entries.pop(key, None)
            return None
        return list(entry.payload)


def put(
    binding_id: int,
    intent: Optional[str],
    payload: List[Any],
    *,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> None:
    with _lock:
        _entries[(binding_id, intent)] = _Entry(
            expires_at=_now() + max(1, ttl_seconds), payload=list(payload)
        )


def flush_binding(binding_id: int) -> int:
    """Drop everything cached for one binding. Returns how many entries went.

    Called on unbind and on revocation. Those are the customer withdrawing permission,
    and leaving five minutes of their mail in a process would make that a suggestion.
    """

    with _lock:
        keys = [k for k in _entries if k[0] == binding_id]
        for key in keys:
            _entries.pop(key, None)
    if keys:
        logger.info("Flushed %s cached mailbox entr(ies) for binding %s", len(keys), binding_id)
    return len(keys)


def flush_all() -> int:
    with _lock:
        count = len(_entries)
        _entries.clear()
    return count


def size() -> int:
    with _lock:
        return len(_entries)
