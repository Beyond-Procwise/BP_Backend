"""A single, honest place to record features that are degraded because a
dependency (usually a DB relation) does not exist.

Several code paths in this codebase historically responded to a missing
table by either (a) letting a broad ``except Exception`` swallow it and log a
full traceback on *every* call, spamming the logs without telling anyone the
feature is unavailable, or (b) silently returning empty results so a caller
could not tell whether "no data" meant "genuinely nothing found" or "the
table this depends on was never built".

This module fixes both problems for good:

- ``mark_degraded(capability, reason)`` records the degradation and logs a
  WARNING exactly once (repeat calls, e.g. once per workflow run, are
  silent) — call this every time the missing dependency is hit so the
  registry always reflects current reality.
- ``mark_available(capability)`` clears a previously degraded capability
  (e.g. useful in tests, or if the dependency is provisioned later in the
  same process).
- ``get_degraded()`` returns the current list of degraded capabilities so a
  human-facing surface (GET /health) can show them honestly.
- ``log_once(key, level, msg, *args)`` is a smaller general-purpose helper
  for de-duplicating any other "expected, not actionable per-call" log line
  (e.g. an unresolvable governance-linked agent slug) without pulling in a
  full capability entry.
"""
from __future__ import annotations

import logging
import threading
from typing import Dict, List

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_degraded: Dict[str, str] = {}
_logged_once: set = set()


def mark_degraded(capability: str, reason: str) -> None:
    """Register ``capability`` as unavailable because of ``reason``.

    Idempotent and safe to call on every request/run — only the first call
    for a given capability actually emits a log line.
    """
    with _lock:
        is_new = _degraded.get(capability) != reason
        _degraded[capability] = reason
    if is_new:
        logger.warning("Capability degraded: %s (%s)", capability, reason)


def mark_available(capability: str) -> None:
    """Clear a previously degraded capability."""
    with _lock:
        _degraded.pop(capability, None)


def get_degraded() -> List[Dict[str, str]]:
    """Return the current degraded capabilities, sorted by name."""
    with _lock:
        return [
            {"capability": name, "reason": reason}
            for name, reason in sorted(_degraded.items())
        ]


def log_once(key: str, level: int, msg: str, *args) -> None:
    """Emit ``msg`` via the module logger at most once per process for ``key``."""
    with _lock:
        if key in _logged_once:
            return
        _logged_once.add(key)
    logger.log(level, msg, *args)
