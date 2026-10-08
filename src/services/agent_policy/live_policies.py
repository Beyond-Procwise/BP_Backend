"""The live agent policies the gate enforces: valid ones only, cached briefly per process.

Same filter as the orchestrator feed (contract.validate against the registry), so a policy
the feed refuses is never enforced either. Any failure to load raises PolicyStoreUnavailable;
the gate turns that into a refusal (fail closed). Saves and retires call invalidate().
"""
from __future__ import annotations

import copy
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from repositories import agent_policy_repo as repo
from services.agent_policy import contract
from services.agent_policy.registry import load_registry

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_cache: Optional[Tuple[float, List[Dict[str, Any]]]] = None
_generation = 0   # bumped by invalidate(); a load that raced a save is not cached


class PolicyStoreUnavailable(RuntimeError):
    """Live agent policies could not be loaded, so no action can be shown to be allowed."""


def invalidate() -> None:
    global _cache, _generation
    with _lock:
        _cache = None
        _generation += 1


def _fetch(conn: Any) -> List[Dict[str, Any]]:
    registry = load_registry(conn)
    good = []
    for doc in repo.live_documents(conn):
        doc_id = doc.get("id") if isinstance(doc, dict) else None
        try:
            problems = contract.validate(doc, registry) if isinstance(doc, dict) else ["not a document"]
        except Exception as exc:  # noqa: BLE001 - one bad row must not hide the others
            problems = [f"could not be validated: {type(exc).__name__}"]
        if problems:
            logger.warning("agent-policy enforcement skips live policy %s: %s", doc_id, problems)
        else:
            good.append(doc)
    return good


def load(conn: Any = None, *, ttl: float = 30) -> List[Dict[str, Any]]:
    global _cache
    with _lock:
        if _cache is not None and time.monotonic() - _cache[0] < ttl:
            return copy.deepcopy(_cache[1])
        started = _generation
    try:
        if conn is None:
            from services.db import get_conn

            with get_conn() as own:
                docs = _fetch(own)
        else:
            docs = _fetch(conn)
    except Exception as exc:  # noqa: BLE001 - every failure is the same answer: cannot check
        raise PolicyStoreUnavailable(f"live agent policies could not be loaded: {type(exc).__name__}") from exc
    with _lock:
        if _generation == started:
            _cache = (time.monotonic(), copy.deepcopy(docs))
    return docs   # callers get their own copy; mutating it never changes the next load
