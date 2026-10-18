"""A per-request trace of what the /ask pipeline did, for debugging.

Recording is cheap and always on; *showing* it is not. The router attaches it to
a response only for an admin, and only when ``ASK_TRACE`` is on or the admin sent
the ``x-ask-trace`` header. A normal user never sees it.

The trace lives in a ContextVar set by ``start()`` on the thread that runs the
pipeline, so code deep in the retrieval stack can add to it with ``current()``
without the trace being threaded through every signature. ``current()`` returns a
harmless stand-in when no trace is active.
"""
from __future__ import annotations

import contextlib
import contextvars
import time
from typing import Any, Dict, Iterator, List, Optional


class Trace:
    def __init__(self) -> None:
        self._t0 = time.perf_counter()
        self.guardrail: Dict[str, Any] = {}
        self.route: Dict[str, Any] = {}
        self.queries: Dict[str, Any] = {}
        self.retrievals: List[Dict[str, Any]] = []
        self.output_filter: Dict[str, Any] = {}
        self.stages: Dict[str, float] = {}      # stage name -> seconds since start when it began
        self.timings: Dict[str, float] = {}     # sub-stage name -> seconds spent

    def stage(self, name: str) -> None:
        self.stages.setdefault(name, round(time.perf_counter() - self._t0, 3))

    @contextlib.contextmanager
    def span(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.timings[name] = round(self.timings.get(name, 0.0) + time.perf_counter() - start, 3)

    def set_guardrail(self, *, decision: str, rule: Optional[str] = None, category: Optional[str] = None,
                      layer: Optional[str] = None, score: Optional[float] = None) -> None:
        self.guardrail = {"decision": decision, "rule": rule, "category": category, "layer": layer,
                          "score": None if score is None else round(float(score), 4)}

    def set_route(self, *, route: str, score: Optional[float] = None, threshold: Optional[float] = None) -> None:
        self.route = {"route": route, "score": score, "threshold": threshold}

    def set_queries(self, *, original: Optional[str] = None, rewritten: Optional[str] = None,
                    standalone: Optional[str] = None) -> None:
        if original is not None:
            self.queries.setdefault("original", original)
        if standalone is not None:
            self.queries["standalone"] = standalone
        if rewritten is not None:
            self.queries["rewritten"] = rewritten

    def add_retrieval(self, *, documents: List[Dict[str, Any]], seconds: float) -> None:
        self.retrievals.append({"count": len(documents), "documents": documents, "seconds": round(seconds, 3)})

    def set_output_filter(self, *, decision: str, rule: Optional[str] = None, reason: Optional[str] = None) -> None:
        self.output_filter = {"decision": decision, "rule": rule, "reason": reason}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "guardrail": self.guardrail,
            "route": self.route,
            "queries": self.queries,
            "retrieval_count": len(self.retrievals),
            "retrievals": self.retrievals,
            "output_filter": self.output_filter,
            "stage_starts_s": self.stages,
            "sub_stage_s": self.timings,
            "total_s": round(time.perf_counter() - self._t0, 3),
        }


class _Null(Trace):
    """What current() returns when nothing is tracing: accepts everything, keeps nothing."""

    def __init__(self) -> None:
        super().__init__()


_current: contextvars.ContextVar[Optional[Trace]] = contextvars.ContextVar("ask_trace", default=None)


def start() -> Trace:
    trace = Trace()
    _current.set(trace)
    return trace


def current() -> Trace:
    return _current.get() or _Null()


def end() -> None:
    _current.set(None)


def visible_to(principal: Any, header_value: Optional[str]) -> bool:
    """May this caller see the trace? Admins only; and only with the flag on or the header sent."""
    from . import ask_flags
    from .report_data.mode import is_admin
    try:
        if not is_admin(principal):
            return False
    except Exception:
        return False
    return ask_flags.enabled(ask_flags.TRACE) or (header_value or "").strip().lower() in ("1", "true", "yes")
