"""One span per render, carrying ids and hashes and never wording.

The span joins the caller's trace. It never starts one: a render with no trace
to join records no span, because a trace begun here would be an orphan that
nothing else in the run belongs to.

WHAT MAY BE ON A SPAN is a closed list (``ALLOWED``), and every value is
checked before it is set: a number, or a short string made only of the
characters ids and hashes are made of. Anything else is replaced by a hash of
itself, so it can still be matched against the same value elsewhere without
being readable. Prompt wording, variable values and context content have no
attribute here to go into, and could not get through the check if they did.

OpenTelemetry is optional. Without the library installed every function here
does nothing, and rendering is unaffected.
"""
from __future__ import annotations

import hashlib
import re
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

try:  # optional dependency
    from opentelemetry import propagate as _propagate
    from opentelemetry import trace as _trace
except Exception:  # pragma: no cover - library not installed
    _propagate = None  # type: ignore[assignment]
    _trace = None  # type: ignore[assignment]

SPAN_NAME = "prompt.render"

ALLOWED = frozenset({
    "prompt.run_id", "prompt.step_id", "prompt.attempt", "prompt.decision_id",
    "prompt.id", "prompt.version", "prompt.hash", "prompt.fragments",
    "prompt.token_count", "prompt.latency_ms", "prompt.error_code",
})
_ID = re.compile(r"^[A-Za-z0-9_.:@/\-]{1,160}$")


def safe(value: Any) -> Any:
    """``value`` if it is a number or looks like an id; otherwise a hash of it."""
    if isinstance(value, bool) or isinstance(value, (int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [str(safe(v)) for v in value]
    text = str(value)
    if _ID.match(text):
        return text
    return "unreadable:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class _Recorder:
    """What the renderer writes to. Drops anything not on the list."""

    def __init__(self, span: Any = None) -> None:
        self._span = span

    def set(self, **attributes: Any) -> None:
        if self._span is None:
            return
        for name, value in attributes.items():
            key = "prompt." + name
            if key not in ALLOWED or value is None:
                continue
            self._span.set_attribute(key, safe(value))


def _parent(trace: Dict[str, Any]) -> Optional[Any]:
    """The context to join, or None if there is no trace to join."""
    carrier = {k: trace[k] for k in ("traceparent", "tracestate", "baggage") if trace.get(k)}
    context = _propagate.extract(carrier) if carrier else None
    current = _trace.get_current_span(context)
    if current is not None and current.get_span_context().is_valid:
        return context
    return None


@contextmanager
def render_span(trace: Optional[Dict[str, Any]]) -> Iterator[_Recorder]:
    """A span for one render, inside the caller's trace if it has one."""
    trace = trace or {}
    if _trace is None:
        yield _Recorder()
        return
    joined = _parent(trace)
    has_current = _trace.get_current_span().get_span_context().is_valid
    if joined is None and not has_current:
        yield _Recorder()          # no trace to join: never start one
        return
    started = time.perf_counter()
    # The library's default is to attach an exception's message and stack to
    # the span as an event. Both are switched off: the code alone is recorded.
    with _trace.get_tracer("procwise.prompt_engine").start_as_current_span(
            SPAN_NAME, context=joined, record_exception=False,
            set_status_on_exception=False) as span:
        recorder = _Recorder(span)
        recorder.set(run_id=trace.get("run_id"), step_id=trace.get("step_id"),
                     attempt=trace.get("attempt"), decision_id=trace.get("decision_id"))
        try:
            yield recorder
        except Exception as exc:
            # The code, never the message.
            recorder.set(error_code=getattr(exc, "code", type(exc).__name__))
            span.set_status(_trace.Status(_trace.StatusCode.ERROR))
            raise
        finally:
            recorder.set(latency_ms=round((time.perf_counter() - started) * 1000, 2))
