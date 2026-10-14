"""Tracing: one span per render, in the caller's trace, with nothing readable in it."""
from __future__ import annotations

import logging
import uuid

import pytest

pytest.importorskip("opentelemetry.sdk.trace")

from opentelemetry import trace as ot
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from services.prompt_engine import client as C
from services.prompt_engine import registry as R
from services.prompt_engine import telemetry as TM
from services.prompt_engine import templates as T

# Marker values: if any of these appears in anything emitted, wording or a value leaked.
WORDING = "ZEBRA-WORDING be brief with"
FRAGMENT = "ZEBRA-FRAGMENT house voice"
VALUE = "ZEBRA-VALUE Acme Ltd, 4111 1111 1111 1111"
SECRET = "ZEBRA-SECRET 123-45-6789"
MARKERS = ("ZEBRA", "4111", "123-45-6789", "Acme")


@pytest.fixture(scope="module")
def exporter():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    ot.set_tracer_provider(provider)
    return exporter


@pytest.fixture()
def spans(exporter):
    exporter.clear()
    return exporter


@pytest.fixture()
def prompt(key):
    voice = f"{key}_voice"
    R.save_version(voice, template=FRAGMENT, author="ada", kind="fragment")
    R.save_version(key, template="{{> %s@1}} %s {{supplier}} / {{customer.ssn}}" % (voice, WORDING),
                   author="ada",
                   variables=[{"name": "supplier", "type": "string"},
                              {"name": "customer", "type": "object"}])
    R.promote(key, 1, approved_by="bob")
    return key


def _render(key, run, **over):
    kw = dict(vars={"supplier": VALUE, "customer": {"ssn": SECRET}},
              obligations=["redact:customer.ssn"],
              trace=C.trace(run, "triage", attempt=2, decision_id="d-9"))
    kw.update(over)
    return C.render(key, **kw)


def _everything_emitted(spans, caplog) -> str:
    parts = [r.getMessage() for r in caplog.records]
    for span in spans.get_finished_spans():
        parts.append(span.name)
        parts.extend(f"{k}={v}" for k, v in (span.attributes or {}).items())
        for event in span.events:
            parts.append(event.name)
            parts.extend(f"{k}={v}" for k, v in (event.attributes or {}).items())
        parts.append(str(span.status.description))
    return "\n".join(parts)


def test_a_render_inside_a_trace_is_one_span_carrying_the_ids_and_the_hash(prompt, spans):
    run = f"run-{uuid.uuid4().hex[:8]}"
    with ot.get_tracer("orchestrator").start_as_current_span("step") as step:
        out = _render(prompt, run)

    (span,) = [s for s in spans.get_finished_spans() if s.name == TM.SPAN_NAME]
    assert span.context.trace_id == step.get_span_context().trace_id
    assert span.parent.span_id == step.get_span_context().span_id
    a = span.attributes
    assert (a["prompt.run_id"], a["prompt.step_id"], a["prompt.attempt"],
            a["prompt.decision_id"]) == (run, "triage", 2, "d-9")
    assert (a["prompt.id"], a["prompt.version"], a["prompt.hash"]) == (
        prompt, 1, out["manifest"]["hash"])
    assert list(a["prompt.fragments"]) == [f"{prompt}_voice@1"]
    assert a["prompt.token_count"] == out["manifest"]["token_count"]
    assert a["prompt.latency_ms"] >= 0
    assert set(a) <= TM.ALLOWED


def test_a_render_with_no_trace_to_join_starts_none(prompt, spans):
    _render(prompt, "run-none")
    assert [s for s in spans.get_finished_spans() if s.name == TM.SPAN_NAME] == []


def test_a_traceparent_from_across_a_queue_is_joined(prompt, spans):
    parent = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
    _render(prompt, "run-q", trace=C.trace("run-q", "triage", traceparent=parent))

    (span,) = [s for s in spans.get_finished_spans() if s.name == TM.SPAN_NAME]
    assert format(span.context.trace_id, "032x") == "0af7651916cd43dd8448eb211c80319c"
    assert format(span.parent.span_id, "016x") == "b7ad6b7169203331"


def test_nothing_emitted_carries_wording_values_or_context(prompt, spans, caplog):
    """The acceptance test: render, fail a render three ways, and read back every
    span attribute, span event, status and log line."""
    with caplog.at_level(logging.DEBUG):
        with ot.get_tracer("orchestrator").start_as_current_span("step"):
            out = _render(prompt, f"run-{uuid.uuid4().hex[:8]}")
            assert "ZEBRA-VALUE" in out["instructions"]          # the text itself has it
            for broken in (
                {"vars": {"supplier": 7, "customer": {"ssn": SECRET}}},          # wrong type
                {"vars": {"supplier": VALUE}},                                   # missing
                {"vars": {"supplier": VALUE, "customer": {}, "extra": VALUE}},   # undeclared
                {"obligations": [f"leak:{VALUE}"]},                              # not in vocabulary
            ):
                with pytest.raises(T.TemplateError):
                    _render(prompt, "run-broken", **broken)

    emitted = _everything_emitted(spans, caplog)
    assert TM.SPAN_NAME in emitted, "nothing was captured, so this proves nothing"
    for marker in MARKERS:
        assert marker not in emitted, f"{marker!r} reached a span or a log"


def test_a_failed_render_is_marked_with_its_code_and_nothing_else(prompt, spans):
    with ot.get_tracer("orchestrator").start_as_current_span("step"):
        with pytest.raises(T.VarMissing):
            _render(prompt, "run-f", vars={"supplier": VALUE})

    (span,) = [s for s in spans.get_finished_spans() if s.name == TM.SPAN_NAME]
    assert span.attributes["prompt.error_code"] == "VAR_MISSING"
    assert span.events == ()
    assert not span.status.is_ok and span.status.description is None


def test_an_id_that_is_not_shaped_like_an_id_is_hashed_not_written(prompt, spans):
    with ot.get_tracer("orchestrator").start_as_current_span("step"):
        _render(prompt, f"a run id that is really a sentence about {VALUE}")

    (span,) = [s for s in spans.get_finished_spans() if s.name == TM.SPAN_NAME]
    assert span.attributes["prompt.run_id"].startswith("unreadable:")
    assert "ZEBRA" not in span.attributes["prompt.run_id"]
