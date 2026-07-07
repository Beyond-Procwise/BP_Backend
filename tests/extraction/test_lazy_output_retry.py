"""context_layer recovers from a lazy/degenerate LLM output.

Fast free-form first; on parse failure, a grammar-constrained retry (Ollama
`format` schema) guarantees valid JSON. Healthy docs never retry.
"""
from src.services.extraction import context_layer as CL


def test_constrained_retry_on_lazy_ellipsis(monkeypatch):
    monkeypatch.setattr(CL, "_CONSTRAINED_DECODING", True)
    calls = []

    def fake_call_llm(prompt, temperature=0.0, fmt=None):
        calls.append(fmt)
        if len(calls) == 1:
            return '{"invoice_id": "INV-1", ...}'  # lazy free-form (fmt=None)
        return '{"invoice_id": "INV-1"}'           # constrained retry (fmt=schema)

    monkeypatch.setattr(CL, "_call_llm", fake_call_llm)
    out = CL.synthesize("invoice", "Invoice number INV-1 total 10", {})
    assert len(calls) == 2, "should retry exactly once on parse failure"
    assert calls[0] is None, "first pass is free-form"
    assert calls[1] is not None, "retry passes a JSON schema (constrained decoding)"
    assert out.get("invoice_id") == "INV-1"


def test_no_retry_when_first_output_valid(monkeypatch):
    calls = []

    def fake_call_llm(prompt, temperature=0.0, fmt=None):
        calls.append(fmt)
        return '{"invoice_id": "INV-1"}'

    monkeypatch.setattr(CL, "_call_llm", fake_call_llm)
    CL.synthesize("invoice", "Invoice number INV-1", {})
    assert len(calls) == 1, "no retry when the first parse succeeds"


def test_returns_raw_candidates_when_retry_also_fails(monkeypatch):
    def fake_call_llm(prompt, temperature=0.0, fmt=None):
        return '{"invoice_id": "INV-1", ...}'  # always invalid

    monkeypatch.setattr(CL, "_call_llm", fake_call_llm)
    raw = {"invoice_id": "SEED"}
    out = CL.synthesize("invoice", "INV-1", raw)
    assert out == raw  # unchanged fallback, no crash


def test_temperature_fallback_when_constrained_disabled(monkeypatch):
    monkeypatch.setattr(CL, "_CONSTRAINED_DECODING", False)
    calls = []

    def fake_call_llm(prompt, temperature=0.0, fmt=None):
        calls.append(temperature)
        if len(calls) == 1:
            return '{"invoice_id": "INV-1", ...}'
        return '{"invoice_id": "INV-1"}'

    monkeypatch.setattr(CL, "_call_llm", fake_call_llm)
    out = CL.synthesize("invoice", "Invoice number INV-1", {})
    assert len(calls) == 2 and calls[1] > 0  # temperature bump fallback
    assert out.get("invoice_id") == "INV-1"
