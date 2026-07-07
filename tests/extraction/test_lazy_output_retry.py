"""context_layer recovers from a lazy/degenerate LLM output via one corrective retry."""
from src.services.extraction import context_layer as CL


def test_retries_on_lazy_ellipsis_output(monkeypatch):
    """First call returns invalid lazy JSON with '...'; retry (temp-bumped) returns valid."""
    calls = []

    def fake_call_llm(prompt, temperature=0.0):
        calls.append(temperature)
        if len(calls) == 1:
            return '{"invoice_id": "INV-1", ...}'  # lazy → invalid JSON
        return '{"invoice_id": "INV-1"}'           # valid on corrective retry

    monkeypatch.setattr(CL, "_call_llm", fake_call_llm)
    out = CL.synthesize("invoice", "Invoice number INV-1 total 10", {})
    assert len(calls) == 2, "should retry exactly once on parse failure"
    assert calls[1] > 0, "retry must bump temperature to break the deterministic loop"
    assert out.get("invoice_id") == "INV-1", "corrective retry recovered the field"


def test_no_retry_when_first_output_valid(monkeypatch):
    """A valid first response must NOT trigger a retry (working docs unchanged)."""
    calls = []

    def fake_call_llm(prompt, temperature=0.0):
        calls.append(temperature)
        return '{"invoice_id": "INV-1"}'

    monkeypatch.setattr(CL, "_call_llm", fake_call_llm)
    CL.synthesize("invoice", "Invoice number INV-1", {})
    assert len(calls) == 1, "no retry when the first parse succeeds"


def test_returns_raw_candidates_when_retry_also_fails(monkeypatch):
    def fake_call_llm(prompt, temperature=0.0):
        return '{"invoice_id": "INV-1", ...}'  # always lazy

    monkeypatch.setattr(CL, "_call_llm", fake_call_llm)
    raw = {"invoice_id": "SEED"}
    out = CL.synthesize("invoice", "INV-1", raw)
    assert out == raw  # unchanged fallback, no crash
