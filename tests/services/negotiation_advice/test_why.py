from src.services.negotiation_advice import advisor as ad


_PLAY = {"lever": "Commercial", "play": "Demand tiered volume discounts",
         "rationale": "deterministic fallback", "state": "ready",
         "evidence": [{"label": "Deal value", "value": "200,000",
                       "source": "deal"}]}
_SIG = {"supplier_name": "Orbis", "deal_value": 200000.0, "currency": "GBP"}


def test_why_uses_the_model_output():
    out = ad.explain_play(_PLAY, _SIG,
                          generate=lambda **kw: "Because volume is material.")
    assert out == "Because volume is material."


def test_prompt_carries_only_grounded_facts_and_forbids_invention():
    seen = {}

    def _gen(**kw):
        seen["prompt"] = kw.get("prompt") or ""
        return "ok"

    ad.explain_play(_PLAY, _SIG, generate=_gen)
    prompt = seen["prompt"]
    assert "Orbis" in prompt
    assert "200,000" in prompt or "200000" in prompt
    assert "Demand tiered volume discounts" in prompt
    low = prompt.lower()
    assert "do not" in low and ("invent" in low or "fabricate" in low)


def test_model_is_called_with_think_false():
    seen = {}
    ad.explain_play(_PLAY, _SIG,
                    generate=lambda **kw: seen.update(kw) or "ok")
    assert seen.get("think") is False


def test_llm_failure_falls_back_to_the_deterministic_rationale():
    def _boom(**kw):
        raise RuntimeError("ollama down")

    assert ad.explain_play(_PLAY, _SIG, generate=_boom) == "deterministic fallback"


def test_empty_model_output_falls_back():
    assert ad.explain_play(_PLAY, _SIG, generate=lambda **kw: "   ") == \
        "deterministic fallback"
