"""Grounding a supplier's sentence, without opening the digit hole.

The obligations guard requires 8+ words because a contract sentence is never
shorter. A supplier reply legitimately is: "We can offer 94,000.00 GBP." is five
words and is the single most important sentence in the thread. The floor has to be
a parameter, not a constant -- and lowering it must not reintroduce
extraction_v3.is_value_grounded's digit-signature fallback, which passes wholly
invented sentences.
"""

# tests/conftest.py owns sys.path (repo root + src/) -- see the note in
# tests/api/test_decisions_email_endpoints.py.
from src.services.obligations.grounding import is_quote_grounded

REPLY = (
    "Thank you for the proposal. We can offer 94,000.00 GBP with 45 day payment "
    "terms and a 14 day lead time."
)


def test_short_supplier_sentence_grounds_with_a_lower_floor():
    assert is_quote_grounded("We can offer 94,000.00 GBP.", REPLY, min_words=4) is False
    # The trailing full stop is not in the source; the sentence itself is.
    assert is_quote_grounded("We can offer 94,000.00 GBP", REPLY, min_words=4) is True


def test_default_floor_is_unchanged_for_contract_callers():
    eight_words = "The Contractor shall indemnify the Authority in full"
    assert is_quote_grounded(eight_words, eight_words) is True
    assert is_quote_grounded("shall indemnify the Authority", eight_words) is False


def test_a_fabricated_sentence_never_grounds():
    # Digits present in the source, sentence not. This is the exact failure mode the
    # obligations guard exists to prevent -- it must survive the new parameter.
    assert is_quote_grounded(
        "We will absorb the 94,000.00 GBP increase entirely", REPLY, min_words=4
    ) is False


def test_empty_source_never_grounds():
    assert is_quote_grounded("We can offer 94,000.00 GBP", "", min_words=4) is False


import json
from types import SimpleNamespace

from src.services.email_intent import ReplyIntent, classify_reply


def _caller(payload):
    """Stand-in for a BaseAgent-like caller: one call_ollama call.

    ``call_ollama`` (src/agents/base_agent.py) returns the ``ollama`` chat shape
    when called with ``messages=`` -- ``{"message": {"content": ...}}`` -- so the
    stub mirrors that instead of the nonexistent ``agent_nick.chat(...)``.
    """
    calls = []

    def call_ollama(*args, **kwargs):
        calls.append((args, kwargs))
        return {"message": {"content": payload}}

    return SimpleNamespace(call_ollama=call_ollama, _calls=calls)


def test_classifies_a_price_change_with_a_grounded_quote():
    caller = _caller(json.dumps({
        "intent": "price_change",
        "confidence": 0.94,
        "quote": "We can offer 94,000.00 GBP",
    }))
    out = classify_reply(REPLY, caller=caller)
    assert out.intent == "price_change"
    assert out.confidence == 0.94
    assert out.grounded is True


def test_an_ungrounded_quote_is_reported_not_trusted():
    caller = _caller(json.dumps({
        "intent": "price_change",
        "confidence": 0.99,
        "quote": "We will absorb the increase entirely",
    }))
    out = classify_reply(REPLY, caller=caller)
    assert out.grounded is False
    # The claim is preserved for inspection; the caller decides (it escalates).
    assert out.intent == "price_change"
    assert "not found" in out.reason.lower()


def test_an_unknown_intent_becomes_unclassified():
    caller = _caller(json.dumps(
        {"intent": "vibes", "confidence": 0.9, "quote": "We can offer 94,000.00 GBP"}
    ))
    out = classify_reply(REPLY, caller=caller)
    assert out.intent == "unclassified"
    assert out.confidence == 0.0


def test_unparseable_model_output_is_unusable_not_guessed():
    out = classify_reply(REPLY, caller=_caller("I think it's a price change!"))
    assert out.intent == "unclassified"
    assert out.confidence == 0.0
    assert out.grounded is False


def test_a_raising_model_is_unusable_not_fatal():
    def boom(*a, **k):
        raise RuntimeError("ollama down")

    out = classify_reply(REPLY, caller=SimpleNamespace(call_ollama=boom))
    assert out.intent == "unclassified"
    assert out.confidence == 0.0


def test_empty_body_is_unusable():
    out = classify_reply("", caller=_caller("{}"))
    assert out.intent == "unclassified"
    assert isinstance(out, ReplyIntent)


def test_a_bare_json_scalar_is_unusable_not_a_crash():
    # json.loads("null") succeeds and returns None, not a dict. Nothing upstream
    # constrains the model to emit an object -- format="json" is inert on this
    # call path (see the comment at the call site) -- so this is a real response
    # shape, not a hypothetical. Must fold into the unusable result, not raise
    # AttributeError out of payload.get(...).
    out = classify_reply(REPLY, caller=_caller("null"))
    assert out.intent == "unclassified"
    assert out.confidence == 0.0
    assert out.grounded is False


def test_a_bare_json_list_is_unusable_not_a_crash():
    out = classify_reply(REPLY, caller=_caller("[1, 2]"))
    assert out.intent == "unclassified"
    assert out.confidence == 0.0


def test_missing_confidence_is_unusable():
    caller = _caller(json.dumps({
        "intent": "price_change",
        "quote": "We can offer 94,000.00 GBP",
    }))
    out = classify_reply(REPLY, caller=caller)
    assert out.intent == "unclassified"
    assert out.confidence == 0.0


def test_non_numeric_confidence_is_unusable():
    caller = _caller(json.dumps({
        "intent": "price_change",
        "confidence": "high",
        "quote": "We can offer 94,000.00 GBP",
    }))
    out = classify_reply(REPLY, caller=caller)
    assert out.intent == "unclassified"
    assert out.confidence == 0.0


# ---------------------------------------------------------------------------
# Two bugs found by calling this with a REAL BaseAgent against live Ollama
# (2026-07-28) rather than a **kwargs stub. Both made every live classification
# unusable, so every supplier reply escalated and the classifier never once ran.
# Fail-closed, but inert. The dict-shaped stub above cannot see either.
# ---------------------------------------------------------------------------
class _Message(SimpleNamespace):
    """Shaped like ollama._types.Message: a pydantic model, NOT a dict."""

    def get(self, key, default=None):
        return getattr(self, key, default)


class _ChatResponse(_Message):
    """Shaped like ollama._types.ChatResponse, which is what ollama.chat returns."""


def test_the_real_ollama_chat_response_shape_is_read():
    """`isinstance(response, dict)` is False for a ChatResponse.

    AgentNick:unified returned exactly the JSON asked for and it was discarded,
    reported as "the classifier did not return usable JSON".
    """
    payload = json.dumps({
        "intent": "price_change",
        "confidence": 0.95,
        "quote": "We can offer 94,000.00 GBP",
    })
    response = _ChatResponse(message=_Message(role="assistant", content=payload))
    # Both levels are models, not dicts -- exactly what ollama.chat returns. If this
    # stub is ever "simplified" back to plain dicts these assertions fail, which is
    # the point: a dict stub is what let both bugs ship.
    assert not isinstance(response, dict)
    assert not isinstance(response.get("message"), dict)
    out = classify_reply(REPLY, caller=SimpleNamespace(call_ollama=lambda **k: response))
    assert out.intent == "price_change"
    assert out.grounded is True


def test_the_generate_response_shape_is_read_when_it_is_not_a_dict_either():
    """The other real shape: ollama.generate() -> GenerateResponse.response."""
    payload = json.dumps({
        "intent": "acknowledge", "confidence": 0.9,
        "quote": "We can offer 94,000.00 GBP",
    })
    response = _ChatResponse(response=payload, message=None)
    assert not isinstance(response, dict)
    out = classify_reply(REPLY, caller=SimpleNamespace(call_ollama=lambda **k: response))
    assert out.intent == "acknowledge"
    assert out.grounded is True


def test_temperature_is_passed_inside_options_not_as_a_bare_kwarg():
    """`call_ollama` splats unknown kwargs into ollama.chat(), which has no
    `temperature` parameter -- a bare temperature=0 raised TypeError every time.
    """
    caller = _caller(json.dumps({
        "intent": "acknowledge", "confidence": 0.9,
        "quote": "We can offer 94,000.00 GBP",
    }))
    classify_reply(REPLY, caller=caller)
    _args, kwargs = caller._calls[0]
    assert "temperature" not in kwargs, "would raise TypeError inside ollama.chat()"
    assert kwargs["options"]["temperature"] == 0


def test_a_caller_whose_chat_rejects_stray_kwargs_still_classifies():
    """The end-to-end guard: a caller with ollama.chat's real strictness."""
    def call_ollama(*, model, messages, options=None, format=None, think=None):
        return {"message": {"content": json.dumps({
            "intent": "acknowledge", "confidence": 0.9,
            "quote": "We can offer 94,000.00 GBP",
        })}}

    out = classify_reply(REPLY, caller=SimpleNamespace(call_ollama=call_ollama))
    assert out.intent == "acknowledge"
    assert out.grounded is True
