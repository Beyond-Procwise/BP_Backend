"""What is this supplier actually asking for -- and can we prove it from their words.

Two rules, inherited from the decision engine and the grounding work:

1. The model classifies; it does not decide. This returns an intent and a confidence,
   and the caller applies the governed policy. Nothing here sends anything.
2. Every classification must carry a verbatim sentence from the supplier's own reply,
   checked against the stored body. A classification we cannot point at is reported
   as ungrounded and the caller escalates. It is never quietly trusted, and never
   quietly discarded either -- an ungrounded claim is itself worth showing a human.

AgentNick is the only model used here. There is no fallback to another model: a
second model would answer differently and no one would know which one spoke.

Why ``caller`` and not ``agent_nick``: ``AgentNick`` (src/agents/base_agent.py) is a
tool-calling loop (``reason()``); it has no one-shot chat method. The one-shot call in
this codebase is ``call_ollama``, defined on ``BaseAgent`` -- so this module takes any
object exposing that method (a ``BaseAgent`` subclass in practice), not AgentNick
itself.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from src.services.obligations.grounding import is_quote_grounded

log = logging.getLogger(__name__)

# The vocabulary the policy is written against. An intent outside this set is
# 'unclassified' -- policy cannot reason about a label it has never heard of, and a
# label the model invented is not evidence of anything.
KNOWN_INTENTS = (
    # consequential
    "price_change", "terms_change", "contract_variation",
    "liability", "dispute", "new_commitment",
    # routine
    "acknowledge", "confirm_receipt", "request_missing_document",
    "chase_no_response", "clarify_lead_time", "out_of_office",
)

UNCLASSIFIED = "unclassified"

# AgentNick is the only permitted model. This mirrors config/settings.py's
# local_primary_model default -- used only when a caller carries no settings at all.
_DEFAULT_MODEL = "BeyondProcwise/AgentNick:unified"

_SYSTEM = (
    "You classify one inbound supplier email for a procurement team. "
    "Reply with JSON only, no prose, with exactly these keys: "
    '{"intent": <one of ' + "|".join(KNOWN_INTENTS) + '>, '
    '"confidence": <number between 0 and 1>, '
    '"quote": <one sentence copied VERBATIM from the email that supports the intent>}. '
    "The quote must be copied character-for-character from the email. Do not "
    "paraphrase, summarise, correct or complete it. If no sentence supports a "
    "classification, return intent 'unclassified' with confidence 0."
)


@dataclass
class ReplyIntent:
    intent: str
    confidence: float
    quote: str
    grounded: bool
    reason: str = ""


def _unusable(reason: str) -> ReplyIntent:
    return ReplyIntent(intent=UNCLASSIFIED, confidence=0.0, quote="", grounded=False, reason=reason)


def _field(obj: Any, key: str) -> Any:
    """One field off a dict OR off an ollama response model.

    ``ollama.chat()`` does not return a dict. It returns a ``ChatResponse``
    (``ollama._types.SubscriptableBaseModel``, a pydantic model with ``get`` and
    ``__getitem__``), and its ``message`` is a ``Message`` model, not a dict. So
    ``isinstance(response, dict)`` is False on the real call path -- which is why
    this is duck-typed on ``.get`` with an attribute fallback rather than
    isinstance-gated. Nothing here imports ollama: the shape is what matters, and
    plain dicts (what the tests pass) must keep working identically.
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    getter = getattr(obj, "get", None)
    if callable(getter):
        try:
            return getter(key)
        except Exception:  # noqa: BLE001 - some other object's unrelated .get
            pass
    return getattr(obj, key, None)


def _extract_ollama_message(response: Any) -> str:
    """Pull the model's text out of whichever shape ``call_ollama`` returned.

    ``call_ollama`` (src/agents/base_agent.py) returns the raw ``ollama`` client
    result: the chat shape (``message.content``) when called with ``messages=``, or
    the generate shape (``response``) otherwise. Missing or empty in both shapes
    returns "", which the caller treats as unusable.

    This started as a copy of ``EmailDraftingAgent._extract_ollama_message`` and is
    deliberately no longer identical to it: that version opens with
    ``if not isinstance(response, dict): return ""``, which discards every real
    ``ollama.chat()`` result (see ``_field``). Verified live 2026-07-28 --
    AgentNick:unified returned exactly the JSON asked for and the extractor threw it
    away, so every classification came back "did not return usable JSON" and every
    reply escalated. The same latent bug is still in email_drafting_agent.py:2827;
    fixing it there is not in this task's scope.
    """
    content = _field(_field(response, "message"), "content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    content = _field(response, "response")
    if isinstance(content, str) and content.strip():
        return content.strip()
    return ""


def _extract_json(text: str) -> dict:
    """Parse the model's JSON object. ``format="json"`` is requested at the call
    site but is currently inert on this call path (see the comment there) --
    nothing constrains the model to emit an object at all, so a syntactically
    valid but non-object response (``"null"``, ``"5"``, ``"[1,2]"``, ``"true"``)
    is a real possibility, not a hypothetical. Falls back to the first ``{...}``
    block in the text for a model that wraps its answer in prose. Either way, the
    result must be a dict -- a bare scalar or list is not "usable JSON" and must
    raise so the caller folds it into the unusable result instead of crashing on
    ``.get()``.
    """
    if not text:
        raise ValueError("empty response")
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        match = re.search(r"\{.*\}", str(text), re.DOTALL)
        if not match:
            raise ValueError("no JSON object in response")
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError(f"parsed JSON is not an object (got {type(parsed).__name__})")
    return parsed


def _resolve_model(caller: Any) -> str:
    """AgentNick's configured name, from the caller's settings if it carries any,
    else the module-level settings singleton. There is no other model to fall back
    to -- if neither source resolves, the hardcoded AgentNick default is used.
    """
    settings_obj = getattr(caller, "settings", None)
    if settings_obj is None:
        try:
            from config.settings import settings as settings_obj  # local: avoid import cost when caller already carries settings
        except Exception:  # noqa: BLE001
            return _DEFAULT_MODEL
    return getattr(settings_obj, "local_primary_model", _DEFAULT_MODEL) or _DEFAULT_MODEL


def classify_reply(body: str, *, caller: Any, min_quote_words: int = 4) -> ReplyIntent:
    """Classify ``body``, requiring a grounded verbatim quote. Never raises.

    ``caller`` is any object exposing ``call_ollama`` (a ``BaseAgent`` subclass in
    practice) -- see the module docstring for why this is not ``agent_nick``.
    """
    text = (body or "").strip()
    if not text:
        return _unusable("the reply has no body to classify")

    model = _resolve_model(caller)
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": f"Email:\n{text}"},
    ]
    try:
        response = caller.call_ollama(
            model=model,
            messages=messages,
            # NOTE: currently inert on this path. BaseAgent.call_ollama
            # (base_agent.py:1150) only forwards `format` to ollama.generate();
            # the messages= branch calls ollama.chat() without it. Kept anyway --
            # harmless, and becomes a real grammar constraint for free if that is
            # ever fixed. Until then, `_extract_json`'s prose fallback below is
            # doing the actual work, not this.
            format="json",
            think=False,  # reasoning models return an empty `response` otherwise
            # temperature belongs INSIDE options. `call_ollama` pops "options" and
            # merges it over agent_nick.ollama_options(); every OTHER kwarg is
            # splatted straight into ollama.chat(), which has no `temperature`
            # parameter and raises TypeError: Client.chat() got an unexpected keyword
            # argument 'temperature'. As `temperature=0` that made EVERY live
            # classification fail ("the classifier was unreachable"), so every reply
            # escalated -- fail-closed, but the classifier never ran once. Caught by
            # calling it with a real BaseAgent; the unit tests take **kwargs and so
            # cannot see it. See tests/engines/test_decide_email_reply.py's live run
            # in the Task 6 report.
            options={"temperature": 0},
        )
    except Exception:  # noqa: BLE001
        log.exception("email intent classification failed")
        return _unusable("the classifier was unreachable")

    raw = _extract_ollama_message(response)
    try:
        payload = _extract_json(raw)
    except Exception:  # noqa: BLE001
        return _unusable("the classifier did not return usable JSON")

    intent = str(payload.get("intent") or "").strip()
    if intent not in KNOWN_INTENTS:
        return _unusable(f"'{intent or 'missing'}' is not a governed intent")

    try:
        confidence = float(payload.get("confidence"))
    except (TypeError, ValueError):
        return _unusable("the classifier returned no usable confidence")
    confidence = max(0.0, min(1.0, confidence))

    quote = str(payload.get("quote") or "").strip()
    grounded = is_quote_grounded(quote, text, min_words=min_quote_words)
    reason = (
        "classified with a grounded quote" if grounded
        else "the supporting sentence was not found verbatim in the reply"
    )
    return ReplyIntent(
        intent=intent, confidence=confidence, quote=quote, grounded=grounded, reason=reason
    )
