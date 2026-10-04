"""Demand Intake: read a procurement request, return values for fields already named.

This agent does ONE thing, and the narrowness is the design. The intake conversation (the
SpendIQ Demand screen) owns the questions: which step comes next, what it asks, which tier
switches it on, and what counts as a valid answer all live in the intake configuration, which an
admin edits. The model is never asked for a question, never asked for an opinion, and never
writes to the record — it reads what the requester typed and proposes values, every one of which
the browser validates against that same configuration before it is kept.

Two rules hold it up:

**The instruction is governed.** The prompt is `demand_intake_extract` in proc.bp_prompt, and
this agent resolves it the way every other agent resolves its own. The browser sends only the
VALUES to fill it with — the categories this tenant uses, the field list, what is already known,
the question just asked, and the requester's text. It used to build the prompt itself and hand it
over, which meant the instruction travelled from the client: anyone who could reach the endpoint
could tell the model to do something else. If the governed row is missing this agent REFUSES,
because falling back to a caller's instruction is worse than not extracting at all — and the
conversation degrades gracefully, asking its next question either way.

**Nothing leaves in a shape the caller has to defend itself against.** A model answers
`{"title": "x"}` as readily as `{"title": {"value": "x"}}`, invents confidence words, and
occasionally answers in prose. All of that is normalised here, so what comes back is always
`{path: {"value": …, "confidence": "high"|"medium"|"low"}}`.

The placeholders are filled with str.replace, NOT str.format: the template ends with a JSON
example, and format() reads `{"fields": …}` as a field reference and raises.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, Optional

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

PROMPT_NAME = "demand_intake_extract"

_CONFIDENCE = ("high", "medium", "low")
# Filled from the caller's context. Every one is DATA about this tenant's configuration or the
# requester's own words — never an instruction.
_PLACEHOLDERS = ("categories", "fields", "known", "asked_field", "text", "today", "currency")
# Keys a JSON object should never carry into a record. The browser validates every path against
# its configuration and would drop these anyway; they are refused here too so that a reply which
# is nonsense cannot be mistaken for one that merely failed validation.
_FORBIDDEN_PATHS = frozenset({"__proto__", "constructor", "prototype"})
_MAX_PATHS = 60


class DemandIntakeUnavailable(RuntimeError):
    """The governed prompt is not installed, so there is nothing authorised to run."""


class DemandIntakeAgent(BaseAgent):
    """Extraction for the demand intake conversation. No questions, no opinions."""

    def extract(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """-> {"fields": {path: {"value", "confidence"}}, "governed": True[, "failed": True]}.

        Raises DemandIntakeUnavailable when the governed prompt is missing. Every other failure
        — a model that is down, a timeout, a reply that is not JSON — returns no fields, because
        one model call is not worth losing the requester's session over.
        """
        template = self.resolve_prompt(PROMPT_NAME)
        if not template:
            raise DemandIntakeUnavailable(
                f"the prompt {PROMPT_NAME} is not installed, so extraction is not authorised")
        prompt = self._render(str(template), context or {})
        try:
            # think=False is required: AgentNick is a reasoning model and returns an empty
            # `response` without it (Model Routing Policy). format="json" is what makes the
            # reply parseable at all.
            raw = self.call_ollama(prompt=prompt, format="json", think=False)
        except Exception:
            logger.debug("demand intake extraction: model call failed", exc_info=True)
            return {"fields": {}, "governed": True, "failed": True}
        parsed, answered = self._parse(raw)
        # `failed` says the model did not answer — NOT that it found nothing. call_ollama does
        # not raise when Ollama refuses; it returns {"response": "", "error": …}, so without this
        # distinction a model that never ran looked exactly like a request with nothing in it,
        # and the screen would tell the requester their words held no values.
        return {"fields": self._fields(parsed), "governed": True, "failed": not answered}

    # ------------------------------------------------------------------
    @staticmethod
    def _one(value: Any) -> str:
        """A context value as one line of prompt text. A list is joined the way the prompt's own
        wording expects ("Valid categories: a | b"); anything else is stringified as given."""
        if isinstance(value, (list, tuple)):
            return " | ".join(str(v) for v in value if str(v).strip())
        if value is None:
            return ""
        return str(value)

    @classmethod
    def _render(cls, template: str, context: Dict[str, Any]) -> str:
        out = template
        for name in _PLACEHOLDERS:
            out = out.replace("{" + name + "}", cls._one(context.get(name)))
        return out

    @staticmethod
    def _parse(raw: Any) -> tuple[Dict[str, Any], bool]:
        """-> (the object the model sent, whether it answered at all).

        call_ollama returns a GenerateResponse, a dict or a string depending on the client — and
        on failure it returns {"response": "", "error": …} rather than raising, which is why the
        second element exists.
        """
        text: Optional[str]
        error: Any = None
        if isinstance(raw, str):
            text = raw
        else:
            text = getattr(raw, "response", None)
            error = getattr(raw, "error", None)
            if hasattr(raw, "get"):
                if text is None:
                    text = raw.get("response")
                if error is None:
                    error = raw.get("error")
        if error:
            logger.debug("demand intake extraction: the model host refused: %s", error)
            return {}, False
        if not isinstance(text, str) or not text.strip():
            return {}, False
        try:
            parsed = json.loads(text)
        except Exception:
            logger.debug("demand intake extraction: reply was not JSON")
            return {}, False
        return (parsed, True) if isinstance(parsed, dict) else ({}, False)

    @classmethod
    def _fields(cls, parsed: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """The fields the model found, in one shape.

        `{"fields": {...}}` is what the prompt asks for; a reply that IS the fields is accepted
        too, because that is the other half of what models actually send back.
        """
        found = parsed.get("fields")
        if not isinstance(found, dict):
            found = parsed if parsed and "fields" not in parsed else {}
        out: Dict[str, Dict[str, Any]] = {}
        for path, entry in list(found.items())[:_MAX_PATHS]:
            key = str(path).strip()
            if not key or key in _FORBIDDEN_PATHS:
                continue
            value, confidence = cls._value_of(entry)
            if not cls._has_value(value):
                continue
            out[key] = {"value": value, "confidence": confidence}
        return out

    @staticmethod
    def _value_of(entry: Any) -> tuple[Any, str]:
        if isinstance(entry, dict):
            value = entry.get("value")
            stated = str(entry.get("confidence") or "").strip().lower()
            # An invented confidence word becomes 'low' rather than travelling: the screen
            # shows confidence to the requester, so "extremely high" would be a claim the
            # model made up about itself.
            return value, stated if stated in _CONFIDENCE else "low"
        # A bare value carries no claim about itself, so it is the lowest confidence there is.
        return entry, "low"

    @staticmethod
    def _has_value(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, (list, tuple, dict)):
            return len(value) > 0
        return True


def extract_fields(agent_nick, context: Dict[str, Any]) -> Dict[str, Any]:
    """One call for a request handler: build the agent, extract, return."""
    return DemandIntakeAgent(agent_nick).extract(context)


__all__ = ["DemandIntakeAgent", "DemandIntakeUnavailable", "extract_fields", "PROMPT_NAME"]
