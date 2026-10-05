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

**An example the instruction quoted is not a value the requester gave.** Measured live on
2026-10-05 against a resident AgentNick, one request came back with thirty fields — every one of
them claiming "high" confidence — and `criteria` was `['99.95% SLA', '≤1 weekly outage',
'≥15% unit-rate reduction']`. Two of the three are copied verbatim out of the governed template,
which offers them as what a measurable outcome looks like; the requester wrote neither. The same
template offers "IT-3300" as what a cost centre looks like, and a cost centre nobody named is
what a demand gets routed for approval by. So a value the INSTRUCTION supplied is dropped — but
only when the requester's own text does not contain it, which is why a genuine reading can never
be caught by this (see `_leaked`).

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
import re
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

# ---------------------------------------------------------------------------
# Fields a procurement REQUEST does not state, and the cues that show it does after all.
#
# Measured live 2026-10-05 against a resident AgentNick. Prompt v2 says in words "Do not
# calculate a saving, split a budget into years, set a target", and the model returned three
# of them anyway on a request that states none:
#
#   finance.saving = 95000  -- 95000 is the CURRENT MPLS cost in that request. Arithmetic
#                              would have said 45000 (3 x 95k = 285k against a 240k budget),
#                              so the figure is both unasked-for and wrong.
#   finance.tco    = 240000 -- the budget, relabelled.
#   finance.type   = 'Hard / cash' -- a finance classification nobody wrote down.
#
# These three set an approval route and feed the savings KPI, so an invented number is worse
# than a blank. Asking the prompt more firmly has been tried.
#
# NOT a deny-list. A requester may well state a saving, and deleting that would be the same
# fault in the other direction, so the value travels when the text carries a cue for THAT
# field — or when the field is the one the conversation just asked about, because then the
# question itself is the claim.
#
# The cues are about MONEY on purpose. "reduce outages" must not license an invented saving,
# which is why the bare words "reduce" and "reduction" are not cues, and "over three years"
# is a budget period rather than a statement of total cost of ownership.
# ---------------------------------------------------------------------------
_CLAIM_CUES: Dict[str, tuple] = {
    "finance.saving": ("save", "saves", "saving", "savings", "cost reduction",
                       "unit-rate reduction", "rate reduction", "cost avoidance",
                       "avoided cost", "cheaper"),
    "finance.tco": ("tco", "total cost", "cost of ownership", "whole-life", "whole life",
                    "lifetime cost", "life-cycle cost", "lifecycle cost"),
    "finance.type": ("hard saving", "cash saving", "cashable", "cost avoidance",
                     "non-financial", "non financial", "hard / cash", "soft saving"),
    # The other five from the same reply. Prompt v2 stopped them, which means they were held
    # back by wording alone and one prompt edit from returning:
    #   finance.phasing = 'Year 1: £80k, Year 2: £80k, Year 3: £80k'
    #   benefit.target  = '£80k/year SD-WAN cost'   (no 80k appears anywhere in that request)
    #   benefit.owner   = 'IT Operations'
    #   pillar          = 'Cost Efficiency'
    #   alignment       = 'Digital Transformation'
    #
    # "£95k a year" and "over three years" say how much and for how long; they are NOT a
    # phasing profile, so no cue here is built out of "year" on its own.
    "finance.phasing": ("phased", "phasing", "year 1", "year one", "first year",
                        "second year", "split over", "split across", "spread over",
                        "spread across", "instalments", "installments"),
    "benefit.target": ("target", "targets", "targeted", "aim", "aims", "aiming", "goal",
                       "goals", "down to", "get to", "reduce to"),
    # NOT bare "owner": this conversation talks about the COST CENTRE owner constantly — it has
    # an ask_owner turn — and that is a different person from whoever owns the benefit.
    "benefit.owner": ("benefit owner", "owned by", "owner is", "will own", "accountable",
                      "responsible for", "sponsor", "sponsored by", "sponsors"),
    "pillar": ("pillar", "pillars", "strategic priority", "strategic priorities"),
    "alignment": ("align", "aligns", "aligned", "alignment", "okr", "okrs", "strategy",
                  "strategic", "objective", "objectives", "initiative", "initiatives"),
}
# DELIBERATELY ABSENT: benefit.baseline. "Today the MPLS circuits cost us £95k a year" is a
# baseline the requester really did state, and putting it in this table would turn a rule against
# fabrication into a rule that destroys evidence.
# Word boundaries matter: a bare "tco" would otherwise match inside "bitcoin".
_CLAIM_PATTERNS = {
    path: re.compile(r"\b(?:" + "|".join(re.escape(c) for c in cues) + r")\b", re.I)
    for path, cues in _CLAIM_CUES.items()
}
# The quoted literals in the instruction's own prose — "IT-3300", "99.95% SLA". Both quote styles,
# because the governed template is edited by hand and uses whichever the editor's keyboard gave.
_QUOTED = re.compile(r'"([^"\n]{2,60})"|“([^”\n]{2,60})”')


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
        # The guard below needs BOTH sides: what the instruction quoted, and what the requester
        # actually wrote. It reads the live template rather than a copy, so an admin who edits
        # the governed row changes the example set in the same edit.
        leaked = self._instruction_examples(str(template))
        said = self._said(context or {})
        offered = self._offered(context or {})
        asked = str((context or {}).get("asked_field") or "").strip()
        # `failed` says the model did not answer — NOT that it found nothing. call_ollama does
        # not raise when Ollama refuses; it returns {"response": "", "error": …}, so without this
        # distinction a model that never ran looked exactly like a request with nothing in it,
        # and the screen would tell the requester their words held no values.
        return {"fields": self._fields(parsed, leaked, said, offered, asked),
                "governed": True, "failed": not answered}

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

    @staticmethod
    def _norm(value: Any) -> str:
        """One spelling for comparing a value against a text: case-folded, whitespace collapsed,
        and stripped of the punctuation a model puts round a value it is quoting."""
        return " ".join(str(value).split()).strip(" .,;:()[]\"\u2018\u2019\u201c\u201d").casefold()

    @classmethod
    def _instruction_examples(cls, template: str) -> frozenset:
        """The literals the INSTRUCTION itself quotes, as its own illustrations.

        The placeholders are blanked FIRST, which is the whole reason this reads the template and
        not the rendered prompt: `{text}` is the requester's words and `{fields}` is this tenant's
        configuration, and a literal quoted in either of those is not something the instruction
        made up. What is left is the governed prose, and the strings it quotes there are examples.
        """
        skeleton = template
        for name in _PLACEHOLDERS:
            skeleton = skeleton.replace("{" + name + "}", " ")
        out = set()
        for match in _QUOTED.finditer(skeleton):
            literal = cls._norm(match.group(1) or match.group(2) or "")
            if literal:
                out.add(literal)
        return frozenset(out)

    @classmethod
    def _said(cls, context: Dict[str, Any]) -> str:
        """The requester's own words, normalised the same way, for the absence half of the test."""
        return cls._norm(context.get("text") or "")

    @classmethod
    def _offered(cls, context: Dict[str, Any]) -> str:
        """What this tenant's configuration offers: the field list and the category names.

        A value the configuration offers belongs to the tenant, not to the instruction. Without
        this the harvested example set — which really does contain "high", "medium" and "low",
        because the template names the confidence words and quotes them — would refuse 'High' on
        a tenant's own High | Medium | Low field whenever the requester said "urgent" instead.
        """
        return cls._norm(" | ".join([cls._one(context.get("fields")),
                                     cls._one(context.get("categories"))]))

    @classmethod
    def _unclaimed(cls, path: str, said: str, asked: str = "") -> bool:
        """Is this a field the request never claims a figure for?

        True only for the handful of fields in _CLAIM_CUES, and only when neither the text nor
        the question just asked shows the requester talking about that field. Every other path
        is none of this method's business.
        """
        pattern = _CLAIM_PATTERNS.get(path)
        if pattern is None:
            return False
        if path == asked:
            # The conversation asked for exactly this, so the answer is about this.
            return False
        return not pattern.search(said or "")

    @classmethod
    def _leaked(cls, value: Any, leaked: frozenset, said: str, offered: str = "") -> bool:
        """Is this value the instruction's example rather than the requester's evidence?

        THREE conditions, and the last two are what make the guard safe. A value is refused only
        when the instruction quoted it, the request does not contain it, and this tenant's
        configuration does not offer it. So a requester who really did ask for a 99.95% SLA keeps
        it; an option the configuration lists is never refused; and a value the model merely
        normalised — a date read out of "March", money stripped of its separators — is not an
        example the instruction quoted at all, so this never sees it.
        """
        norm = cls._norm(value)
        return (bool(norm) and norm in leaked
                and norm not in said and norm not in offered)

    @classmethod
    def _fields(cls, parsed: Dict[str, Any], leaked: frozenset = frozenset(),
                said: str = "", offered: str = "",
                asked: str = "") -> Dict[str, Dict[str, Any]]:
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
            if cls._unclaimed(key, said, asked):
                logger.debug("demand intake extraction: %s is not something this request "
                             "claims, so the model's figure is dropped", key)
                continue
            value, confidence = cls._value_of(entry)
            if isinstance(value, (list, tuple)):
                # Each element on its own: the live reply mixed two of the template's examples in
                # with one real reading, and the real one must survive.
                kept = [v for v in value if not cls._leaked(v, leaked, said, offered)]
                if len(kept) != len(value):
                    logger.debug("demand intake extraction: dropped %d quoted example(s) from %s",
                                 len(value) - len(kept), key)
                value = kept
            elif cls._leaked(value, leaked, said, offered):
                logger.debug("demand intake extraction: %s came back as the instruction's own "
                             "example and is not in the request", key)
                continue
            # A field left with nothing but examples falls out here rather than travelling empty.
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
