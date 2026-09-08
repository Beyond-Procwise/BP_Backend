"""The headline: the one sentence a model is allowed to write, and its gate.

Everything else in an analytic answer is arithmetic. This is the single place
a model earns its keep — it is shown the facts and says which of them matters —
and it is therefore the only place a model can put a figure in front of a
customer. So the rule is not "please only use the numbers given"; the rule is
that a sentence carrying a number the payload does not support is thrown away,
and the templated headline the answer already shipped with stands instead.

Three things follow from that, and all three are the point of this module:

  * **The writer never sees a row.** It is shown the scope line and the facts,
    each already rendered by the shared formatter. A model shown the raw
    amounts is a model formatting money again.
  * **The check is mechanical.** ``AnalyticAnswer.unquoted_numbers`` strips the
    entity names, then compares every remaining figure against the ones the
    facts and scope actually carry. Naming "Kestrel Supplies 8" licences no
    "8% of spend".
  * **Failure is free.** No output, unparseable output, a hedge, a third
    sentence, a recommendation, a model that is down — every one of them lands
    on the same fallback, and the reader gets the templated headline rather
    than an apology or a gap.

Both outcomes are written to the audit spine with the hash of the prompt that
produced them, so a sentence a customer saw can be traced to the payload it was
written from.

The instruction text below is the built-in. Governance may override it: the
caller resolves ``proc.bp_prompt`` and passes ``template`` (see
``load_governed_instructions``), which keeps this module pure and keeps the DB
out of the path that renders an answer.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

from src.services.agent_actions import record_action
from src.services.analytics.models import AnalyticAnswer, Headline

logger = logging.getLogger(__name__)

AGENT = "analytic_insight_writer"
PHASE = "analytics"
ACTION_TYPE = "insight_write"
PROMPT_TYPE = "analytic_insight"

# Grammar-constrained: the model cannot return anything but this one field, so
# there is no prose wrapper, no preamble and no second key to parse around.
INSIGHT_SCHEMA: dict = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
}

INSTRUCTIONS = """You write the one-line finding above a procurement answer.

You are given the scope of the answer and the facts that were measured from it.
Write at most two sentences saying which fact matters and why, for the reader
named below.

Rules:
- Use only the figures printed in the facts, exactly as they are printed there.
  Any other number, or the same number rounded differently, is a fabrication.
- State what is true. Do not recommend, advise or suggest anything.
- Do not hedge: no "may", "might", "could", "appears to", "likely".
- No preamble, no restating the question, no offer to help further.
"""

# A hedge is the model declining to say what the arithmetic already settled;
# a recommendation is it answering a question nobody asked. Both are refused
# because the answer under them is exact.
_HEDGES = re.compile(
    r"\b(may|might|could|possibly|potentially|perhaps|probably|likely|seems?|"
    r"appears?|suggests?\s+that|roughly|approximately|around)\b", re.I)
# Colour standing in for a measurement. "Elevated supply risk", "a significant
# share", "healthy competition" — none of these is a figure, so the grounding
# check cannot see them, and none of them is measured anywhere in the answer.
# A claim the reader cannot check is the same defect as a number they cannot
# check, so it is refused unless the answer itself uses the word: the caveats
# under the table are written by this system, and a sentence repeating one is
# quoting the answer rather than editorialising over it.
_JUDGEMENTS = re.compile(
    r"\b(risks?|risky|exposures?|significant\w*|substantial\w*|material(ly)?|"
    r"concerning|worrying|troubling|alarming|healthy|unhealthy|strong\w*|weak\w*|"
    r"poor|excellent|impressive|elevated|critical|severe|vulnerable|fragile|"
    r"over-?rel\w+|over-?depend\w+|depend(ence|ency)|dominant|dominates?)\b", re.I)

_RECOMMENDATIONS = re.compile(
    r"\b(should|shouldn't|must|ought|recommend\w*|advise\w*|suggest|consider|"
    r"needs?\s+to|worth\s+\w+ing|we\s+can\s+help)\b", re.I)

MAX_SENTENCES = 2


@dataclass(frozen=True)
class Rejection:
    """Why a sentence was thrown away. Recorded, never shown to the reader."""

    reason: str
    detail: str = ""


def build_prompt(answer: AnalyticAnswer, persona: str, template: Optional[str] = None) -> str:
    """What the writer is shown: the scope, the facts, and who is reading.

    Facts only — the table rows carry unrounded amounts, and a model shown
    those is back to choosing precision and glyphs for itself.
    """
    lines = [(template or INSTRUCTIONS).strip(), "", f"Reader: {persona}", "",
             f"Scope: {answer.scope.line()}", "", "Facts:"]
    if answer.facts:
        for fact in answer.facts:
            note = ", ".join(part for part in (fact.entity, fact.unit) if part)
            lines.append(f"- {fact.code.value}: {fact.display}" + (f" ({note})" if note else ""))
    else:
        lines.append("- none")
    if answer.anomalies:
        lines.extend(["", "Caveats the reader is already shown:"])
        lines.extend(f"- {anomaly.text}" for anomaly in answer.anomalies)
    return "\n".join(lines)


def _sentences(text: str) -> list[str]:
    """Sentences, counted on a terminator that ends a word — not on a decimal.

    "£1.2M" and "57.1%" both carry a full stop that no reader hears as the end
    of a sentence, and a splitter that does hears three sentences in one.
    """
    return [part for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part]


def _answer_vocabulary(answer: AnalyticAnswer) -> str:
    """Everything the answer itself already says, for the judgement check."""
    parts = [anomaly.text for anomaly in answer.anomalies]
    parts.extend(fact.code.value.replace("_", " ") for fact in answer.facts)
    parts.extend(fact.unit for fact in answer.facts if fact.unit)
    parts.extend(column.label for column in answer.table.columns)
    return " ".join(parts).lower()


def _unsupported_judgement(answer: AnalyticAnswer, text: str) -> Optional[str]:
    """The first judgement word the answer does not itself use, if any."""
    vocabulary = _answer_vocabulary(answer)
    for match in _JUDGEMENTS.finditer(text):
        word = match.group(0).lower()
        if word not in vocabulary:
            return word
    return None


def validate_insight(answer: AnalyticAnswer, text: str) -> Optional[Rejection]:
    """``None`` if the sentence may be shown, otherwise why it may not."""
    candidate = (text or "").strip()
    if not candidate:
        return Rejection("empty")
    if len(_sentences(candidate)) > MAX_SENTENCES:
        return Rejection("too_many_sentences", str(len(_sentences(candidate))))
    hedge = _HEDGES.search(candidate)
    if hedge:
        return Rejection("hedged", hedge.group(0))
    recommendation = _RECOMMENDATIONS.search(candidate)
    if recommendation:
        return Rejection("recommendation", recommendation.group(0))
    judgement = _unsupported_judgement(answer, candidate)
    if judgement:
        return Rejection("unsupported_claim", judgement)
    unquoted = answer.unquoted_numbers(candidate)
    if unquoted:
        return Rejection("ungrounded_number", ", ".join(sorted(unquoted)))
    return None


def _parse(raw: Any) -> tuple[str, Optional[Rejection]]:
    if raw is None or not str(raw).strip():
        return "", Rejection("no_output")
    try:
        payload = json.loads(str(raw))
    except (ValueError, TypeError):
        return "", Rejection("unparseable", str(raw)[:200])
    if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
        return "", Rejection("unparseable", str(raw)[:200])
    return payload["text"].strip(), None


# One attempt, and not a long one. This call sits on the request path with a
# templated headline already written behind it, so a model that is slow or down
# costs the reader a sentence, not their answer. The client's own defaults are
# three attempts at ten minutes each — right for an extraction job running in
# the background, wrong here. (Today, live: every ollama_generate call returns
# 500 "memory layout cannot be allocated with num_gpu = 999" because the card
# cannot hold this 30B model whole alongside the API's own 3.8GB — under the
# defaults that failure would have held the answer for thirty minutes.)
WRITER_TIMEOUT_SECONDS = 45


def _ollama_writer(prompt: str) -> Optional[str]:
    """The platform model, constrained to the schema. think=False: a reasoning
    model otherwise answers in a field this call does not read."""
    from src.services.ollama_client import ollama_generate

    return ollama_generate(prompt, format=INSIGHT_SCHEMA, temperature=0,
                           num_predict=256, think=False, retries=1,
                           timeout=WRITER_TIMEOUT_SECONDS)


def load_governed_instructions(conn: Any) -> Optional[str]:
    """The instruction text from ``proc.bp_prompt``, if governance set one.

    Read by the caller rather than here: an answer is rendered on the request
    path, and a governance row that cannot be read must cost that path nothing.
    """
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT prompts_desc FROM proc.bp_prompt "
                "WHERE prompt_type = %s AND COALESCE(prompts_status, 1) = 1 "
                "ORDER BY prompt_id DESC LIMIT 1", (PROMPT_TYPE,))
            row = cur.fetchone()
    except Exception:
        logger.debug("analytic insight prompt lookup failed; using the built-in", exc_info=True)
        return None
    if not row or not row[0]:
        return None
    payload = row[0] if isinstance(row[0], dict) else json.loads(row[0])
    if isinstance(payload, dict):
        template = payload.get("prompt_template") or payload.get("template")
        if template and str(template).strip():
            return str(template)
    return None


def write_insight(
    answer: AnalyticAnswer,
    *,
    persona: str = "default",
    generate: Optional[Callable[[str], Optional[str]]] = None,
    audit: Callable[..., None] = record_action,
    template: Optional[str] = None,
) -> AnalyticAnswer:
    """The answer with a written headline, or exactly the answer it was given."""
    prompt = build_prompt(answer, persona, template)
    text = ""
    rejection: Optional[Rejection] = None

    try:
        raw = (generate or _ollama_writer)(prompt)
    except Exception as exc:  # the model being down is not the reader's problem
        logger.warning("analytic insight writer unavailable: %s", exc)
        rejection = Rejection("model_unavailable", str(exc)[:200])
    else:
        text, rejection = _parse(raw)
        if rejection is None:
            rejection = validate_insight(answer, text)

    written = answer if rejection else answer.model_copy(update={
        "headline": Headline(text=text, confidence=answer.headline.confidence)})

    details = {"persona": persona,
               "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()}
    if rejection:
        details["reason"] = rejection.reason
        details["detail"] = rejection.detail
        details["text"] = text
    try:
        audit(phase=PHASE, action_type=ACTION_TYPE, agent=AGENT, field_name="headline",
              trace_id=answer.answer_id, status="rejected" if rejection else "ok",
              summary=written.headline.text, details=details)
    except Exception as exc:  # an audit that will not write costs the reader nothing
        logger.warning("analytic insight audit write failed: %s", exc)
    return written
