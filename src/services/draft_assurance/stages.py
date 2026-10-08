"""Stages 1, 3 and 4 as model calls, each wrapped in validation. Never raises.

Every function takes ``ask(system, user) -> text`` (the agent supplies one over its model; tests
supply fakes that return good AND bad output) and the governed prompt TEXT. A missing prompt means
the stage is ``unavailable`` -- it is never run from a prompt written into the code, because a
governed instruction that has quietly become a hardcoded one is how governance rots.

Every result carries a ``status``: ``captured`` (it ran and the output passed validation),
``invalid`` (it ran and the output was refused, with the reason) or ``unavailable`` (it could not
run). A caller can therefore tell "the model said nothing useful" from "nothing was asked".
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional

from .brief import BriefInvalid, normalise_brief
from .classify import ClassificationInvalid, clarification_for, parse_classification
from .judge import JudgeInvalid, parse_judgement

logger = logging.getLogger(__name__)
Ask = Callable[[str, str], str]


def _fill(template: str, **values: Any) -> str:
    """Named placeholders only. ``str.format`` would trip on the JSON braces a prompt must contain."""
    for key, value in values.items():
        template = template.replace("{" + key + "}", value if isinstance(value, str) else json.dumps(value, default=str))
    return template


def _json(text: str) -> Any:
    from src.services.email_intent import _extract_json
    return _extract_json(text)


def _unavailable(reason: str) -> Dict[str, Any]:
    return {"status": "unavailable", "reason": reason}


def _run(ask: Ask, system: str, user: str):
    try:
        return _json(ask(system, user)), None
    except Exception as exc:  # noqa: BLE001 - a dead model, an empty answer and prose are all "no usable JSON"
        logger.warning("stage model call failed: %s", exc)
        return None, f"no usable JSON from the model ({type(exc).__name__})"


def classify_request(ask: Ask, template: Optional[str], request: str,
                     families: Dict[str, str], labels: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Stage 1 for free text. ``families`` is {family_id: description}, read from config at run time."""

    if not template:
        return _unavailable("the governed classifier prompt is not installed")
    if not families:
        return _unavailable("no families are configured")
    listing = "\n".join(f"- {fid}: {desc}" for fid, desc in sorted(families.items()))
    system = _fill(template, families=listing, request=request)
    raw, err = _run(ask, system, request)
    if err:
        return {"status": "invalid", "reason": err}
    try:
        c = parse_classification(raw, request, families)
    except ClassificationInvalid as exc:
        return {"status": "invalid", "reason": str(exc)}
    labels = {fid: (labels or {}).get(fid) or desc.split(".")[0].strip() or fid for fid, desc in families.items()}
    return {"status": "captured", "classification": c.as_dict(),
            "clarification": clarification_for(c, labels)}


def plan_brief(ask: Ask, template: Optional[str], *, family_id: str, tone: Optional[Dict[str, Any]],
               instruction: str, facts: Dict[str, Any], context: Dict[str, Any], request: str) -> Dict[str, Any]:
    """Stage 3 for free text. Facts go in as authoritative; the answer is held to the brief schema."""

    if not template:
        return _unavailable("the governed planner prompt is not installed")
    system = _fill(template, family=family_id, tone=(tone or {}).get("values") or {},
                   instruction=instruction or "none", facts=facts, context=context)
    raw, err = _run(ask, system, request)
    if err:
        return {"status": "invalid", "reason": err}
    try:
        brief = normalise_brief(raw, fact_keys=facts, context_keys=context)
    except BriefInvalid as exc:
        return {"status": "invalid", "reason": str(exc)}
    if brief["status"] == "missing":
        return {"status": "captured", "brief": brief}
    # The planner is told not to put figures in the brief that are not in the facts or the request.
    # It does not always obey, so the same deterministic check that guards the email guards the plan.
    from . import validator as V
    prose = " ".join(str(brief.get(k) or "") for k in ("goal", "explicit_ask", "deadline", "tone_rationale")
                     ) + " " + " ".join(brief.get("key_points") or [])
    allowed_nums = V.numbers_in(facts) | V.numbers_in(request) | V.numbers_in(context)
    allowed_dates = V.dates_in([str(v) for v in facts.values()] + [request] + [str(v) for v in context.values()])
    allowed_refs = {str(v) for v in facts.values()} | set(V._REF.findall(request or ""))
    bad = V.check_figures(prose, allowed_nums, allowed_dates, allowed_refs)
    if bad:
        return {"status": "invalid",
                "reason": "the brief states figures that are in no fact and not in the request: "
                          + ", ".join(sorted({b["detail"] for b in bad}))}
    return {"status": "captured", "brief": brief}


def judge_draft(ask: Ask, template: Optional[str], rubric: List[str], *, text: str,
                brief: Optional[Dict[str, Any]], facts: Dict[str, Any]) -> Dict[str, Any]:
    """Stage 4: score the draft 1..5 against the family's rubric. Never invents a score."""

    if not template:
        return _unavailable("the governed judge prompt is not installed")
    if not rubric:
        return _unavailable("the family defines no rubric")
    system = _fill(template, rubric=rubric, brief=brief or {}, facts=facts)
    raw, err = _run(ask, system, text)
    if err:
        return {"status": "invalid", "reason": err}
    try:
        return parse_judgement(raw, rubric)
    except JudgeInvalid as exc:
        return {"status": "invalid", "reason": str(exc)}
