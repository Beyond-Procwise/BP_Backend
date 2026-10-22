"""The context builder: assembles the system prompt in labelled layers.

The orchestrator, not the model, decides what AgentNick is told. Layers, in a fixed order:
``base`` (the standing rules), ``extra`` (what the caller added), ``skill`` (the playbook the router
chose, in full, with its settings) and ``memory`` (what was remembered of this conversation). With no
skill and no extra, the prompt is exactly the base: nothing about a plain request changes.

A skill is injected whole or not at all. A playbook cut short would be followed halfway, so one that
does not fit the budget is skipped and the model is left to read it itself. A skill is also skipped
when it needs a tool this agent was not given: its steps would name something that cannot be called.
"""
from __future__ import annotations

import json
import re
from datetime import date
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

MAX_SYSTEM_CHARS = 24000
_TOOL_REF = re.compile(r"`tool:([a-z_0-9]+)`")
_NOT_AVAILABLE = re.compile(r"NOT AVAILABLE: ([a-z_0-9]+)")
# Settings only the router needs; the model gains nothing from reading them.
_INTERNAL_SETTINGS = ("routing",)


@dataclass(frozen=True)
class Layer:
    name: str
    text: str
    source: str


@dataclass(frozen=True)
class Context:
    system: str
    layers: List[Layer] = field(default_factory=list)
    skill: Optional[str] = None
    skipped: Optional[str] = None

    def describe(self) -> List[Dict[str, Any]]:
        return [{"layer": l.name, "chars": len(l.text), "source": l.source} for l in self.layers]


def tools_needed(body: str) -> Set[str]:
    """The tools a playbook tells the agent to call, leaving out any it says are not available."""
    unavailable = set(_NOT_AVAILABLE.findall(body))
    return set(_TOOL_REF.findall(body)) - unavailable


def can_inject(skill: Any, offered: Set[str]) -> Tuple[bool, str]:
    missing = sorted(tools_needed(skill.body) - set(offered))
    if missing:
        return False, f"the skill needs tools this agent was not given: {', '.join(missing)}"
    return True, ""


def _skill_text(skill: Any) -> str:
    settings = {k: v for k, v in (skill.config or {}).items() if k not in _INTERNAL_SETTINGS}
    return (
        f"Skill loaded for this request: {skill.name}\n"
        "Follow it. It is already here in full, so do not call list_skills or use_skill for it. "
        "Use its settings as given and no others. Where it says to read a policy, call get_policy once for "
        "that topic; when nothing governs the case, say so and carry on with the playbook's own defaults "
        "rather than trying other slugs. 'SC-007' is the name of the policy-hooks section, not a policy.\n\n"
        f"Settings:\n{json.dumps(settings, indent=2, default=str)}\n\n"
        f"{skill.body.strip()}"
    )


def today_line(today: date) -> str:
    """The one fact the model cannot get from a tool. Live, 2026-10-22: asked for 'the last two years' it
    chose 2022-04-01 to 2024-03-31, because nothing had told it the date."""
    return (f"Today is {today.strftime('%A')} {today.isoformat()}. Work any period a person names ('this year', "
            "'the last two years', 'since January') out from this date and pass it to tools as start and end "
            "dates; never guess the year.")


def build(*, base: str, extra: Optional[str] = None, skill: Any = None, memory: Optional[str] = None,
          today: Optional[date] = None, max_chars: int = MAX_SYSTEM_CHARS) -> Context:
    layers: List[Layer] = [Layer("base", base, "orchestrator")]
    if today is not None:
        layers.append(Layer("today", today_line(today), "orchestrator"))
    if extra:
        layers.append(Layer("extra", extra, "caller"))
    skipped: Optional[str] = None
    chosen: Optional[str] = None
    if skill is not None:
        text = _skill_text(skill)
        fixed = sum(len(l.text) for l in layers) + len(memory or "")
        if fixed + len(text) > max_chars:
            skipped = f"the skill is too long for the prompt budget ({fixed + len(text)} of {max_chars} characters)"
        else:
            layers.append(Layer("skill", text, f"skill:{skill.name}"))
            chosen = skill.name
    if memory:
        layers.append(Layer("memory", memory, "conversation memory"))
    return Context("\n\n".join(l.text for l in layers), layers, chosen, skipped)
