"""Who answers a question typed into an Ask box.

Until this module, every question went down one pipeline whatever screen it was
asked on, and Chat's agent picker was a label: the agent a conversation was
started "with" was never sent. Now the person's choice wins, then the agent the
screen belongs to, then General, which is the pipeline as it always was.

`choose_responder` is pure, so the whole table is tested without a database or a
model. Spec: specs/2026-10-15-ask-agent-routing-design.md.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

logger = logging.getLogger(__name__)

GENERAL = "general"

#: The governed policy holding the screen->agent map (proc.bp_policy).
POLICY = "ask_screen_agents"

#: The only tools an agent answering a question may use (Nick, 2026-10-15: Ask
#: is read-only). A list of what is allowed, not of what is barred, so a tool
#: added later stays out until someone puts it here. Every agent tool (run_*) is
#: absent because running an agent writes its own records and results, and
#: evaluate_opportunity_lever is absent because it records the lever on the
#: bucket. run_calculation stays: it writes only an audit line saying it ran.
READ_ONLY_TOOLS = frozenset({
    "list_governance", "get_policy", "get_prompt",
    "get_corpus_facts",
    "list_opportunity_levers", "get_opportunity_bucket",
    "list_calculations", "run_calculation",
})

#: Told to every answering agent, so one whose instructions say "send" or
#: "record" does not claim to have done it.
READ_ONLY_NOTE = (
    "You are answering a question in the Ask box. Here you can look things up "
    "but not change anything: do not say you have sent, saved, recorded or "
    "approved anything. If the person wants a change made, say where in the "
    "app they can make it."
)


def _read_only_tools(agent: Any) -> List[str]:
    """The reading tools, narrowed to the agent's own list if it was given one."""
    allowed = getattr(agent, "allowed_tools", None)
    if allowed is None:
        return sorted(READ_ONLY_TOOLS)
    return sorted(READ_ONLY_TOOLS & set(allowed))


#: Agents that send, write records, or exist as pipeline steps. They are never
#: offered a question, even when a map row names one, and an agent answering a
#: question is never given them as tools: an Ask must not be the way an email
#: goes out or an approval is filed.
NOT_ASKABLE = frozenset({
    "email_dispatch", "email_drafting", "email_watcher",
    "data_extraction", "supplier_interaction", "approvals",
    # pipeline-internal: General is the rag path, the other two check opportunities
    "rag", "opportunity_critic", "opportunity_evidence",
})

#: How long the screen map is trusted before it is read again, so an admin's edit
#: applies within a minute without a restart.
MAP_TTL_SECONDS = 60.0
_MAP_CACHE: Dict[str, Any] = {"at": None, "value": {}}


def _norm(value: Optional[str]) -> Optional[str]:
    text = str(value or "").strip().lower()
    return text or None


def agent_label(slug: Optional[str]) -> str:
    if not slug:
        return "General"
    return slug.replace("_", " ").replace("-", " ").strip().capitalize() + " agent"


@dataclass(frozen=True)
class Responder:
    slug: Optional[str]          # None means General
    source: str                  # chosen | screen | fallback | default | records
    note: Optional[str] = None

    def answered_by(self, label: Optional[str] = None) -> Dict[str, Any]:
        return {"agent": self.slug, "label": label or agent_label(self.slug),
                "source": self.source, "note": self.note}


def _unavailable(slug: str) -> Responder:
    return Responder(None, "fallback",
                     f"{agent_label(slug)} is not available, so this was answered generally.")


def askable_agents(registry: Mapping[str, Any], user_id: Optional[str] = None) -> List[str]:
    """The agents this person may ask. A draft is offered to its creator only, the
    rule BaseAgent.execute enforces for every other way of running one."""
    out = []
    for slug, agent in registry.items():
        if slug in NOT_ASKABLE:
            continue
        owner = getattr(agent, "draft_owner", None)
        if owner is not None and (not user_id or owner != user_id):
            continue
        out.append(slug)
    return sorted(out)


def choose_responder(agent: Optional[str], screen: Optional[str],
                     available: Iterable[str], screen_map: Mapping[str, str]) -> Responder:
    pool = {s.lower() for s in available}
    chosen = _norm(agent)
    if chosen == GENERAL:
        return Responder(None, "chosen")
    if chosen:
        return Responder(chosen, "chosen") if chosen in pool else _unavailable(chosen)
    mapped = _norm((screen_map or {}).get(_norm(screen) or ""))
    if mapped:
        return Responder(mapped, "screen") if mapped in pool else _unavailable(mapped)
    return Responder(None, "default")


def screen_agent_map(read: Optional[Callable[[str], Dict[str, Any]]] = None, *,
                     now: Callable[[], float] = time.monotonic) -> Dict[str, str]:
    """The screen->agent map an admin governs, or {} when it cannot be read.

    Empty is safe here, unlike a governed limit: no map means every screen
    answers as General, which is the Ask box as it was. A routing outage must
    not make every question fail.
    """
    if read is None:
        at = _MAP_CACHE["at"]
        if at is not None and now() - at < MAP_TTL_SECONDS:
            return dict(_MAP_CACHE["value"])
        value = screen_agent_map(_fresh_rules)
        _MAP_CACHE.update(at=now(), value=value)
        return dict(value)
    try:
        screens = (read(POLICY) or {}).get("screens")
    except Exception as exc:  # noqa: BLE001
        logger.warning("ask screen map unreadable; every screen answers as General: %s", exc)
        return {}
    if not isinstance(screens, dict):
        return {}
    return {str(k).strip().lower(): str(v).strip().lower()
            for k, v in screens.items() if str(k).strip() and str(v).strip()}


def _fresh_rules(policy: str) -> Dict[str, Any]:
    from src.services import governed_limits
    return governed_limits.rules(policy, fresh=True)


def _tool_exclusions(slug: Optional[str]) -> tuple:
    try:
        from agents.base_agent import NODE_TOOL_EXCLUSIONS
    except ImportError:  # pragma: no cover - imported as src.services
        from src.agents.base_agent import NODE_TOOL_EXCLUSIONS
    own = (f"run_{slug}",) if slug else ()
    return tuple(NODE_TOOL_EXCLUSIONS) + tuple(f"run_{s}" for s in sorted(NOT_ASKABLE)) + own


def agent_answer(agent: Any, query: str, *, user_id: Optional[str],
                 context: Optional[str], slug: Optional[str] = None) -> Optional[str]:
    """The agent's answer to one question, or None, meaning "answer it generally".

    The agent reasons under its own instructions and policies
    (BaseAgent.reason), with the reading tools only (READ_ONLY_TOOLS). `user_id` reaches the agent-policy gate on every tool
    call, which is the per-person check on what this agent may do for them.
    """
    # Never the sending or writing agents, nor itself (a built-in agent has no
    # governance_slug, so BaseAgent.reason cannot exclude it for us).
    # And only the reading tools: an Ask changes nothing.
    kwargs: Dict[str, Any] = {"user_id": user_id, "exclude": _tool_exclusions(slug),
                              "only": _read_only_tools(agent),
                              "extra_system": "\n\n".join(p for p in (context, READ_ONLY_NOTE) if p)}
    try:
        result = agent.reason(query, **kwargs)
    except Exception:  # noqa: BLE001 - a broken agent is a General answer, not a 500
        logger.exception("ask agent %s failed", type(agent).__name__)
        return None
    text = str(getattr(result, "answer", "") or "").strip()
    if getattr(result, "error", None) and not text:
        logger.warning("ask agent %s errored: %s", type(agent).__name__, result.error)
        return None
    return text or None
