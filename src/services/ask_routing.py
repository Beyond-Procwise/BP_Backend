"""Who answers a question typed into an Ask box.

Until this module, every question went down one pipeline whatever screen it was
asked on, and Chat's agent picker was a label: the agent a conversation was
started "with" was never sent. Now the person's choice wins, then the agent the
screen belongs to, then General, which is the pipeline as it always was.

`choose_responder` is pure, so the whole table is tested without a database or a
model. Spec: specs/2026-10-15-ask-agent-routing-design.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional

GENERAL = "general"

#: Agents that send, act on documents, or exist as pipeline steps. They are never
#: offered a question, even when a map row names one: an Ask must not be the way
#: an email goes out.
NOT_ASKABLE = frozenset({
    "email_dispatch", "email_drafting", "email_watcher",
    "data_extraction", "supplier_interaction",
})


def _norm(value: Optional[str]) -> Optional[str]:
    text = str(value or "").strip().lower()
    return text or None


def agent_label(slug: Optional[str]) -> str:
    if not slug:
        return "General"
    return slug.replace("_", " ").strip().capitalize() + " agent"


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


def askable_agents(registry: Mapping[str, Any]) -> List[str]:
    return sorted(s for s in registry.keys() if s not in NOT_ASKABLE)


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
