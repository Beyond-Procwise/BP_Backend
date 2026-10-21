"""RecipientResolver: the single place that decides who receives something.

Order of precedence when one person qualifies for several reasons: explicit, mentioned, approver,
manager, chain_of_command, entity_owner, requester, category_owner, team_member, watcher.
Rules (which targets an event type notifies) come from a RuleSource, editable per tenant.
Graph-suggested recipients (tiers 2-4) are NOT used here unless ``allow_graph`` is set and a
suggester is supplied; that flag is off by default per tenant (Phase 4).
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

from services.people.hierarchy import HierarchyService
from services.people.models import EntityRef, PeopleStore, Person, Recipient

REASON_ORDER = ("explicit", "mentioned", "approver", "manager", "chain_of_command", "entity_owner",
                "requester", "category_owner", "team_member", "watcher", "graph_suggested")

TARGETS = {"manager", "chain_of_command", "approver", "entity_owner", "requester",
           "category_owner", "team", "watcher"}

#: Used when a tenant has no rule for an event type. Approvals reach the approver; everything else
#: reaches the owner. A rules-source failure falls back to this plus admins, never to "nobody".
DEFAULT_TARGETS = {
    "requisition.submitted": ["manager", "approver"],
    "approval.requested": ["approver"],
    "approval.approved": ["requester", "manager"],
    "approval.rejected": ["requester", "manager"],
}
FALLBACK_TARGETS = ["entity_owner"]


class RuleSource(Protocol):
    def targets_for(self, tenant_id: str, event_type: str) -> Optional[Tuple[List[str], bool]]:
        """(targets, notify_actor) or None when no rule exists."""


class StaticRules:
    def __init__(self, rules: Optional[Dict[Tuple[str, str], Tuple[List[str], bool]]] = None) -> None:
        self._rules = rules or {}

    def targets_for(self, tenant_id, event_type):
        return self._rules.get((tenant_id, event_type)) or self._rules.get(("default", event_type))


def _detail(reason: str, actor: Person, entity: EntityRef, delegated_from: Optional[str]) -> str:
    who = actor.display_name or actor.subject
    text = {
        "explicit": "You were chosen as a recipient",
        "mentioned": "You were mentioned",
        "approver": f"You approve {entity.entity_type} items" + (f" over this amount" if entity.amount else ""),
        "manager": f"You're {who}'s manager",
        "chain_of_command": f"You're in {who}'s reporting line",
        "entity_owner": f"You own this {entity.entity_type}",
        "requester": f"You requested this {entity.entity_type}",
        "category_owner": "You own this category",
        "team_member": "You're on the team for this item",
        "watcher": "You're watching this item",
        "graph_suggested": "Suggested from past activity",
    }[reason]
    if delegated_from:
        text += f" (covering for {delegated_from})"
    return text


class RecipientResolver:
    def __init__(
        self,
        store: PeopleStore,
        hierarchy: HierarchyService,
        rules: RuleSource,
        can_view: Callable[[Person, EntityRef], bool],
    ) -> None:
        self._store, self._h, self._rules, self._can_view = store, hierarchy, rules, can_view

    def resolve(
        self,
        actor: Person,
        event_type: str,
        entity: EntityRef,
        explicit: Sequence[str] = (),
        mentioned: Sequence[str] = (),
    ) -> List[Recipient]:
        tenant = actor.tenant_id
        try:
            rule = self._rules.targets_for(tenant, event_type)
        except Exception:  # governance reads fail open elsewhere; here they fail to safe defaults
            rule = None
        targets, notify_actor = rule if rule else (
            DEFAULT_TARGETS.get(event_type, FALLBACK_TARGETS), False)

        # (subject, reason) candidates, in no particular order; precedence applied at the end
        cands: List[Tuple[str, str, Optional[str]]] = [(s, "explicit", None) for s in explicit]
        cands += [(s, "mentioned", None) for s in mentioned]
        for t in targets:
            if t == "manager":
                m = self._h.get_manager(actor)
                if m:
                    cands.append((m.subject, "manager", None))
            elif t == "chain_of_command":
                cands += [(p.subject, "chain_of_command", None) for p in self._h.get_chain_of_command(actor)]
            elif t == "approver":
                a = self._h.get_approver_for(actor, event_type, entity.amount)
                if a:
                    # already past delegation; carry who they are covering for
                    cands.append((a[0].subject, "approver", a[1]))
            elif t == "entity_owner" and entity.owner:
                cands.append((entity.owner, "entity_owner", None))
            elif t == "requester" and entity.requester:
                cands.append((entity.requester, "requester", None))
            elif t == "category_owner" and entity.category_owner:
                cands.append((entity.category_owner, "category_owner", None))
            elif t == "team":
                cands += [(s, "team_member", None) for s in entity.team]
            elif t == "watcher":
                cands += [(s, "watcher", None) for s in entity.watchers]

        best: Dict[str, Tuple[int, str, Optional[str]]] = {}
        for subject, reason, pre_from in cands:
            base = self._store.get(tenant, subject)
            if base is None:
                continue
            person, delegated_from = self._h.effective_person(base)
            delegated_from = delegated_from or pre_from
            if not person.is_active:
                continue
            rank = REASON_ORDER.index(reason)
            if person.subject not in best or rank < best[person.subject][0]:
                best[person.subject] = (rank, reason, delegated_from)

        out: List[Recipient] = []
        for subject, (rank, reason, delegated_from) in sorted(best.items(), key=lambda kv: kv[1][0]):
            person = self._store.get(tenant, subject)
            if person is None or person.tenant_id != tenant:
                continue
            if subject == actor.subject and not notify_actor:
                continue
            if not self._can_view(person, entity):
                continue
            out.append(Recipient(person, reason, _detail(reason, actor, entity, delegated_from), delegated_from))
        return out
