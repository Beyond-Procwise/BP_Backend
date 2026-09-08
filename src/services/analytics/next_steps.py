"""What to offer the reader next — chosen by rule, never written by a model.

The chips under an answer today are three LLM-written questions. They are not
grounded in the figures above them, they carry no entity, and clicking one
re-submits a sentence that has to be parsed back into an intent, which is
exactly where the subject of the question gets lost. And a model asked for
three follow-ups produces three, whether or not anything is worth asking.

So this engine is pure and table-driven: the same answer yields the same steps,
every time, with no model in the path. It returns at most two, and it returns
none rather than pad.

The ladder
----------
An analysis has an order. Having seen who the largest suppliers are, the next
useful question is how concentrated that is, then how it moved, then what the
money bought — not a jump to risk scoring. The ladder encodes that order as
data, one rung per question, each naming the action a click dispatches to and
the data that action needs to exist.

Selection, in order:

  1. An anomaly outranks everything. If the figures cannot be trusted until
     something is resolved, resolving it is the first step offered — and only
     the worst one, so two anomalies cannot consume both slots.
  2. A fact that crossed a threshold promotes its rung, with the value in the
     label.
  3. Otherwise, walk the ladder.
  4. Never more than one eligible rung is skipped, so a persona cannot vault
     the reader from "who is largest" to "what is our risk exposure".
  5. Anything the caller is not entitled to, or whose data is absent, is
     dropped — quietly and without error, because a step that dispatches
     nowhere is worse than no step.
  6. Two at most. Nothing qualifying means none.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional, Protocol, Sequence, Set, Tuple

from src.services.analytics.models import AnalyticAnswer, AnomalyCode, FactCode, NextStep

# The verbs a step label may open with. A label is an instruction, so it starts
# with the thing the click does. The optional LLM label writer is held to this
# same list.
VERBS: FrozenSet[str] = frozenset({
    "Review", "Compare", "Break", "Check", "Show", "Set", "Pick", "Clear",
    "Widen", "Open", "Flag",
})

MAX_STEPS = 2

# How far ahead of the current rung a step may reach, counted over the rungs
# that are actually eligible. One may be passed over; five may not.
MAX_SKIP = 1


@dataclass(frozen=True)
class Rung:
    """One question on the ladder, and the action that answers it."""

    name: str
    action_id: str
    label_template: str
    # Data that must exist for the action to lead anywhere. A rung whose data is
    # absent is dropped rather than offered as a dead end.
    requires_data: Tuple[str, ...] = ()


# The ladder for SUPPLIER_SPEND_RANKING, in the order a procurement analysis
# actually proceeds. `scope` is the answer already on screen, never a step.
LADDER: Tuple[Rung, ...] = (
    Rung("scope", "analytic.supplier_spend_ranking", "Show top suppliers by spend"),
    Rung("concentration", "analytic.supplier_concentration",
         "Review supplier concentration", ("spend",)),
    Rung("period_trend", "analytic.supplier_spend_trend",
         "Compare against last year", ("period_comparison",)),
    Rung("composition", "analytic.spend_composition",
         "Break down what we buy", ("category",)),
    Rung("contract_coverage", "analytic.contract_coverage",
         "Check contract coverage", ("contract_link",)),
    Rung("relationship_owner", "analytic.relationship_owner",
         "Show who owns these suppliers", ("relationship_owner",)),
    Rung("risk_exposure", "analytic.supplier_risk_exposure",
         "Review risk exposure", ("risk_score",)),
)

_RUNG_INDEX = {rung.name: i for i, rung in enumerate(LADDER)}

# Every prerequisite the ladder knows about. The live set is probed from the
# database at the call site: in this corpus, for instance, bp_invoice_trgt
# carries contract_id on 0 of 12,408 rows, so nothing links spend to the 3,051
# contracts on record and `contract_link` is absent however many contracts exist.
ALL_DATA: FrozenSet[str] = frozenset(
    key for rung in LADDER for key in rung.requires_data
)

# Which rung a given answer already stands on, so the ladder knows where it is.
# A chip leads to an answer that stands one rung higher, and the next chip has
# to start from there — otherwise the reader is offered the screen they are
# already looking at.
_QUERY_REF_RUNG: Dict[str, str] = {
    "supplier_spend_ranking/v1": "scope",
    "supplier_concentration/v1": "concentration",
    "supplier_spend_trend/v1": "period_trend",
}


@dataclass(frozen=True)
class AnomalyAction:
    action_id: str
    label_template: str


# Resolving the answer's own problems. These are not ladder rungs: they do not
# advance the analysis, they make it trustworthy, which is why they outrank it.
ANOMALY_ACTIONS: Dict[AnomalyCode, AnomalyAction] = {
    AnomalyCode.UNCONVERTED_CURRENCY: AnomalyAction(
        "currency.set_rate", "Set a rate for {subject}"),
    AnomalyCode.RANKING_NOT_COMPARABLE: AnomalyAction(
        "currency.select", "Pick a display currency"),
    AnomalyCode.MANUAL_FX_RATE: AnomalyAction(
        "currency.clear_manual", "Clear the manual rate"),
    AnomalyCode.MISSING_PERIOD_DATA: AnomalyAction(
        "period.widen", "Widen the period"),
}

# Persona priorities. These live as data beside the ladder so a profile can
# reorder what it reaches first without any of them being able to reach past
# rule 4. `default` walks the ladder as written.
#
# The specification's finance persona prioritises payment terms; there is no
# payment-terms rung on this intent's ladder, so finance's reachable priority
# here is the period comparison. When that rung exists, it goes in this table
# and nothing else changes.
PERSONA_PRIORITIES: Dict[str, Tuple[str, ...]] = {
    "default": (),
    "cpo": ("concentration", "period_trend"),
    "category_manager": ("composition",),
    "finance": ("period_trend",),
    "risk": ("risk_exposure",),
}


class Entitlements(Protocol):
    """Whether the caller may run an action. Implementations must fail closed."""

    def allows(self, action_id: str) -> bool: ...


class AllowAll:
    """Every action permitted.

    This platform has no entitlement service and no tenant dimension, so this
    is the stand-in — passed explicitly at the call site rather than defaulted
    to, so that the day a real gate exists there is exactly one line to change
    and it is visible in review. Passing ``None`` fails closed.
    """

    def allows(self, action_id: str) -> bool:  # noqa: D102
        return True


@dataclass(frozen=True)
class AllowList:
    """Only the named actions are permitted."""

    allowed: Set[str]

    def allows(self, action_id: str) -> bool:  # noqa: D102
        return action_id in self.allowed


def _entity_refs(answer: AnalyticAnswer) -> List[str]:
    """The suppliers on screen, in the order they are ranked."""
    refs = []
    for row in answer.table.rows:
        ref = row.get("entity_ref")
        if isinstance(ref, str) and ref and ref not in refs:
            refs.append(ref)
    return refs


def _anomaly_step(answer: AnalyticAnswer, entitlements: Entitlements) -> Optional[NextStep]:
    """A step that resolves the worst anomaly, if there is one and it is actionable."""
    for anomaly in answer.anomalies:  # already ordered worst-first by the contract
        action = ANOMALY_ACTIONS.get(anomaly.code)
        if action is None or not entitlements.allows(action.action_id):
            continue
        label = action.label_template.format(subject=anomaly.subject or "")
        return NextStep(
            action_id=action.action_id,
            label=" ".join(label.split()),
            rung="scope",
            reason=f"anomaly:{anomaly.code.value}",
            entity_refs=list(anomaly.entity_refs),
        )
    return None


def _eligible(
    answer: AnalyticAnswer,
    entitlements: Entitlements,
    available_data: Iterable[str],
) -> List[Rung]:
    """The rungs above this answer that lead somewhere for this caller."""
    available = set(available_data)
    current = _RUNG_INDEX.get(_QUERY_REF_RUNG.get(answer.provenance.query_ref, "scope"), 0)
    return [
        rung
        for rung in LADDER[current + 1:]
        if set(rung.requires_data) <= available and entitlements.allows(rung.action_id)
    ]


def _ordered_by_persona(eligible: Sequence[Rung], persona: str) -> List[Rung]:
    """Persona preferences first, but only among the rungs rule 4 lets them reach."""
    priorities = PERSONA_PRIORITIES.get(persona or "default")
    if priorities is None:
        priorities = PERSONA_PRIORITIES["default"]

    reachable = {rung.name for rung in eligible[: MAX_SKIP + 1]}
    preferred = [rung for name in priorities for rung in eligible
                 if rung.name == name and rung.name in reachable]
    rest = [rung for rung in eligible if rung not in preferred]
    return preferred + rest


def select_next_steps(
    answer: AnalyticAnswer,
    *,
    persona: str = "default",
    entitlements: Optional[Entitlements] = None,
    available_data: Iterable[str] = (),
) -> List[NextStep]:
    """At most two steps, chosen deterministically from the answer itself."""

    # An absent gate is not an open gate.
    if entitlements is None:
        return []

    # An answer with nothing in it has nothing to follow up. Offering a step
    # here would be padding of exactly the kind this replaces.
    if not answer.table.rows:
        return []

    steps: List[NextStep] = []
    taken: Set[str] = set()

    anomaly_step = _anomaly_step(answer, entitlements)
    if anomaly_step is not None:
        steps.append(anomaly_step)
        taken.add(anomaly_step.action_id)

    breach = next((f for f in answer.facts
                   if f.code is FactCode.CONCENTRATION_THRESHOLD_BREACHED), None)
    eligible = _eligible(answer, entitlements, available_data)
    ordered = _ordered_by_persona(eligible, persona)

    # A crossed threshold promotes its rung to the front of what is left, with
    # the value in the label so the chip states the finding rather than hinting
    # at it.
    promoted: Optional[str] = None
    if breach is not None:
        promoted = "concentration"
        ordered = ([r for r in ordered if r.name == promoted]
                   + [r for r in ordered if r.name != promoted])

    refs = _entity_refs(answer)
    for rung in ordered:
        if len(steps) >= MAX_STEPS:
            break
        if rung.action_id in taken:
            continue
        if rung.name == promoted and breach is not None:
            label = f"{rung.label_template} ({breach.display})"
            reason = f"fact:{breach.code.value}"
        else:
            label = rung.label_template
            reason = f"ladder:{persona or 'default'}"
        steps.append(NextStep(action_id=rung.action_id, label=label, rung=rung.name,
                              reason=reason, entity_refs=list(refs)))
        taken.add(rung.action_id)

    return steps[:MAX_STEPS]
