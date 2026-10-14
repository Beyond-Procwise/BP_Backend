"""What we can walk away to, and how much weight that answer bears.

Salvaged from ``NegotiationStrategyEngine._build_batna`` and
``select_strategy`` before that module was deleted. Those two fields ---
``alternative_quotes`` and ``supplier_history_count`` --- were the only BATNA
reasoning in the codebase, and they were reached by nothing: the engine's sole
call site sat inside ``ReasoningEngine._rule_based_plan``, whose caller
``create_plan`` has none.

Two corrections were made in the move, both deliberate:

1. **Missing is no longer zero.** The engine took its inputs as
   ``int(task.get("alternative_quotes", 0) or 0)``, so "we never looked for an
   alternative supplier" and "we looked and there is none" arrived as the same
   integer. Here the first is UNASSESSED.

2. **No BATNA no longer licenses the hardest anchor.** Zero alternatives and
   zero history selected ``STRATEGY_ANCHORING``, whose ``target_discount`` of
   0.15 was the largest of the six strategies. Having nowhere else to go is the
   weakest position at a negotiating table, not the strongest; it now scores
   0.0.

The prose the engine generated is kept in substance and not in wording: it
stated counts unconditionally, so an unassessed BATNA read as "No established
relationship or alternative quotes exist."
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

from src.services.formulas.unassessed import UNASSESSED, Confidence

#: Alternatives at or above this count read as a strong BATNA. From the
#: engine's ``select_strategy`` rung 2 and ``_build_batna`` rung 1.
STRONG_BATNA_ALTERNATIVES = 2

#: Orders placed with this supplier at or above which the engine read the
#: relationship as established (``select_strategy`` rung 1).
ESTABLISHED_RELATIONSHIP_ORDERS = 5

_STRENGTH_SCORE = {"strong": 1.0, "moderate": 0.5, "weak": 0.2, "none": 0.0}


@dataclass(frozen=True)
class BatnaAssessment:
    """Our best alternative, and how much the answer can be leaned on.

    ``strength`` and ``score`` are both UNASSESSED together or neither: a
    strength with no score would invite ``score or 0.0``.
    """

    strength: Union[str, Any]
    score: Union[float, Any]
    confidence: Confidence
    narrative: str
    reasons: List[str] = field(default_factory=list)
    findings: List[str] = field(default_factory=list)
    alternative_quotes: Optional[int] = None
    supplier_history_count: Optional[int] = None

    @property
    def is_assessed(self) -> bool:
        return self.strength is not UNASSESSED


def _count(value: Any) -> Optional[int]:
    """A non-negative integer, or None. Never a coerced zero.

    ``bool`` is rejected before ``int`` because ``isinstance(True, int)`` is
    True and ``True`` is not a count of anything.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def assess_batna(
    *,
    alternative_quotes: Any = None,
    supplier_history_count: Any = None,
    supplier_name: Optional[str] = None,
    source: Confidence = Confidence.UNVERIFIED,
) -> BatnaAssessment:
    """Assess the buyer's walk-away position.

    ``alternative_quotes`` is required: a BATNA is an alternative, and no
    number of past orders with *this* supplier establishes one. Absent it, the
    result is UNASSESSED rather than a weak BATNA, because a buyer told "your
    position is weak" behaves differently from one told "we have not checked".

    ``source`` is the provenance of the counts and defaults to UNVERIFIED --
    silence about where a number came from is not a claim about where it came
    from. Pass OBSERVED for a count from a system of record, ASSERTED for one
    read out of correspondence by a model.
    """
    alternatives = _count(alternative_quotes)
    history = _count(supplier_history_count)

    if alternatives is None:
        findings = [
            "batna.alternative_quotes missing or invalid: cannot establish a "
            "walk-away position. Negotiating leverage is UNASSESSED, not weak."
        ]
        if history is None:
            findings.append(
                "batna.supplier_history_count missing or invalid: the "
                "relationship signal is unavailable too."
            )
        return BatnaAssessment(
            strength=UNASSESSED,
            score=UNASSESSED,
            confidence=Confidence.UNVERIFIED,
            narrative=(
                "BATNA not assessed: the number of alternative suppliers for "
                "this requirement has not been established."
            ),
            reasons=[],
            findings=findings,
            alternative_quotes=None,
            supplier_history_count=history,
        )

    who = supplier_name or "this supplier"
    reasons: List[str] = []

    if alternatives >= STRONG_BATNA_ALTERNATIVES:
        strength = "strong"
        narrative = (
            f"{alternatives} qualified alternative suppliers have been "
            "identified for this requirement. Failure to agree can be met by "
            "awarding elsewhere."
        )
        reasons.append(
            f"{alternatives} alternative quote(s) at or above the "
            f"{STRONG_BATNA_ALTERNATIVES}-quote strong-BATNA bar"
        )
    elif alternatives == 1:
        strength = "moderate"
        narrative = (
            "One alternative supplier quote is available. Failure to agree "
            "may be met by switching, on a single option."
        )
        reasons.append("1 alternative quote — a switch is possible but unproven")
    elif history is not None and history > 0:
        strength = "weak"
        narrative = (
            f"No alternative quote has been sourced. Continued business with "
            f"{who} rests on the existing relationship rather than on "
            "competitive tension."
        )
        reasons.append(f"no alternative quotes; {history} prior order(s) with {who}")
        if history >= ESTABLISHED_RELATIONSHIP_ORDERS:
            reasons.append(
                f"relationship established at or above the "
                f"{ESTABLISHED_RELATIONSHIP_ORDERS}-order bar"
            )
    else:
        strength = "none"
        narrative = (
            "No alternative quote and no prior trading history. There is no "
            "established walk-away position; sourcing an alternative is the "
            "first move, not a harder price ask."
        )
        reasons.append("no alternative quotes and no prior orders — no walk-away exists")

    return BatnaAssessment(
        strength=strength,
        score=_STRENGTH_SCORE[strength],
        confidence=source,
        narrative=narrative,
        reasons=reasons,
        findings=[],
        alternative_quotes=alternatives,
        supplier_history_count=history,
    )
