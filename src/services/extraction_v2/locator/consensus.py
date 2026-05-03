"""Consensus voting across multiple locators for a single field.

The voting policy is intentionally conservative: unless a value has
strong support (≥2 locators OR a single locator with conf ≥ 0.95),
the consensus runner ABSTAINS. Abstention is a first-class output —
it routes the field to the residual / review queue rather than
silently committing a guess.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from src.services.extraction_v2.locator.base import Locator, LocatorOutput

logger = logging.getLogger(__name__)


__all__ = ["ConsensusResult", "run_locators"]


@dataclass(frozen=True)
class ConsensusResult:
    """Outcome of running all locators for one field."""
    field_name: str
    value: Any                                 # the agreed value (or None if abstained)
    confidence: float                          # 0.0–1.0; computed from voting outcome
    abstained: bool                            # True ⇒ residual / no commit
    candidates: tuple[LocatorOutput, ...]      # all proposals (for audit / review queue)
    winning_strategies: tuple[str, ...]        # locators whose values agreed
    why: str                                   # human-readable summary


# Confidence thresholds used by the voting policy
_SINGLE_STRATEGY_FLOOR = 0.90   # one high-trust locator suffices if ≥ this
_GROUP_AGREEMENT_FLOOR = 2      # need ≥ this many agreeing for a multi-locator commit


def run_locators(
    field_name: str,
    locators: list[Locator],
    doc: Any,
) -> ConsensusResult:
    """Run all locators for `field_name` and vote on the result.

    Args:
        field_name: the schema field being resolved (e.g., "invoice_id").
        locators:   strategies to run; each independent.
        doc:        the parsed document — passed to every locator.

    Returns:
        ConsensusResult capturing the vote outcome and all proposals.
    """
    if not locators:
        return _abstain(
            field_name, (), "no_locators_registered",
            "no strategies registered for this field",
        )

    candidates: list[LocatorOutput] = []
    for loc in locators:
        try:
            out = loc.locate(doc)
        except Exception:
            logger.exception(
                "locator %r raised on field %r; treating as abstain",
                getattr(loc, "name", repr(loc)), field_name,
            )
            out = None
        if out is not None:
            candidates.append(out)

    if not candidates:
        return _abstain(
            field_name, (), "all_locators_abstained",
            f"all {len(locators)} strategies returned None",
        )

    # Group candidates by canonical value-equality. Two candidates with
    # the same value are votes for the same answer.
    groups: dict[Any, list[LocatorOutput]] = {}
    for c in candidates:
        key = _canonical_key(c.value)
        groups.setdefault(key, []).append(c)

    # Pick the largest group; tiebreak by mean confidence.
    largest = max(
        groups.values(),
        key=lambda g: (len(g), sum(c.confidence for c in g) / max(1, len(g))),
    )

    largest_size = len(largest)
    total_locators = len(locators)
    max_conf = max(c.confidence for c in largest)

    # Apply voting policy
    if largest_size >= _GROUP_AGREEMENT_FLOOR:
        # Multi-locator agreement → commit.
        # Use max_conf rather than mean: when N independent strategies
        # converge on the same value, the highest-trust strategy is
        # corroborated, not diluted by lower-trust ones. Each additional
        # agreeing voter beyond the first adds a small bonus.
        confidence = max_conf + (largest_size - 1) * 0.05
        return ConsensusResult(
            field_name=field_name,
            value=largest[0].value,
            confidence=min(confidence, 1.0),
            abstained=False,
            candidates=tuple(candidates),
            winning_strategies=tuple(c.evidence.raw_text or "" for c in largest),
            why=f"{largest_size}/{total_locators} locators agreed",
        )

    if largest_size == 1 and largest[0].confidence >= _SINGLE_STRATEGY_FLOOR:
        # Single high-confidence strategy → commit (cap confidence at 0.9 since
        # there's no second-source agreement)
        return ConsensusResult(
            field_name=field_name,
            value=largest[0].value,
            confidence=min(largest[0].confidence, 0.9),
            abstained=False,
            candidates=tuple(candidates),
            winning_strategies=(largest[0].why,),
            why=f"single high-confidence locator ({largest[0].why})",
        )

    # No consensus, no high-confidence singleton → abstain
    return _abstain(
        field_name, tuple(candidates), "no_consensus",
        f"{len(groups)} distinct candidates from {len(candidates)} locators; "
        f"largest group size={largest_size} below threshold",
    )


_NAME_SUFFIXES = {
    "ltd", "limited", "llc", "inc", "incorporated", "corp", "corporation",
    "co", "company", "plc", "gmbh", "ag", "sa", "srl", "bv", "nv", "kg",
}


def _canonical_key(value: Any) -> Any:
    """Canonicalize a value for vote-equality comparison.

    String values: lowercase + strip + trim recognised company suffixes
    (Ltd / LLC / Inc / Corp …) from the right so 'Acme' and 'Acme Ltd'
    vote together. Otherwise the full normalised string is used — this
    is conservative; values that are merely substrings of one another
    do NOT match, by design.
    """
    if not isinstance(value, str):
        return value
    s = value.strip().lower()
    if not s:
        return s
    words = [w.strip(",.;:") for w in s.split() if w.strip(",.;:")]
    while words and words[-1] in _NAME_SUFFIXES:
        words.pop()
    return " ".join(words) if words else s


def _abstain(
    field_name: str,
    candidates: tuple[LocatorOutput, ...],
    code: str,
    why: str,
) -> ConsensusResult:
    return ConsensusResult(
        field_name=field_name,
        value=None,
        confidence=0.0,
        abstained=True,
        candidates=candidates,
        winning_strategies=(),
        why=f"{code}: {why}",
    )
