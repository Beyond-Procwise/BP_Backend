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
_SINGLE_STRATEGY_FLOOR = 0.95   # one locator suffices only if ≥ this
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
    mean_conf = sum(c.confidence for c in largest) / max(1, largest_size)

    # Apply voting policy
    if largest_size >= _GROUP_AGREEMENT_FLOOR:
        # Multi-locator agreement → commit
        # Confidence scaled by agreement rate
        agreement_rate = largest_size / total_locators
        confidence = mean_conf * (0.7 + 0.3 * agreement_rate)
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


def _canonical_key(value: Any) -> Any:
    """Canonicalize a value for vote-equality comparison."""
    if isinstance(value, str):
        return value.strip().lower()
    return value


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
