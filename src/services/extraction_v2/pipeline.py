"""End-to-end extraction pipeline V2.

Glues:
    - parsing (existing structural_extractor parsers)
    - locator framework (multi-strategy consensus)
    - verification network (math/date rules)
    - typed values (Money / IsoDate / etc.)

Output: an :class:`ExtractionResultV2` carrying:
    - committed:  per-field typed value + confidence + provenance
    - residuals:  per-field rejection records (no commit; review queue)
    - rule_trace: which verification rules ran and their outcomes
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from src.services.extraction_v2.locator.base import AnchorRef, LocatorOutput
from src.services.extraction_v2.locator.consensus import ConsensusResult, run_locators
from src.services.extraction_v2.locator.registry import build_locators
from src.services.extraction_v2.verification.network import (
    Rule, RuleResult, run_verification, VerificationOutcome,
)
from src.services.extraction_v2.verification.rules.dates import ALL_DATE_RULES
from src.services.extraction_v2.verification.rules.math import ALL_MATH_RULES
from src.services.structural_extractor.parsing.model import ParsedDocument

logger = logging.getLogger(__name__)


__all__ = ["CommittedField", "ResidualField", "ExtractionResultV2", "ExtractionPipelineV2"]


# Threshold above which a field is auto-committed.
# Set just below the consensus floor so 2-locator agreement at moderate
# confidence (e.g. 0.55 + 0.75 = ~0.55 consensus) routes to residual,
# while structural-extractor-alone (0.92) commits cleanly.
_COMMIT_THRESHOLD = 0.75
# How much demotion costs in confidence (per failed rule).
# 0.10 was chosen so a single failed verification rule drops a high-conf
# (0.92) commit to 0.82 — still above threshold — while two failures
# (0.72) correctly route to residual.
_DEMOTION_PENALTY = 0.10


@dataclass(frozen=True)
class CommittedField:
    """A field that the pipeline asserts a value for."""
    field: str
    value: Any
    confidence: float
    evidence: AnchorRef
    why: str
    locator_count: int


@dataclass(frozen=True)
class ResidualField:
    """A field the pipeline could not commit to. Routes to review queue."""
    field: str
    candidates: tuple[LocatorOutput, ...]
    why: str


@dataclass
class ExtractionResultV2:
    doc_type: str
    committed: dict[str, CommittedField] = field(default_factory=dict)
    residuals: list[ResidualField] = field(default_factory=list)
    rule_trace: list[RuleResult] = field(default_factory=list)

    def as_header_dict(self) -> dict[str, Any]:
        """Flatten the committed values into a dict suitable for persistence."""
        return {f: cf.value for f, cf in self.committed.items()}


class ExtractionPipelineV2:
    """Orchestrates the V2 extraction pipeline."""

    def __init__(self, commit_threshold: float = _COMMIT_THRESHOLD):
        self.commit_threshold = commit_threshold

    def extract(self, doc: ParsedDocument, doc_type: str) -> ExtractionResultV2:
        """Run multi-strategy consensus + verification on `doc`.

        Returns an ExtractionResultV2 with the committed fields, residuals,
        and the rule-trace for audit.
        """
        result = ExtractionResultV2(doc_type=doc_type)

        locators_per_field = build_locators(doc_type)
        if not locators_per_field:
            logger.warning("no locators registered for doc_type=%r", doc_type)
            return result

        # Step 1: run consensus voting per field
        consensus_per_field: dict[str, ConsensusResult] = {}
        for field_name, locators in locators_per_field.items():
            consensus_per_field[field_name] = run_locators(field_name, locators, doc)

        # Step 2: build the candidate values dict for verification
        committed_candidates: dict[str, Any] = {}
        for fname, cr in consensus_per_field.items():
            if not cr.abstained:
                committed_candidates[fname] = cr.value

        # Step 3: run verification network — applies demote / abstain
        rules = self._rules_for(doc_type)
        verification = run_verification(committed_candidates, rules)
        result.rule_trace = verification.rule_results

        # Step 4: assemble final committed + residuals based on rule outcomes
        for fname, cr in consensus_per_field.items():
            if cr.abstained:
                result.residuals.append(ResidualField(
                    field=fname, candidates=cr.candidates, why=cr.why,
                ))
                continue

            # If rules said abstain — eject from commit
            if fname in verification.abstained_fields:
                result.residuals.append(ResidualField(
                    field=fname, candidates=cr.candidates,
                    why=f"verification rule rejected: {cr.why}",
                ))
                continue

            # Apply demotion penalty per failed rule touching this field
            confidence = cr.confidence
            if fname in verification.demoted_fields:
                # Count rules that touched this field and failed
                penalty_count = sum(
                    1 for r in verification.rule_results
                    if not r.passed and fname in r.fields and r.on_fail == "demote"
                )
                confidence -= _DEMOTION_PENALTY * penalty_count

            if confidence < self.commit_threshold:
                # Demoted below commit threshold → residual
                result.residuals.append(ResidualField(
                    field=fname, candidates=cr.candidates,
                    why=f"confidence {confidence:.2f} below threshold "
                        f"after {len(verification.rule_results)} rules",
                ))
                continue

            # Pick the winning candidate's evidence/why
            winner = self._winning_candidate(cr)
            result.committed[fname] = CommittedField(
                field=fname,
                value=cr.value,
                confidence=min(confidence, 1.0),
                evidence=winner.evidence,
                why=cr.why,
                locator_count=len(cr.candidates),
            )

        return result

    @staticmethod
    def _winning_candidate(cr: ConsensusResult) -> LocatorOutput:
        """Pick the highest-confidence candidate matching the agreed value."""
        if not cr.candidates:
            # Should never happen for non-abstained results, but defensive
            return LocatorOutput(
                value=cr.value, confidence=cr.confidence,
                evidence=AnchorRef(), why="no candidates",
            )
        # Find ones whose value matches the consensus
        matching = [c for c in cr.candidates if c.value == cr.value]
        if not matching:
            matching = list(cr.candidates)
        return max(matching, key=lambda c: c.confidence)

    @staticmethod
    def _rules_for(doc_type: str) -> list[Rule]:
        """Return the verification rules applicable to this doc type."""
        return ALL_MATH_RULES + ALL_DATE_RULES
