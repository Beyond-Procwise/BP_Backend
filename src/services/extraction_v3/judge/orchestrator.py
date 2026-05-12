"""Judge orchestrator. Decides which judge invocations to fire per field
based on the candidate set, then runs the schema-coherence judge on the
assembled record. Enforces a per-document cost ceiling.

Circuit breaker: after `_FAIL_FAST_THRESHOLD` consecutive judge call
failures within a single document, all remaining judge calls are skipped
for that document. Prevents Ollama outages from causing 15-minute
per-document timeouts during contention.
"""
from __future__ import annotations
import logging
import os
from dataclasses import dataclass, field as dataclass_field
from typing import Optional
from src.services.extraction_v3.schemas.candidate import Candidate
from src.services.extraction_v3.schemas.result import JudgeAction, ResidualReason
from src.services.extraction_v3.yaml_schema.loader import DocSchema, FieldSpec
from src.services.extraction_v3.judge.tiebreaker import call_tiebreaker_judge
from src.services.extraction_v3.judge.grounded_last_resort import call_grounded_last_resort
from src.services.extraction_v3.judge.schema_coherence import call_coherence_judge
from src.services.extraction_v3.judge.contracts import InvariantResultSummary, CoherenceOutput

log = logging.getLogger(__name__)


@dataclass
class FieldOutcome:
    field_name: str
    chosen: Candidate | None  # None = residual
    judge_actions: list[JudgeAction] = dataclass_field(default_factory=list)
    residual_reason: ResidualReason | None = None
    candidates_seen: list[Candidate] = dataclass_field(default_factory=list)


@dataclass
class OrchestratorResult:
    field_outcomes: dict[str, FieldOutcome]
    coherence: CoherenceOutput | None
    judge_calls: int


def _candidates_agree(cands: list[Candidate]) -> bool:
    """All candidates share the same (normalized) value."""
    if len(cands) < 2:
        return True
    norms = {(c.value or "").strip().lower() for c in cands}
    return len(norms) == 1


def _highest_confidence(cands: list[Candidate]) -> Candidate:
    return max(cands, key=lambda c: c.confidence)


def run_judge_orchestrator(
    candidates_by_field: dict[str, list[Candidate]],
    schema: DocSchema,
    parsed_full_text: str,
    invariant_results: list[InvariantResultSummary] | None = None,
    file_path: str | None = None,
) -> OrchestratorResult:
    """Decide per-field outcomes and run the coherence pass.

    Returns OrchestratorResult with per-field commit/residual decisions
    and the coherence verdict.
    """
    fields_by_name: dict[str, FieldSpec] = {f.name: f for f in schema.fields}
    outcomes: dict[str, FieldOutcome] = {}
    judge_calls = 0

    # Circuit breaker: skip all remaining judge calls for this doc if Ollama
    # is unavailable (returns None too many times in a row). Set
    # EXTRACTION_V3_DISABLE_JUDGE=1 to skip the judge layer entirely.
    judge_disabled = os.getenv("EXTRACTION_V3_DISABLE_JUDGE") == "1"
    consecutive_failures = 0
    _FAIL_FAST_THRESHOLD = 3

    def _judge_open() -> bool:
        return (not judge_disabled) and consecutive_failures < _FAIL_FAST_THRESHOLD

    # Pass 1: per-field commit/residual + targeted judge calls
    for field_name, field_spec in fields_by_name.items():
        cands = candidates_by_field.get(field_name, [])
        out = FieldOutcome(field_name=field_name, chosen=None, candidates_seen=cands)

        if len(cands) == 0:
            # No candidates: if required, try grounded last-resort
            if field_spec.required and field_spec.judge.grounded_last_resort and _judge_open():
                grounded = call_grounded_last_resort(field_spec, parsed_full_text, file_path=file_path)
                judge_calls += 1
                if grounded is not None:
                    out.chosen = grounded
                    out.judge_actions.append("grounded_last_resort")
                    consecutive_failures = 0
                else:
                    out.residual_reason = "required_field_missing_no_grounding"
                    consecutive_failures += 1
            elif field_spec.required:
                out.residual_reason = "required_field_missing_no_grounding"
            # optional + 0 candidates → just leave chosen=None, no residual
        elif len(cands) == 1:
            out.chosen = cands[0]
        elif _candidates_agree(cands):
            out.chosen = _highest_confidence(cands)
        else:
            # Disagreement → tiebreaker
            if field_spec.judge.tiebreaker and _judge_open():
                chosen = call_tiebreaker_judge(
                    cands, field_name, field_spec.type, parsed_full_text,
                )
                judge_calls += 1
                if chosen is not None:
                    out.chosen = chosen
                    out.judge_actions.append("tiebreaker")
                    consecutive_failures = 0
                else:
                    # Tiebreaker returned None — fall back to highest confidence
                    out.chosen = _highest_confidence(cands)
                    consecutive_failures += 1
            else:
                out.chosen = _highest_confidence(cands)

        outcomes[field_name] = out

    # Pass 2: schema-coherence on the assembled record
    record_dict = {fn: out.chosen.value for fn, out in outcomes.items() if out.chosen is not None}
    coherence: CoherenceOutput | None = None
    if record_dict and _judge_open():
        coherence = call_coherence_judge(
            schema.doc_type,
            record_dict,
            invariant_results=invariant_results,
            file_path=file_path,
            doc_full_text=parsed_full_text,
        )
        if coherence is not None:
            judge_calls += 1
            # Mark each field with judge_action="schema_coherence" so the
            # provenance writer knows the coherence pass ran on the field
            for fn, out in outcomes.items():
                if out.chosen is not None:
                    out.judge_actions.append("schema_coherence")

    return OrchestratorResult(
        field_outcomes=outcomes,
        coherence=coherence,
        judge_calls=judge_calls,
    )
