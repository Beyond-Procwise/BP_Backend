"""Frozen JSON contracts for the three LLM judge invocations.

These are I/O envelopes — strict Pydantic v2 models that validate the
prompt-input we send to the LLM and the JSON we receive back. Strict mode
+ closed Literals = parse failure if the LLM goes off-script.
"""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict


# ---- Tiebreaker ---------------------------------------------------------

class TiebreakerCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: str
    model: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str
    page: int = Field(ge=0)
    bbox: tuple[float, float, float, float]


class TiebreakerInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: str
    field_type: str
    candidates: list[TiebreakerCandidate]
    context_text: str  # surrounding ~200 chars from each candidate's page


class TiebreakerOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chosen_candidate_index: int | None = Field(default=None, ge=0)  # validate against len(candidates) at call site
    rationale: str


# ---- Grounded last-resort -----------------------------------------------

class GroundedConstraints(BaseModel):
    model_config = ConfigDict(extra="forbid")
    must_be_verbatim_substring_of_doc_full_text: bool = True
    max_length: int = 64


class GroundedInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: str
    field_type: str
    field_canonical_labels: list[str]
    doc_full_text: str
    constraints: GroundedConstraints = GroundedConstraints()


class GroundedOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: str | None = None
    evidence_text: str | None = None
    rationale: str

    def is_null(self) -> bool:
        return self.value is None and self.evidence_text is None


# ---- Schema coherence ---------------------------------------------------

class InvariantResultSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    passed: bool
    delta: float | None = None
    severity: str | None = None
    message: str | None = None


class CoherenceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    doc_type: str
    extracted_record: dict  # JSON-serializable record
    invariant_results: list[InvariantResultSummary]


class CoherenceIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: str
    issue: str


class CoherenceOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verdict: Literal["coherent", "incoherent"]
    issues: list[CoherenceIssue] = []
