from typing import Literal
from pydantic import BaseModel
from .candidate import Candidate

ResidualReason = Literal[
    "unsupported_layout",
    "required_field_missing_no_grounding",
    "invariant_critical_failed",
    "judge_incoherent",
    "bind_error_no_resolution",
]

JudgeAction = Literal["tiebreaker", "grounded_last_resort", "schema_coherence"]

class CommittedField(BaseModel):
    field_path: str
    value: str
    page: int
    bbox: tuple[float, float, float, float]
    evidence_text: str
    model: str
    model_confidence: float
    judge_actions: list[JudgeAction] = []
    final_confidence: float

class ResidualField(BaseModel):
    field_path: str
    reason: ResidualReason
    candidates: list[Candidate] = []

class ExtractionResult(BaseModel):
    doc_type: str
    doc_pk: str | None
    committed: list[CommittedField]
    residuals: list[ResidualField]
    judge_calls: int
    pipeline_version: str
