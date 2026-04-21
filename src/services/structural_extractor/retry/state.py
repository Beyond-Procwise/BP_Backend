from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from src.services.structural_extractor.types import ExtractedValue


@dataclass
class AttemptOutput:
    attempt: int
    source: Literal[
        "structural", "pattern_cached", "nlu_ner", "nlu_table",
        "nlu_layout", "llm_fallback"
    ]
    extracted: dict[str, ExtractedValue]
    line_items: Optional[list[dict[str, ExtractedValue]]]
    validation_failures: list[str]
    residual_unresolved: list[str]
    latency_ms: int


@dataclass
class RetryState:
    doc: Any  # ParsedDocument — typed as Any to avoid import cycles in tests
    doc_type: str
    target_fields: set[str]
    attempts: list[AttemptOutput] = field(default_factory=list)
    accepted_header: dict[str, ExtractedValue] = field(default_factory=dict)
    accepted_line_items: Optional[list[dict[str, ExtractedValue]]] = None
    unresolved: set[str] = field(default_factory=set)

    def residual_fields(self) -> set[str]:
        return self.target_fields - self.accepted_header.keys()
