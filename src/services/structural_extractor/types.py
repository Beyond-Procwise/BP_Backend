from dataclasses import dataclass
from typing import Any, Literal, Optional

from src.services.structural_extractor.parsing.model import AnchorRef


@dataclass
class ExtractedValue:
    value: Any
    provenance: Literal["extracted", "derived", "inferred", "lookup"]
    anchor_text: Optional[str] = None
    anchor_ref: Optional[AnchorRef] = None
    derivation_trace: Optional[dict] = None
    confidence: float = 1.0
    source: Literal[
        "structural", "pattern_cached", "nlu_ner", "nlu_table", "nlu_layout",
        "llm_fallback", "derivation_registry", "lookup_api", "lookup_db"
    ] = "structural"
    attempt: int = 1


@dataclass
class ExtractionResult:
    header: dict[str, ExtractedValue]
    line_items: list[dict[str, ExtractedValue]]
    parsed_text: str
    unresolved_fields: list[str]
    attempts: int
    pattern_id_used: Optional[int] = None
    layout_signature: str = ""
    process_monitor_id: Optional[int] = None
    doc_type: str = ""
