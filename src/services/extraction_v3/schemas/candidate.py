from typing import Literal
from pydantic import BaseModel, Field
from .parsed_document import BBox

ExtractorName = Literal[
    "layoutlmv3", "layoutlmv3_finetuned", "table_transformer", "sbert_anchor",
    "spacy_ner", "qa_roberta", "vendor_template"
]

class Candidate(BaseModel):
    field: str
    value: str
    page: int
    bbox: BBox
    evidence_text: str
    model: ExtractorName
    confidence: float = Field(ge=0.0, le=1.0)
