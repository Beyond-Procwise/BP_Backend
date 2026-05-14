"""V4 hybrid extraction pipeline.

Public API:
    run_data_extraction(file_path, doc_type=None) -> dict
    detect_document_type(file_path) -> "invoice" | "po" | "quote"
    to_extraction_result(engine_output, doc_type) -> ExtractionResult
    PIPELINE_VERSION
"""
from src.services.extraction_v3.extraction_v4.engine import (
    detect_document_type,
    run_data_extraction,
)
from src.services.extraction_v3.extraction_v4.adapter import (
    PIPELINE_VERSION,
    to_extraction_result,
)

__all__ = [
    "run_data_extraction",
    "detect_document_type",
    "to_extraction_result",
    "PIPELINE_VERSION",
]
