from __future__ import annotations

from typing import Optional

from src.services.structural_extractor.parsing import parse
from src.services.structural_extractor.retry.driver import run_retry_loop
from src.services.structural_extractor.types import ExtractionResult


def extract(
    file_bytes: bytes,
    filename: str,
    doc_type: str,
    max_attempts: int = 10,
    db_conn=None,
    process_monitor_id: Optional[int] = None,
) -> ExtractionResult:
    """Public entry point. Parses file -> runs retry loop -> optionally writes provenance + parks review."""
    doc = parse(file_bytes, filename)
    result = run_retry_loop(doc, doc_type, max_attempts=max_attempts)
    result.process_monitor_id = process_monitor_id
    # Wire provenance + review-queue if DB connection provided
    if db_conn is not None:
        if result.unresolved_fields:
            from src.services.structural_extractor.review_queue import park_in_review_queue
            park_in_review_queue(db_conn, result, filename)
        else:
            # Successful: write provenance if there's a parent PK
            pk = None
            if doc_type == "Invoice":
                pk = result.header.get("invoice_id")
            elif doc_type == "Purchase_Order":
                pk = result.header.get("po_id")
            elif doc_type == "Quote":
                pk = result.header.get("quote_id")
            if pk and pk.value:
                from src.services.structural_extractor.provenance import write_provenance
                write_provenance(db_conn, f"bp_{doc_type.lower()}", pk.value, result.header)
    return result


__all__ = ["extract", "ExtractionResult"]
