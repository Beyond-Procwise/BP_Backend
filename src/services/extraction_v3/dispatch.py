# src/services/extraction_v3/dispatch.py
"""V3 document dispatch.

All EXTRACTION_PIPELINE_<category> env vars are now set to 'v3'.
The legacy AgentNick fallback branch has been removed.

Routes a document to PipelineV3 and returns a dict the watcher consumes.
The adapter maps ExtractionResult → legacy dict shape so ProcessMonitorWatcher
doesn't need to change:
  - status: "ok" | "partial" | "error"
  - pk: the doc primary key (e.g. "INV-005-41")
  - header_persisted: True iff the header row was written
  - confidence: average of committed final_confidences (0.0 if none)
  - errors: count of residuals
"""
from __future__ import annotations
import os
import logging
from typing import Any
from src.services.extraction_v3.pipeline import PipelineV3
from src.services.extraction_v3.persistence import persist as persist_v3
from src.services.extraction_v3.schemas.result import ExtractionResult

log = logging.getLogger(__name__)

_CAT_ALIASES: dict[str, str] = {
    "po": "purchase_order",
    "purchase order": "purchase_order",
    "purchase_order": "purchase_order",
    "invoice": "invoice",
    "quote": "quote",
    "quotes": "quote",
    "contract": "contract",
    "contracts": "contract",
}


def _normalise_category(category: str) -> str:
    cat = (category or "").lower().replace(" ", "_")
    return _CAT_ALIASES.get(cat, cat)


def _adapt_v3_result(result: ExtractionResult) -> dict[str, Any]:
    """Convert ExtractionResult into the legacy dict shape the watcher expects."""
    has_required_residual = any(
        r.reason in (
            "required_field_missing_no_grounding",
            "invariant_critical_failed",
            "judge_incoherent",
            "bind_error_no_resolution",
        )
        for r in result.residuals
    )
    status = "partial" if has_required_residual else "ok"
    avg_conf = (
        sum(cf.final_confidence for cf in result.committed) / len(result.committed)
        if result.committed
        else 0.0
    )
    return {
        "status": status,
        "pk": result.doc_pk or "",
        "header_persisted": status == "ok" and result.doc_pk is not None,
        "confidence": avg_conf,
        "errors": len(result.residuals),
        "doc_type": result.doc_type,
        "judge_calls": result.judge_calls,
        "pipeline_version": result.pipeline_version,
        "_v3_result": result,
    }


def dispatch_document(
    agent_nick,
    file_path: str,
    category: str,
    user_id: str | None = None,
) -> dict[str, Any]:
    """Dispatch a document to PipelineV3.

    Args:
        agent_nick: kept for API compatibility (not used by v3).
        file_path: path/key for the document.
        category: invoice / purchase_order / quote / contract (or synonyms).
        user_id: optional user attribution.

    Returns:
        Result dict with keys: status, pk, header_persisted, confidence, errors.
    """
    doc_type = _normalise_category(category)
    try:
        log.info("Dispatching %s (%s) to extraction_v3", file_path, doc_type)
        result = PipelineV3().run(file_path, doc_type)
        adapted = _adapt_v3_result(result)
        if adapted["header_persisted"]:
            persist_v3(result)
        return adapted
    except Exception as exc:
        log.exception("extraction_v3 pipeline failed for %s", file_path)
        return {
            "status": "error",
            "pk": "",
            "error": str(exc),
            "header_persisted": False,
            "confidence": 0.0,
            "errors": 1,
        }
