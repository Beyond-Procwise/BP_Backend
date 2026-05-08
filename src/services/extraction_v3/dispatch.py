# src/services/extraction_v3/dispatch.py
"""Feature-flag dispatch: route a document to v3 or legacy AgentNick.

The per-category env var EXTRACTION_PIPELINE_{CATEGORY} (uppercase) decides:
  - "v3"        → Pipeline V3 (new schema-driven, judge-as-judge pipeline)
  - "agentnick" → legacy AgentNickOrchestrator (default)

Both pipelines return a dict the watcher consumes. V3 ExtractionResult is
adapted to the legacy dict shape so the watcher doesn't need to know the
difference. The adapter sets:
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


def _flag_for_category(category: str) -> str:
    """Return the pipeline flag for the given category, default 'agentnick'."""
    cat = (category or "").lower().replace(" ", "_")
    # Normalize known synonyms
    cat = {
        "po": "purchase_order", "purchase order": "purchase_order",
        "purchase_order": "purchase_order",
        "invoice": "invoice", "quote": "quote", "quotes": "quote",
        "contract": "contract", "contracts": "contract",
    }.get(cat, cat)
    env_var = f"EXTRACTION_PIPELINE_{cat.upper()}"
    return os.getenv(env_var, "agentnick").lower()


def _adapt_v3_result(result: ExtractionResult) -> dict[str, Any]:
    """Convert ExtractionResult into the legacy dict shape the watcher expects."""
    has_required_residual = any(
        r.reason in ("required_field_missing_no_grounding", "invariant_critical_failed",
                      "judge_incoherent", "bind_error_no_resolution")
        for r in result.residuals
    )
    status = "partial" if has_required_residual else "ok"
    avg_conf = (
        sum(cf.final_confidence for cf in result.committed) / len(result.committed)
        if result.committed else 0.0
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
        "_v3_result": result,  # keep the full result for downstream use if needed
    }


def dispatch_document(
    agent_nick,
    file_path: str,
    category: str,
    user_id: str | None = None,
) -> dict[str, Any]:
    """Dispatch a document to the configured pipeline.

    Args:
        agent_nick: AgentNickSettings (passed through to legacy path).
        file_path: path/key for the document.
        category: invoice / purchase_order / quote / contract (or synonyms).
        user_id: optional user attribution.

    Returns:
        Legacy-shaped result dict. See module docstring.
    """
    flag = _flag_for_category(category)

    if flag == "v3":
        try:
            log.info("Dispatching %s (%s) to extraction_v3", file_path, category)
            doc_type = (category or "").lower().replace(" ", "_")
            doc_type = {"po": "purchase_order", "purchase order": "purchase_order",
                        "quotes": "quote", "contracts": "contract"}.get(doc_type, doc_type)
            result = PipelineV3().run(file_path, doc_type)
            adapted = _adapt_v3_result(result)
            if adapted["header_persisted"]:
                persist_v3(result)
            return adapted
        except Exception as exc:
            log.exception("extraction_v3 pipeline failed for %s", file_path)
            return {"status": "error", "pk": "", "error": str(exc),
                    "header_persisted": False, "confidence": 0.0, "errors": 1}

    # Default: legacy
    from src.services.agent_nick_orchestrator import AgentNickOrchestrator
    nick = AgentNickOrchestrator(agent_nick)
    return nick.process_document(file_path, category, user_id=user_id)
