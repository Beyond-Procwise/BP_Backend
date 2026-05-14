"""V3 document dispatch.

Routes a document to one of two extraction engines:

  - 'hybrid_v4' (default): the v4 hybrid pipeline at extraction_v4.engine —
    PyMuPDF block layout + spaCy NER + NuExtract (Ollama) + regex + Excel
    cell scanner with confidence-weighted voting. Output is wrapped via
    extraction_v4.adapter into the same ExtractionResult schema the persist()
    pipeline expects, so supplier resolution, region/USD derivation, and
    discrepancy detection all run unchanged.

  - 'qwen_vlm': the previous Qwen2.5-VL single-call pipeline (PipelineV3).
    Kept available for rollback while the v4 hybrid is being validated on
    real procwise traffic.

Engine selection: env var EXTRACTION_V3_ENGINE in {'hybrid_v4', 'qwen_vlm'}.
Default is 'hybrid_v4'.

The watcher contract is unchanged:
  status: "ok" | "partial" | "error"
  pk: doc primary key
  header_persisted: bool
  confidence: float
  errors: count of residuals
"""
from __future__ import annotations

import logging
import os
from typing import Any

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


def _adapt_v3_result(result: ExtractionResult, raw_id: int | None = None) -> dict[str, Any]:
    """Convert ExtractionResult into the legacy dict shape the watcher expects.

    Status semantics:
      ok        -- extraction promoted to _stg, header row present
      no_pk     -- _raw row written (raw_id set) but doc_pk could not be
                   extracted; manual review required. Watcher should treat
                   this as a soft success (data is preserved in _raw).
      partial   -- doc_pk present but required residuals blocked _stg promotion.
                   _raw row exists. Watcher may flag.
      error     -- nothing could be persisted; treat as failure.
    """
    pk_missing = result.doc_pk is None
    has_required_residual = any(
        r.reason in (
            "required_field_missing_no_grounding",
            "invariant_critical_failed",
            "judge_incoherent",
        )
        for r in result.residuals
    )
    if pk_missing:
        status = "no_pk"
    elif has_required_residual:
        status = "partial"
    else:
        status = "ok"
    avg_conf = (
        sum(cf.final_confidence for cf in result.committed) / len(result.committed)
        if result.committed
        else 0.0
    )
    return {
        "status": status,
        "pk": result.doc_pk or "",
        "header_persisted": status == "ok" and result.doc_pk is not None,
        "raw_persisted": raw_id is not None,
        "raw_id": raw_id,
        "confidence": avg_conf,
        "errors": len(result.residuals),
        "doc_type": result.doc_type,
        "judge_calls": result.judge_calls,
        "pipeline_version": result.pipeline_version,
        "_v3_result": result,
    }


def _select_engine() -> str:
    """Return the active extraction engine name.

    Defaults to 'hybrid_v4'. Set EXTRACTION_V3_ENGINE=qwen_vlm for rollback.
    """
    raw = (os.environ.get("EXTRACTION_V3_ENGINE") or "hybrid_v4").lower().strip()
    if raw in ("hybrid_v4", "v4", "hybrid", "new"):
        return "hybrid_v4"
    if raw in ("qwen_vlm", "v3", "qwen", "vlm", "old"):
        return "qwen_vlm"
    log.warning("Unknown EXTRACTION_V3_ENGINE=%r; defaulting to hybrid_v4", raw)
    return "hybrid_v4"


_TYPE_TO_CAT = {"po": "purchase_order", "purchase_order": "purchase_order",
                "invoice": "invoice", "quote": "quote", "contract": "contract"}


def _run_hybrid_v4(file_path: str, doc_type: str) -> ExtractionResult:
    """Run the v4 hybrid engine and wrap output as ExtractionResult.

    Adds two safety nets on top of the regex/layout extraction:

    1. LLM fallback (BeyondProcwise/AgentNick:extract via Ollama) fills
       missing required fields. Substring-grounded — values that don't
       appear in the source text are rejected.

    2. Doc-type re-route: if no doc_pk after LLM fallback, sniff content
       and retry with the detected type (handles miscategorized files).
    """
    from src.services.extraction_v3.extraction_v4 import (
        run_data_extraction,
        detect_document_type,
        to_extraction_result,
    )
    from src.services.extraction_v3.extraction_v4.engine import PDFParser, DOCXParser
    from src.services.extraction_v3.extraction_v4.llm_extractor import (
        llm_fill_missing_required,
    )
    import os as _os

    # The v4 engine's "po" alias matches purchase_order.
    engine_doc_type = "po" if doc_type == "purchase_order" else doc_type
    raw_full = run_data_extraction(file_path, doc_type=engine_doc_type)

    # --- LLM safety net for missing required fields -----------------------
    # raw_extraction is the engine's raw dict (BEFORE the mapper). It has
    # field names like 'invoice_number', 'supplier_name' etc. that the LLM
    # extractor knows. We fill any missing required fields here, THEN re-map.
    raw_engine = raw_full.get("_raw_extraction") or {}
    if raw_engine:
        # Pull the freshest source text — flat PDF/DOCX, not layout-rebuilt
        try:
            ext = _os.path.splitext(file_path)[1].lower()
            if ext == ".pdf":
                source_text = PDFParser.extract_text(file_path)
            elif ext == ".docx":
                source_text = DOCXParser.extract_text(file_path)
            else:
                source_text = raw_engine.get("_raw_text", "") or ""
        except Exception:
            source_text = raw_engine.get("_raw_text", "") or ""
        try:
            llm_fill_missing_required(raw_engine, engine_doc_type, source_text)
        except Exception as exc:
            log.warning("LLM fallback raised: %s -- continuing without it", exc)
        # If LLM filled fields, re-map so the new values land in the mapped header.
        from src.services.extraction_v3.extraction_v4.engine import (
            map_invoice, map_line_items_invoice,
            map_purchase_order, map_po_line_items,
            map_quote, map_quote_line_items,
        )
        if engine_doc_type == "invoice":
            raw_full["invoice_data"] = map_invoice(raw_engine)
            raw_full["line_items"] = map_line_items_invoice(raw_engine)
        elif engine_doc_type == "po":
            raw_full["po_data"] = map_purchase_order(raw_engine)
            raw_full["line_items"] = map_po_line_items(raw_engine)
        elif engine_doc_type == "quote":
            raw_full["quote_data"] = map_quote(raw_engine)
            raw_full["line_items"] = map_quote_line_items(raw_engine)

    result = to_extraction_result(raw_full, doc_type, source_file=file_path)
    if doc_type == "quote":
        result = _resolve_quote_supplier(result)

    # Auto-reroute on type mismatch: if no PK, sniff content; if it disagrees
    # with the requested category and the new type is different, retry.
    if result.doc_pk is None:
        try:
            detected = detect_document_type(file_path)
            detected_cat = _TYPE_TO_CAT.get(detected, detected)
            if detected_cat and detected_cat != doc_type and detected_cat != "contract":
                log.warning(
                    "doc-type mismatch: requested category=%r but content "
                    "looks like %r -- re-routing %s",
                    doc_type, detected_cat, file_path,
                )
                detected_engine_type = "po" if detected_cat == "purchase_order" else detected_cat
                raw2 = run_data_extraction(file_path, doc_type=detected_engine_type)
                result2 = to_extraction_result(raw2, detected_cat, source_file=file_path)
                if detected_cat == "quote":
                    result2 = _resolve_quote_supplier(result2)
                if result2.doc_pk is not None:
                    log.info(
                        "doc-type re-route SUCCESS: pk=%r recovered as %s",
                        result2.doc_pk, detected_cat,
                    )
                    return result2
                log.warning(
                    "doc-type re-route attempted but second pass also no_pk -- "
                    "keeping original category=%r result", doc_type,
                )
        except Exception as exc:
            log.warning("doc-type sniff/reroute failed: %s -- using original result", exc)

    return result


def _resolve_quote_supplier(result: ExtractionResult) -> ExtractionResult:
    """Resolve quote 'supplier_id' value (currently a name) to a SUP- id."""
    new_committed = []
    needs_resolve = False
    for cf in result.committed:
        if cf.field_path == "supplier_id" and cf.value and not cf.value.startswith("SUP-"):
            needs_resolve = True
            break
    if not needs_resolve:
        return result
    try:
        from src.services.db import get_conn
        from src.services.extraction_v3.supplier_resolver import (
            resolve_or_create_supplier,
        )
        with get_conn() as conn:
            for cf in result.committed:
                if cf.field_path == "supplier_id" and cf.value and not cf.value.startswith("SUP-"):
                    resolved = resolve_or_create_supplier(cf.value, conn)
                    if resolved:
                        log.info(
                            "quote pre-resolve: supplier_id '%s' -> '%s'",
                            cf.value, resolved,
                        )
                        new_committed.append(cf.model_copy(update={"value": resolved}))
                        continue
                new_committed.append(cf)
            conn.commit()  # commit any auto-create from resolver
    except Exception as exc:
        log.warning("quote supplier pre-resolve failed: %s -- using raw value", exc)
        return result
    return result.model_copy(update={"committed": new_committed})


def _run_qwen_vlm(file_path: str, doc_type: str) -> ExtractionResult:
    """Run the legacy Qwen2.5-VL single-call pipeline."""
    from src.services.extraction_v3.pipeline import PipelineV3
    return PipelineV3().run(file_path, doc_type)


def dispatch_document(
    agent_nick,
    file_path: str,
    category: str,
    user_id: str | None = None,
) -> dict[str, Any]:
    """Dispatch a document to the active extraction engine.

    Args:
        agent_nick: kept for API compatibility (not used).
        file_path: path/key for the document.
        category: invoice / purchase_order / quote / contract (or synonyms).
        user_id: optional user attribution.

    Returns:
        Result dict with keys: status, pk, header_persisted, confidence, errors.
    """
    doc_type = _normalise_category(category)
    engine = _select_engine()

    try:
        log.info(
            "Dispatching %s (%s) to extraction engine=%s",
            file_path, doc_type, engine,
        )
        if engine == "hybrid_v4":
            # Contract docs aren't yet supported by v4; fall back to qwen_vlm.
            if doc_type == "contract":
                log.info("Contract doc -- routing to qwen_vlm (v4 doesn't support contracts yet)")
                result = _run_qwen_vlm(file_path, doc_type)
            else:
                result = _run_hybrid_v4(file_path, doc_type)
        else:
            result = _run_qwen_vlm(file_path, doc_type)

        # Always call persist_v3 — it now handles missing PK by writing _raw
        # with promotion_status='no_pk' so the audit trail is always preserved.
        raw_id = persist_v3(result, source_file=file_path)
        adapted = _adapt_v3_result(result, raw_id=raw_id)
        return adapted
    except Exception as exc:
        log.exception(
            "extraction pipeline failed (engine=%s) for %s",
            engine, file_path,
        )
        return {
            "status": "error",
            "pk": "",
            "error": str(exc),
            "header_persisted": False,
            "confidence": 0.0,
            "errors": 1,
        }
