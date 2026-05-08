"""AgentNick Orchestrator — Primary agentic intelligence for ProcWise.

AgentNick is the primary agent that:
1. Receives document upload events from ProcessMonitorWatcher
2. Dispatches DataExtractionAgent as a sub-agent
3. Validates and enriches the extraction result using LLM intelligence
4. Resolves suppliers against bp_supplier
5. Persists to bp_ tables with audit columns
6. Triggers downstream agents (discrepancy detection, ranking, etc.)
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
AGENT_NICK_MODEL = os.getenv("PROCWISE_EXTRACTION_MODEL", "BeyondProcwise/AgentNick:latest")

# bp_ table schemas — source of truth for validation
BP_REQUIRED_FIELDS = {
    "Invoice": ["invoice_id", "supplier_id", "invoice_total_incl_tax"],
    "Purchase_Order": ["po_id", "supplier_name", "total_amount"],
    "Quote": ["quote_id", "supplier_id", "total_amount"],
    "Contract": ["contract_id", "supplier_id", "contract_title"],
}

BP_TABLES = {
    "Invoice": "proc.bp_invoice",
    "Purchase_Order": "proc.bp_purchase_order",
    "Quote": "proc.bp_quote",
    "Contract": "proc.bp_contracts",
}

BP_LINE_TABLES = {
    "Invoice": "proc.bp_invoice_line_items",
    "Purchase_Order": "proc.bp_po_line_items",
    "Quote": "proc.bp_quote_line_items",
}

PK_MAP = {
    "Invoice": "invoice_id",
    "Purchase_Order": "po_id",
    "Quote": "quote_id",
    "Contract": "contract_id",
}


class AgentNickOrchestrator:
    """Primary orchestrating agent for ProcWise document processing."""

    def __init__(self, agent_nick) -> None:
        self._agent_nick = agent_nick
        self._product_catalog = None   # lazy init on first use
        self._pattern_store = None     # lazy init on first use

    def _get_product_catalog(self):
        """Lazy-initialize ProductCatalogService on first use."""
        if self._product_catalog is None:
            from services.product_catalog_service import ProductCatalogService
            self._product_catalog = ProductCatalogService(
                self._agent_nick.get_db_connection
            )
        return self._product_catalog

    def _get_pattern_store(self):
        """Lazy-initialize ExtractionPatternStore on first use."""
        if self._pattern_store is None:
            from services.extraction_pattern_store import ExtractionPatternStore
            self._pattern_store = ExtractionPatternStore(
                self._agent_nick.get_db_connection
            )
        return self._pattern_store

    def process_document(
        self,
        file_path: str,
        category: str,
        *,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Orchestrate full document processing as the primary agent.

        Flow:
        1. Determine document type from category
        2. Dispatch DataExtractionAgent (sub-agent)
        3. Validate extraction result
        4. Enrich with supplier resolution and field completion
        5. Persist to bp_ tables
        6. Return result with status
        """
        category_map = {
            "invoice": "Invoice",
            "po": "Purchase_Order",
            "purchase_order": "Purchase_Order",
            "quote": "Quote",
            "quotes": "Quote",
            "contract": "Contract",
            "contracts": "Contract",
        }
        doc_type = category_map.get(category.strip().lower(), "")
        if not doc_type:
            return {"status": "error", "error": f"Unknown category: {category}"}

        logger.info(
            "[AgentNick] Processing %s as %s", file_path, doc_type
        )

        # Step 1: Dispatch DataExtractionAgent
        extraction = self._dispatch_extraction(file_path, doc_type)
        if not extraction:
            return {
                "status": "error",
                "error": "DataExtractionAgent returned no data",
                "file_path": file_path,
            }

        header = extraction.get("header", {})
        line_items = extraction.get("line_items", [])
        logger.info(
            "[AgentNick] DataExtractionAgent returned %d header fields, %d line items",
            len(header), len(line_items),
        )

        # Step 2: Validate — check required fields
        required = BP_REQUIRED_FIELDS.get(doc_type, [])
        missing = [f for f in required if not header.get(f)]

        # Step 3: Enrich — resolve supplier, fill gaps with LLM
        header = self._resolve_supplier(header, doc_type)

        # Auto-create supplier if not found in bp_supplier
        supplier_val = header.get("supplier_id") or header.get("supplier_name") or ""
        if supplier_val and len(supplier_val) >= 3:
            new_id = self._auto_create_supplier(header, doc_type)
            if new_id:
                header["supplier_id"] = new_id

        # Step 3a: Resolve buyer to canonical SUP-ID. The LLM/LangExtract
        # may set either header["buyer_id"] (raw company name) or
        # header["buyer_name"]; we want to land a canonical SUP-ID in
        # buyer_id so cross-document joins work. Mirrors _resolve_supplier
        # but writes to buyer_id instead.
        self._resolve_buyer(header)

        # Re-check after supplier resolution
        missing = [f for f in required if not header.get(f)]
        if missing:
            logger.info(
                "[AgentNick] Missing fields %s — asking LLM to fill gaps",
                missing,
            )
            header = self._llm_fill_gaps(
                header, missing, extraction.get("_source_text", ""), doc_type
            )

        # Step 3b: Validate PK using source text — format-agnostic
        # Instead of regex patterns, check if the extracted PK actually
        # appears in the document. This works for ANY document format.
        pk_col = PK_MAP.get(doc_type)
        source_text = extraction.get("_source_text", "")
        if pk_col:
            current_pk = str(header.get(pk_col, "") or "").strip()
            pk_valid = self._validate_pk_against_source(current_pk, source_text, doc_type)

            if not pk_valid:
                if current_pk:
                    logger.warning(
                        "[AgentNick] PK '%s' not found in source text — trying filename hint",
                        current_pk,
                    )
                # Fallback: try filename-based PK as hint
                fname_id = self._extract_pk_from_filename(file_path, doc_type)
                if fname_id:
                    header[pk_col] = fname_id
                    logger.info("[AgentNick] Filled %s from filename: %s", pk_col, fname_id)

        # Step 3c: Extract cross-references from filename (PO links etc.)
        self._fill_cross_refs_from_filename(header, file_path, doc_type)

        # Step 3d: Supplier sanity check — reject bank/payment terms as supplier
        _bad_supplier_markers = (
            "bank name", "bank account", "sort code", "iban", "swift",
            "payable to", "payment", "remittance",
        )
        for field in ("supplier_id", "supplier_name"):
            val = header.get(field, "")
            if val and any(m in val.lower() for m in _bad_supplier_markers):
                logger.warning(
                    "[AgentNick] Rejected bad supplier '%s' (contains bank/payment terms)",
                    val,
                )
                header[field] = ""

        # Step 3e: Filename-based supplier validation
        # The filename is the ground truth: "{SUPPLIER} PO{num} for QUT{num}.pdf"
        rescued_fields: set = set()
        fname_supplier = self._extract_supplier_from_filename(file_path)
        if fname_supplier:
            extracted_sup = (
                header.get("supplier_name")
                or header.get("supplier_id")
                or ""
            )
            # If no supplier extracted, or supplier doesn't match filename
            if not extracted_sup:
                header["supplier_name"] = fname_supplier
                if not header.get("supplier_id"):
                    header["supplier_id"] = fname_supplier
                logger.info(
                    "[AgentNick] Filled supplier from filename: %s", fname_supplier
                )
                rescued_fields.update({"supplier_name", "supplier_id"})
            elif fname_supplier.lower().split()[0] not in extracted_sup.lower():
                # Filename supplier's first word doesn't appear in extracted value
                # This catches: extracted="Assurity Ltd" but filename="PERRY PO526689"
                logger.warning(
                    "[AgentNick] Supplier mismatch: extracted='%s' but filename says '%s' — using filename",
                    extracted_sup, fname_supplier,
                )
                header["supplier_name"] = fname_supplier
                header["supplier_id"] = fname_supplier
                rescued_fields.update({"supplier_name", "supplier_id"})

        # Step 3f: Vendor-template override.
        # If we have seen this layout before, apply stored field hints
        # (typically supplier_name/buyer_name) over whatever the LLM
        # produced. Templates are durable across restarts via Postgres.
        try:
            from src.services.extraction_v2.template_service import (
                get_template_service,
            )
            from src.services.structural_extractor.parsing import (
                parse as _v2_parse,
            )
            _file_bytes = extraction.get("_file_bytes")
            _filename = extraction.get("_filename") or os.path.basename(file_path)
            template_fingerprint: Optional[str] = None
            template_overrides: Dict[str, str] = {}
            if _file_bytes:
                try:
                    parsed = _v2_parse(_file_bytes, _filename)
                    template_service = get_template_service()
                    template_fingerprint = template_service.fingerprint(parsed)
                    header, template_overrides = template_service.apply_template(
                        header, template_fingerprint, line_items=line_items,
                    )
                    rescued_fields.update(template_overrides.keys())
                except Exception as exc:
                    logger.warning(
                        "[AgentNick] template apply skipped for %s: %s",
                        file_path, exc,
                    )
        except Exception:
            # If the template module is unavailable, fall through silently;
            # the rest of the pipeline must keep working.
            template_fingerprint = None
            template_overrides = {}

        # Step 3f: Derive computable fields from document content
        # Pass source text so derivation can find "valid for 30 days" etc.
        header["_source_notes"] = extraction.get("_source_text", "")
        self._derive_fields(header, doc_type)
        header.pop("_source_notes", None)  # remove internal field before persistence

        # Step 4: Multi-pass validation and correction
        source_text = extraction.get("_source_text", "")
        try:
            from services.extraction_validator import ExtractionValidator
            validator = ExtractionValidator(self._agent_nick)
            header, line_items, discrepancies = validator.validate_and_correct(
                header, line_items, doc_type, source_text, file_path=file_path,
            )
        except Exception:
            logger.exception("[AgentNick] Validation failed, proceeding with raw extraction")
            discrepancies = []

        # Step 5: Cross-validation engine — auto-fix math errors, flag source issues
        header, line_items = self._cross_validate(header, line_items, doc_type)

        # Step 6: Flag source data issues (NOT auto-fix — these are problems in the document)
        source_warnings = self._flag_source_issues(header, line_items, doc_type)
        if source_warnings:
            header["_source_data_warnings"] = source_warnings
            for w in source_warnings:
                logger.warning("[AgentNick] SOURCE DATA ISSUE: %s", w)

        # Re-check after validation
        still_missing = [f for f in required if not header.get(f)]
        if still_missing:
            logger.warning(
                "[AgentNick] Still missing required fields after validation: %s",
                still_missing,
            )

        # Step 5: Set audit columns (force-set, don't use setdefault
        # because extraction may return empty strings for these fields)
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        audit_user = "AgentNick"
        header["created_date"] = now
        header["created_by"] = audit_user
        header["last_modified_date"] = now
        header["last_modified_by"] = audit_user

        for item in line_items:
            item["created_date"] = now
            item["created_by"] = audit_user
            item["last_modified_date"] = now
            item["last_modified_by"] = audit_user

        # Step 6: Final re-verification before persistence
        # Ensure critical fields are valid before writing to database.
        # This is the last gate — anything that passes here becomes
        # the official record.
        pk_col = PK_MAP.get(doc_type)
        pk_value = header.get(pk_col, "") if pk_col else ""

        # PK NORMALIZATION — collapse INV148769 / 148769 onto a single
        # canonical form so the same physical document never produces
        # two rows in bp_*. See pk_normalizer for the rules.
        if pk_col and pk_value:
            from services.pk_normalizer import normalize_pk
            normalized = normalize_pk(str(pk_value), doc_type)
            if normalized != str(pk_value):
                logger.info(
                    "[AgentNick] Normalized %s: %r → %r",
                    pk_col, pk_value, normalized,
                )
                header[pk_col] = normalized
                pk_value = normalized

        verification_issues = []
        if not pk_value:
            verification_issues.append(f"MISSING_PK: {pk_col} is empty")
        for amt_field in ("invoice_amount", "tax_amount", "invoice_total_incl_tax",
                          "total_amount", "total_amount_incl_tax"):
            val = header.get(amt_field)
            if val is not None:
                try:
                    float(val)
                except (ValueError, TypeError):
                    verification_issues.append(f"NON_NUMERIC: {amt_field}='{val}'")
                    header[amt_field] = None  # null out non-numeric amounts

        for date_field in ("invoice_date", "due_date", "order_date", "quote_date",
                           "expected_delivery_date", "validity_date"):
            val = header.get(date_field)
            if val is not None and not re.match(r"^\d{4}-\d{2}-\d{2}", str(val)):
                verification_issues.append(f"INVALID_DATE: {date_field}='{val}'")

        if verification_issues:
            logger.warning(
                "[AgentNick] PRE-PERSIST ISSUES for %s %s: %s",
                doc_type, pk_value, verification_issues,
            )

        # Step 6c: Field-type validation gate — sanitizes garbage party
        # names, fixes po_id format, normalizes payment_terms, nulls
        # tax==subtotal and far-future dates. Rejected values are logged
        # to bp_discrepancy_data so they remain visible for review.
        sanitizer_rejections = []
        try:
            from services.extraction_sanitizer import ExtractionSanitizer
            sanitizer = ExtractionSanitizer()
            header, line_items, sanitizer_rejections = sanitizer.sanitize(
                header, line_items, doc_type,
            )
            if sanitizer_rejections:
                self._log_sanitizer_rejections(
                    sanitizer_rejections, doc_type, pk_value, file_path,
                )
        except Exception:
            logger.exception("[AgentNick] Sanitizer pass failed (non-fatal)")

        # Step 6c-2: Field recovery from parsed text. Fills NULL header
        # fields (payment_terms, validity/delivery dates, incoterm, tax
        # rate/amount, country/region/city, postcode) ONLY when there is
        # explicit evidence in the parsed text. Never overwrites a
        # non-NULL value; never fabricates. Runs before persist so
        # recovered values land in the DB.
        try:
            from src.services.extraction_v2.field_recovery import recover_fields
            parsed_text = extraction.get("_source_text", "")
            recovery_report = recover_fields(
                header, parsed_text=parsed_text,
                doc_type=doc_type, file_path=file_path,
            )
            if recovery_report.fields_recovered:
                logger.info(
                    "[AgentNick] field_recovery filled %d NULL(s) for %s %s",
                    len(recovery_report.fields_recovered), doc_type, pk_value,
                )
                header["_field_recovery"] = [
                    {"field": f, "value": v[:80], "source": s}
                    for (f, v, s) in recovery_report.fields_recovered
                ]
        except Exception:
            logger.exception("[AgentNick] field_recovery skipped (non-fatal)")

        # Step 6c-3a: PDF-table line-items recovery. For docs whose
        # line-item table is laid out in columns that the structural
        # extractor flattened (I-37 — WADE quotes etc.), use pdfplumber's
        # word-level positions to reconstruct rows by Y-coordinate. Only
        # fires when we have raw PDF bytes and a target subtotal; emits
        # qty/unit_price when math reconciles.
        try:
            from src.services.extraction_v2.pdf_table_recovery import (
                recover_lines_from_pdf_table,
            )
            # Compute subtotal target from header. The line-items table in
            # the source document typically holds pre-tax amounts, so:
            # 1) Prefer explicit pre-tax fields (subtotal / invoice_amount /
            #    total_amount / total_amount_excl_tax).
            # 2) If only the post-tax total (invoice_total_incl_tax /
            #    total_amount_incl_tax) is available AND tax_percent is set,
            #    derive pre-tax target = post_tax / (1 + rate).
            tax_pct = header.get("tax_percent")
            try:
                tax_pct = float(tax_pct) if tax_pct is not None else None
            except (ValueError, TypeError):
                tax_pct = None

            pdf_target = None
            for k in ("subtotal", "invoice_amount", "total_amount",
                      "total_amount_excl_tax"):
                v = header.get(k)
                if v is not None:
                    try:
                        f = float(v)
                        if f > 0:
                            pdf_target = f
                            break
                    except (ValueError, TypeError):
                        continue
            if pdf_target is None:
                # Pre-tax not found; try post-tax + tax_pct derivation
                for k in ("invoice_total_incl_tax", "total_amount_incl_tax"):
                    v = header.get(k)
                    if v is None:
                        continue
                    try:
                        f = float(v)
                        if f <= 0:
                            continue
                        if tax_pct is not None and tax_pct > 0:
                            rate = tax_pct / 100.0 if tax_pct > 1 else tax_pct
                            pdf_target = f / (1.0 + rate)
                        else:
                            pdf_target = f
                        break
                    except (ValueError, TypeError):
                        continue
            pdf_bytes = extraction.get("_file_bytes")
            if pdf_bytes and pdf_target:
                pdf_lr = recover_lines_from_pdf_table(
                    pdf_bytes, target_total=pdf_target, tax_percent=tax_pct,
                    file_path=file_path,
                )
                # Compute existing sum to compare
                existing_sum = 0.0
                for it in (line_items or []):
                    for k in ("line_total", "line_amount", "total_amount"):
                        v = it.get(k)
                        if v is not None:
                            try:
                                existing_sum += float(v)
                                break
                            except (ValueError, TypeError):
                                continue
                if pdf_lr.items_recovered > 0:
                    # Replace if PDF table extractor's sum is closer to target
                    # than the existing line items, OR if both sums match the
                    # target equally well but PDF found MORE granular line
                    # items (e.g. LLM emitted 1 line summing to subtotal but
                    # the doc actually has 3 — replace with the 3).
                    pdf_diff = abs(pdf_lr.sum_recovered - pdf_target)
                    existing_diff = abs(existing_sum - pdf_target)
                    pdf_better = (
                        pdf_diff < existing_diff
                        or (
                            # Both sums equally close (within 50p) and PDF
                            # has more line items than existing.
                            abs(pdf_diff - existing_diff) < 0.50
                            and pdf_lr.items_recovered > len(line_items or [])
                        )
                    )
                    if not line_items or pdf_better:
                        logger.info(
                            "[AgentNick] pdf_table_recovery filled %d items "
                            "for %s %s (sum=%.2f, target=%.2f, replaced=%.2f)",
                            pdf_lr.items_recovered, doc_type, pk_value,
                            pdf_lr.sum_recovered, pdf_target, existing_sum,
                        )
                        line_items = pdf_lr.items
                        header["_line_recovery"] = {
                            "source": "pdf_table_recovery",
                            "count": pdf_lr.items_recovered,
                            "sum": pdf_lr.sum_recovered,
                            "target": pdf_target,
                            "replaced_existing_sum": existing_sum,
                        }
        except Exception:
            logger.exception("[AgentNick] pdf_table_recovery skipped (non-fatal)")

        # Step 6c-3b: Text-pattern line-items recovery. Fires when the
        # PDF-table path didn't help and either:
        #   (a) LLM/structural extractor returned line_items=[] — empty,
        #       but the doc clearly has line data the layout obscured.
        #   (b) line_items is non-empty but their total doesn't match the
        #       header subtotal — the LLM got SOME items but missed
        #       others (partial extraction).
        # The recovery only emits items if their amounts sum to within
        # 5% of the header subtotal — and only REPLACES the existing
        # set if its sum is closer to the target than the existing sum.
        try:
            from src.services.extraction_v2.line_recovery import (
                recover_line_items,
            )
            target = None
            for k in ("subtotal", "invoice_amount", "total_amount",
                      "total_amount_excl_tax",
                      "invoice_total_incl_tax", "total_amount_incl_tax"):
                v = header.get(k)
                if v is not None:
                    try:
                        f = float(v)
                        if f > 0:
                            target = f
                            break
                    except (ValueError, TypeError):
                        continue

            existing_sum = 0.0
            if line_items and target:
                for it in line_items:
                    for k in ("line_total", "line_amount", "total_amount"):
                        v = it.get(k)
                        if v is not None:
                            try:
                                existing_sum += float(v)
                                break
                            except (ValueError, TypeError):
                                continue
            existing_off_target = bool(
                target and (
                    not line_items or
                    abs(existing_sum - target) > max(target * 0.05, 0.50)
                )
            )

            if not line_items or existing_off_target:
                lr = recover_line_items(
                    header,
                    parsed_text=extraction.get("_source_text", ""),
                    doc_type=doc_type, file_path=file_path,
                )
                if lr.items_recovered > 0:
                    # Only replace existing set if recovery's sum is closer
                    # to the target than what's already there.
                    existing_diff = abs(existing_sum - (lr.target_total or 0))
                    recovered_diff = abs(lr.sum_recovered - (lr.target_total or 0))
                    if not line_items or recovered_diff < existing_diff:
                        logger.info(
                            "[AgentNick] line_recovery filled %d items for %s %s "
                            "(sum=%.2f, target=%.2f, replaced existing sum=%.2f)",
                            lr.items_recovered, doc_type, pk_value,
                            lr.sum_recovered, lr.target_total or 0,
                            existing_sum,
                        )
                        line_items = lr.items
                        header["_line_recovery"] = {
                            "count": lr.items_recovered,
                            "sum": lr.sum_recovered,
                            "target": lr.target_total,
                            "replaced_existing_sum": existing_sum,
                        }
                    else:
                        logger.debug(
                            "[AgentNick] line_recovery candidates found but "
                            "existing set is closer to target (%.2f vs %.2f)",
                            existing_diff, recovered_diff,
                        )
                elif lr.skipped_reason:
                    logger.debug(
                        "[AgentNick] line_recovery skipped for %s %s: %s",
                        doc_type, pk_value, lr.skipped_reason,
                    )
        except Exception:
            logger.exception("[AgentNick] line_recovery skipped (non-fatal)")

        # Step 6d: Critical-field confidence gate. If the sanitizer nulled
        # a critical field (PK, supplier_id, total_amount), enqueue the
        # record for human review. Persistence still proceeds so the
        # pipeline doesn't block — the queue is a parallel signal.
        try:
            self._enqueue_for_review_if_needed(
                header, line_items, doc_type, file_path,
                sanitizer_rejections, source_text=extraction.get("_source_text", ""),
                rescued_fields=rescued_fields,
            )
        except Exception:
            logger.exception("[AgentNick] Review-queue enqueue failed (non-fatal)")

        # Step 6b: Populate item_id via product catalog
        if line_items:
            catalog = self._get_product_catalog()
            for item in line_items:
                desc = item.get("item_description", "")
                doc_item_id = item.get("item_id")
                price = None
                try:
                    price = float(item.get("unit_price") or 0) or None
                except (ValueError, TypeError):
                    pass
                product_id = catalog.match_or_create(
                    item_description=desc,
                    item_id_from_doc=doc_item_id,
                    unit_price=price,
                    currency=header.get("currency"),
                    unit_of_measure=item.get("unit_of_measure"),
                    doc_type=doc_type,
                    doc_id=pk_value,
                )
                item["item_id"] = product_id

        header_ok = self._persist_header(header, doc_type)
        # Header must succeed before line items are written. Writing lines
        # under a missing header produces orphan rows in bp_*_line_items that
        # cannot be cross-referenced back to a document — silent data loss.
        if header_ok:
            lines_ok = self._persist_line_items(line_items, doc_type, pk_value)
        else:
            lines_ok = False
            logger.error(
                "[AgentNick] Header persist FAILED for %s %s — skipping %d line items to avoid orphans",
                doc_type, pk_value, len(line_items),
            )

        # Record per-field provenance for the persisted record so
        # downstream consumers can tell which fields were template-applied
        # vs. filename-rescued vs. sanitizer-nulled vs. LLM-derived. Best-
        # effort: a provenance write failure must not fail the extraction.
        if header_ok and pk_value:
            try:
                from services.db import get_conn as _prov_get_conn
                from src.services.extraction_v2.provenance import (
                    record_extraction_provenance,
                )
                # Default-source heuristic: if the structural extractor
                # supplied the bytes (USE_STRUCTURAL_EXTRACTOR path), we
                # call the default ``structural``; otherwise the LLM path
                # produced everything by default.
                default_src = "structural" if (
                    extraction.get("_file_bytes") is not None
                    and os.getenv("USE_STRUCTURAL_EXTRACTOR", "false").lower()
                        in ("true", "1", "yes")
                ) else "llm"
                record_extraction_provenance(
                    _prov_get_conn,
                    record_id=str(pk_value), doc_type=doc_type, header=header,
                    rescued_fields=rescued_fields,
                    template_overrides=template_overrides,
                    sanitizer_rejections=sanitizer_rejections or [],
                    default_source=default_src,
                )
            except Exception as exc:
                logger.warning(
                    "[AgentNick] provenance write skipped for %s: %s",
                    pk_value, exc,
                )

        # Step 7: Learn vendor profile
        self._learn_vendor_profile(header, doc_type)

        # Step 7-pre-A3: line-level currency normalization. Sets
        # header["currency"] + per-line "currency" from the parsed text
        # / vendor country if not already specified. The CurrencyConsistency
        # invariant below uses these to detect mixed-currency anomalies.
        try:
            from src.services.extraction_v2.currency import (
                normalize_currency_in_place,
            )
            normalize_currency_in_place(
                header, line_items,
                parsed_text=extraction.get("_source_text", ""),
            )
        except Exception:
            logger.debug("[AgentNick] currency normalization skipped", exc_info=True)

        # Step 7-pre-A1+A4: run the procurement invariant chain. Each
        # validator returns (passed, residual, severity). Critical
        # failures auto-route to review regardless of other signals.
        validation_critical_count = 0
        validation_warning_count = 0
        try:
            from services.db import get_conn as _val_get_conn
            from src.services.extraction_v2.invariants import (
                DEFAULT_VALIDATORS, ValidatorChain,
            )
            from src.services.extraction_v2.po_linkage import PoLinkage
            chain = ValidatorChain(
                DEFAULT_VALIDATORS + [PoLinkage(get_conn=_val_get_conn)]
            )
            report = chain.run(header, line_items, doc_type)
            validation_critical_count = len(report.critical_failures)
            validation_warning_count = len(report.warnings)
            if report.critical_failures:
                header["needs_review"] = True
                for r in report.critical_failures:
                    logger.warning(
                        "[AgentNick] INVARIANT_FAIL(%s) %s %s: %s",
                        r.severity, doc_type, pk_value, r.message,
                    )
            elif report.warnings:
                for r in report.warnings:
                    logger.info(
                        "[AgentNick] invariant warn(%s) %s %s: %s",
                        r.severity, doc_type, pk_value, r.message,
                    )
            header["_invariant_report"] = {
                "pass_rate": round(report.pass_rate, 3),
                "critical_count": validation_critical_count,
                "warning_count": validation_warning_count,
                "failures": [
                    {"name": r.name, "severity": r.severity,
                     "message": r.message[:200]}
                    for r in report.results if not r.passed
                ],
            }
        except Exception:
            logger.debug("[AgentNick] invariant chain skipped", exc_info=True)

        # Step 7-pre: compute the calibrated confidence score and write
        # it onto the header BEFORE we report the orchestration result.
        # This replaces the static 0.85 with a real signal-driven score.
        try:
            from src.services.extraction_v2.confidence import calibrated_confidence
            calib = calibrated_confidence(
                header=header, line_items=line_items, doc_type=doc_type,
                sanitizer_rejections=sanitizer_rejections or [],
                rescued_fields=rescued_fields,
                template_overrides=template_overrides,
            )
            # Invariant penalties: each warning -0.03, each critical -0.10.
            invariant_penalty = (
                0.03 * validation_warning_count
                + 0.10 * validation_critical_count
            )
            score = max(0.0, calib.score - invariant_penalty)
            header["confidence_score"] = round(score, 3)
            if score < 0.65 or validation_critical_count > 0:
                logger.warning(
                    "[AgentNick] LOW_CONFIDENCE for %s %s: score=%.2f "
                    "(calib=%.2f - invariant_penalty=%.2f) notes=%s",
                    doc_type, pk_value, score, calib.score,
                    invariant_penalty, calib.notes,
                )
                header["needs_review"] = True
        except Exception:
            logger.debug("[AgentNick] confidence calibration skipped", exc_info=True)

        # Step 7a: Snapshot a candidate vendor template if extraction
        # quality is good. Only successful header persistence + at least
        # one rescued or LLM-confirmed canonical value qualifies — this
        # prevents bad extractions from poisoning the template store.
        if header_ok and template_fingerprint:
            try:
                from src.services.extraction_v2.template_service import (
                    get_template_service,
                )
                template_service = get_template_service()
                template_service.learn_from_extraction(
                    fingerprint=template_fingerprint,
                    header=header,
                    line_items=line_items,
                    doc_type=doc_type,
                    vendor_name_hint=fname_supplier,
                    rescued_fields=rescued_fields,
                )
            except Exception as exc:
                logger.warning(
                    "[AgentNick] template learn skipped for %s: %s",
                    pk_value, exc,
                )

        error_count = sum(1 for d in discrepancies if d.severity == "error")
        result = {
            "status": "success" if header_ok else "partial",
            "file_path": file_path,
            "doc_type": doc_type,
            "pk": pk_value,
            "header_fields": len(header),
            "line_items": len(line_items),
            "header_persisted": header_ok,
            "lines_persisted": lines_ok,
            "missing_fields": still_missing,
            "discrepancies": len(discrepancies),
            "errors": error_count,
            "confidence": header.get("confidence_score", 0),
            "needs_review": header.get("needs_review", False),
            "template_fingerprint": template_fingerprint,
            "template_overrides": list(template_overrides.keys()),
            "rescued_fields": sorted(rescued_fields),
        }

        logger.info(
            "[AgentNick] Completed: %s %s=%s, %d fields, %d lines, missing=%s",
            doc_type, pk_col, pk_value,
            len(header), len(line_items), still_missing,
        )
        return result

    # ------------------------------------------------------------------
    # Sub-agent dispatch — DocWain + LLM in parallel
    # ------------------------------------------------------------------
    def _dispatch_extraction(
        self, file_path: str, doc_type: str
    ) -> Optional[Dict[str, Any]]:
        """Extract document content using intelligent structure-preserving pipeline.

        Strategy:
        1. Download file from S3
        2. Parse into structured representation (tables, metadata, totals)
        3. Send structured data to LLM for intelligent mapping to schema
        4. Verify extraction against source structure
        5. Retry with targeted guidance if verification finds errors
        6. Fall back to legacy extraction engine as last resort
        """
        # Feature-flag routing: structural extractor path
        if os.getenv("USE_STRUCTURAL_EXTRACTOR", "false").lower() in ("true", "1", "yes"):
            try:
                from services.direct_extraction_service import DirectExtractionService
                _svc = DirectExtractionService(self._agent_nick)
                _text, _file_bytes = _svc._get_document_text(file_path)
                if _file_bytes:
                    from src.services.structural_extractor import extract as _extract
                    from src.services.structural_extractor.provenance import write_provenance
                    filename = os.path.basename(file_path)
                    result = _extract(_file_bytes, filename, doc_type)
                    # Required-field gate: only fall back to legacy if a CRITICAL
                    # field is unresolved. Optional fields (delivery_address_line1,
                    # incoterm, expected_delivery_date, payment_terms, etc.) are
                    # allowed to be NULL — falling back to the hallucinating LLM
                    # path would overwrite correct structural values with wrong
                    # ones (the Staedtler pen regression on Duncan PO).
                    _critical = {
                        "Invoice": {"invoice_id", "invoice_amount", "invoice_total_incl_tax"},
                        "Purchase_Order": {"po_id", "total_amount"},
                        "Quote": {"quote_id", "total_amount"},
                        "Contract": {"contract_id"},
                    }.get(doc_type, set())
                    _missing_critical = _critical - result.header.keys()
                    if not _missing_critical:
                        # --- Country / region derivation from full document text ---
                        # The structural extractor's country_from_postcode rule
                        # expects an "_address_text" input; populate it from the
                        # parsed document's full_text so the rule can fire and
                        # produce fully-traced country + region values.
                        try:
                            from src.services.structural_extractor.derivation_rules import misc as _derivation_misc
                            from src.services.structural_extractor.types import ExtractedValue as _EV
                            _addr_text = result.parsed_text or ""
                            for _rule in (_derivation_misc._country, _derivation_misc._region):
                                _field = {"_country": "country", "_region": "region"}[_rule.__name__]
                                if _field in result.header:
                                    continue  # already extracted
                                _val = _rule({"_address_text": _addr_text})
                                if _val is None:
                                    continue
                                result.header[_field] = _EV(
                                    value=_val, provenance="inferred",
                                    derivation_trace={
                                        "rule_id": f"{_field}_from_postcode" if _field == "country" else "region_from_address",
                                        "inputs": {"_address_text_sample": _addr_text[:120]},
                                    },
                                    source="derivation_registry", confidence=1.0, attempt=1,
                                )
                            # Purchase_Order schema uses ship_to_country /
                            # delivery_region instead of country / region.
                            # Copy the derived values across so they aren't
                            # silently dropped by the persist layer's
                            # unknown-column filter (that was the cause of
                            # 96% of POs showing NULL ship_to_country).
                            if doc_type == "Purchase_Order":
                                if "country" in result.header and "ship_to_country" not in result.header:
                                    result.header["ship_to_country"] = result.header["country"]
                                if "region" in result.header and "delivery_region" not in result.header:
                                    result.header["delivery_region"] = result.header["region"]
                        except Exception as _addr_exc:
                            logger.debug("[structural] country/region derivation skipped: %s", _addr_exc)

                        # Per-doc-type line-total column name mapping.
                        # bp_invoice_line_items uses `line_amount`, bp_po_line_items
                        # and bp_quote_line_items use `line_total`.
                        _lt_key = "line_amount" if doc_type == "Invoice" else "line_total"
                        header_dict = {
                            k: v.value for k, v in result.header.items()
                            if v.value is not None
                        }
                        line_items_dicts: list = []
                        for item in result.line_items:
                            li = {}
                            for k, v in item.items():
                                val = v.value if hasattr(v, "value") else v
                                # Map line_total → line_amount for invoices;
                                # downstream persistence (data_extraction_agent._persist_line_items_to_postgres)
                                # expects the per-doc-type column name.
                                if k == "line_total" and _lt_key == "line_amount":
                                    li["line_amount"] = val
                                else:
                                    li[k] = val
                            # Inherit country/region from header into line items
                            # (bp_invoice_line_items has country/region columns;
                            # bp_po_line_items and bp_quote_line_items do not).
                            if doc_type == "Invoice":
                                for _inherit in ("country", "region"):
                                    if _inherit not in li and header_dict.get(_inherit):
                                        li[_inherit] = header_dict[_inherit]
                            # Per-line tax derivation: tax_percent inherits from
                            # header; tax_amount = line_amount × tax_percent/100;
                            # total_amount_incl_tax (invoices) or total_amount
                            # (POs/quotes) = line_amount + tax_amount. Applies
                            # when header has a single tax rate.
                            _hdr_pct = header_dict.get("tax_percent")
                            _line_amt = li.get("line_amount") or li.get("line_total")
                            if _hdr_pct is not None and _line_amt is not None:
                                try:
                                    _pct = float(_hdr_pct)
                                    _amt = float(_line_amt)
                                    if "tax_percent" not in li:
                                        li["tax_percent"] = _pct
                                    if "tax_amount" not in li:
                                        li["tax_amount"] = round(_amt * _pct / 100, 2)
                                    _line_incl = round(_amt + li["tax_amount"], 2)
                                    if doc_type == "Invoice":
                                        if "total_amount_incl_tax" not in li:
                                            li["total_amount_incl_tax"] = _line_incl
                                    else:  # Purchase_Order / Quote
                                        if "total_amount" not in li:
                                            li["total_amount"] = _line_incl
                                except (TypeError, ValueError):
                                    pass
                            line_items_dicts.append(li)
                        # Provenance: write one row per populated field (header + line items).
                        # Uses a fresh connection from AgentNick's DB pool; best-effort —
                        # failure to write provenance never blocks extraction persistence.
                        pk_field = {
                            "Invoice": "invoice_id",
                            "Purchase_Order": "po_id",
                            "Quote": "quote_id",
                            "Contract": "contract_id",
                        }.get(doc_type)
                        parent_pk = result.header.get(pk_field).value if pk_field and result.header.get(pk_field) else None
                        if parent_pk:
                            try:
                                _prov_conn = self._agent_nick.get_db_connection()
                                try:
                                    parent_table = {
                                        "Invoice": "bp_invoice",
                                        "Purchase_Order": "bp_purchase_order",
                                        "Quote": "bp_quote",
                                        "Contract": "bp_contracts",
                                    }.get(doc_type, f"bp_{doc_type.lower()}")
                                    write_provenance(_prov_conn, parent_table, str(parent_pk), result.header)
                                    for idx, item in enumerate(result.line_items, 1):
                                        line_pk = f"{parent_pk}-{idx}"
                                        line_parent_table = {
                                            "Invoice": "bp_invoice_line_items",
                                            "Purchase_Order": "bp_po_line_items",
                                            "Quote": "bp_quote_line_items",
                                        }.get(doc_type)
                                        if line_parent_table:
                                            write_provenance(_prov_conn, line_parent_table, line_pk, item)
                                    _prov_conn.commit()
                                finally:
                                    _prov_conn.close()
                            except Exception as _prov_exc:
                                logger.warning(
                                    "[structural] provenance write failed for %s: %s (non-blocking)",
                                    parent_pk, _prov_exc,
                                )
                        # Line-item salvage: structural extractor sometimes
                        # returns zero line items on summary-style quotes
                        # and one-page POs even though lines exist in the
                        # body. Run a narrowly-scoped LLM pass to recover
                        # them before falling through to persistence.
                        # Trigger broadened: salvage on ANY substantive
                        # parsed_text (>200 chars), even if total is null,
                        # so quotes without explicit totals still get lines.
                        if (not line_items_dicts
                                and result.parsed_text and len(result.parsed_text) > 200):
                            # Compute fingerprint + vendor hint for the
                            # salvage prompt. This is the per-vendor
                            # accuracy lever for line items: when we have
                            # seen this layout before, the LLM's prompt
                            # gets the column layout + expected min-rows
                            # injected so it knows where to look.
                            _vendor_hint = None
                            try:
                                from src.services.extraction_v2.template_service import (
                                    get_template_service,
                                )
                                from src.services.structural_extractor.parsing import (
                                    parse as _v2_parse_for_lines,
                                )
                                _parsed_for_hint = _v2_parse_for_lines(
                                    _file_bytes, filename,
                                )
                                _ts = get_template_service()
                                _fp = _ts.fingerprint(_parsed_for_hint)
                                _vendor_hint = _ts.line_items_prompt_hint(_fp)
                            except Exception:
                                _vendor_hint = None
                            try:
                                salvaged = self._llm_salvage_line_items(
                                    result.parsed_text, doc_type, header_dict,
                                    vendor_hint=_vendor_hint,
                                )
                                if salvaged:
                                    line_items_dicts = salvaged
                                    logger.info(
                                        "[structural] line-item salvage recovered %d rows for %s%s",
                                        len(salvaged), file_path,
                                        " (template-aided)" if _vendor_hint else "",
                                    )
                                else:
                                    # Salvage couldn't find a tabular or
                                    # multi-row layout. Try the narrow
                                    # single-service fallback for
                                    # summary-style invoices.
                                    single = self._llm_extract_single_service(
                                        result.parsed_text, doc_type, header_dict,
                                    )
                                    if single:
                                        line_items_dicts = single
                                        logger.info(
                                            "[structural] single-service salvage recovered "
                                            "1 row for %s (description=%r)",
                                            file_path, single[0].get("item_description"),
                                        )
                            except Exception as _salvage_exc:
                                logger.debug(
                                    "[structural] line-item salvage failed: %s",
                                    _salvage_exc,
                                )
                        return {
                            "header": header_dict,
                            "line_items": line_items_dicts,
                            "_source_text": result.parsed_text,
                            "_file_bytes": _file_bytes,
                            "_filename": filename,
                        }
                    # Missing CRITICAL field: legacy fallback (last resort)
                    logger.warning(
                        "[structural] %s: missing critical %s (unresolved %s) — falling back to legacy",
                        file_path, _missing_critical, result.unresolved_fields,
                    )
            except Exception as exc:
                logger.warning(
                    "[structural] failed for %s: %s — falling back", file_path, exc
                )
        # end structural hook — existing legacy path follows below unchanged
        try:
            from services.direct_extraction_service import (
                DirectExtractionService,
                TABLE_SCHEMAS,
            )
            from services.intelligent_extractor import IntelligentExtractor

            service = DirectExtractionService(self._agent_nick)
            schema = TABLE_SCHEMAS.get(doc_type)
            if not schema:
                logger.error("[AgentNick] No schema for doc_type=%s", doc_type)
                return None

            # Step 1: Download file
            text, file_bytes = service._get_document_text(file_path)
            ext = os.path.splitext(file_path)[1].lower() or ".pdf"

            if not file_bytes and not text.strip():
                logger.error("[AgentNick] No content extracted from %s", file_path)
                return None

            # Step 2: Parse document into structured representation
            extractor = IntelligentExtractor(
                service, pattern_store=self._get_pattern_store()
            )
            doc_structure = extractor.parse_document(file_bytes, ext)

            # If structured parsing produced no tables and no metadata,
            # fall back to raw text from the existing extraction pipeline
            if (
                not doc_structure.tables
                and not doc_structure.metadata
                and not doc_structure.raw_text
            ):
                if text.strip():
                    doc_structure.raw_text = text
                else:
                    logger.error(
                        "[AgentNick] No text or structure extracted from %s",
                        file_path,
                    )
                    return None

            # Inject vendor profile context (legacy: filename-keyed
            # vendor_profile table aggregating prior extractions).
            supplier_hint = self._extract_supplier_from_filename(file_path)
            vendor_context = self._get_vendor_context(supplier_hint or "")
            if vendor_context:
                doc_structure.metadata.insert(
                    0, {"label": "VENDOR_PROFILE", "value": vendor_context}
                )

            # Augment with vendor-template context (V2: fingerprint-keyed
            # hints learned automatically from successful extractions or
            # entered manually via /vendors/onboard). This improves the
            # FIRST-pass legacy-LLM accuracy for known vendor layouts.
            try:
                from src.services.extraction_v2.template_service import (
                    get_template_service,
                )
                from src.services.structural_extractor.parsing import (
                    parse as _v2_parse_for_legacy,
                )
                if file_bytes:
                    parsed = _v2_parse_for_legacy(file_bytes, os.path.basename(file_path))
                    ts = get_template_service()
                    fp = ts.fingerprint(parsed)
                    template_context = ts.legacy_extraction_prompt_context(fp)
                    if template_context:
                        doc_structure.metadata.insert(
                            0, {"label": "VENDOR_TEMPLATE", "value": template_context}
                        )
            except Exception as _tpl_exc:
                logger.debug("[AgentNick] template context skipped: %s", _tpl_exc)

            # Step 3-4: Intelligent extraction with verification
            result = extractor.extract(doc_structure, doc_type, schema)

            if result:
                header = result.get("header", {})
                line_items = result.get("line_items", [])
                verification = result.get("_verification", {})
                logger.info(
                    "[AgentNick] Intelligent extraction: %d header fields, "
                    "%d line items, %d issues",
                    len(header),
                    len(line_items),
                    verification.get("critical_count", 0)
                    + verification.get("warning_count", 0),
                )

                # If no line items, try focused extraction as fallback
                # Use the legacy flat-text extraction first (more reliable for PDFs)
                # then focused line-item extraction
                if not line_items and (text or doc_structure.raw_text):
                    fallback_text = text or doc_structure.raw_text
                    logger.info(
                        "[AgentNick] No line items — trying legacy + focused extraction"
                    )
                    # Try 1: Legacy full extraction (sometimes finds lines the intelligent extractor misses)
                    try:
                        from services.direct_extraction_service import DirectExtractionService
                        legacy_result = service._llm_extract(
                            self._build_enhanced_text(
                                fallback_text,
                                os.path.splitext(os.path.basename(file_path))[0],
                                doc_type,
                            ),
                            doc_type, schema,
                        )
                        if legacy_result and legacy_result.get("line_items"):
                            line_items = legacy_result["line_items"]
                            logger.info(
                                "[AgentNick] Legacy extraction found %d line items",
                                len(line_items),
                            )
                    except Exception:
                        pass

                    # Try 2: Focused line-item-only extraction
                    if not line_items:
                        line_items = self._llm_extract_line_items(
                            fallback_text, doc_type, header
                        )

                return {
                    "header": self._sanitize_header(header),
                    "line_items": self._sanitize_items(line_items),
                    "_source_text": text or doc_structure.raw_text,
                    "_file_bytes": file_bytes,
                    "_filename": os.path.basename(file_path),
                }

            # Step 5: Fall back to legacy extraction
            logger.info(
                "[AgentNick] Intelligent extraction failed — trying legacy"
            )
            if text.strip():
                basename = os.path.splitext(os.path.basename(file_path))[0]
                enhanced_text = self._build_enhanced_text(
                    text, basename, doc_type
                )
                llm_result = service._llm_extract(
                    enhanced_text, doc_type, schema
                )
                if llm_result:
                    return {
                        "header": self._sanitize_header(
                            llm_result.get("header", {})
                        ),
                        "line_items": self._sanitize_items(
                            llm_result.get("line_items", [])
                        ),
                        "_source_text": text,
                    }

            logger.error(
                "[AgentNick] All extraction methods failed for %s", file_path
            )
            return None

        except Exception as exc:
            logger.exception(
                "[AgentNick] DataExtractionAgent dispatch failed: %s", exc
            )
            return None

    @staticmethod
    def _build_enhanced_text(text: str, filename: str, doc_type: str) -> str:
        """Build optimized extraction text with filename context and smart windowing.

        Uses document intelligence to detect sections and prioritize
        line items (never truncated) over boilerplate.
        """
        from services.document_intelligence import build_smart_text

        hint = (
            f"FILENAME: {filename}\n"
            f"(The filename may contain the document ID, supplier name, "
            f"and related document references — use as context.)\n\n"
        )
        smart_text = build_smart_text(text, max_chars=6000)
        return hint + smart_text

    @staticmethod
    @staticmethod
    def _sanitize_header(header: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize extracted header — fix known LLM mistakes."""
        cleaned = {k: (v if v != "" else None) for k, v in header.items()}

        # Fix: buyer_id / supplier_id / supplier_name must be company names
        _address_indicators = {
            "street", "road", "lane", "way", "park", "unit ", "floor",
            "department", "suite", "building", "house", "avenue", "drive",
            "close", "crescent", "place", "terrace", "square", "redkiln",
            "kingsway", "regent", "market st",
        }
        # Postcode patterns (UK, US, etc.)
        _postcode_re = re.compile(
            r"[A-Z]{1,2}\d{1,2}\s?\d[A-Z]{2}|"  # UK: RH13 5QH
            r"\d{5}(-\d{4})?",                     # US: 10001
            re.IGNORECASE,
        )
        # Company name suffixes — if the value has one, it's likely a company
        _company_suffixes = {
            "ltd", "limited", "llc", "inc", "plc", "corp", "gmbh",
            "pty", "co.", "group", "partners", "llp", "trading",
            "solutions", "services", "consulting", "agency", "studio",
        }

        for field in ("buyer_id", "supplier_id", "supplier_name"):
            val = cleaned.get(field)
            if not val or not isinstance(val, str):
                continue
            val_lower = val.lower().strip()

            # Skip if value has a company suffix — it's a company name
            if any(suf in val_lower for suf in _company_suffixes):
                continue

            # Check 1: Contains postcode → address
            if _postcode_re.search(val):
                logger.warning(
                    "Sanitize: %s contains postcode, clearing: '%s'",
                    field, val[:60],
                )
                cleaned[field] = None
                continue

            # Check 2: Contains address words → address
            addr_words = sum(1 for w in _address_indicators if w in val_lower)
            if addr_words >= 1:
                logger.warning(
                    "Sanitize: %s looks like address, clearing: '%s'",
                    field, val[:60],
                )
                cleaned[field] = None
                continue

            # Check 3: Truncated values (< 4 chars)
            if len(val.strip()) < 4:
                logger.warning(
                    "Sanitize: %s too short, clearing: '%s'",
                    field, val,
                )
                cleaned[field] = None

        return cleaned

    @staticmethod
    def _sanitize_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {k: (v if v != "" else None) for k, v in item.items()}
            for item in items
        ]

    def _sanitize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "header": self._sanitize_header(result.get("header", {})),
            "line_items": self._sanitize_items(result.get("line_items", [])),
            "_source_text": result.get("_source_text", ""),
        }

    @staticmethod
    def _map_engine_result(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Map extraction engine result format to standard header/line_items."""
        header = {}
        line_items = []
        if "header" in result:
            header = dict(result.get("header", {}))
            line_items = list(result.get("line_items", []))
        elif result.get("document_type") == "invoice":
            header = dict(result.get("invoice_data", {}))
            line_items = list(result.get("line_items", []))
        elif result.get("document_type") == "po":
            header = dict(result.get("po_data", {}))
            line_items = list(result.get("line_items", []))
        elif result.get("document_type") == "quote":
            header = dict(result.get("quote_data", {}))
            line_items = list(result.get("line_items", []))
        if not header:
            return None
        return {
            "header": {k: (v if v != "" else None) for k, v in header.items()},
            "line_items": [
                {k: (v if v != "" else None) for k, v in item.items()}
                for item in line_items
            ],
        }

    def _llm_extract_direct(
        self, text: str, doc_type: str
    ) -> Optional[Dict[str, Any]]:
        """Fallback: Use AgentNick LLM to extract directly from text."""
        try:
            from services.direct_extraction_service import (
                DirectExtractionService,
                TABLE_SCHEMAS,
            )

            schema = TABLE_SCHEMAS.get(doc_type)
            if not schema:
                return None

            service = DirectExtractionService(self._agent_nick)
            result = service._llm_extract(text, doc_type, schema)
            if result:
                result["_source_text"] = text
            return result
        except Exception:
            logger.exception("[AgentNick] Direct LLM extraction failed")
            return None

    # ------------------------------------------------------------------
    # Enrichment
    # ------------------------------------------------------------------
    @staticmethod
    def _fill_cross_refs_from_filename(
        header: Dict[str, Any], file_path: str, doc_type: str
    ) -> None:
        """Extract cross-reference IDs from filename patterns like 'for PO398434'."""
        basename = os.path.splitext(os.path.basename(file_path))[0]

        # Invoice/Quote → PO link
        if doc_type in ("Invoice", "Quote") and not header.get("po_id"):
            m = re.search(r"(?:for\s+)?PO\s*(\d{4,})", basename, re.IGNORECASE)
            if m:
                header["po_id"] = m.group(1)
                logger.info("[AgentNick] Linked %s to po_id=%s from filename", doc_type, m.group(1))

        # PO → Quote link (stored on line items, but set header hint for orchestration)
        if doc_type == "Purchase_Order" and not header.get("_quote_ref"):
            m = re.search(r"(?:for\s+)?QUT\s*([\d][\d\-]{2,})", basename, re.IGNORECASE)
            if m:
                header["_quote_ref"] = m.group(1)

    @staticmethod
    def _validate_pk_against_source(
        pk_value: str, source_text: str, doc_type: str
    ) -> bool:
        """Validate that extracted PK actually appears in the source document.

        Format-agnostic: instead of regex patterns, simply checks whether
        the PK string (or a normalized version) is present in the document
        text. This works for ANY document format.
        """
        if not pk_value or len(pk_value) < 3:
            return False

        # Reject generic/hallucinated values
        _HALLUCINATED = {
            "invoice", "invoice1", "invoice2", "quote", "quote1",
            "po", "purchase order", "contract", "n/a", "na", "none",
            "unknown", "null", "test",
        }
        if pk_value.lower() in _HALLUCINATED:
            return False

        # Short pure digits are likely hallucinated
        if pk_value.isdigit() and len(pk_value) < 4:
            return False

        if not source_text:
            # No source text to validate against — accept if it looks reasonable
            return len(pk_value) >= 3

        # Check if PK appears in source text (case-insensitive)
        text_lower = source_text.lower()
        pk_lower = pk_value.lower()

        # Direct match
        if pk_lower in text_lower:
            return True

        # Try without whitespace/hyphens (e.g. "INV-25-058" vs "INV25058")
        pk_normalized = re.sub(r"[\s\-]", "", pk_lower)
        text_normalized = re.sub(r"[\s\-]", "", text_lower)
        if pk_normalized in text_normalized:
            return True

        # Try just the numeric portion for IDs like "INV600255"
        numeric_part = re.sub(r"[^\d]", "", pk_value)
        if numeric_part and len(numeric_part) >= 4 and numeric_part in source_text:
            return True

        return False

    @staticmethod
    def _derive_fields(header: Dict[str, Any], doc_type: str) -> None:
        """Derive computable fields from document content.

        Derives values ONLY when the source document provides enough
        information to compute them accurately. Falls back to business
        defaults only when the document has no payment/date information.

        Derivations:
        - due_date from payment_terms + invoice_date
        - validity_date from quote_date + "valid for X days"
        - tax_amount from total_amount * tax_percent / 100 (when missing)
        - total_amount_incl_tax from total_amount + tax_amount (when missing)
        - country/region from address fields
        """
        from datetime import timedelta

        source_text = str(header.get("_source_notes", "") or "")

        # ---- INVOICE DERIVATIONS ----
        if doc_type == "Invoice":
            inv_date_str = header.get("invoice_date")

            # Derive due_date from payment_terms
            if not header.get("due_date"):
                terms = str(header.get("payment_terms", "") or "").strip()
                days = None

                if terms:
                    # "Net 30", "Net 60", "Net 90"
                    m = re.search(r"[Nn]et\s+(\d+)", terms)
                    if m:
                        days = int(m.group(1))
                    # "30 days", "60 days", "14 days"
                    if not days:
                        m = re.search(r"(\d+)\s*[Dd]ays?", terms)
                        if m:
                            days = int(m.group(1))
                    # "within 30 days"
                    if not days:
                        m = re.search(r"within\s+(\d+)\s*[Dd]ays?", terms)
                        if m:
                            days = int(m.group(1))
                    # "Pay by {date}" — extract explicit date
                    if not days:
                        m = re.search(
                            r"[Pp]ay\s+by\s+(\d{1,2}\s+\w+\s+\d{4})", terms
                        )
                        if m:
                            try:
                                from dateutil.parser import parse as dateparse
                                due = dateparse(m.group(1), dayfirst=True)
                                header["due_date"] = due.strftime("%Y-%m-%d")
                                logger.info(
                                    "[Derive] due_date=%s from terms='%s'",
                                    header["due_date"], terms,
                                )
                            except Exception:
                                pass
                    # "Due on receipt" / "immediately"
                    if not days and not header.get("due_date"):
                        if "receipt" in terms.lower() or "immediate" in terms.lower():
                            if inv_date_str:
                                header["due_date"] = inv_date_str
                                logger.info("[Derive] due_date=%s (due on receipt)", inv_date_str)

                # Also check source text for payment info
                if not days and not header.get("due_date") and source_text:
                    m = re.search(r"[Nn]et\s+(\d+)", source_text)
                    if m:
                        days = int(m.group(1))
                        if not terms:
                            header["payment_terms"] = f"Net {days}"
                    if not days:
                        m = re.search(r"(\d+)\s*[Dd]ays?", source_text)
                        if m:
                            days = int(m.group(1))
                            if not terms:
                                header["payment_terms"] = f"{days} Days"

                # Compute due_date from days + invoice_date
                if days and inv_date_str and not header.get("due_date"):
                    try:
                        from datetime import datetime as dt
                        inv_date = dt.strptime(str(inv_date_str)[:10], "%Y-%m-%d")
                        due = inv_date + timedelta(days=days)
                        header["due_date"] = due.strftime("%Y-%m-%d")
                        logger.info(
                            "[Derive] due_date=%s from invoice_date + %d days",
                            header["due_date"], days,
                        )
                    except Exception:
                        pass

                # Business default: invoice_date + 90 days when NO payment info at all
                if not header.get("due_date") and inv_date_str:
                    try:
                        from datetime import datetime as dt
                        inv_date = dt.strptime(str(inv_date_str)[:10], "%Y-%m-%d")
                        due = inv_date + timedelta(days=90)
                        header["due_date"] = due.strftime("%Y-%m-%d")
                        if not header.get("payment_terms"):
                            header["payment_terms"] = "Net 90"
                        logger.info(
                            "[Derive] due_date=%s (default: invoice_date + 90 days)",
                            header["due_date"],
                        )
                    except Exception:
                        pass

        # ---- QUOTE DERIVATIONS ----
        if doc_type == "Quote":
            if not header.get("validity_date"):
                quote_date_str = header.get("quote_date")
                if quote_date_str:
                    # Check source for "valid for X days"
                    m = re.search(
                        r"[Vv]alid\s+(?:for\s+)?(\d+)\s*[Dd]ays?",
                        source_text,
                    )
                    if m:
                        try:
                            from datetime import datetime as dt
                            qd = dt.strptime(str(quote_date_str)[:10], "%Y-%m-%d")
                            vd = qd + timedelta(days=int(m.group(1)))
                            header["validity_date"] = vd.strftime("%Y-%m-%d")
                            logger.info(
                                "[Derive] validity_date=%s from quote_date + %s days",
                                header["validity_date"], m.group(1),
                            )
                        except Exception:
                            pass

        # ---- COMMON DERIVATIONS (all doc types) ----

        # Derive tax_amount when we have total_amount and tax_percent
        subtotal_field = "invoice_amount" if doc_type == "Invoice" else "total_amount"
        total_field = "invoice_total_incl_tax" if doc_type == "Invoice" else "total_amount_incl_tax"
        subtotal = header.get(subtotal_field)
        tax_pct = header.get("tax_percent")
        tax_amt = header.get("tax_amount")
        total_incl = header.get(total_field)

        if subtotal is not None and tax_pct is not None and tax_amt is None:
            try:
                derived_tax = round(float(subtotal) * float(tax_pct) / 100, 2)
                header["tax_amount"] = derived_tax
                logger.info("[Derive] tax_amount=%s from %s × %s%%", derived_tax, subtotal, tax_pct)
            except (ValueError, TypeError):
                pass

        # Derive total_amount_incl_tax
        if subtotal is not None and header.get("tax_amount") is not None and total_incl is None:
            try:
                derived_total = round(float(subtotal) + float(header["tax_amount"]), 2)
                header[total_field] = derived_total
                logger.info("[Derive] %s=%s", total_field, derived_total)
            except (ValueError, TypeError):
                pass

        # Derive subtotal from total_incl_tax - tax_amount
        if subtotal is None and total_incl is not None and tax_amt is not None:
            try:
                derived_sub = round(float(total_incl) - float(tax_amt), 2)
                header[subtotal_field] = derived_sub
                logger.info("[Derive] %s=%s from total - tax", subtotal_field, derived_sub)
            except (ValueError, TypeError):
                pass

        # ---- ADDRESS DERIVATIONS (PO) ----
        if doc_type == "Purchase_Order":
            postal = str(header.get("postal_code", "") or "")
            addr1 = str(header.get("delivery_address_line1", "") or "")
            addr2 = str(header.get("delivery_address_line2", "") or "")
            city = str(header.get("delivery_city", "") or "")

            # Derive ship_to_country from postcode pattern
            if not header.get("ship_to_country") and postal:
                if re.match(r"[A-Z]{1,2}\d{1,2}\s?\d[A-Z]{2}", postal, re.I):
                    header["ship_to_country"] = "United Kingdom"
                    logger.info("[Derive] ship_to_country=United Kingdom from UK postcode %s", postal)

            # delivery_region is derived later by field_recovery using the
            # delivery postcode itself — that's the only reliable signal,
            # since UK county names appear in many address lines (supplier
            # AND buyer) and a keyword scan picks the wrong one.

    @staticmethod
    def _extract_supplier_from_filename(file_path: str) -> Optional[str]:
        """Extract supplier name from filename — it's always the leading portion."""
        basename = os.path.splitext(os.path.basename(file_path))[0]
        # Remove trailing "for POxxxx" / "for QUTxxxx"
        basename = re.sub(r"\s+for\s+(?:PO|QUT)\S*\s*$", "", basename, flags=re.IGNORECASE).strip()
        # Remove the document ID (INV/PO/QUT + number)
        supplier_part = re.sub(
            r"\s*(?:INV|PO|QUT)[\-\s]*[\w\-]+\s*$", "", basename, flags=re.IGNORECASE
        ).strip()
        # Clean up trailing numbers/punctuation
        supplier_part = re.sub(r"\s*\d+\s*$", "", supplier_part).strip()
        if supplier_part and len(supplier_part) > 2:
            return supplier_part.title()
        return None

    @staticmethod
    def _extract_pk_from_filename(file_path: str, doc_type: str) -> Optional[str]:
        """Extract document PK from filename, preserving the alphabetic prefix.

        Patterns are written so group(1) is the FULL identifier including its
        alphabetic prefix (PO, INV, QUT, …). Prefix-less captures break
        cross-references — INV600820 must not become "600820".
        """
        basename = os.path.splitext(os.path.basename(file_path))[0]
        patterns = {
            "Invoice": [
                r"(INV[\-\s]?[\w\-]+)",          # INV-123, INV123, INV 123-456
                r"(Invoice[\-\s]*\d{3,})",        # Invoice-001
                r"(BILL[\-\s]*\d{3,})",           # BILL-001
            ],
            "Purchase_Order": [
                r"(PO[\-\s]?\d{4,})",             # PO526809, PO 526809  (FULL match w/ prefix)
                r"(PUR[\-\s]*\d{3,})",            # PUR-001
            ],
            "Quote": [
                r"(QUT[\-\s]?[\d][\d\-]{2,})",    # QUT30746, QUT-25-032
                r"(QTE[\-\s]?[\d][\d\-]{2,})",    # QTE-2026-00487
                r"(QUOTE[\-\s]?[\d][\d\-]{2,})",  # QUOTE-123
                r"(Q\d{4,})",                     # Q10483
            ],
        }
        for pat in patterns.get(doc_type, []):
            m = re.search(pat, basename, re.IGNORECASE)
            if m:
                pk = re.sub(r"\s+", "", m.group(1).strip())
                if pk.lower() in ("invoice", "quote", "po", "purchase"):
                    continue
                return pk
        return None

    def _resolve_buyer(self, header: Dict[str, Any]) -> None:
        """Resolve buyer to canonical SUP-* id.

        The LangExtract narrative pass surfaces ``buyer_name`` (a verbatim
        company name from the document body). The schema only has
        ``buyer_id`` so we must look up an existing canonical row and
        rewrite the field. If no canonical match exists, auto-create one
        — buyers and suppliers share the same bp_supplier table because
        they are all parties in the procurement graph.

        Mutates ``header`` in place. Never raises.
        """
        # Source of truth: prefer the langextract-grounded buyer_name,
        # fall back to whatever was extracted into buyer_id raw.
        raw = (
            (header.get("buyer_name") or "").strip()
            or (header.get("buyer_id") or "").strip()
        )
        if not raw:
            return
        # If buyer_id is already a canonical SUP- match, leave it alone.
        existing_id = (header.get("buyer_id") or "").strip()
        if existing_id.startswith("SUP-") and len(existing_id) >= 6:
            return
        # Reject obvious garbage early — sanitizer would null these
        # anyway, but doing it here avoids a needless DB round-trip.
        try:
            from services.extraction_sanitizer import is_garbage_party_name
            if is_garbage_party_name(raw):
                return
        except Exception:
            pass

        try:
            conn = self._agent_nick.get_db_connection()
            try:
                with conn.cursor() as cur:
                    # Exact case-insensitive match
                    cur.execute(
                        "SELECT supplier_id FROM proc.bp_supplier "
                        "WHERE LOWER(supplier_name)=LOWER(%s) OR LOWER(trading_name)=LOWER(%s) "
                        "LIMIT 1",
                        (raw, raw),
                    )
                    row = cur.fetchone()
                    if row:
                        header["buyer_id"] = row[0]
                        return
                    # Fuzzy match (handles "Assurity" → "Assurity Ltd")
                    cur.execute(
                        "SELECT supplier_id FROM proc.bp_supplier "
                        "WHERE supplier_name ILIKE %s OR trading_name ILIKE %s "
                        "LIMIT 1",
                        (f"%{raw}%", f"%{raw}%"),
                    )
                    row = cur.fetchone()
                    if row:
                        header["buyer_id"] = row[0]
                        return
                    # No canonical match — create one. Reuse the supplier
                    # canonicalization helpers so buyer SUP-IDs follow the
                    # same naming rules.
                    canonical = self._canonical_supplier_key(raw)
                    if not canonical:
                        return
                    slug = re.sub(r"\s+", "", canonical.title())[:40] or "Unknown"
                    new_id = f"SUP-{slug}"
                    cur.execute(
                        """
                        INSERT INTO proc.bp_supplier
                            (supplier_id, supplier_name, trading_name, default_currency,
                             country, created_date, created_by)
                        VALUES (%s, %s, %s, %s, %s, NOW(), %s)
                        ON CONFLICT DO NOTHING
                        """,
                        (new_id, raw, raw, header.get("currency"),
                         header.get("country"), "AgentNick-AutoBuyer"),
                    )
                    header["buyer_id"] = new_id
                    logger.info(
                        "[AgentNick] Auto-created buyer: %s → %s", raw, new_id,
                    )
            finally:
                conn.close()
        except Exception:
            logger.debug("buyer resolution failed", exc_info=True)

    def _resolve_supplier(
        self, header: Dict[str, Any], doc_type: str
    ) -> Dict[str, Any]:
        """Cross-reference supplier against bp_supplier table."""
        supplier_field = "supplier_id" if doc_type != "Purchase_Order" else "supplier_name"
        raw_name = header.get(supplier_field, "") or header.get("supplier_name", "") or header.get("supplier_id", "")

        if not raw_name:
            return header

        try:
            conn = self._agent_nick.get_db_connection()
            try:
                with conn.cursor() as cur:
                    # Try exact match
                    cur.execute(
                        "SELECT supplier_id, supplier_name FROM proc.bp_supplier "
                        "WHERE LOWER(supplier_name) = LOWER(%s) OR LOWER(trading_name) = LOWER(%s) "
                        "LIMIT 1",
                        (raw_name, raw_name),
                    )
                    row = cur.fetchone()
                    if row:
                        header["supplier_id"] = row[0]
                        logger.info(
                            "[AgentNick] Supplier resolved: '%s' → id=%s",
                            raw_name, row[0],
                        )
                    else:
                        # Try fuzzy match
                        cur.execute(
                            "SELECT supplier_id, supplier_name FROM proc.bp_supplier "
                            "WHERE supplier_name ILIKE %s OR trading_name ILIKE %s "
                            "LIMIT 1",
                            (f"%{raw_name}%", f"%{raw_name}%"),
                        )
                        row = cur.fetchone()
                        if row:
                            header["supplier_id"] = row[0]
                            logger.info(
                                "[AgentNick] Supplier fuzzy-matched: '%s' → id=%s",
                                raw_name, row[0],
                            )
                        else:
                            # No match in bp_supplier — set supplier_id
                            # from extracted name but do NOT overwrite
                            # supplier_name
                            if not header.get("supplier_id"):
                                header["supplier_id"] = raw_name
            finally:
                conn.close()
        except Exception:
            logger.debug("[AgentNick] Supplier resolution failed", exc_info=True)

        # Ensure supplier_id is always populated
        if not header.get("supplier_id") and header.get("supplier_name"):
            header["supplier_id"] = header["supplier_name"]

        return header

    @staticmethod
    def _flag_source_issues(
        header: Dict[str, Any],
        line_items: List[Dict[str, Any]],
        doc_type: str,
    ) -> List[str]:
        """Comprehensive source data integrity audit — flag anomalies, NEVER modify.

        Logs every data quality issue found in the extracted values so
        humans can investigate. Values are preserved exactly as extracted.
        """
        warnings: List[str] = []

        def _f(v) -> float:
            try:
                return float(v) if v else 0.0
            except (ValueError, TypeError):
                return 0.0

        amt_key = "invoice_amount" if doc_type == "Invoice" else "total_amount"
        tax_key = "tax_amount"
        total_key = (
            "invoice_total_incl_tax" if doc_type == "Invoice"
            else "total_amount_incl_tax" if doc_type == "Quote"
            else "total_amount"
        )

        amt = _f(header.get(amt_key))
        tax = _f(header.get(tax_key))
        total = _f(header.get(total_key))

        # --- Amount arithmetic check ---
        if amt > 0 and tax > 0 and total > 0:
            expected = round(amt + tax, 2)
            if abs(total - expected) > 1.0:
                warnings.append(
                    f"AMOUNT_MISMATCH: {total_key}={total} but "
                    f"{amt_key}({amt}) + {tax_key}({tax}) = {expected}"
                )

        # --- Tax/Amount swap indicator ---
        if tax > amt > 0:
            warnings.append(
                f"TAX_EXCEEDS_SUBTOTAL: {tax_key}({tax}) > {amt_key}({amt}) — "
                f"possible field swap in source document"
            )

        # --- Tax percent anomaly ---
        tax_pct = _f(header.get("tax_percent"))
        if tax_pct > 30:
            actual_pct = round((tax / amt) * 100, 1) if amt > 0 and tax > 0 else 0
            warnings.append(
                f"TAX_PERCENT_HIGH: tax_percent={tax_pct}% (computed from amounts: {actual_pct}%) — "
                f"may be misread from document"
            )

        # --- Date logic checks ---
        invoice_date = header.get("invoice_date") or header.get("order_date") or header.get("quote_date")
        due_date = header.get("due_date") or header.get("validity_date") or header.get("expected_delivery_date")
        if invoice_date and due_date and str(due_date) < str(invoice_date):
            warnings.append(
                f"DATE_LOGIC: due/validity date ({due_date}) is before issue date ({invoice_date})"
            )

        # --- Missing critical amounts ---
        if amt == 0 and total == 0:
            warnings.append("MISSING_AMOUNTS: both subtotal and total are zero/missing")
        elif amt == 0 and total > 0:
            warnings.append(f"MISSING_SUBTOTAL: {amt_key} is missing, only total available")

        # --- Line items sum vs header subtotal ---
        if line_items and amt > 0:
            line_sum = sum(
                _f(li.get("line_amount") or li.get("line_total"))
                for li in line_items
            )
            if line_sum > 0 and abs(line_sum - amt) > 1.0:
                warnings.append(
                    f"LINE_SUM_MISMATCH: line items total ({line_sum:.2f}) != {amt_key} ({amt:.2f})"
                )

        # --- Line item math checks ---
        for i, li in enumerate(line_items):
            qty = _f(li.get("quantity"))
            price = _f(li.get("unit_price"))
            line_amt = _f(li.get("line_amount") or li.get("line_total"))
            desc = str(li.get("item_description", ""))[:50]

            if qty > 0 and price > 0 and line_amt > 0:
                expected = round(qty * price, 2)
                if abs(expected - line_amt) > 1.0:
                    warnings.append(
                        f"LINE_MATH: item '{desc}' qty({qty}) x price({price}) = {expected} but amount={line_amt}"
                    )

            if qty > 10000:
                warnings.append(f"QUANTITY_HIGH: item '{desc}' has qty={qty} — verify this is a count")

            if not desc and (qty > 0 or price > 0):
                warnings.append(f"LINE_INCOMPLETE: line {i+1} has amounts but no description")

        # --- Supplier sanity check ---
        supplier = header.get("supplier_id", "") or header.get("supplier_name", "")
        if supplier and len(str(supplier)) < 3:
            warnings.append(f"SUPPLIER_SHORT: supplier_id='{supplier}' is suspiciously short")

        return warnings

    @staticmethod
    def _cross_validate(
        header: Dict[str, Any],
        line_items: List[Dict[str, Any]],
        doc_type: str,
    ) -> tuple:
        """Validate extracted data integrity — LOG anomalies, NEVER modify source values.

        Principle: Values from the document are persisted exactly as extracted.
        Only derive values when they are completely ABSENT from the document.
        Inaccurate data is logged to discrepancy records for human review.
        """

        def _f(v) -> float:
            try:
                return float(v) if v else 0.0
            except (ValueError, TypeError):
                return 0.0

        amt_key = "invoice_amount" if doc_type == "Invoice" else "total_amount"
        tax_key = "tax_amount"
        total_key = (
            "invoice_total_incl_tax" if doc_type == "Invoice"
            else "total_amount_incl_tax" if doc_type == "Quote"
            else "total_amount"
        )

        amt = _f(header.get(amt_key))
        tax = _f(header.get(tax_key))
        total = _f(header.get(total_key))

        # --- LOG anomalies but DO NOT modify source values ---
        if tax > amt > 0:
            logger.warning(
                "[CrossVal] ANOMALY: tax_amount(%.2f) > %s(%.2f) — possible swap in source document",
                tax, amt_key, amt,
            )

        if amt > 0 and tax > 0 and total_key != amt_key:
            expected_total = round(amt + tax, 2)
            if total > 0 and abs(total - expected_total) > 0.50:
                logger.warning(
                    "[CrossVal] ANOMALY: %s(%.2f) != %s(%.2f) + %s(%.2f) = %.2f",
                    total_key, total, amt_key, amt, tax_key, tax, expected_total,
                )

        if header.get("tax_percent"):
            tax_pct = _f(header.get("tax_percent"))
            if tax_pct > 30:
                logger.warning(
                    "[CrossVal] ANOMALY: tax_percent=%.1f%% is unusually high — verify source document",
                    tax_pct,
                )

        # --- Line item anomaly logging ---
        for item in line_items:
            qty = _f(item.get("quantity"))
            price = _f(item.get("unit_price"))
            line_amt = _f(item.get("line_amount") or item.get("line_total"))

            if qty > 0 and price > 0 and line_amt > 0:
                expected = round(qty * price, 2)
                if abs(expected - line_amt) > 1.0:
                    logger.warning(
                        "[CrossVal] ANOMALY: line item '%s' qty(%.0f) x price(%.2f) = %.2f but amount=%.2f",
                        str(item.get("item_description", ""))[:40], qty, price, expected, line_amt,
                    )

        # --- DERIVE only when value is completely ABSENT ---
        # tax_percent: derive from amounts ONLY if not present in document
        if not header.get("tax_percent") and amt > 0 and tax > 0:
            header["tax_percent"] = round((tax / amt) * 100, 2)
            logger.info("[CrossVal] Derived missing tax_percent: %.2f%%", header["tax_percent"])

        # total_incl_tax: derive ONLY if absent
        if total_key != amt_key:
            if not header.get(total_key) and amt > 0 and tax > 0:
                header[total_key] = round(amt + tax, 2)
                logger.info("[CrossVal] Derived missing %s: %.2f", total_key, header[total_key])
            elif not header.get(tax_key) and total > 0 and amt > 0:
                header[tax_key] = round(total - amt, 2)
                logger.info("[CrossVal] Derived missing %s: %.2f", tax_key, header[tax_key])
            elif not header.get(amt_key) and total > 0 and tax > 0:
                header[amt_key] = round(total - tax, 2)
                logger.info("[CrossVal] Derived missing %s: %.2f", amt_key, header[amt_key])

        # line_amount: derive ONLY if absent
        for item in line_items:
            qty = _f(item.get("quantity"))
            price = _f(item.get("unit_price"))
            has_amt = item.get("line_amount") or item.get("line_total")
            if qty > 0 and price > 0 and not has_amt:
                calculated = round(qty * price, 2)
                item["line_amount"] = calculated
                logger.info("[CrossVal] Derived missing line_amount: %.2f", calculated)

        # Currency: default GBP only if completely absent
        if not header.get("currency"):
            header["currency"] = "GBP"

        return header, line_items

    # External Docwain API for high-accuracy extraction of structured formats
    DOCWAIN_URL = os.getenv("DOCWAIN_API_URL", "http://198.145.127.234:8000/api/v1/docwain/extract")
    DOCWAIN_KEY = os.getenv("DOCWAIN_API_KEY", "dw_0b40bd8bb676e2dec6dc63134e45154db4111d5302be094d")

    def _docwain_extract(
        self, file_bytes: bytes, file_path: str, doc_type: str
    ) -> Optional[Dict[str, Any]]:
        """Extract via external Docwain API — higher accuracy for Excel/DOCX."""
        if not self.DOCWAIN_KEY:
            return None

        fname = os.path.basename(file_path)
        schema_hints = {
            "Quote": "quote_id, supplier_name, supplier_address, buyer_name, buyer_address, quote_date, validity_date, currency, payment_terms, subtotal (before tax), tax_percent, tax_amount, total_incl_tax, vat_registration, company_registration, account_manager, phone, email",
            "Invoice": "invoice_id, supplier_name, supplier_address, po_id, invoice_date, due_date, currency, invoice_amount (subtotal), tax_percent, tax_amount, total_incl_tax, payment_terms, phone, email",
            "Purchase_Order": "po_id, supplier_name, supplier_address, buyer_name, buyer_address, order_date, currency, subtotal, tax_amount, total_amount, payment_terms, delivery_address",
        }
        prompt = (
            f"Extract ALL fields from this {doc_type} as JSON:\n"
            f"Header: {schema_hints.get(doc_type, '')}\n"
            f"Line items array: line_no, item_description, quantity, unit_price, line_total\n"
            f"CRITICAL: Extract ALL addresses, phone numbers, email addresses.\n"
            f"quantity is a count (small number), unit_price is cost per item. tax_amount < subtotal."
        )

        try:
            resp = requests.post(
                self.DOCWAIN_URL,
                headers={"X-Api-Key": self.DOCWAIN_KEY},
                files={"file": (fname, file_bytes)},
                data={"mode": "entities", "output_format": "json", "prompt": prompt},
                timeout=120,
            )
            data = resp.json()
            if "error" in data:
                logger.debug("[Docwain] Error: %s", data["error"])
                return None

            result = data.get("result", {})
            if not result:
                return None

            # Map Docwain result to our standard format
            header = {k: v for k, v in result.items() if k != "line_items"}
            line_items = result.get("line_items", [])

            # Normalize field names to match our schema
            field_map = {
                "supplier_name": "supplier_id" if doc_type != "Purchase_Order" else "supplier_name",
                "buyer_name": "buyer_id",
                "subtotal": "invoice_amount" if doc_type == "Invoice" else "total_amount",
                "total_incl_tax": "invoice_total_incl_tax" if doc_type == "Invoice" else "total_amount_incl_tax",
            }
            for old_key, new_key in field_map.items():
                if old_key in header and old_key != new_key:
                    header[new_key] = header.pop(old_key)

            # Ensure supplier_name is preserved alongside supplier_id
            if "supplier_id" in header and "supplier_name" not in header:
                header["supplier_name"] = header["supplier_id"]

            # Generate item_id for each line item
            from agents.extraction_engine import _generate_item_id
            for li in line_items:
                desc = li.get("item_description", "")
                if desc and not li.get("item_id"):
                    li["item_id"] = _generate_item_id(desc)

            logger.info(
                "[Docwain] Extracted %d header fields, %d line items from %s",
                len(header), len(line_items), fname,
            )
            return {"header": header, "line_items": line_items, "_source_text": ""}

        except Exception:
            logger.debug("[Docwain] Request failed", exc_info=True)
            return None

    def _llm_extract_line_items(
        self, text: str, doc_type: str, header: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Focused LLM call to extract ONLY line items when header succeeded but items are empty."""
        prompt = (
            f"Extract ALL line items from this {doc_type} document.\n"
            f"Return a JSON array of objects with: line_no, item_description, quantity, unit_price, line_total\n"
            f"RULES:\n"
            f"- quantity is a COUNT (small number like 1, 2, 5, 10, 100)\n"
            f"- unit_price is COST PER ITEM (£ amount)\n"
            f"- line_total: extract the ACTUAL value from the document, do NOT compute\n"
            f"- Stop at Subtotal/Tax/Total rows — those are NOT line items\n"
            f"- Return ONLY the JSON array, nothing else\n\n"
            f"Document:\n{text}"
        )
        try:
            from services.ollama_client import ollama_generate
            raw = ollama_generate(
                prompt,
                model=AGENT_NICK_MODEL,
                num_predict=4096,
                timeout=600,
            )
            if not raw:
                return []
            # Parse JSON array
            import re as _re
            arr_match = _re.search(r"\[[\s\S]*\]", raw)
            if arr_match:
                items = json.loads(arr_match.group())
                if isinstance(items, list) and items:
                    logger.info("[AgentNick] Focused LLM extracted %d line items", len(items))
                    return items
        except Exception:
            logger.debug("[AgentNick] Focused line item extraction failed", exc_info=True)
        return []

    def _llm_extract_tabular(
        self, text: str, doc_type: str
    ) -> Optional[Dict[str, Any]]:
        """Extract structured data from Excel/CSV text using LLM directly."""
        schema_hints = {
            "Quote": "quote_id, supplier_id, supplier_address, buyer_id, buyer_address, quote_date, validity_date, currency, total_amount (subtotal before tax), tax_percent, tax_amount, total_amount_incl_tax, payment_terms",
            "Invoice": "invoice_id, supplier_id, supplier_address, po_id, invoice_date, due_date, currency, invoice_amount (subtotal), tax_percent, tax_amount, invoice_total_incl_tax, payment_terms",
            "Purchase_Order": "po_id, supplier_name, supplier_address, buyer_id, buyer_address, order_date, currency, total_amount (the FINAL total from summary section — look for Subtotal/Total rows at the bottom), tax_amount, payment_terms",
        }
        fields = schema_hints.get(doc_type, "")
        prompt = (
            f"Extract ALL data from this {doc_type} document. Return ONLY valid JSON.\n\n"
            f"Header fields: {fields}\n"
            f"Line items: array of {{line_no, item_description, quantity, unit_price, line_total}}\n\n"
            f"RULES:\n"
            f"- VENDOR section contains the supplier name and address\n"
            f"- BILL TO section contains the buyer name and address (NOT the supplier)\n"
            f"- Look for Subtotal/Total rows in a SEPARATE summary table — this is the total_amount\n"
            f"- For POs: total_amount is the final total from the summary section, NOT a line item\n"
            f"- Extract ALL line items from the line items table (skip header rows and summary rows)\n"
            f"- tax_amount must be LESS than subtotal\n"
            f"- If source has data issues (empty totals, orphaned rows), extract what IS there\n\n"
            f"Document content:\n{text[:6000]}"
        )
        try:
            from services.ollama_client import ollama_generate
            raw = ollama_generate(
                prompt,
                model=AGENT_NICK_MODEL,
                num_predict=2048,
            )
            if not raw:
                return None
            import re as _re
            json_match = _re.search(r"\{[\s\S]*\}", raw)
            if json_match:
                data = json.loads(json_match.group())
                header = data.get("header", {k: v for k, v in data.items() if k != "line_items"})
                line_items = data.get("line_items", [])
                header["_source_text"] = text[:3000]
                return {
                    "header": header,
                    "line_items": line_items,
                    "_source_text": text[:3000],
                }
        except Exception:
            logger.exception("[AgentNick] Tabular LLM extraction failed")
        return None

    def _llm_fill_gaps(
        self,
        header: Dict[str, Any],
        missing: List[str],
        text: str,
        doc_type: str,
    ) -> Dict[str, Any]:
        """Ask AgentNick LLM to fill missing required fields."""
        if not text or not missing:
            return header

        fields_desc = ", ".join(missing)
        prompt = (
            f"Extract these specific fields from the {doc_type} document below.\n"
            f"Fields needed: {fields_desc}\n\n"
            f"Return ONLY a JSON object with the field names as keys.\n"
            f"Dates in YYYY-MM-DD format. Amounts as numbers.\n"
            f"Currency as 3-letter ISO code.\n\n"
            f"Document:\n{text[:6000]}"
        )

        try:
            from services.ollama_client import ollama_generate
            raw = ollama_generate(
                prompt,
                model=AGENT_NICK_MODEL,
                num_predict=512,
            )
            if not raw:
                return header

            # Parse JSON from response
            cleaned = re.sub(r"^```(?:json)?\s*\n?", "", raw)
            cleaned = re.sub(r"\n?```\s*$", "", cleaned)
            match = re.search(r"\{[\s\S]*?\}", cleaned)
            if match:
                data = json.loads(match.group())
                for field, value in data.items():
                    if field in missing and value is not None and str(value).strip():
                        header[field] = value
                        logger.info(
                            "[AgentNick] Filled gap: %s = %s", field, value
                        )
        except Exception:
            logger.debug("[AgentNick] LLM gap-fill failed", exc_info=True)

        return header

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _persist_header(
        self, header: Dict[str, Any], doc_type: str
    ) -> bool:
        """Persist header to bp_ table using DirectExtractionService."""
        try:
            from services.direct_extraction_service import (
                DirectExtractionService,
                TABLE_SCHEMAS,
            )

            schema = TABLE_SCHEMAS.get(doc_type)
            if not schema:
                return False

            service = DirectExtractionService(self._agent_nick)
            audit = {
                "created_date": header.get("created_date"),
                "created_by": header.get("created_by"),
                "last_modified_by": header.get("last_modified_by"),
                "last_modified_date": header.get("last_modified_date"),
            }
            return service._persist_header(header, schema, audit)
        except Exception:
            logger.exception("[AgentNick] Header persistence failed")
            return False

    def _persist_line_items(
        self, line_items: List[Dict], doc_type: str, pk_value: str
    ) -> int:
        """Persist line items to bp_ line table."""
        if not line_items:
            return 0
        try:
            from services.direct_extraction_service import (
                DirectExtractionService,
                TABLE_SCHEMAS,
            )

            schema = TABLE_SCHEMAS.get(doc_type)
            if not schema or not schema.get("line_table"):
                return 0

            service = DirectExtractionService(self._agent_nick)
            audit = {}
            if line_items:
                audit = {
                    "created_date": line_items[0].get("created_date"),
                    "created_by": line_items[0].get("created_by"),
                    "last_modified_by": line_items[0].get("last_modified_by"),
                    "last_modified_date": line_items[0].get("last_modified_date"),
                }
            return service._persist_line_items(line_items, schema, pk_value, audit)
        except Exception:
            logger.exception("[AgentNick] Line items persistence failed")
            return 0

    def _learn_vendor_profile(
        self, header: Dict[str, Any], doc_type: str
    ) -> None:
        """Auto-learn vendor extraction patterns."""
        try:
            from services.vendor_profile_service import VendorProfileService

            supplier_name = header.get("supplier_name", "") or header.get("supplier_id", "")
            if not supplier_name or not doc_type:
                return

            vps = VendorProfileService(self._agent_nick)
            vps.learn_from_extraction(
                supplier_name=supplier_name,
                supplier_id=header.get("supplier_id", ""),
                doc_type=doc_type,
                currency_hint=header.get("currency", ""),
            )
        except Exception:
            logger.debug("[AgentNick] Vendor profile learning failed", exc_info=True)

        # Also update the v2 vendor profile table
        self._update_vendor_profile(header, doc_type)

    def _get_vendor_context(self, supplier_hint: str) -> str:
        """Look up vendor profile and return context string for LLM prompt."""
        if not supplier_hint or len(supplier_hint) < 3:
            return ""
        try:
            conn = self._agent_nick.get_db_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT profile_data, extraction_count, avg_confidence "
                        "FROM proc.vendor_profile WHERE supplier_name ILIKE %s LIMIT 1",
                        (f"%{supplier_hint}%",),
                    )
                    row = cur.fetchone()
                    if not row:
                        return ""
                    profile = row[0] if isinstance(row[0], dict) else {}
                    count = row[1] or 0
                    if count < 2:
                        return ""
                    parts = [f"VENDOR CONTEXT (from {count} previous extractions):"]
                    if profile.get("default_currency"):
                        parts.append(f"- Currency: {profile['default_currency']}")
                    if profile.get("typical_tax_rate"):
                        parts.append(f"- Typical tax rate: {profile['typical_tax_rate']}%")
                    if profile.get("date_format_hint"):
                        parts.append(f"- Date format: {profile['date_format_hint']}")
                    if profile.get("id_pattern"):
                        parts.append(f"- Document ID pattern: {profile['id_pattern']}")
                    return "\n".join(parts)
            finally:
                conn.close()
        except Exception:
            logger.debug("Vendor profile lookup failed", exc_info=True)
            return ""

    def _update_vendor_profile(self, header: Dict[str, Any], doc_type: str) -> None:
        """Update v2 vendor profile with patterns from this extraction."""
        import json as _json
        supplier = header.get("supplier_id") or header.get("supplier_name") or ""
        if not supplier or len(supplier) < 3:
            return
        try:
            conn = self._agent_nick.get_db_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT profile_data, extraction_count, avg_confidence "
                        "FROM proc.vendor_profile WHERE supplier_name = %s",
                        (supplier,),
                    )
                    row = cur.fetchone()
                    profile = row[0] if row and isinstance(row[0], dict) else {}
                    count = (row[1] or 0) if row else 0
                    old_conf = float(row[2] or 0) if row else 0.0

                    profile["default_currency"] = header.get("currency") or profile.get("default_currency")
                    if header.get("tax_percent"):
                        try:
                            profile["typical_tax_rate"] = float(header["tax_percent"])
                        except (ValueError, TypeError):
                            pass
                    profile["last_doc_type"] = doc_type

                    new_count = count + 1
                    conf = float(header.get("confidence_score", 0) or 0)
                    new_avg = ((old_conf * count) + conf) / new_count if new_count > 0 else 0

                    cur.execute("""
                        INSERT INTO proc.vendor_profile
                            (supplier_name, profile_data, extraction_count, avg_confidence,
                             last_extraction, last_modified_date)
                        VALUES (%s, %s, %s, %s, NOW(), NOW())
                        ON CONFLICT (supplier_name) DO UPDATE SET
                            profile_data = %s,
                            extraction_count = %s,
                            avg_confidence = %s,
                            last_extraction = NOW(),
                            last_modified_date = NOW()
                    """, (supplier, _json.dumps(profile, default=str), new_count, round(new_avg, 3),
                          _json.dumps(profile, default=str), new_count, round(new_avg, 3)))
            finally:
                conn.close()
        except Exception:
            logger.debug("Vendor profile update failed", exc_info=True)

    def _llm_salvage_line_items(
        self, source_text: str, doc_type: str, header: dict,
        *, vendor_hint: Optional[str] = None,
    ) -> list:
        """Narrow LLM pass that extracts ONLY line items from a document.

        Used when the structural extractor returns zero line items but the
        header has a meaningful total — a strong signal that lines exist
        in the document but the table detector didn't find them. The
        prompt is tightly scoped (no header fields, no narrative) so the
        model produces minimal, JSON-parseable output.

        ``vendor_hint`` is an optional vendor-specific prompt suffix
        produced by the template service when the document's fingerprint
        matches a known template. Injecting the vendor's column layout
        into the prompt is the lever that pulls 0-line-items vendors
        (NEXASPARK, AQUARIUS, NEWPORT, RUBILOGY) out of silent failure.
        """
        amt_total = (
            header.get("invoice_total_incl_tax")
            or header.get("total_amount_incl_tax")
            or header.get("total_amount")
            or header.get("invoice_amount")
        )
        # Salvage runs whenever the document looks substantive enough to
        # have line items: either a non-trivial total OR a long enough
        # source body. Previously we skipped salvage for any total < £10,
        # which silently dropped legitimate small-value invoices.
        try:
            if amt_total is not None and float(amt_total) < 1.0:
                return []
        except (ValueError, TypeError):
            pass

        # Doc-type-specific line column names
        if doc_type == "Invoice":
            line_cols = "line_no, item_description, quantity, unit_price, line_amount"
        else:
            line_cols = "line_number, item_description, quantity, unit_price, line_total"

        vendor_block = (
            f"\nVENDOR CONTEXT:\n{vendor_hint}\n" if vendor_hint else ""
        )
        prompt = f"""Extract ALL line items / products / services from this {doc_type}.

The document may be in one of THREE shapes — handle all three:

  (A) TABULAR: a clear products/services table with a header row (Description,
      Qty, Unit Price, ...). Extract every body row, skip subtotal/tax/total
      rows.

  (B) SUMMARY-STYLE (single service): the invoice describes ONE main
      service in prose (e.g. "Digital Marketing Package", "Advanced Package
      (3 months)") followed by a price (often the SUB TOTAL). This is a
      LINE ITEM. Extract it as one row: item_description from the service
      title, quantity=1, unit_price = SUBTOTAL (pre-tax) amount,
      {('line_amount' if doc_type == "Invoice" else 'line_total')} = same.

  (C) MULTI-SERVICE LIST: each service is a labelled paragraph or bullet
      point with a price. Extract each service as a separate row.

DO NOT return header fields (invoice_id, supplier, dates, totals).
DO NOT emit subtotal / tax / TOTAL lines themselves as line items.
{vendor_block}
Return this JSON shape ONLY:
{{
  "line_items": [
    {{ {line_cols} }},
    ...
  ]
}}

Rules:
- quantity is a small count (1, 2, 10). NOT a price. Default to 1 when
  the doc describes a single service without an explicit count.
- unit_price is the per-unit cost. For SUMMARY-style invoices (one
  service), unit_price = the SUB TOTAL (pre-tax) amount.
- item_description is the product/service NAME (the title of the service
  block, not the marketing prose around it).
- If a row truly has no description and no price, SKIP it.
- Only return {{ "line_items": [] }} if the document genuinely has no
  service/product described at all (rare).

DOCUMENT TEXT:
{source_text[:8000]}"""

        try:
            from services.ollama_client import ollama_generate
            import json as _json
            raw = ollama_generate(prompt, num_predict=2048)
            if not raw:
                return []
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
                cleaned = re.sub(r"\n?```\s*$", "", cleaned)
            try:
                data = _json.loads(cleaned)
            except Exception:
                m = re.search(r"\{[\s\S]*\}", cleaned)
                if not m:
                    return []
                data = _json.loads(m.group())
            items = data.get("line_items") or []
            if not isinstance(items, list):
                return []
            # Coerce and map line_total→line_amount for invoices
            out = []
            for i, li in enumerate(items, start=1):
                if not isinstance(li, dict):
                    continue
                desc = str(li.get("item_description", "")).strip()
                if not desc:
                    continue
                row = {"item_description": desc}
                if doc_type == "Invoice":
                    row["line_no"] = li.get("line_no") or i
                    for k in ("quantity", "unit_price", "line_amount"):
                        if li.get(k) is not None:
                            row[k] = li[k]
                    if "line_total" in li and "line_amount" not in row:
                        row["line_amount"] = li["line_total"]
                else:
                    row["line_number"] = li.get("line_number") or i
                    for k in ("quantity", "unit_price", "line_total"):
                        if li.get(k) is not None:
                            row[k] = li[k]
                out.append(row)
            return out
        except Exception:
            logger.debug("line-item salvage LLM call failed", exc_info=True)
            return []

    def _llm_extract_single_service(
        self, source_text: str, doc_type: str, header: dict,
    ) -> list:
        """Last-resort fallback for SUMMARY-style invoices.

        Runs ONLY when the regular salvage returned zero rows and the
        header has a non-zero total — a signal that the document is a
        single-service invoice whose ONE line item is described in prose
        rather than in a tabular structure.

        Asks the LLM a much narrower question: "what is the one service
        being invoiced and what is its pre-tax price?" Returns a single-
        element list synthesised from the response.
        """
        amt_total = (
            header.get("invoice_total_incl_tax")
            or header.get("total_amount_incl_tax")
            or header.get("total_amount")
            or header.get("invoice_amount")
        )
        try:
            if amt_total is None or float(amt_total) < 1.0:
                return []
        except (ValueError, TypeError):
            return []

        prompt = f"""This {doc_type} has NO structured line-items table.
It describes ONE main product or service in prose — your task is to find it.

Look for:
  - A service or package name (often in capitals, e.g. "DIGITAL MARKETING
    PACKAGE", "Advanced Package", "Professional Services")
  - The pre-tax amount (often labelled "SUB TOTAL" or appears next to the
    service name)

Return JSON ONLY:
{{
  "description": "the service or product name as written in the document",
  "amount":      <pre-tax amount as a number, no currency symbol>
}}

If you genuinely cannot find any service description, return {{ "description": "", "amount": 0 }}.

DOCUMENT TEXT:
{source_text[:6000]}"""
        try:
            from services.ollama_client import ollama_generate
            import json as _json
            raw = ollama_generate(prompt, num_predict=512)
            if not raw:
                return []
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
                cleaned = re.sub(r"\n?```\s*$", "", cleaned)
            try:
                data = _json.loads(cleaned)
            except Exception:
                m = re.search(r"\{[\s\S]*\}", cleaned)
                if not m:
                    return []
                data = _json.loads(m.group())
            description = str(data.get("description") or "").strip()
            try:
                amount = float(data.get("amount") or 0)
            except (ValueError, TypeError):
                amount = 0.0
            if not description or amount < 1.0:
                return []
            row = {
                "item_description": description,
                "quantity": 1,
            }
            if doc_type == "Invoice":
                row["line_no"] = 1
                row["unit_price"] = amount
                row["line_amount"] = amount
            else:
                row["line_number"] = 1
                row["unit_price"] = amount
                row["line_total"] = amount
            return [row]
        except Exception:
            logger.debug("single-service salvage LLM call failed", exc_info=True)
            return []

    # Critical fields per doc_type — when sanitizer nulls one of these,
    # the record must be reviewed by a human before downstream agents
    # (ranking, opportunities, negotiation) can trust it.
    _CRITICAL_FIELDS = {
        "Invoice": {"invoice_id", "supplier_id", "invoice_total_incl_tax"},
        "Purchase_Order": {"po_id", "supplier_id", "total_amount_incl_tax"},
        "Quote": {"quote_id", "supplier_id", "total_amount_incl_tax"},
    }

    def _enqueue_for_review_if_needed(
        self, header: dict, line_items: list, doc_type: str, file_path: str,
        sanitizer_rejections: list, source_text: str = "",
        rescued_fields: Optional[set] = None,
    ) -> None:
        """Push to extraction_review_queue when extraction quality fails an
        invariant. Idempotent — re-extraction of the same doc updates the
        existing queue row.

        Trigger conditions (any one fires):
          1. A critical field was nulled by the sanitizer.
          2. A critical field is missing from the header.
          3. ``len(line_items) == 0`` for an invoice/quote/PO whose
             total_amount is non-zero (silent line-data loss invariant).
          4. ``len(rescued_fields) >= 3`` — too many fields needed
             filename/template rescue, suggesting overall extraction
             quality is poor.
        """
        critical = self._CRITICAL_FIELDS.get(doc_type, set())
        rescued_fields = rescued_fields or set()

        # 1. Critical field nulled by sanitizer
        rejected = {r.field for r in sanitizer_rejections} & critical
        # 2. Critical field missing
        missing = {f for f in critical if header.get(f) in (None, "", 0)}

        # 3. Zero-line-items invariant for line-bearing doc types.
        # An invoice/quote/PO with a non-zero total but zero line items is
        # almost certainly a parser/prompt failure rather than a real
        # zero-line document. Mark this as a failure mode.
        zero_lines_violation = False
        if doc_type in ("Invoice", "Quote", "Purchase_Order"):
            total_value = (
                header.get("total_amount") or header.get("total_amount_incl_tax")
                or header.get("invoice_total_incl_tax") or header.get("invoice_amount")
                or 0
            )
            try:
                total_float = float(total_value or 0)
            except (ValueError, TypeError):
                total_float = 0
            if (not line_items) and total_float > 0:
                zero_lines_violation = True

        # 4. Too-many-rescues invariant
        too_many_rescues = len(rescued_fields) >= 3

        failed = sorted(rejected | missing)
        if zero_lines_violation:
            failed.append("__lines_empty_with_total")
        if too_many_rescues:
            failed.append("__excessive_rescues")
        if not failed:
            return

        try:
            import json as _json
            conn = self._agent_nick.get_db_connection()
            try:
                with conn.cursor() as cur:
                    # Get the originating process_monitor row (if any)
                    cur.execute(
                        "SELECT id FROM proc.process_monitor WHERE file_path=%s "
                        "ORDER BY id DESC LIMIT 1",
                        (file_path,),
                    )
                    pm_row = cur.fetchone()
                    pm_id = pm_row[0] if pm_row else None

                    signals = {
                        "rejected_critical_fields": sorted(rejected),
                        "missing_critical_fields": sorted(missing),
                        "rejection_reasons": [
                            {"field": r.field,
                             "reason": r.reason,
                             "extracted_value": str(r.extracted_value)[:200]
                                                  if r.extracted_value is not None else None}
                            for r in sanitizer_rejections
                        ],
                        "zero_lines_violation": zero_lines_violation,
                        "rescued_fields": sorted(rescued_fields),
                        "rescued_count": len(rescued_fields),
                    }

                    # UPSERT — same file may be re-extracted multiple times.
                    cur.execute(
                        """
                        INSERT INTO proc.extraction_review_queue
                          (process_monitor_id, file_path, doc_type, partial_header,
                           partial_line_items, failed_fields, parsed_text,
                           attempt_count, last_attempt_at, signals_json)
                        VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, 1, NOW(), %s::jsonb)
                        """,
                        (pm_id, file_path, doc_type,
                         _json.dumps(header, default=str),
                         _json.dumps(line_items, default=str),
                         failed,
                         (source_text or "")[:8000],
                         _json.dumps(signals)),
                    )
                conn.commit()
                logger.warning(
                    "[ReviewQueue] %s %s enqueued — failed critical fields: %s",
                    doc_type, file_path, failed,
                )
            finally:
                conn.close()
        except Exception:
            logger.debug("Review queue enqueue failed", exc_info=True)

    def _log_sanitizer_rejections(
        self, rejections: list, doc_type: str, pk_value: str, file_path: str,
    ) -> None:
        """Write sanitizer rejections to bp_discrepancy_data so cleanup
        actions are auditable and reviewable. Best-effort — never fails
        the extraction."""
        if not rejections:
            return
        try:
            conn = self._agent_nick.get_db_connection()
            try:
                with conn.cursor() as cur:
                    for r in rejections:
                        cur.execute(
                            "INSERT INTO proc.bp_discrepancy_data "
                            "(doc_type, record_id, field_name, rule_name, severity, "
                            " extracted_value, message, file_path) "
                            "VALUES (%s, %s, %s, 'sanitizer', %s, %s, %s, %s)",
                            (doc_type, str(pk_value or ""), r.field, r.severity,
                             str(r.extracted_value)[:500] if r.extracted_value is not None else None,
                             r.reason, file_path),
                        )
                conn.commit()
                logger.info(
                    "[Sanitizer] %s %s: nulled %d garbage field(s): %s",
                    doc_type, pk_value, len(rejections),
                    ", ".join(f"{r.field}={r.reason}" for r in rejections),
                )
            finally:
                conn.close()
        except Exception:
            logger.debug("Sanitizer rejection logging failed", exc_info=True)

    # Document-type labels and filename artifacts that the LLM sometimes
    # captures as supplier_name when no real supplier is found. Block them
    # before they pollute bp_supplier with garbage IDs.
    _SUPPLIER_GARBAGE_PATTERNS = (
        re.compile(r"^(invoice|quote|purchase\s*order|po|property\s+invoice|"
                   r"resource\s+rate\s+card|estimated)\s*$", re.I),
        re.compile(r"^(quote|invoice|po)[_\-\s]*scenario", re.I),
        re.compile(r"_watermark|_split[_\s]+tables|_duplicate", re.I),
        re.compile(r"^(invoice|quote)\s*no\s*[:.]?\s*\d+", re.I),
    )
    # Token sequences inside supplier_name that the LLM mashed together with
    # the company name (e.g. "Duncan LLC INVOICENO: 132666"). We strip them.
    _SUPPLIER_NOISE_RE = re.compile(
        r"\b(invoice\s*no\.?|quote\s*no\.?|po\s*no\.?|purchase\s*order\s*no\.?|"
        r"inv\s*no\.?|qte\s*no\.?|date|address|bill\s*to|ship\s*to)\s*[:#]?\s*[\w\-/]*",
        re.I,
    )

    @staticmethod
    def _canonical_supplier_key(name: str) -> str:
        """Normalize a supplier name for fuzzy lookup: lowercase, strip suffix
        terms (Ltd/Inc/LLC/Corp), collapse whitespace, drop punctuation."""
        s = name.lower()
        s = re.sub(r"\b(ltd\.?|limited|inc\.?|incorporated|llc|llp|corp\.?|"
                   r"corporation|plc|gmbh|s\.?a\.?|co\.?|company)\b", "", s)
        s = re.sub(r"[^a-z0-9]+", " ", s)
        return " ".join(s.split())

    @classmethod
    def _is_garbage_supplier(cls, name: str) -> bool:
        if not name or len(name.strip()) < 3:
            return True
        # Local pattern check (legacy)
        for p in cls._SUPPLIER_GARBAGE_PATTERNS:
            if p.search(name):
                return True
        # Pure digits / mostly digits — never a supplier name
        digits = sum(c.isdigit() for c in name)
        if digits and digits / len(name) > 0.5:
            return True
        # Delegate to the sanitizer's stricter check so the orchestrator
        # and sanitizer agree on what counts as a party-name garbage.
        # Catches things like:
        #   "Quote Number QUT10253 Nov 2024"
        #   "10 Redkiln Way Horsham RH13 5QH"
        #   "PO526809"
        try:
            from services.extraction_sanitizer import is_garbage_party_name
            if is_garbage_party_name(name):
                return True
        except Exception:
            pass
        # Additional: any captured supplier_name containing a recognised
        # document-id pattern (PO12345 / INV-2024 / QUT103107) is almost
        # always a label+value mash, not a real company.
        if re.search(r"\b(?:PO|INV|QUT|QTE)[\-\s]?\d{4,}\b", name, re.I):
            return True
        # Label-with-id mashes that the original regex misses ("Number"
        # synonym and prefix-id values).
        if re.match(r"^\s*(invoice|quote|po|purchase\s*order)\s*"
                    r"(?:no\.?|number|num|#|ref(?:erence)?)\s*[:#]?\s*\S+",
                    name, re.I):
            return True
        return False

    def _auto_create_supplier(self, header: Dict[str, Any], doc_type: str) -> Optional[str]:
        """Resolve to canonical supplier_id. Looks up an existing canonical
        match before creating a new row. Rejects garbage names so they never
        become SUP-IDs. Returns the canonical supplier_id (existing or new).
        """
        supplier_name = (header.get("supplier_name")
                         or header.get("supplier_id") or "").strip()

        # Short-circuit: the upstream may already have given us a canonical
        # supplier_id (starts with "SUP-"). Re-canonicalising it produces
        # garbage like "SUP-ThriveStudiosLLC" → "SUP-SupThrivestudiosllc".
        # Verify it exists in bp_supplier and reuse as-is, never re-slug.
        if supplier_name.startswith("SUP-") and len(supplier_name) >= 6:
            try:
                conn = self._agent_nick.get_db_connection()
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT 1 FROM proc.bp_supplier WHERE supplier_id=%s LIMIT 1",
                            (supplier_name,),
                        )
                        if cur.fetchone():
                            return supplier_name
                finally:
                    conn.close()
            except Exception:
                pass
            # Existing SUP- not in the table — return None so caller
            # doesn't pollute by re-slugging this canonical-shaped string.
            logger.debug(
                "[AgentNick] supplier_id %r looks canonical but isn't in bp_supplier — skipping create",
                supplier_name,
            )
            return None

        # Strip embedded label noise like "Duncan LLC INVOICENO: 132666"
        if supplier_name:
            cleaned = self._SUPPLIER_NOISE_RE.sub("", supplier_name).strip(" ,;:-")
            if cleaned and len(cleaned) >= 3:
                supplier_name = cleaned

        # Reject bank names captured from payment-detail footers.
        # The supplier on a procurement document is the COMPANY ISSUING
        # the invoice, never the bank receiving payment.
        _bank_markers = (
            "bank ", "bank,", "banking", " bank", "trust ", " trust",
            "credit union", "savings", "branch", "sort code", "iban",
            "swift", "bsb", "routing", "fedwire",
        )
        low = supplier_name.lower()
        if any(m in low for m in _bank_markers):
            logger.warning(
                "[AgentNick] Rejected bank-name as supplier: %r", supplier_name,
            )
            return None

        if self._is_garbage_supplier(supplier_name):
            logger.warning(
                "[AgentNick] Rejected garbage supplier name: %r", supplier_name,
            )
            return None

        canonical = self._canonical_supplier_key(supplier_name)
        if not canonical:
            return None

        try:
            conn = self._agent_nick.get_db_connection()
            try:
                with conn.cursor() as cur:
                    # 1. Exact name/trading match (case-insensitive)
                    cur.execute(
                        "SELECT supplier_id FROM proc.bp_supplier "
                        "WHERE LOWER(supplier_name)=LOWER(%s) OR LOWER(trading_name)=LOWER(%s) "
                        "LIMIT 1",
                        (supplier_name, supplier_name),
                    )
                    row = cur.fetchone()
                    if row:
                        return row[0]

                    # 2. Canonical-key match — same business entity under
                    # different formatting (TechWorld vs Techworld vs TechWorld Ltd)
                    cur.execute(
                        "SELECT supplier_id, supplier_name FROM proc.bp_supplier "
                        "WHERE LOWER(supplier_name) ~ %s OR LOWER(trading_name) ~ %s "
                        "LIMIT 5",
                        (re.escape(canonical[:40]), re.escape(canonical[:40])),
                    )
                    for row in cur.fetchall():
                        existing_canon = self._canonical_supplier_key(row[1] or "")
                        if existing_canon == canonical:
                            logger.info(
                                "[AgentNick] Canonicalised supplier '%s' → existing %s (%s)",
                                supplier_name, row[0], row[1],
                            )
                            return row[0]

                    # 3. Create new — slug derived from canonical key, not
                    # truncated mid-word. Up to 40 chars after the SUP- prefix.
                    slug = re.sub(r"\s+", "", canonical.title())[:40] or "Unknown"
                    supplier_id = f"SUP-{slug}"

                    cur.execute("""
                        INSERT INTO proc.bp_supplier
                            (supplier_id, supplier_name, trading_name, default_currency,
                             country, created_date, created_by)
                        VALUES (%s, %s, %s, %s, %s, NOW(), %s)
                        ON CONFLICT DO NOTHING
                    """, (
                        supplier_id, supplier_name, supplier_name,
                        header.get("currency"),
                        header.get("country"),
                        "AgentNick-AutoDiscovery",
                    ))

                    cur.execute("""
                        INSERT INTO proc.supplier
                            (supplier_id, supplier_name, trading_name, default_currency,
                             created_date, created_by)
                        VALUES (%s, %s, %s, %s, NOW(), %s)
                        ON CONFLICT DO NOTHING
                    """, (supplier_id, supplier_name, supplier_name,
                          header.get("currency"), "AgentNick-AutoDiscovery"))

                    logger.info(
                        "[AgentNick] Auto-created supplier: %s → %s",
                        supplier_name, supplier_id,
                    )
                    return supplier_id
            finally:
                conn.close()
        except Exception:
            logger.debug("Supplier auto-creation failed", exc_info=True)
            return None
