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
            elif fname_supplier.lower().split()[0] not in extracted_sup.lower():
                # Filename supplier's first word doesn't appear in extracted value
                # This catches: extracted="Assurity Ltd" but filename="PERRY PO526689"
                logger.warning(
                    "[AgentNick] Supplier mismatch: extracted='%s' but filename says '%s' — using filename",
                    extracted_sup, fname_supplier,
                )
                header["supplier_name"] = fname_supplier
                header["supplier_id"] = fname_supplier

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
        lines_ok = self._persist_line_items(line_items, doc_type, pk_value)

        # Step 7: Learn vendor profile
        self._learn_vendor_profile(header, doc_type)

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

            # Inject vendor profile context
            supplier_hint = self._extract_supplier_from_filename(file_path)
            vendor_context = self._get_vendor_context(supplier_hint or "")
            if vendor_context:
                doc_structure.metadata.insert(
                    0, {"label": "VENDOR_PROFILE", "value": vendor_context}
                )

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
    @staticmethod
    def _derive_fields(header: Dict[str, Any], doc_type: str) -> None:
        """Derive computable fields from document content.

        Only derives when the source data clearly supports it:
        - due_date from payment_terms + invoice_date (e.g., "Net 30" + date)
        - validity_date from quote_date + "valid for X days"
        """
        from datetime import timedelta

        # Derive due_date for invoices
        if doc_type == "Invoice" and not header.get("due_date"):
            terms = str(header.get("payment_terms", "") or "").strip()
            inv_date_str = header.get("invoice_date")
            if terms and inv_date_str:
                days = None
                # "Net 30", "Net 60", "Net 90"
                m = re.search(r"[Nn]et\s+(\d+)", terms)
                if m:
                    days = int(m.group(1))
                # "30 days", "60 days", "14 days"
                if not days:
                    m = re.search(r"(\d+)\s*[Dd]ays?", terms)
                    if m:
                        days = int(m.group(1))
                # "Pay by {date}" — extract the explicit date
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
                # "Due on receipt" / "immediately" → same as invoice_date
                if not days and ("receipt" in terms.lower() or "immediate" in terms.lower()):
                    header["due_date"] = inv_date_str
                    logger.info(
                        "[Derive] due_date=%s (due on receipt)", inv_date_str
                    )

                if days:
                    try:
                        from datetime import datetime as dt
                        inv_date = dt.strptime(str(inv_date_str)[:10], "%Y-%m-%d")
                        due = inv_date + timedelta(days=days)
                        header["due_date"] = due.strftime("%Y-%m-%d")
                        logger.info(
                            "[Derive] due_date=%s from invoice_date=%s + %d days",
                            header["due_date"], inv_date_str, days,
                        )
                    except Exception:
                        pass

            # Default: if still no due_date and we have invoice_date,
            # apply business default of invoice_date + 90 days
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

        # Derive validity_date for quotes
        if doc_type == "Quote" and not header.get("validity_date"):
            quote_date_str = header.get("quote_date")
            # Check notes/metadata for "valid for X days"
            source = str(header.get("_source_notes", ""))
            if quote_date_str:
                m = re.search(r"[Vv]alid\s+(?:for\s+)?(\d+)\s*[Dd]ays?", source)
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
        """Extract document PK from filename patterns like 'WADE PO526809 for QUT30746.pdf'."""
        basename = os.path.splitext(os.path.basename(file_path))[0]
        patterns = {
            "Invoice": [
                r"(INV[\-]?\s*[\w\-]+)",        # INV-123, INV123, INV 123-456
                r"(Invoice[\-\s]*\d{3,})",       # Invoice-001, Invoice 12345
                r"(BILL[\-\s]*\d{3,})",          # BILL-001
            ],
            "Purchase_Order": [
                r"PO\s*(\d{4,})",                # PO526809, PO 526809
                r"(PUR[\-\s]*\d{3,})",           # PUR-001
            ],
            "Quote": [
                r"(QUT[\-\s]*[\d][\d\-]{2,})",   # QUT30746, QUT-25-032
                r"(QTE[\-\s]*[\d][\d\-]{2,})",   # QTE-2026-00487
                r"(QUOTE[\-\s]*[\d][\d\-]{2,})", # QUOTE-123
                r"Q\s*(\d{4,})",                 # Q10483
                r"(\d{5,})",                     # bare numeric IDs (136700)
            ],
        }
        for pat in patterns.get(doc_type, []):
            m = re.search(pat, basename, re.IGNORECASE)
            if m:
                pk = re.sub(r"\s+", "", m.group(1).strip())
                # Reject overly generic values
                if pk.lower() in ("invoice", "quote", "po", "purchase"):
                    continue
                return pk
        return None

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

    def _auto_create_supplier(self, header: Dict[str, Any], doc_type: str) -> Optional[str]:
        """Auto-create a new supplier in bp_supplier if not found.

        Only creates from explicitly extracted document values — never guesses.
        Returns the new supplier_id or None.
        """
        supplier_name = header.get("supplier_id") or header.get("supplier_name") or ""
        if not supplier_name or len(supplier_name) < 3:
            return None

        try:
            conn = self._agent_nick.get_db_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT supplier_id FROM proc.bp_supplier "
                        "WHERE LOWER(supplier_name) = LOWER(%s) OR LOWER(trading_name) = LOWER(%s) "
                        "LIMIT 1",
                        (supplier_name, supplier_name),
                    )
                    if cur.fetchone():
                        return None

                    clean = re.sub(r"[^a-zA-Z0-9]", "", supplier_name)
                    supplier_id = f"SUP-{clean[:20]}"

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
