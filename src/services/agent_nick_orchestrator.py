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

        if missing:
            logger.info(
                "[AgentNick] Missing fields %s — asking LLM to fill gaps",
                missing,
            )
            header = self._llm_fill_gaps(
                header, missing, extraction.get("_source_text", ""), doc_type
            )

        # Step 3b: Filename-based PK — the filename is the most reliable
        # source for the document ID (e.g. "PERRY QUT136586 .pdf").
        # Always prefer filename PK over LLM-extracted values which can
        # be wrong (e.g. "10", "INVOICE", empty).
        pk_col = PK_MAP.get(doc_type)
        if pk_col:
            fname_id = self._extract_pk_from_filename(file_path, doc_type)
            if fname_id:
                existing = header.get(pk_col, "")
                if existing != fname_id:
                    if existing:
                        logger.info(
                            "[AgentNick] Overriding %s: '%s' → '%s' (from filename)",
                            pk_col, existing, fname_id,
                        )
                    else:
                        logger.info(
                            "[AgentNick] Filled %s from filename: %s",
                            pk_col, fname_id,
                        )
                    header[pk_col] = fname_id

        # Step 3c: Extract cross-references and supplier from filename
        self._fill_cross_refs_from_filename(header, file_path, doc_type)

        # Step 3d: Extract supplier name from filename as last resort
        # Filename pattern: "SUPPLIER_NAME DOC_ID for REF.pdf"
        if not header.get("supplier_id") and not header.get("supplier_name"):
            fname_supplier = self._extract_supplier_from_filename(file_path)
            if fname_supplier:
                header["supplier_id"] = fname_supplier
                header["supplier_name"] = fname_supplier
                logger.info("[AgentNick] Filled supplier from filename: %s", fname_supplier)

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

        # Step 6: Persist to bp_ tables
        pk_col = PK_MAP.get(doc_type)
        pk_value = header.get(pk_col, "") if pk_col else ""

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
    # Sub-agent dispatch
    # ------------------------------------------------------------------
    def _dispatch_extraction(
        self, file_path: str, doc_type: str
    ) -> Optional[Dict[str, Any]]:
        """Dispatch DataExtractionAgent to extract document content."""
        try:
            from services.direct_extraction_service import DirectExtractionService

            service = DirectExtractionService(self._agent_nick)
            text, file_bytes = service._get_document_text(file_path)

            if not text.strip():
                logger.error("[AgentNick] No text extracted from %s", file_path)
                return None

            # Run the extraction engine
            import tempfile
            from agents.extraction_engine import (
                run_data_extraction,
                set_db_connection_func,
            )

            # Ensure DB connection is available for supplier resolution
            set_db_connection_func(self._agent_nick.get_db_connection)

            suffix = os.path.splitext(file_path)[1].lower() or ".pdf"

            # Excel/CSV: extraction engine doesn't support these formats,
            # so use LLM directly on the tabular text
            if suffix in (".xlsx", ".xls", ".csv"):
                result = self._llm_extract_tabular(text, doc_type)
            else:
                with tempfile.NamedTemporaryFile(
                    suffix=suffix, delete=False
                ) as tmp:
                    tmp.write(file_bytes)
                    tmp_path = tmp.name

                try:
                    result = run_data_extraction(tmp_path)
                finally:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

            if not result or result.get("error"):
                logger.warning(
                    "[AgentNick] Extraction engine error: %s",
                    result.get("error") if result else "None",
                )
                # Fallback: use AgentNick LLM directly
                return self._llm_extract_direct(text, doc_type)

            # Map extraction result to standard format
            header = {}
            line_items = []

            # Tabular extraction (Excel/CSV) already returns header/line_items
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

            # Sanitize: strip empty strings to None
            header = {
                k: (v if v != "" else None)
                for k, v in header.items()
            }
            line_items = [
                {k: (v if v != "" else None) for k, v in item.items()}
                for item in line_items
            ]

            return {
                "header": header,
                "line_items": line_items,
                "_source_text": text,
            }

        except Exception as exc:
            logger.exception(
                "[AgentNick] DataExtractionAgent dispatch failed: %s", exc
            )
            return None

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
            "Invoice": [r"(INV[\-]?\s*[\w\-]+)"],
            "Purchase_Order": [r"PO\s*(\d{4,})"],
            "Quote": [r"QUT\s*([\d][\d\-]{2,})", r"Q\s*(\d{4,})"],
        }
        for pat in patterns.get(doc_type, []):
            m = re.search(pat, basename, re.IGNORECASE)
            if m:
                return re.sub(r"\s+", "", m.group(1).strip())
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
        """Detect issues in the SOURCE document — flag, don't fix."""
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

        # Check: total should equal subtotal + tax
        if amt > 0 and tax > 0 and total > 0:
            expected = round(amt + tax, 2)
            if abs(total - expected) > 1.0:
                warnings.append(
                    f"Total mismatch: {total_key}={total} but "
                    f"{amt_key}({amt}) + {tax_key}({tax}) = {expected}"
                )

        # Check: line items sum vs header subtotal
        if line_items and amt > 0:
            line_sum = sum(
                _f(li.get("line_amount") or li.get("line_total") or li.get("line_total_amount"))
                for li in line_items
            )
            if line_sum > 0 and abs(line_sum - amt) > 1.0:
                warnings.append(
                    f"Line items sum ({line_sum:.2f}) != {amt_key} ({amt:.2f})"
                )

        # Check: empty/missing critical amounts
        if amt == 0 and total == 0:
            warnings.append("Both subtotal and total are missing or zero")
        elif amt == 0 and total > 0:
            warnings.append(f"Subtotal ({amt_key}) is missing, only total available")

        # Check: line items with missing data
        incomplete = 0
        for li in line_items:
            qty = _f(li.get("quantity"))
            price = _f(li.get("unit_price"))
            desc = li.get("item_description", "")
            if not desc and (qty > 0 or price > 0):
                incomplete += 1
            elif desc and qty == 0 and price == 0:
                incomplete += 1
        if incomplete:
            warnings.append(f"{incomplete} line item(s) have incomplete data (missing description, qty, or price)")

        return warnings

    @staticmethod
    def _cross_validate(
        header: Dict[str, Any],
        line_items: List[Dict[str, Any]],
        doc_type: str,
    ) -> tuple:
        """Cross-validate extracted data: fix math errors, swap misplaced values."""

        def _f(v) -> float:
            try:
                return float(v) if v else 0.0
            except (ValueError, TypeError):
                return 0.0

        # --- Tax/Amount swap detection (all doc types) ---
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

        if tax > amt > 0:
            header[amt_key], header[tax_key] = tax, amt
            amt, tax = tax, amt
            logger.info("[CrossVal] Fixed tax/amount swap on %s", doc_type)

        # Recalculate total if it doesn't add up
        if amt > 0 and tax > 0:
            expected_total = round(amt + tax, 2)
            if total_key != amt_key and abs(total - expected_total) > 0.50:
                header[total_key] = expected_total
                logger.info("[CrossVal] Recalculated %s: %.2f", total_key, expected_total)

        # Derive tax_percent from amounts if missing
        if amt > 0 and tax > 0 and not header.get("tax_percent"):
            header["tax_percent"] = round((tax / amt) * 100, 2)

        # --- Line item validation ---
        for item in line_items:
            qty = _f(item.get("quantity"))
            price = _f(item.get("unit_price"))
            line_amt = _f(item.get("line_amount") or item.get("line_total") or item.get("line_total_amount"))

            # Quantity/price swap: if qty > price and both > 0, they're likely swapped
            if qty > 0 and price > 0 and qty > price * 10:
                item["quantity"], item["unit_price"] = price, qty
                qty, price = price, qty
                logger.info("[CrossVal] Fixed qty/price swap: qty=%.0f, price=%.2f", qty, price)

            # Calculate line_amount if missing
            if qty > 0 and price > 0 and line_amt == 0:
                calculated = round(qty * price, 2)
                for k in ("line_amount", "line_total", "line_total_amount"):
                    if k in item:
                        item[k] = calculated
                        break

        # --- Header total vs line items sum ---
        if line_items and amt == 0:
            line_sum = sum(
                _f(li.get("line_amount") or li.get("line_total") or li.get("line_total_amount"))
                for li in line_items
            )
            if line_sum > 0:
                header[amt_key] = round(line_sum, 2)
                logger.info("[CrossVal] Derived %s from line items: %.2f", amt_key, line_sum)

        # --- Currency: default GBP if not set (UK procurement) ---
        if not header.get("currency"):
            header["currency"] = "GBP"

        return header, line_items

    def _llm_extract_tabular(
        self, text: str, doc_type: str
    ) -> Optional[Dict[str, Any]]:
        """Extract structured data from Excel/CSV text using LLM directly."""
        schema_hints = {
            "Quote": "quote_id, supplier_id, buyer_id, quote_date, validity_date, currency, total_amount (subtotal before tax), tax_percent, tax_amount, total_amount_incl_tax",
            "Invoice": "invoice_id, supplier_id, po_id, invoice_date, due_date, currency, invoice_amount, tax_percent, tax_amount, invoice_total_incl_tax",
            "Purchase_Order": "po_id, supplier_name, order_date, currency, total_amount",
        }
        fields = schema_hints.get(doc_type, "")
        prompt = (
            f"Extract ALL data from this {doc_type} spreadsheet. Return ONLY valid JSON.\n\n"
            f"Header fields: {fields}\n"
            f"Line items: array of {{line_no, item_description, quantity, unit_price, line_total}}\n\n"
            f"RULES:\n"
            f"- supplier is the company providing the quote (usually at the top)\n"
            f"- buyer is the company receiving the quote (delivery/billing address)\n"
            f"- tax_amount must be LESS than total_amount\n"
            f"- Extract ALL line items from the table\n"
            f"- If source has data issues (empty totals, orphaned rows), extract what IS there\n\n"
            f"Spreadsheet content:\n{text[:6000]}"
        )
        try:
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": AGENT_NICK_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0, "num_predict": 2048, "num_gpu": 99},
                },
                timeout=600,
            )
            raw = resp.json().get("response", "")
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
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": AGENT_NICK_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0, "num_predict": 512, "num_gpu": 99},
                },
                timeout=120,
            )
            response.raise_for_status()
            raw = response.json().get("response", "").strip()

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
