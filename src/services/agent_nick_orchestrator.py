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

            # Run the extraction engine (the proven docs/extraction.py logic)
            import tempfile
            from agents.extraction_engine import (
                run_data_extraction,
                set_db_connection_func,
            )

            # Ensure DB connection is available for supplier resolution
            set_db_connection_func(self._agent_nick.get_db_connection)

            suffix = os.path.splitext(file_path)[1].lower() or ".pdf"
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

            # Map extraction engine output to standard format
            header = {}
            line_items = []

            if result.get("document_type") == "invoice":
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
                        header["supplier_id"] = row[0] or row[1]
                        header["supplier_name"] = row[1]
                        logger.info(
                            "[AgentNick] Supplier resolved: '%s' → '%s' (id=%s)",
                            raw_name, row[1], row[0],
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
                            header["supplier_id"] = row[0] or row[1]
                            header["supplier_name"] = row[1]
                            logger.info(
                                "[AgentNick] Supplier fuzzy-matched: '%s' → '%s'",
                                raw_name, row[1],
                            )
            finally:
                conn.close()
        except Exception:
            logger.debug("[AgentNick] Supplier resolution failed", exc_info=True)

        return header

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
