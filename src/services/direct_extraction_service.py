"""Direct extraction service — schema-driven, LLM-powered, accurate.

Extracts document content, sends to AgentNick with the exact bp_ table
schema, and persists directly to the database. No temp files, no
multi-strategy fallbacks, no staging tables.

Flow:
1. Download file from S3
2. Extract text (pdfplumber → OCR fallback → S3 canonical fallback)
3. Send text + exact column definitions to AgentNick LLM
4. Parse JSON response
5. INSERT/UPDATE directly into bp_ tables with audit columns
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EXTRACTION_MODEL = os.getenv("PROCWISE_EXTRACTION_MODEL", "BeyondProcwise/AgentNick:latest")

# Exact table schemas from the database — source of truth
TABLE_SCHEMAS = {
    "Invoice": {
        "header_table": "proc.bp_invoice",
        "line_table": "proc.bp_invoice_line_items",
        "pk": "invoice_id",
        "line_pk": "invoice_line_id",
        "line_fk": "invoice_id",
        "line_seq": "line_no",
        "header_columns": {
            "invoice_id": "text", "po_id": "text", "supplier_id": "text",
            "buyer_id": "text", "requisition_id": "text", "requested_by": "text",
            "requested_date": "date", "invoice_date": "date", "due_date": "date",
            "invoice_paid_date": "date", "payment_terms": "text",
            "currency": "varchar", "invoice_amount": "numeric",
            "tax_percent": "numeric", "tax_amount": "numeric",
            "invoice_total_incl_tax": "numeric", "exchange_rate_to_usd": "numeric",
            "converted_amount_usd": "numeric", "country": "text", "region": "text",
            "invoice_status": "text",
        },
        "line_columns": {
            "invoice_line_id": "text", "invoice_id": "text", "line_no": "integer",
            "item_id": "text", "item_description": "text", "quantity": "integer",
            "unit_of_measure": "text", "unit_price": "numeric",
            "line_amount": "numeric", "tax_percent": "numeric",
            "tax_amount": "numeric", "total_amount_incl_tax": "numeric",
            "po_id": "text", "delivery_date": "date", "country": "text",
            "region": "text",
        },
    },
    "Purchase_Order": {
        "header_table": "proc.bp_purchase_order",
        "line_table": "proc.bp_po_line_items",
        "pk": "po_id",
        "line_pk": "po_line_id",
        "line_fk": "po_id",
        "line_seq": "line_number",
        "header_columns": {
            "po_id": "text", "supplier_name": "text", "buyer_id": "text",
            "requisition_id": "text", "requested_by": "text",
            "requested_date": "date", "currency": "varchar",
            "order_date": "date", "expected_delivery_date": "date",
            "ship_to_country": "text", "delivery_region": "text",
            "incoterm": "text", "incoterm_responsibility": "text",
            "total_amount": "numeric", "delivery_address_line1": "text",
            "delivery_address_line2": "text", "delivery_city": "text",
            "postal_code": "text", "payment_terms": "varchar",
            "po_status": "varchar", "contract_id": "text",
        },
        "line_columns": {
            "po_line_id": "text", "po_id": "text", "line_number": "integer",
            "item_id": "text", "item_description": "text", "quote_number": "text",
            "quantity": "numeric", "unit_price": "numeric",
            "unit_of_measue": "text", "currency": "varchar",
            "line_total": "numeric", "tax_percent": "smallint",
            "tax_amount": "numeric", "total_amount": "numeric",
        },
    },
    "Quote": {
        "header_table": "proc.bp_quote",
        "line_table": "proc.bp_quote_line_items",
        "pk": "quote_id",
        "line_pk": "quote_line_id",
        "line_fk": "quote_id",
        "line_seq": "line_number",
        "header_columns": {
            "quote_id": "text", "deal_id": "text", "supplier_id": "text",
            "buyer_id": "text", "supplier_address": "text",
            "buyer_address": "text", "quote_date": "date",
            "validity_date": "date", "currency": "varchar",
            "total_amount": "numeric", "tax_percent": "numeric",
            "tax_amount": "numeric", "total_amount_incl_tax": "numeric",
            "po_id": "text", "country": "text", "region": "text",
        },
        "line_columns": {
            "quote_line_id": "text", "quote_id": "text",
            "line_number": "integer", "item_id": "text",
            "item_description": "text", "quantity": "integer",
            "unit_of_measure": "text", "unit_price": "numeric",
            "line_total": "numeric", "tax_percent": "numeric",
            "tax_amount": "numeric", "total_amount": "numeric",
            "currency": "varchar",
        },
    },
    "Contract": {
        "header_table": "proc.bp_contracts",
        "line_table": None,
        "pk": "contract_id",
        "line_pk": None,
        "line_fk": None,
        "line_seq": None,
        "header_columns": {
            "contract_id": "text", "contract_title": "text",
            "contract_type": "text", "supplier_id": "text",
            "buyer_org_id": "text", "contract_start_date": "date",
            "contract_end_date": "date", "currency": "text",
            "total_contract_value": "numeric", "spend_category": "text",
            "business_unit_id": "text", "cost_centre_id": "text",
            "is_amendment": "text", "parent_contract_id": "text",
            "auto_renew_flag": "text", "renewal_term": "text",
            "contract_lifecycle_status": "text",
        },
        "line_columns": {},
    },
}

CATEGORY_MAP = {
    "invoice": "Invoice",
    "po": "Purchase_Order",
    "purchase_order": "Purchase_Order",
    "quote": "Quote",
    "quotes": "Quote",
    "contract": "Contract",
    "contracts": "Contract",
}


class DirectExtractionService:
    """Schema-driven extraction: text → LLM → bp_ tables."""

    def __init__(self, agent_nick) -> None:
        self._agent_nick = agent_nick
        self._s3_bucket = agent_nick.settings.s3_bucket_name

    def extract_and_persist(
        self,
        file_path: str,
        category: str,
        *,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Extract document content and persist to bp_ tables.

        Returns dict with status, extracted fields, and any errors.
        """
        doc_type = CATEGORY_MAP.get(category.strip().lower(), "")
        if not doc_type:
            return {"status": "error", "error": f"Unknown category: {category}"}

        schema = TABLE_SCHEMAS.get(doc_type)
        if not schema:
            return {"status": "error", "error": f"No schema for doc_type: {doc_type}"}

        # Step 1: Download and extract text
        text, file_bytes = self._get_document_text(file_path)
        if not text.strip():
            return {
                "status": "error",
                "error": f"No text could be extracted from {file_path}",
                "file_path": file_path,
            }

        logger.info(
            "Extracted %d chars from %s for %s extraction",
            len(text), file_path, doc_type,
        )

        # Step 2: Send to LLM with exact schema
        extraction = self._llm_extract(text, doc_type, schema)
        if not extraction:
            return {
                "status": "error",
                "error": "LLM extraction returned no data",
                "file_path": file_path,
            }

        header = extraction.get("header", {})
        line_items = extraction.get("line_items", [])

        # Step 3: Persist to bp_ tables
        now = datetime.now(timezone.utc)
        audit = {
            "created_date": now,
            "created_by": user_id or "AgentNick",
            "last_modified_by": user_id or "AgentNick",
            "last_modified_date": now,
        }

        pk_value = header.get(schema["pk"], "")
        if not pk_value:
            return {
                "status": "error",
                "error": f"Missing primary key '{schema['pk']}' in extraction",
                "header": header,
            }

        header_result = self._persist_header(header, schema, audit)
        line_result = self._persist_line_items(
            line_items, schema, pk_value, audit
        )

        return {
            "status": "success",
            "file_path": file_path,
            "doc_type": doc_type,
            "pk": pk_value,
            "header_fields": len(header),
            "line_items": len(line_items),
            "header_persisted": header_result,
            "lines_persisted": line_result,
        }

    # ------------------------------------------------------------------
    # Text extraction
    # ------------------------------------------------------------------
    def _get_document_text(self, file_path: str) -> Tuple[str, bytes]:
        """Download from S3 and extract text. Try canonical fallback for corrupt PDFs."""
        file_bytes = self._download_s3(file_path)
        if not file_bytes:
            return "", b""

        ext = os.path.splitext(file_path)[1].lower()

        # CSV/Excel — convert to text representation
        if ext in (".csv", ".xlsx", ".xls"):
            return self._tabular_to_text(file_bytes, ext), file_bytes

        # PDF/DOCX/Image
        text = self._extract_text_from_bytes(file_bytes, ext)

        # Fallback: try canonical S3 path if empty
        if not text.strip() and file_path.startswith("documents/"):
            canonical_map = {
                "documents/invoice/": "Invoice/",
                "documents/po/": "Purchase_Order/",
                "documents/quote/": "Invoice/",
            }
            for prefix, s3_prefix in canonical_map.items():
                if file_path.startswith(prefix):
                    filename = file_path[len(prefix):]
                    alt_bytes = self._download_s3(s3_prefix + filename)
                    if alt_bytes:
                        alt_text = self._extract_text_from_bytes(alt_bytes, ext)
                        if alt_text.strip():
                            logger.info(
                                "Canonical fallback '%s%s' succeeded (%d chars)",
                                s3_prefix, filename, len(alt_text),
                            )
                            return alt_text, alt_bytes
                    break

        return text, file_bytes

    def _download_s3(self, key: str) -> Optional[bytes]:
        """Download a file from S3."""
        try:
            s3 = self._agent_nick.reserve_s3_connection().__enter__()
            obj = s3.get_object(Bucket=self._s3_bucket, Key=key)
            return obj["Body"].read()
        except Exception:
            logger.debug("S3 download failed for '%s'", key)
            return None

    def _extract_text_from_bytes(self, file_bytes: bytes, ext: str) -> str:
        """Extract text from PDF/DOCX/image bytes."""
        if ext == ".pdf":
            return self._extract_pdf_text(file_bytes)
        elif ext == ".docx":
            return self._extract_docx_text(file_bytes)
        elif ext in (".png", ".jpg", ".jpeg"):
            return self._extract_image_text(file_bytes)
        return ""

    def _extract_pdf_text(self, file_bytes: bytes) -> str:
        """Extract text from PDF using pdfplumber, with OCR fallback."""
        import pdfplumber
        text_parts = []
        try:
            with pdfplumber.open(BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        text_parts.append(page_text)
        except Exception:
            logger.debug("pdfplumber failed", exc_info=True)

        if text_parts:
            return "\n".join(text_parts)

        # OCR fallback for scanned PDFs
        try:
            from pdf2image import convert_from_bytes
            import pytesseract
            images = convert_from_bytes(file_bytes, dpi=300)
            for img in images:
                ocr_text = pytesseract.image_to_string(img)
                if ocr_text.strip():
                    text_parts.append(ocr_text)
        except Exception:
            logger.debug("OCR fallback failed", exc_info=True)

        return "\n".join(text_parts)

    def _extract_docx_text(self, file_bytes: bytes) -> str:
        """Extract text from DOCX."""
        try:
            from docx import Document
            doc = Document(BytesIO(file_bytes))
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except Exception:
            return ""

    def _extract_image_text(self, file_bytes: bytes) -> str:
        """Extract text from image using OCR."""
        try:
            from PIL import Image
            import pytesseract
            img = Image.open(BytesIO(file_bytes))
            return pytesseract.image_to_string(img)
        except Exception:
            return ""

    def _tabular_to_text(self, file_bytes: bytes, ext: str) -> str:
        """Convert CSV/Excel to text for LLM processing."""
        try:
            if ext == ".csv":
                for enc in ("utf-8", "latin-1", "cp1252"):
                    try:
                        df = pd.read_csv(BytesIO(file_bytes), encoding=enc)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    return ""
            else:
                df = pd.read_excel(BytesIO(file_bytes))

            header = " | ".join(str(c) for c in df.columns)
            rows = []
            for _, row in df.iterrows():
                rows.append(" | ".join(
                    str(v) for v in row.values if pd.notna(v)
                ))
            return f"COLUMNS: {header}\n\n" + "\n".join(rows)
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # LLM extraction
    # ------------------------------------------------------------------
    def _llm_extract(
        self, text: str, doc_type: str, schema: Dict
    ) -> Optional[Dict[str, Any]]:
        """Send document text + exact schema to AgentNick for extraction."""
        header_cols = list(schema["header_columns"].keys())
        line_cols = list(schema.get("line_columns", {}).keys())

        prompt = self._build_extraction_prompt(
            text, doc_type, header_cols, line_cols, schema
        )

        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": EXTRACTION_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0,
                        "num_predict": 4096,
                        "num_gpu": 99,
                    },
                },
                timeout=300,
            )
            response.raise_for_status()
            raw = response.json().get("response", "").strip()
            return self._parse_llm_response(raw, schema)
        except Exception as exc:
            logger.exception("LLM extraction failed: %s", exc)
            return None

    def _build_extraction_prompt(
        self,
        text: str,
        doc_type: str,
        header_cols: List[str],
        line_cols: List[str],
        schema: Dict,
    ) -> str:
        """Build a precise extraction prompt with exact column names and types."""
        header_spec = "\n".join(
            f"  - {col} ({schema['header_columns'][col]})"
            for col in header_cols
        )
        line_spec = ""
        if line_cols:
            line_spec = "\nLINE ITEM COLUMNS (extract ALL line items):\n" + "\n".join(
                f"  - {col} ({schema['line_columns'][col]})"
                for col in line_cols
            )

        return f"""Extract structured data from this {doc_type} document.

HEADER COLUMNS (use these exact field names):
{header_spec}
{line_spec}

RULES:
- Return ONLY valid JSON, no other text
- Dates must be YYYY-MM-DD format
- Amounts must be numbers without currency symbols (e.g., 1234.56 not £1,234.56)
- Currency must be 3-letter ISO code (GBP, USD, EUR)
- The supplier/vendor is the SELLER (who issued the document), NOT the buyer
- If a field is not found in the document, omit it from the response
- Extract ALL line items, stop at subtotal/total rows
- tax_percent should be the percentage number (e.g., 20 for 20%, not 0.20)

RESPONSE FORMAT:
{{
  "header": {{ ... header fields using exact column names above ... }},
  "line_items": [ {{ ... line item fields ... }}, ... ]
}}

DOCUMENT TEXT:
{text[:8000]}"""

    def _parse_llm_response(
        self, raw: str, schema: Dict
    ) -> Optional[Dict[str, Any]]:
        """Parse LLM JSON response, handling markdown code blocks."""
        # Strip markdown code fences
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```\s*$", "", cleaned)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to find JSON object in the response
            match = re.search(r"\{[\s\S]*\}", cleaned)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    logger.error("Failed to parse LLM response as JSON")
                    return None
            else:
                logger.error("No JSON found in LLM response")
                return None

        header = data.get("header", {})
        line_items = data.get("line_items", [])

        # Validate: only keep columns that exist in schema
        valid_header = {}
        for k, v in header.items():
            if k in schema["header_columns"] and v is not None:
                valid_header[k] = v

        valid_lines = []
        line_cols = set(schema.get("line_columns", {}).keys())
        for item in line_items:
            valid_item = {k: v for k, v in item.items() if k in line_cols and v is not None}
            if valid_item:
                valid_lines.append(valid_item)

        return {"header": valid_header, "line_items": valid_lines}

    # ------------------------------------------------------------------
    # Database persistence
    # ------------------------------------------------------------------
    def _persist_header(
        self,
        header: Dict[str, Any],
        schema: Dict,
        audit: Dict[str, Any],
    ) -> bool:
        """Insert or update header record in the bp_ table."""
        table = schema["header_table"]
        pk_col = schema["pk"]
        pk_value = header.get(pk_col)
        if not pk_value:
            return False

        schema_name, table_name = table.split(".", 1)
        columns = dict(schema["header_columns"])

        # Build payload with type coercion
        payload = {}
        for col, val in header.items():
            if col not in columns:
                continue
            payload[col] = self._coerce_value(val, columns[col])

        # Add audit columns
        payload.update(audit)

        try:
            conn = self._agent_nick.get_db_connection()
            conn.autocommit = True
            with conn.cursor() as cur:
                col_names = list(payload.keys())
                placeholders = ["%s"] * len(col_names)
                values = [payload[c] for c in col_names]

                # UPSERT: insert or update on PK conflict
                insert_sql = (
                    f'INSERT INTO {table} ({", ".join(col_names)}) '
                    f'VALUES ({", ".join(placeholders)})'
                )

                # Check if PK has unique constraint
                cur.execute(
                    "SELECT 1 FROM pg_indexes WHERE schemaname=%s AND tablename=%s "
                    "AND indexdef LIKE %s LIMIT 1",
                    (schema_name, table_name, f"%{pk_col}%"),
                )
                has_pk = cur.fetchone() is not None

                if has_pk:
                    update_cols = [c for c in col_names if c != pk_col]
                    if update_cols:
                        update_clause = ", ".join(
                            f"{c} = EXCLUDED.{c}" for c in update_cols
                        )
                        insert_sql += (
                            f" ON CONFLICT ({pk_col}) DO UPDATE SET {update_clause}"
                        )
                    else:
                        insert_sql += f" ON CONFLICT ({pk_col}) DO NOTHING"
                else:
                    insert_sql += " ON CONFLICT DO NOTHING"

                cur.execute(insert_sql, values)
            conn.close()
            logger.info(
                "Persisted %s header: %s=%s (%d fields)",
                table, pk_col, pk_value, len(payload),
            )
            return True
        except Exception as exc:
            logger.exception("Failed to persist header to %s: %s", table, exc)
            return False

    def _persist_line_items(
        self,
        line_items: List[Dict[str, Any]],
        schema: Dict,
        pk_value: str,
        audit: Dict[str, Any],
    ) -> int:
        """Insert line items into the bp_ line items table."""
        line_table = schema.get("line_table")
        if not line_table or not line_items:
            return 0

        line_fk = schema["line_fk"]
        line_pk = schema["line_pk"]
        line_seq = schema["line_seq"]
        columns = dict(schema.get("line_columns", {}))
        count = 0

        try:
            conn = self._agent_nick.get_db_connection()
            conn.autocommit = True
            with conn.cursor() as cur:
                for idx, item in enumerate(line_items, start=1):
                    payload = {}
                    for col, val in item.items():
                        if col not in columns:
                            continue
                        payload[col] = self._coerce_value(val, columns[col])

                    # Set FK and sequence
                    payload[line_fk] = pk_value
                    if line_seq and line_seq not in payload:
                        payload[line_seq] = idx
                    if line_pk and line_pk not in payload:
                        payload[line_pk] = f"{pk_value}-{idx}"

                    payload.update(audit)

                    col_names = list(payload.keys())
                    placeholders = ["%s"] * len(col_names)
                    values = [payload[c] for c in col_names]

                    insert_sql = (
                        f'INSERT INTO {line_table} ({", ".join(col_names)}) '
                        f'VALUES ({", ".join(placeholders)}) '
                        f"ON CONFLICT DO NOTHING"
                    )
                    cur.execute(insert_sql, values)
                    count += 1

            conn.close()
            logger.info(
                "Persisted %d line items to %s for %s=%s",
                count, line_table, line_fk, pk_value,
            )
        except Exception as exc:
            logger.exception("Failed to persist line items to %s: %s", line_table, exc)

        return count

    @staticmethod
    def _coerce_value(val: Any, col_type: str) -> Any:
        """Coerce a value to the target column type."""
        if val is None:
            return None

        if col_type in ("numeric", "integer", "smallint"):
            if isinstance(val, (int, float)):
                return val
            text = str(val).strip()
            # Handle European format
            if re.match(r"^\d{1,3}(?:\.\d{3})+,\d{1,2}$", text):
                text = text.replace(".", "").replace(",", ".")
            elif "," in text and "." in text:
                if text.rfind(",") > text.rfind("."):
                    text = text.replace(".", "").replace(",", ".")
                else:
                    text = text.replace(",", "")
            elif "," in text:
                parts = text.split(",")
                if len(parts) == 2 and len(parts[1]) <= 2:
                    text = text.replace(",", ".")
                else:
                    text = text.replace(",", "")
            text = re.sub(r"[£$€¥₹%\s]", "", text)
            # Handle parenthesized negatives
            paren = re.match(r"^\((.+)\)$", text)
            if paren:
                text = "-" + paren.group(1)
            try:
                if col_type in ("integer", "smallint"):
                    return int(float(text))
                return float(text)
            except (ValueError, TypeError):
                return None

        if col_type == "date":
            if isinstance(val, str):
                text = val.strip()
                # Try common formats
                for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",
                            "%m/%d/%Y", "%d %B %Y", "%d %b %Y", "%B %d, %Y",
                            "%b %d, %Y"):
                    try:
                        from datetime import datetime as dt
                        return dt.strptime(text, fmt).date()
                    except ValueError:
                        continue
                # Fallback to dateutil
                try:
                    from dateutil import parser
                    return parser.parse(text, dayfirst=True).date()
                except Exception:
                    return None
            return val

        # Text types — just convert to string
        return str(val).strip() if val is not None else None
