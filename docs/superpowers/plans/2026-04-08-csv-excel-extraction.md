# CSV/Excel Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable the data extraction pipeline to process CSV and Excel files, mapping columns to procurement DB schemas using category-driven routing, with Neo4j KG fallback for unrecognized categories.

**Architecture:** Add `CATEGORY_TO_DOC_TYPE` mapping in `procurement_schema.py`, a `_process_tabular_document()` method in `DataExtractionAgent`, a new `KGIngestionService` for Neo4j fallback, and thread `category` from `process_monitor` through the orchestrator into the extraction pipeline.

**Tech Stack:** pandas (existing), openpyxl (for xlsx), neo4j (existing driver), psycopg2 (existing), existing fuzzy matching in `procurement_schema.py`.

---

### Task 1: Add Category-to-DocType Mapping in procurement_schema.py

**Files:**
- Modify: `utils/procurement_schema.py:380-400`

- [ ] **Step 1: Add CATEGORY_TO_DOC_TYPE mapping and normalize_category helper**

After `DOC_TYPE_TO_TABLE` (line 400), add:

```python
CATEGORY_TO_DOC_TYPE: Dict[str, str] = {
    "invoice": "Invoice",
    "po": "Purchase_Order",
    "purchase_order": "Purchase_Order",
    "quote": "Quote",
    "quotes": "Quote",
    "contract": "Contract",
    "contracts": "Contract",
}


def normalize_category(category: str) -> str | None:
    """Normalize a process_monitor category to a DOC_TYPE_TO_TABLE key.

    Returns None if category is not recognized (should go to KG).
    """
    if not category:
        return None
    return CATEGORY_TO_DOC_TYPE.get(category.strip().lower())


def map_columns_to_schema(
    columns: list[str],
    table_key: str,
    threshold: float = 0.55,
) -> Dict[str, str]:
    """Map CSV/Excel column names to schema column names.

    Returns a dict of {csv_column: schema_column} for matched columns.
    Unmatched columns are excluded.
    """
    schema = PROCUREMENT_SCHEMAS.get(table_key)
    if schema is None:
        return {}
    used: set[str] = set()
    mapping: Dict[str, str] = {}
    for col in columns:
        normalised = _normalise(col.strip().replace("-", " ").replace("_", " "))
        # Exact match first
        if col.strip().lower().replace(" ", "_") in {c.lower() for c in schema.columns}:
            for sc in schema.columns:
                if sc.lower() == col.strip().lower().replace(" ", "_") and sc not in used:
                    mapping[col] = sc
                    used.add(sc)
                    break
            continue
        # Fuzzy match
        best_col, best_score = _best_schema_match(normalised, schema, used)
        if best_col and best_score >= threshold:
            mapping[col] = best_col
            used.add(best_col)
    return mapping
```

- [ ] **Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('utils/procurement_schema.py').read()); print('OK')"`

- [ ] **Step 3: Commit**

```bash
git add utils/procurement_schema.py
git commit -m "feat: add category-to-doc-type mapping and column mapper for tabular extraction

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Thread Category Through Orchestrator

**Files:**
- Modify: `src/services/process_monitor_watcher.py:233-260`
- Modify: `src/orchestration/orchestrator.py:186-203`

- [ ] **Step 1: Update ProcessMonitorWatcher._process_record to pass category**

In `src/services/process_monitor_watcher.py`, replace the `_process_record` method (lines 233-260):

```python
    def _process_record(self, record: Dict[str, Any]) -> None:
        """Run extraction for a claimed record."""
        record_id = record["id"]
        file_path = record.get("file_path", "")
        category = record.get("category", "")
        document_type = record.get("document_type", "")
        user_id = record.get("user_id")
        logger.info(
            "Starting extraction for record %s: file_path=%s category=%s",
            record_id,
            file_path,
            category,
        )
        try:
            orchestrator = self._orchestrator
            if orchestrator is None:
                raise RuntimeError("Orchestrator not available")
            result = orchestrator.execute_extraction_flow(
                s3_object_key=file_path,
                category=category,
                document_type=document_type,
                user_id=str(user_id) if user_id is not None else None,
            )
            status = "error"
            if isinstance(result, dict):
                status = str(result.get("status", "error")).lower()
            if status in ("blocked", "error", "failed"):
                raise RuntimeError(f"Extraction {status}: {result.get('reason', result.get('error', 'unknown'))}")
            self._mark_extracted(record_id)
            logger.info("Extraction completed for record %s", record_id)
        except Exception as exc:
            logger.exception("Extraction failed for record %s", record_id)
            self._mark_failed(record_id, str(exc))
```

- [ ] **Step 2: Update orchestrator.execute_extraction_flow**

In `src/orchestration/orchestrator.py`, replace `execute_extraction_flow` (lines 186-195):

```python
    def execute_extraction_flow(
        self,
        s3_prefix: Optional[str] = None,
        s3_object_key: Optional[str] = None,
        *,
        category: Optional[str] = None,
        document_type: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict:
        """Public wrapper for the document extraction workflow."""
        payload: Dict[str, Any] = {
            "s3_prefix": s3_prefix,
            "s3_object_key": s3_object_key,
        }
        if category:
            payload["category"] = category
        if document_type:
            payload["document_type"] = document_type
        if user_id:
            payload["user_id"] = user_id
        return self.execute_workflow("document_extraction", payload)
```

Also update `execute_extraction_workflow` alias (lines 197-203):

```python
    def execute_extraction_workflow(
        self,
        s3_prefix: Optional[str] = None,
        s3_object_key: Optional[str] = None,
        **kwargs,
    ) -> Dict:
        """Backward compatible alias for :meth:`execute_extraction_flow`."""
        return self.execute_extraction_flow(s3_prefix, s3_object_key, **kwargs)
```

- [ ] **Step 3: Verify syntax**

Run: `python -c "import ast; [ast.parse(open(f).read()) for f in ['src/services/process_monitor_watcher.py', 'src/orchestration/orchestrator.py']]; print('OK')"`

- [ ] **Step 4: Commit**

```bash
git add src/services/process_monitor_watcher.py src/orchestration/orchestrator.py
git commit -m "feat: thread category/document_type/user_id through extraction pipeline

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Add Tabular Processing to DataExtractionAgent

**Files:**
- Modify: `src/agents/data_extraction_agent.py:1546` (supported_exts)
- Modify: `src/agents/data_extraction_agent.py:1582-1640` (_process_single_document)

- [ ] **Step 1: Add CSV/XLSX to supported extensions**

In `src/agents/data_extraction_agent.py`, at line 1546, change:

```python
        supported_exts = {".pdf", ".doc", ".docx", ".png", ".jpg", ".jpeg"}
```

to:

```python
        supported_exts = {".pdf", ".doc", ".docx", ".png", ".jpg", ".jpeg", ".csv", ".xlsx", ".xls"}
```

- [ ] **Step 2: Add the _process_tabular_document method**

Add the following method after `_process_single_document` (after line ~1879). This method needs to be added to the `DataExtractionAgent` class:

```python
    def _process_tabular_document(
        self,
        file_bytes: bytes,
        object_key: str,
        *,
        category: str = "",
        context: AgentContext | None = None,
    ) -> Optional[Dict[str, str]]:
        """Process CSV/Excel files: map columns to schema, persist to DB or KG."""
        from utils.procurement_schema import (
            normalize_category,
            map_columns_to_schema,
            DOC_TYPE_TO_TABLE,
            PROCUREMENT_SCHEMAS,
        )

        workflow_id = getattr(context, "workflow_id", None) if context else None
        ext = os.path.splitext(object_key)[1].lower()

        # --- Read into DataFrame(s) ---
        sheets: Dict[str, pd.DataFrame] = {}
        try:
            if ext == ".csv":
                for encoding in ("utf-8", "latin-1", "cp1252"):
                    try:
                        df = pd.read_csv(BytesIO(file_bytes), encoding=encoding)
                        sheets["default"] = df
                        break
                    except UnicodeDecodeError:
                        continue
                if not sheets:
                    logger.error("Failed to decode CSV %s with any encoding", object_key)
                    return {"object_key": object_key, "status": "error", "error": "encoding_failure"}
            elif ext in (".xlsx", ".xls"):
                try:
                    xls = pd.ExcelFile(BytesIO(file_bytes))
                    for sheet_name in xls.sheet_names:
                        sheets[sheet_name] = xls.parse(sheet_name)
                except Exception as exc:
                    logger.error("Failed to read Excel %s: %s", object_key, exc)
                    return {"object_key": object_key, "status": "error", "error": str(exc)}
        except Exception as exc:
            logger.error("Failed to read tabular file %s: %s", object_key, exc)
            return {"object_key": object_key, "status": "error", "error": str(exc)}

        # --- Resolve target schema ---
        doc_type = normalize_category(category)
        total_rows = 0
        total_mapped = 0

        for sheet_name, df in sheets.items():
            if df.empty:
                logger.warning("Empty sheet '%s' in %s", sheet_name, object_key)
                continue

            # Check if sheet name hints at a doc type
            sheet_doc_type = doc_type or normalize_category(sheet_name)

            if sheet_doc_type:
                # --- Known category: map to DB ---
                table_info = DOC_TYPE_TO_TABLE.get(sheet_doc_type)
                if not table_info:
                    logger.warning("No table mapping for doc_type '%s'", sheet_doc_type)
                    self._ingest_to_kg(df, category or sheet_name, object_key, context=context)
                    continue

                header_table = table_info[0]
                column_mapping = map_columns_to_schema(
                    list(df.columns), header_table
                )

                if not column_mapping:
                    logger.warning(
                        "No columns matched schema for %s (category=%s). "
                        "Columns: %s. Falling back to KG.",
                        object_key, category, list(df.columns),
                    )
                    self._ingest_to_kg(df, category or sheet_name, object_key, context=context)
                    continue

                unmapped = [c for c in df.columns if c not in column_mapping]
                if unmapped:
                    logger.info(
                        "Unmapped columns for %s: %s", object_key, unmapped
                    )

                logger.info(
                    "Processing tabular file %s: sheet=%s rows=%d matched=%d/%d",
                    object_key, sheet_name, len(df), len(column_mapping), len(df.columns),
                )

                # Persist rows via staging pattern
                schema_name, table_name = header_table.split(".", 1)
                row_count = self._persist_tabular_rows(
                    df, column_mapping, schema_name, table_name
                )
                total_rows += row_count
                total_mapped += len(column_mapping)
            else:
                # --- Unknown category: push to Neo4j KG ---
                logger.info(
                    "Unrecognized category '%s' for %s — ingesting to KG",
                    category, object_key,
                )
                self._ingest_to_kg(df, category or sheet_name, object_key, context=context)
                total_rows += len(df)

        # Vectorize for RAG regardless of path
        try:
            text_repr = self._tabular_to_text(sheets)
            if text_repr.strip():
                self._vectorize_document(
                    text_repr,
                    os.path.basename(object_key),
                    doc_type or category or "tabular",
                    "tabular_data",
                    object_key,
                )
        except Exception:
            logger.warning("Failed to vectorize tabular file %s", object_key, exc_info=True)

        return {
            "object_key": object_key,
            "status": "completed",
            "doc_type": doc_type or category or "unknown",
            "total_rows": str(total_rows),
            "mapped_columns": str(total_mapped),
        }

    def _persist_tabular_rows(
        self,
        df: pd.DataFrame,
        column_mapping: Dict[str, str],
        schema_name: str,
        table_name: str,
    ) -> int:
        """Persist DataFrame rows to the target table via staging pattern."""
        schema = PROCUREMENT_SCHEMAS.get(f"{schema_name}.{table_name}")
        pk_col = schema.required[0] if schema and schema.required else None
        row_count = 0
        chunk_size = 10_000

        for start in range(0, len(df), chunk_size):
            chunk = df.iloc[start : start + chunk_size]
            try:
                with self.agent_nick.get_db_connection() as conn:
                    conn.autocommit = True
                    with conn.cursor() as cur:
                        for _, row in chunk.iterrows():
                            payload: Dict[str, Any] = {}
                            for csv_col, schema_col in column_mapping.items():
                                val = row.get(csv_col)
                                if pd.isna(val):
                                    continue
                                payload[schema_col] = val

                            if not payload:
                                continue

                            # Generate PK if missing
                            if pk_col and pk_col not in payload:
                                payload[pk_col] = uuid.uuid4().hex[:8]

                            conflict_cols = [pk_col] if pk_col and self._has_unique_constraint(
                                cur, schema_name, table_name, [pk_col]
                            ) else []
                            update_cols = [
                                c for c in payload.keys() if c not in set(conflict_cols)
                            ]
                            self._merge_from_staging(
                                cur,
                                target_schema=schema_name,
                                target_table=table_name,
                                payload=payload,
                                conflict_cols=conflict_cols,
                                update_cols=update_cols,
                            )
                            row_count += 1
            except Exception:
                logger.exception(
                    "Failed to persist chunk starting at row %d for %s.%s",
                    start, schema_name, table_name,
                )
        return row_count

    def _tabular_to_text(self, sheets: Dict[str, pd.DataFrame]) -> str:
        """Convert tabular data to text representation for vectorization."""
        parts: List[str] = []
        for sheet_name, df in sheets.items():
            if df.empty:
                continue
            header = " | ".join(str(c) for c in df.columns)
            rows = []
            for _, row in df.head(50).iterrows():
                rows.append(" | ".join(str(v) for v in row.values if pd.notna(v)))
            parts.append(f"Sheet: {sheet_name}\n{header}\n" + "\n".join(rows))
        return "\n\n".join(parts)

    def _ingest_to_kg(
        self,
        df: pd.DataFrame,
        category: str,
        object_key: str,
        *,
        context: AgentContext | None = None,
    ) -> None:
        """Ingest a DataFrame into Neo4j Knowledge Graph."""
        try:
            from services.kg_ingestion_service import KGIngestionService

            kg_service = KGIngestionService(self.agent_nick)
            kg_service.ingest_dataframe(df, category, source=object_key)
        except Exception:
            logger.exception(
                "Failed to ingest %s (category=%s) to KG", object_key, category
            )
```

- [ ] **Step 3: Add file extension branching in _process_single_document**

In `_process_single_document` (around line 1639, after `file_bytes = body.read()`), add a branch before the text extraction. Insert right after the `file_bytes` is read (after the finally block at line 1620, before `force_ocr_vendors` at line 1633):

```python
        # --- Tabular file fast path ---
        ext = os.path.splitext(object_key)[1].lower()
        if ext in (".csv", ".xlsx", ".xls"):
            category = ""
            if context and hasattr(context, "input_data") and isinstance(context.input_data, dict):
                category = context.input_data.get("category", "")
            return self._process_tabular_document(
                file_bytes, object_key, category=category, context=context
            )
```

- [ ] **Step 4: Add import for uuid at top of file if not present**

Check if `uuid` is already imported. If not, add:

```python
import uuid
```

- [ ] **Step 5: Verify syntax**

Run: `python -c "import ast; ast.parse(open('src/agents/data_extraction_agent.py').read()); print('OK')"`

- [ ] **Step 6: Commit**

```bash
git add src/agents/data_extraction_agent.py
git commit -m "feat: add tabular document processing for CSV/Excel extraction

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Create KG Ingestion Service

**Files:**
- Create: `src/services/kg_ingestion_service.py`

- [ ] **Step 1: Create the KG ingestion service**

```python
"""Knowledge Graph ingestion service for unrecognized document categories.

Ingests tabular data into Neo4j as supplier-centric nodes and relationships.
Used as fallback when process_monitor.category doesn't map to a known schema.
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Set

import pandas as pd

logger = logging.getLogger(__name__)

# Column name patterns that identify supplier-related fields
SUPPLIER_ID_PATTERNS = re.compile(
    r"^(supplier_id|vendor_id|supplier_code|vendor_code)$", re.IGNORECASE
)
SUPPLIER_NAME_PATTERNS = re.compile(
    r"^(supplier_name|vendor_name|vendor|supplier|manufacturer)$", re.IGNORECASE
)
FK_PATTERNS: Dict[str, str] = {
    "contract_id": "Contract",
    "po_id": "PurchaseOrder",
    "purchase_order_id": "PurchaseOrder",
    "category_id": "Category",
    "item_id": "Item",
    "invoice_id": "Invoice",
}
SUPPLIER_PROPERTY_PATTERNS = re.compile(
    r"^(country|region|currency|brand|contact|email|phone|address|city|state|zip)$",
    re.IGNORECASE,
)


def _label_from_category(category: str) -> str:
    """Convert a category string to a valid Neo4j label."""
    cleaned = re.sub(r"[^a-zA-Z0-9]", "", category.strip().title())
    return cleaned or "UnknownData"


class KGIngestionService:
    """Ingests tabular data into Neo4j as a supplier-centric knowledge graph."""

    def __init__(self, agent_nick) -> None:
        self._agent_nick = agent_nick
        self._driver = self._get_neo4j_driver()

    def _get_neo4j_driver(self):
        """Create a Neo4j driver from agent_nick settings."""
        try:
            from neo4j import GraphDatabase

            settings = self._agent_nick.settings
            uri = getattr(settings, "neo4j_uri", "bolt://localhost:7687")
            username = getattr(settings, "neo4j_username", "neo4j")
            password = getattr(settings, "neo4j_password", "neo4j")
            return GraphDatabase.driver(uri, auth=(username, password))
        except Exception:
            logger.exception("Failed to create Neo4j driver")
            return None

    def ingest_dataframe(
        self,
        df: pd.DataFrame,
        category: str,
        *,
        source: str = "",
    ) -> int:
        """Ingest a DataFrame into Neo4j.

        Returns the number of rows ingested.
        """
        if df.empty or self._driver is None:
            return 0

        label = _label_from_category(category)
        columns = list(df.columns)

        # Identify supplier columns
        supplier_id_col = self._find_column(columns, SUPPLIER_ID_PATTERNS)
        supplier_name_col = self._find_column(columns, SUPPLIER_NAME_PATTERNS)
        supplier_prop_cols = [
            c for c in columns if SUPPLIER_PROPERTY_PATTERNS.match(c)
        ]

        # Identify FK columns
        fk_cols: Dict[str, str] = {}
        for col in columns:
            col_lower = col.strip().lower()
            if col_lower in FK_PATTERNS:
                fk_cols[col] = FK_PATTERNS[col_lower]

        row_count = 0
        batch_size = 500
        for start in range(0, len(df), batch_size):
            batch = df.iloc[start : start + batch_size]
            try:
                with self._driver.session() as session:
                    for _, row in batch.iterrows():
                        self._ingest_row(
                            session,
                            row=row,
                            label=label,
                            source=source,
                            supplier_id_col=supplier_id_col,
                            supplier_name_col=supplier_name_col,
                            supplier_prop_cols=supplier_prop_cols,
                            fk_cols=fk_cols,
                            columns=columns,
                        )
                        row_count += 1
            except Exception:
                logger.exception(
                    "Failed to ingest batch starting at row %d for category '%s'",
                    start, category,
                )

        logger.info(
            "KG ingestion complete: category=%s label=%s rows=%d source=%s",
            category, label, row_count, source,
        )
        return row_count

    def _ingest_row(
        self,
        session,
        *,
        row,
        label: str,
        source: str,
        supplier_id_col: str | None,
        supplier_name_col: str | None,
        supplier_prop_cols: List[str],
        fk_cols: Dict[str, str],
        columns: List[str],
    ) -> None:
        """Ingest a single row: create data node, supplier node, and relationships."""
        row_id = uuid.uuid4().hex

        # Build properties dict (all non-null values)
        props: Dict[str, Any] = {"row_id": row_id, "source": source}
        for col in columns:
            val = row.get(col)
            if pd.notna(val):
                safe_key = re.sub(r"[^a-zA-Z0-9_]", "_", col.strip().lower())
                props[safe_key] = self._serialize_value(val)

        # Create data node
        session.run(
            f"MERGE (n:{label} {{row_id: $row_id}}) SET n += $props",
            row_id=row_id,
            props=props,
        )

        # Create/merge Supplier node and link
        supplier_id = None
        if supplier_id_col:
            supplier_id = row.get(supplier_id_col)
        supplier_name = None
        if supplier_name_col:
            supplier_name = row.get(supplier_name_col)

        if pd.notna(supplier_id) or pd.notna(supplier_name):
            supplier_props: Dict[str, Any] = {}
            if pd.notna(supplier_id):
                supplier_props["supplier_id"] = str(supplier_id)
            if pd.notna(supplier_name):
                supplier_props["name"] = str(supplier_name)
            for prop_col in supplier_prop_cols:
                val = row.get(prop_col)
                if pd.notna(val):
                    safe_key = re.sub(r"[^a-zA-Z0-9_]", "_", prop_col.strip().lower())
                    supplier_props[safe_key] = self._serialize_value(val)

            # Merge supplier by ID if available, else by name
            if pd.notna(supplier_id):
                session.run(
                    "MERGE (s:Supplier {supplier_id: $sid}) "
                    "SET s += $props "
                    f"WITH s MATCH (n:{label} {{row_id: $row_id}}) "
                    f"MERGE (s)-[:HAS_{label.upper()}]->(n)",
                    sid=str(supplier_id),
                    props=supplier_props,
                    row_id=row_id,
                )
            elif pd.notna(supplier_name):
                session.run(
                    "MERGE (s:Supplier {name: $sname}) "
                    "SET s += $props "
                    f"WITH s MATCH (n:{label} {{row_id: $row_id}}) "
                    f"MERGE (s)-[:HAS_{label.upper()}]->(n)",
                    sname=str(supplier_name),
                    props=supplier_props,
                    row_id=row_id,
                )

        # Create FK relationships
        for fk_col, target_label in fk_cols.items():
            fk_val = row.get(fk_col)
            if pd.notna(fk_val):
                fk_key = fk_col.strip().lower()
                rel_type = f"LINKED_TO_{target_label.upper()}"
                session.run(
                    f"MATCH (n:{label} {{row_id: $row_id}}) "
                    f"MERGE (t:{target_label} {{{fk_key}: $fk_val}}) "
                    f"MERGE (n)-[:{rel_type}]->(t)",
                    row_id=row_id,
                    fk_val=str(fk_val),
                )

    @staticmethod
    def _find_column(columns: List[str], pattern: re.Pattern) -> str | None:
        """Find the first column matching a regex pattern."""
        for col in columns:
            if pattern.match(col.strip()):
                return col
        return None

    @staticmethod
    def _serialize_value(val: Any) -> Any:
        """Convert value to a Neo4j-safe type."""
        if isinstance(val, (int, float, str, bool)):
            return val
        return str(val)

    def close(self) -> None:
        """Close the Neo4j driver."""
        if self._driver:
            try:
                self._driver.close()
            except Exception:
                pass
```

- [ ] **Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('src/services/kg_ingestion_service.py').read()); print('OK')"`

- [ ] **Step 3: Commit**

```bash
git add src/services/kg_ingestion_service.py
git commit -m "feat: add KGIngestionService for Neo4j ingestion of unrecognized categories

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Write Tests

**Files:**
- Create: `tests/test_tabular_extraction.py`

- [ ] **Step 1: Write tests for category mapping, column matching, and tabular processing**

```python
"""Tests for CSV/Excel tabular extraction pipeline."""

import io
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from utils.procurement_schema import (
    normalize_category,
    map_columns_to_schema,
    CATEGORY_TO_DOC_TYPE,
    DOC_TYPE_TO_TABLE,
)


class TestNormalizeCategory:
    def test_known_categories(self):
        assert normalize_category("invoice") == "Invoice"
        assert normalize_category("po") == "Purchase_Order"
        assert normalize_category("quote") == "Quote"
        assert normalize_category("quotes") == "Quote"
        assert normalize_category("contract") == "Contract"
        assert normalize_category("purchase_order") == "Purchase_Order"

    def test_case_insensitive(self):
        assert normalize_category("INVOICE") == "Invoice"
        assert normalize_category("Po") == "Purchase_Order"
        assert normalize_category("  Quote  ") == "Quote"

    def test_unknown_returns_none(self):
        assert normalize_category("spend") is None
        assert normalize_category("item") is None
        assert normalize_category("") is None
        assert normalize_category("unknown_type") is None


class TestMapColumnsToSchema:
    def test_exact_match(self):
        mapping = map_columns_to_schema(
            ["invoice_id", "supplier_name", "invoice_total_incl_tax"],
            "proc.invoice_agent",
        )
        assert mapping["invoice_id"] == "invoice_id"
        assert mapping["supplier_name"] == "supplier_name"
        assert mapping["invoice_total_incl_tax"] == "invoice_total_incl_tax"

    def test_synonym_match(self):
        mapping = map_columns_to_schema(
            ["invoice number", "vendor name"],
            "proc.invoice_agent",
        )
        assert mapping.get("invoice number") == "invoice_id"
        assert mapping.get("vendor name") == "supplier_name"

    def test_no_match_excluded(self):
        mapping = map_columns_to_schema(
            ["totally_random_column", "invoice_id"],
            "proc.invoice_agent",
        )
        assert "totally_random_column" not in mapping
        assert mapping["invoice_id"] == "invoice_id"

    def test_unknown_table_returns_empty(self):
        mapping = map_columns_to_schema(
            ["invoice_id"],
            "proc.nonexistent_table",
        )
        assert mapping == {}

    def test_po_columns(self):
        mapping = map_columns_to_schema(
            ["po_id", "supplier_name", "total_amount"],
            "proc.purchase_order_agent",
        )
        assert "po_id" in mapping
        assert "supplier_name" in mapping


class TestCategoryToDocTypeIntegration:
    def test_all_categories_have_tables(self):
        for category, doc_type in CATEGORY_TO_DOC_TYPE.items():
            assert doc_type in DOC_TYPE_TO_TABLE, (
                f"Category '{category}' maps to '{doc_type}' "
                f"which is not in DOC_TYPE_TO_TABLE"
            )


class TestKGIngestionService:
    def test_label_from_category(self):
        from services.kg_ingestion_service import _label_from_category

        assert _label_from_category("spend") == "Spend"
        assert _label_from_category("item_master") == "ItemMaster"
        assert _label_from_category("") == "UnknownData"

    def test_find_column(self):
        import re
        from services.kg_ingestion_service import KGIngestionService

        pattern = re.compile(r"^(supplier_id|vendor_id)$", re.IGNORECASE)
        assert KGIngestionService._find_column(
            ["name", "supplier_id", "amount"], pattern
        ) == "supplier_id"
        assert KGIngestionService._find_column(
            ["name", "amount"], pattern
        ) is None

    def test_ingest_returns_zero_for_empty_df(self):
        nick = SimpleNamespace(
            settings=SimpleNamespace(
                neo4j_uri="bolt://localhost:7687",
                neo4j_username="neo4j",
                neo4j_password="neo4j",
            )
        )
        with patch("services.kg_ingestion_service.GraphDatabase") as mock_gdb:
            mock_gdb.driver.return_value = MagicMock()
            from services.kg_ingestion_service import KGIngestionService
            service = KGIngestionService.__new__(KGIngestionService)
            service._agent_nick = nick
            service._driver = MagicMock()
            result = service.ingest_dataframe(pd.DataFrame(), "spend")
            assert result == 0
```

- [ ] **Step 2: Run the tests**

Run: `cd /home/muthu/PycharmProjects/BP_Backend && python -m pytest tests/test_tabular_extraction.py -v`

- [ ] **Step 3: Commit**

```bash
git add tests/test_tabular_extraction.py
git commit -m "test: add tests for tabular extraction pipeline

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: End-to-End Verification

- [ ] **Step 1: Syntax check all files**

Run:
```bash
python -c "
import ast
for f in [
    'utils/procurement_schema.py',
    'src/services/process_monitor_watcher.py',
    'src/orchestration/orchestrator.py',
    'src/agents/data_extraction_agent.py',
    'src/services/kg_ingestion_service.py',
]:
    ast.parse(open(f).read())
    print(f'{f}: OK')
"
```

- [ ] **Step 2: Run all tests**

Run: `python -m pytest tests/test_tabular_extraction.py tests/test_process_monitor_watcher.py -v`

- [ ] **Step 3: Verify category mapping completeness**

Run:
```python
python -c "
from utils.procurement_schema import CATEGORY_TO_DOC_TYPE, DOC_TYPE_TO_TABLE, normalize_category
for cat in ['invoice', 'po', 'quote', 'quotes', 'contract', 'purchase_order', 'contracts']:
    dt = normalize_category(cat)
    table = DOC_TYPE_TO_TABLE.get(dt) if dt else None
    print(f'{cat} -> {dt} -> {table}')
for cat in ['spend', 'item', 'other']:
    dt = normalize_category(cat)
    print(f'{cat} -> {dt} (KG fallback)')
"
```
