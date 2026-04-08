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
