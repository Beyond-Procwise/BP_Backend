"""Procurement Knowledge Graph Builder.

Creates Neo4j nodes from live bp_ table data and connects them using
the relationship model defined in the KG Excel workbook.

Entity-to-Table mapping:
  Supplier        → proc.bp_supplier
  Contract        → proc.bp_contracts
  Invoice         → proc.bp_invoice
  InvoiceLine     → proc.bp_invoice_line_items
  PurchaseOrder   → proc.bp_purchase_order
  POLine          → proc.bp_po_line_items
  Quote           → proc.bp_quote
  QuoteLine       → proc.bp_quote_line_items
  Category        → proc.bp_category
  Approval        → proc.bp_approvals
  Policy          → proc.bp_policy

Runs unsupervised via BackendScheduler.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

KG_WORKBOOK_PATH = os.getenv(
    "KG_WORKBOOK_PATH",
    "/home/muthu/Downloads/Procurement Knowledge Graphv2 (1).xlsx",
)

# Entity → (table, pk_column, neo4j_label)
ENTITY_TABLE_MAP = {
    "Supplier": ("proc.bp_supplier", "supplier_id", "Supplier"),
    "Contract": ("proc.bp_contracts", "contract_id", "Contract"),
    "Invoice": ("proc.bp_invoice", "invoice_id", "Invoice"),
    "InvoiceLine": ("proc.bp_invoice_line_items", "invoice_line_id", "InvoiceLine"),
    "PurchaseOrder": ("proc.bp_purchase_order", "po_id", "PurchaseOrder"),
    "POLine": ("proc.bp_po_line_items", "po_line_id", "POLine"),
    "Quote": ("proc.bp_quote", "quote_id", "Quote"),
    "QuoteLine": ("proc.bp_quote_line_items", "quote_line_id", "QuoteLine"),
    "Category": ("proc.bp_category", "category_id", "Category"),
    "Approval": ("proc.bp_approvals", "id", "Approval"),
    "Policy": ("proc.bp_policy", "id", "Policy"),
}

# FK-based relationships: (from_label, rel_type, to_label, from_fk, to_pk)
FK_RELATIONSHIPS = [
    # Supplier → Contract
    ("Contract", "SUPPLIER_PARTY_TO_CONTRACT", "Supplier", "supplier_id", "supplier_id"),
    # Invoice → Supplier
    ("Invoice", "INVOICE_FROM_SUPPLIER", "Supplier", "supplier_id", "supplier_id"),
    # Invoice → PO
    ("Invoice", "INVOICE_REFERENCES_PO", "PurchaseOrder", "po_id", "po_id"),
    # PO → Supplier (by name)
    ("PurchaseOrder", "PO_FROM_SUPPLIER", "Supplier", "supplier_name", "supplier_name"),
    # Quote → Supplier
    ("Quote", "QUOTE_FROM_SUPPLIER", "Supplier", "supplier_id", "supplier_id"),
    # Line items → parent
    ("InvoiceLine", "LINE_OF_INVOICE", "Invoice", "invoice_id", "invoice_id"),
    ("POLine", "LINE_OF_PO", "PurchaseOrder", "po_id", "po_id"),
    ("QuoteLine", "LINE_OF_QUOTE", "Quote", "quote_id", "quote_id"),
    # Quote → PO (quote references the PO it was created for)
    ("Quote", "QUOTE_FOR_PO", "PurchaseOrder", "po_id", "po_id"),
    # PO line → Quote (PO line items reference the quote they originated from)
    ("POLine", "PO_LINE_FROM_QUOTE", "Quote", "quote_number", "quote_id"),
    # Category hierarchy
    ("Category", "CATEGORY_HAS_PARENT", "Category", "parent_category_id", "category_id"),
    # Contract → Category (via spend_category)
    ("Contract", "CONTRACT_COVERS_CATEGORY", "Category", "spend_category", "category_name"),
]


class ProcurementKGBuilder:
    """Builds the procurement KG from live bp_ table data."""

    def __init__(self, agent_nick) -> None:
        self._agent_nick = agent_nick
        self._driver = self._get_driver()

    def _get_driver(self):
        try:
            from neo4j import GraphDatabase
            s = self._agent_nick.settings
            uri = getattr(s, "neo4j_uri", "bolt://localhost:7687")
            user = getattr(s, "neo4j_username", "neo4j")
            pwd = getattr(s, "neo4j_password", "procwise2026")
            driver = GraphDatabase.driver(uri, auth=(user, pwd))
            driver.verify_connectivity()
            return driver
        except Exception:
            logger.exception(
                "Failed to connect to Neo4j — the knowledge graph is unavailable. "
                "Ensure Neo4j is running at the configured URI."
            )
            return None

    def build_full_graph(self) -> Dict[str, int]:
        """Build complete KG from live data. Returns node/rel counts."""
        if not self._driver:
            return {"error": "No Neo4j driver"}

        counts = {}

        # 1. Create indexes for fast lookups
        self._create_indexes()

        # 2. Load all entities from bp_ tables
        for entity_name, (table, pk, label) in ENTITY_TABLE_MAP.items():
            n = self._load_entity(table, pk, label)
            counts[entity_name] = n

        # 3. Create FK-based relationships
        for from_label, rel_type, to_label, from_fk, to_pk in FK_RELATIONSHIPS:
            n = self._create_relationships(from_label, rel_type, to_label, from_fk, to_pk)
            counts[f"rel_{rel_type}"] = n

        # 4. Load approval levels from Excel
        if os.path.exists(KG_WORKBOOK_PATH):
            counts.update(self._load_excel_reference_data())

        # 5. Create inferred cross-document relationships
        counts["rel_PO_REFERENCES_QUOTE"] = self._link_po_to_quotes()

        # 6. Infer supplier nodes from extracted data if bp_supplier is empty
        counts["inferred_suppliers"] = self._infer_suppliers()

        logger.info("[KG Builder] Complete: %s", counts)
        return counts

    def _create_indexes(self) -> None:
        """Create indexes for all entity labels."""
        with self._driver.session() as session:
            for _, (_, pk, label) in ENTITY_TABLE_MAP.items():
                try:
                    session.run(
                        f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON (n.{pk})"
                    )
                except Exception:
                    pass
            # Extra indexes for relationship lookups
            for idx in [
                "CREATE INDEX IF NOT EXISTS FOR (n:Supplier) ON (n.supplier_name)",
                "CREATE INDEX IF NOT EXISTS FOR (n:Category) ON (n.category_name)",
            ]:
                try:
                    session.run(idx)
                except Exception:
                    pass

    def _load_entity(self, table: str, pk: str, label: str) -> int:
        """Load all rows from a bp_ table as Neo4j nodes."""
        try:
            conn = self._agent_nick.get_db_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT * FROM {table} LIMIT 5000")
                    rows = cur.fetchall()
                    cols = [d.name for d in cur.description]
            finally:
                conn.close()
        except Exception:
            logger.debug("Failed to read %s", table, exc_info=True)
            return 0

        if not rows:
            return 0

        count = 0
        with self._driver.session() as session:
            for row in rows:
                data = dict(zip(cols, row))
                pk_val = data.get(pk)
                if not pk_val:
                    continue

                props = {}
                for k, v in data.items():
                    if v is not None:
                        if isinstance(v, (int, float)):
                            props[k] = v
                        else:
                            s = str(v).strip()
                            if s:
                                props[k] = s

                session.run(
                    f"MERGE (n:{label} {{{pk}: $pk_val}}) SET n += $props",
                    pk_val=str(pk_val),
                    props=props,
                )
                count += 1

        logger.info("[KG] Loaded %d %s nodes from %s", count, label, table)
        return count

    def _create_relationships(
        self,
        from_label: str,
        rel_type: str,
        to_label: str,
        from_fk: str,
        to_pk: str,
    ) -> int:
        """Create relationships between existing nodes based on FK values."""
        count = 0
        try:
            with self._driver.session() as session:
                result = session.run(
                    f"MATCH (a:{from_label}) WHERE a.{from_fk} IS NOT NULL "
                    f"MATCH (b:{to_label} {{{to_pk}: a.{from_fk}}}) "
                    f"MERGE (a)-[r:{rel_type}]->(b) "
                    f"RETURN count(r) as cnt"
                )
                record = result.single()
                count = record["cnt"] if record else 0
        except Exception:
            logger.debug("Failed to create %s relationships", rel_type, exc_info=True)

        if count > 0:
            logger.info("[KG] Created %d %s relationships", count, rel_type)
        return count

    def _link_po_to_quotes(self) -> int:
        """Create PO → Quote relationships inferred from PO line item quote references."""
        count = 0
        try:
            with self._driver.session() as session:
                # PO line items carry a quote_number that maps to quote_id.
                # Create a direct PO_REFERENCES_QUOTE edge at the document level.
                result = session.run(
                    "MATCH (pl:POLine)-[:LINE_OF_PO]->(po:PurchaseOrder) "
                    "WHERE pl.quote_number IS NOT NULL "
                    "MATCH (q:Quote {quote_id: pl.quote_number}) "
                    "MERGE (po)-[r:PO_REFERENCES_QUOTE]->(q) "
                    "RETURN count(r) as cnt"
                )
                count = (result.single() or {}).get("cnt", 0)
                if count:
                    logger.info("[KG] Created %d PO_REFERENCES_QUOTE relationships", count)
        except Exception:
            logger.debug("PO→Quote linking failed", exc_info=True)
        return count

    def _infer_suppliers(self) -> int:
        """Create Supplier nodes from extracted data if bp_supplier is empty."""
        count = 0
        try:
            with self._driver.session() as session:
                # From invoices
                result = session.run(
                    "MATCH (i:Invoice) WHERE i.supplier_id IS NOT NULL "
                    "MERGE (s:Supplier {supplier_id: i.supplier_id}) "
                    "ON CREATE SET s.supplier_name = i.supplier_id, s.source = 'inferred_from_invoice' "
                    "MERGE (i)-[:INVOICE_FROM_SUPPLIER]->(s) "
                    "RETURN count(s) as cnt"
                )
                count += (result.single() or {}).get("cnt", 0)

                # From POs
                result = session.run(
                    "MATCH (p:PurchaseOrder) WHERE p.supplier_name IS NOT NULL "
                    "MERGE (s:Supplier {supplier_name: p.supplier_name}) "
                    "ON CREATE SET s.supplier_id = p.supplier_name, s.source = 'inferred_from_po' "
                    "MERGE (p)-[:PO_FROM_SUPPLIER]->(s) "
                    "RETURN count(s) as cnt"
                )
                count += (result.single() or {}).get("cnt", 0)

                # From quotes
                result = session.run(
                    "MATCH (q:Quote) WHERE q.supplier_id IS NOT NULL "
                    "MERGE (s:Supplier {supplier_id: q.supplier_id}) "
                    "ON CREATE SET s.supplier_name = q.supplier_id, s.source = 'inferred_from_quote' "
                    "MERGE (q)-[:QUOTE_FROM_SUPPLIER]->(s) "
                    "RETURN count(s) as cnt"
                )
                count += (result.single() or {}).get("cnt", 0)

                # From contracts
                result = session.run(
                    "MATCH (c:Contract) WHERE c.supplier_id IS NOT NULL "
                    "MERGE (s:Supplier {supplier_id: c.supplier_id}) "
                    "ON CREATE SET s.supplier_name = c.supplier_id, s.source = 'inferred_from_contract' "
                    "MERGE (c)-[:SUPPLIER_PARTY_TO_CONTRACT]->(s) "
                    "RETURN count(s) as cnt"
                )
                count += (result.single() or {}).get("cnt", 0)

        except Exception:
            logger.debug("Supplier inference failed", exc_info=True)

        if count > 0:
            logger.info("[KG] Inferred %d supplier nodes from extracted data", count)
        return count

    def _load_excel_reference_data(self) -> Dict[str, int]:
        """Load reference data from Excel: approval levels, risk model, finance hierarchy."""
        counts = {}
        try:
            xls = pd.ExcelFile(KG_WORKBOOK_PATH)
        except Exception:
            return counts

        # Finance approval levels
        if "Finance_Approval_Levels" in xls.sheet_names:
            counts["finance_approvals"] = self._load_levels(
                xls, "Finance_Approval_Levels", "FinanceApprovalLevel"
            )

        # Procurement approval levels
        if "Procurement_Approval_Levels" in xls.sheet_names:
            counts["procurement_approvals"] = self._load_levels(
                xls, "Procurement_Approval_Levels", "ProcurementApprovalLevel"
            )

        # Risk scoring model
        if "Risk_Scoring_Model" in xls.sheet_names:
            df = xls.parse("Risk_Scoring_Model")
            n = 0
            with self._driver.session() as session:
                for _, row in df.iterrows():
                    comp = row.get("Component", "")
                    if not comp:
                        continue
                    session.run(
                        "MERGE (r:RiskComponent {name: $name}) "
                        "SET r.weight = $w, r.example_score = $s",
                        name=str(comp),
                        w=float(row.get("Weight", 0)),
                        s=float(row.get("Example_Score", 0)),
                    )
                    n += 1
            counts["risk_components"] = n

        # Finance hierarchy
        if "Finance_Budget_Hierarchy" in xls.sheet_names:
            df = xls.parse("Finance_Budget_Hierarchy")
            n = 0
            with self._driver.session() as session:
                for _, row in df.iterrows():
                    nid = row.get("finance_budget_hierarchy_id", "")
                    if not nid:
                        continue
                    props = {}
                    for col in df.columns:
                        v = row.get(col)
                        if pd.notna(v):
                            props[col.lower()] = float(v) if isinstance(v, (int, float)) else str(v)
                    session.run(
                        "MERGE (n:FinanceBudgetNode {node_id: $nid}) SET n += $props",
                        nid=str(nid), props=props,
                    )
                    parent = row.get("parent_budget_id")
                    if pd.notna(parent) and str(parent).strip():
                        session.run(
                            "MATCH (c:FinanceBudgetNode {node_id: $c}) "
                            "MATCH (p:FinanceBudgetNode {node_id: $p}) "
                            "MERGE (c)-[:ROLLS_UP_TO]->(p)",
                            c=str(nid), p=str(parent),
                        )
                    n += 1
            counts["finance_hierarchy"] = n

        # TPRM profiles
        if "TPRM_Profile" in xls.sheet_names:
            df = xls.parse("TPRM_Profile")
            n = 0
            with self._driver.session() as session:
                for _, row in df.iterrows():
                    sid = row.get("Supplier_ID", "")
                    if not sid:
                        continue
                    session.run(
                        "MERGE (p:TPRMProfile {supplier_id: $sid}) "
                        "SET p.risk_tier = $tier, p.overall_score = $score, "
                        "p.risk_owner = $owner "
                        "WITH p "
                        "MERGE (s:Supplier {supplier_id: $sid}) "
                        "MERGE (s)-[:HAS_RISK_PROFILE]->(p)",
                        sid=str(sid),
                        tier=str(row.get("Risk_Tier", "")),
                        score=float(row.get("Overall_Score", 0)),
                        owner=str(row.get("Risk_Owner", "")),
                    )
                    n += 1
            counts["tprm_profiles"] = n

        return counts

    def _load_levels(self, xls, sheet: str, label: str) -> int:
        df = xls.parse(sheet)
        n = 0
        with self._driver.session() as session:
            for _, row in df.iterrows():
                level = row.get("Level", "")
                if not level:
                    continue
                session.run(
                    f"MERGE (a:{label} {{level: $level}}) "
                    "SET a.threshold_min = $tmin, a.threshold_max = $tmax, "
                    "a.approver_role = $role, a.description = $desc",
                    level=str(level),
                    tmin=float(row.get("Threshold_Min", 0)),
                    tmax=str(row.get("Threshold_Max", "")),
                    role=str(row.get("Approver_Role", "")),
                    desc=str(row.get("Description", "")),
                )
                n += 1
        return n

    def close(self):
        if self._driver:
            self._driver.close()
