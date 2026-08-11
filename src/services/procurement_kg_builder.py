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
# Source tables for the graph's nodes: (table, primary key, Neo4j label).
#
# THESE MUST BE THE _trgt TIER. The six document entries used to name
# proc.bp_invoice / bp_quote / bp_purchase_order and their line-item tables —
# the pre-renovation names. Those tables were dropped when the pipeline moved to
# _stg -> _trgt, so every document loader read a table that did not exist,
# caught the error, logged it at DEBUG, and returned 0. The graph stopped
# gaining documents on 2026-07-31 and the job kept reporting success. Verified
# 2026-08-11: all six were MISSING from information_schema.
#
# The PK names are the real column names, checked against information_schema
# rather than assumed. Two were also wrong before: bp_category has no
# `category_id` (its columns are item_description, category) and bp_policy has
# `policy_id`, not `id` — so Policy nodes never loaded either, silently, from 19
# live rows.
#
# proc.bp_approvals does not exist at all and the Approval entry is removed
# rather than left to fail quietly every run.
ENTITY_TABLE_MAP = {
    "Supplier": ("proc.bp_supplier", "supplier_id", "Supplier"),
    "Contract": ("proc.bp_contracts", "contract_id", "Contract"),
    "Invoice": ("proc.bp_invoice_trgt", "invoice_id", "Invoice"),
    "InvoiceLine": ("proc.bp_invoice_line_items_trgt", "invoice_line_id", "InvoiceLine"),
    "PurchaseOrder": ("proc.bp_purchase_order_trgt", "po_id", "PurchaseOrder"),
    "POLine": ("proc.bp_po_line_items_trgt", "po_line_id", "POLine"),
    "Quote": ("proc.bp_quote_trgt", "quote_id", "Quote"),
    "QuoteLine": ("proc.bp_quote_line_items_trgt", "quote_line_id", "QuoteLine"),
    "Policy": ("proc.bp_policy", "policy_id", "Policy"),
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

    # Rows fetched per round trip. The previous code used `LIMIT 5000` with no
    # offset, which was not a batch size but a silent cap: bp_supplier has 5,028
    # rows, so 28 suppliers never reached the graph and nothing said so. The
    # line-item tables are far worse — bp_quote_line_items_trgt alone has
    # 115,814 rows, so a 5,000 cap would have loaded 4% of it and reported
    # success.
    _PAGE = 5000

    def _load_entity(self, table: str, pk: str, label: str) -> int:
        """Load every row from a source table as Neo4j nodes. Paged, not capped."""
        try:
            conn = self._agent_nick.get_db_connection()
            try:
                rows: list = []
                with conn.cursor() as cur:
                    offset = 0
                    while True:
                        cur.execute(
                            f"SELECT * FROM {table} ORDER BY {pk} "
                            f"LIMIT {self._PAGE} OFFSET {offset}"
                        )
                        page = cur.fetchall()
                        if not page:
                            break
                        if not rows:
                            cols = [d.name for d in cur.description]
                        rows.extend(page)
                        if len(page) < self._PAGE:
                            break
                        offset += self._PAGE
            finally:
                conn.close()
        except Exception:
            # WARNING, not DEBUG. This was the line that hid a broken graph for
            # eleven days: six tables failed to read on every run and produced
            # no output at any normal log level.
            logger.warning(
                "KG: could not read %s — no %s nodes will be loaded",
                table, label, exc_info=True,
            )
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

    def _supplier_master_is_empty(self) -> bool:
        """True when proc.bp_supplier has no rows to build Supplier nodes from."""
        try:
            conn = self._agent_nick.get_db_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT count(*) FROM proc.bp_supplier")
                    return (cur.fetchone() or [0])[0] == 0
            finally:
                conn.close()
        except Exception:
            # Cannot tell — assume it is populated. Inferring suppliers into a
            # graph that already has a real supplier master is the damaging
            # direction, so that is the one to avoid on uncertainty.
            logger.warning("KG: could not size proc.bp_supplier; "
                           "skipping supplier inference", exc_info=True)
            return False

    def _infer_suppliers(self) -> int:
        """Create Supplier nodes from extracted data, ONLY if bp_supplier is empty.

        The docstring has always said "if bp_supplier is empty". The check was
        never written, so this ran on every rebuild against a populated supplier
        master, and it fabricates identifiers:

            MERGE (s:Supplier {supplier_name: p.supplier_name})
            ON CREATE SET s.supplier_id = p.supplier_name

        That sets a supplier_id to a company NAME. Such a node can never join to
        proc.bp_supplier, inflates every supplier count taken from the graph,
        and puts a third id shape alongside the real SUP-* ones. Measured
        2026-08-11: the graph held 5,324 suppliers against 5,028 in the master.

        The invoice/quote/contract branches are milder but the same idea in
        reverse — they set supplier_name to the supplier_id, so a node's name is
        "SUP-Northgate". Also fabrication, also only defensible when there is no
        master to read.

        So the guard the docstring promised is now real. With a populated master
        this returns 0 and the supplier set in the graph matches the source.
        """
        if not self._supplier_master_is_empty():
            logger.info(
                "KG: proc.bp_supplier is populated — skipping supplier "
                "inference (it fabricates ids from names)"
            )
            return 0

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
