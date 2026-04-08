"""Procurement Knowledge Graph Builder.

Reads the KG schema from the Excel workbook and builds the complete
Neo4j knowledge graph with all entities, relationships, attributes,
approval levels, TPRM profiles, finance hierarchy, and views.

Runs unsupervised — called by BackendScheduler on startup and periodically.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

KG_WORKBOOK_PATH = os.getenv(
    "KG_WORKBOOK_PATH",
    "/home/muthu/Downloads/Procurement Knowledge Graphv2 (1).xlsx",
)


class ProcurementKGBuilder:
    """Builds and maintains the procurement knowledge graph in Neo4j."""

    def __init__(self, agent_nick) -> None:
        self._agent_nick = agent_nick
        self._driver = self._get_driver()

    def _get_driver(self):
        try:
            from neo4j import GraphDatabase
            s = self._agent_nick.settings
            return GraphDatabase.driver(
                getattr(s, "neo4j_uri", "bolt://localhost:7687"),
                auth=(getattr(s, "neo4j_username", "neo4j"),
                      getattr(s, "neo4j_password", "neo4j")),
            )
        except Exception:
            logger.exception("Failed to create Neo4j driver")
            return None

    def build_full_graph(self) -> Dict[str, int]:
        """Build the complete KG from the Excel workbook. Returns counts."""
        if not self._driver:
            return {"error": "No Neo4j driver"}

        if not os.path.exists(KG_WORKBOOK_PATH):
            logger.warning("KG workbook not found at %s", KG_WORKBOOK_PATH)
            return {"error": f"Workbook not found: {KG_WORKBOOK_PATH}"}

        xls = pd.ExcelFile(KG_WORKBOOK_PATH)
        counts = {}

        # 1. Schema nodes (Entity types as schema reference)
        counts["entity_types"] = self._load_entity_types(xls)

        # 2. Relationships schema
        counts["relationship_types"] = self._load_relationship_types(xls)

        # 3. Approval levels
        counts["finance_approvals"] = self._load_approval_levels(
            xls, "Finance_Approval_Levels", "FinanceApprovalLevel"
        )
        counts["procurement_approvals"] = self._load_approval_levels(
            xls, "Procurement_Approval_Levels", "ProcurementApprovalLevel"
        )

        # 4. Supplier attributes
        counts["supplier_attributes"] = self._load_keyed_attributes(
            xls, "Supplier_Attributes", "Supplier", "Supplier_ID"
        )

        # 5. Category attributes
        counts["category_attributes"] = self._load_keyed_attributes(
            xls, "Category_Attributes", "Category", "Category_ID"
        )

        # 6. TPRM profiles
        counts["tprm_profiles"] = self._load_tprm(xls)

        # 7. Risk scoring model
        counts["risk_model"] = self._load_risk_model(xls)

        # 8. Finance hierarchy
        counts["finance_hierarchy"] = self._load_finance_hierarchy(xls)

        # 9. Extended nodes and relationships
        counts["extended_nodes"] = self._load_extended_nodes(xls)
        counts["extended_relationships"] = self._load_extended_relationships(xls)

        # 10. Approval workflow relationships
        counts["approval_relationships"] = self._load_approval_relationships(xls)

        # 11. KG relationships (Supplier-Category, TPRM, Finance)
        counts["supplier_category_rel"] = self._load_generic_relationships(
            xls, "KG_Supplier_Category_Rel"
        )
        counts["tprm_relationships"] = self._load_generic_relationships(
            xls, "KG_TPRM_Relationships"
        )
        counts["finance_relationships"] = self._load_generic_relationships(
            xls, "KG_Finance_Relationships"
        )

        # 12. Live data from bp_ tables
        counts["bp_suppliers"] = self._sync_bp_suppliers()
        counts["bp_invoices"] = self._sync_bp_data("bp_invoice", "Invoice", "invoice_id")
        counts["bp_purchase_orders"] = self._sync_bp_data("bp_purchase_order", "PurchaseOrder", "po_id")
        counts["bp_quotes"] = self._sync_bp_data("bp_quote", "Quote", "quote_id")
        counts["bp_contracts"] = self._sync_bp_data("bp_contracts", "Contract", "contract_id")

        logger.info("[KG Builder] Complete. Counts: %s", counts)
        return counts

    # ------------------------------------------------------------------
    # Schema loading
    # ------------------------------------------------------------------
    def _load_entity_types(self, xls: pd.ExcelFile) -> int:
        df = xls.parse("Entities")
        count = 0
        with self._driver.session() as session:
            for _, row in df.iterrows():
                code = row.get("Entity Code", "")
                if not code:
                    continue
                session.run(
                    "MERGE (e:EntityType {code: $code}) "
                    "SET e.name = $name, e.description = $desc, "
                    "e.functional_group = $group, e.source_systems = $sources",
                    code=str(code),
                    name=str(row.get("Entity Name", "")),
                    desc=str(row.get("Description", "")),
                    group=str(row.get("Functional Group", "")),
                    sources=str(row.get("Typical Source Systems", "")),
                )
                count += 1

        # Also load attributes for each entity
        if "Attributes" in xls.sheet_names:
            attr_df = xls.parse("Attributes")
            with self._driver.session() as session:
                for _, row in attr_df.iterrows():
                    entity_code = row.get("Entity Code", "")
                    attr_code = row.get("Attribute Code", "")
                    if not entity_code or not attr_code:
                        continue
                    session.run(
                        "MATCH (e:EntityType {code: $entity_code}) "
                        "MERGE (a:Attribute {code: $attr_code, entity: $entity_code}) "
                        "SET a.data_type = $dtype, a.description = $desc, "
                        "a.required = $required, a.example = $example "
                        "MERGE (e)-[:HAS_ATTRIBUTE]->(a)",
                        entity_code=str(entity_code),
                        attr_code=str(attr_code),
                        dtype=str(row.get("Data Type", "")),
                        desc=str(row.get("Description", "")),
                        required=bool(row.get("Required", 0)),
                        example=str(row.get("Example", "")),
                    )
        return count

    def _load_relationship_types(self, xls: pd.ExcelFile) -> int:
        df = xls.parse("Relationships")
        count = 0
        with self._driver.session() as session:
            for _, row in df.iterrows():
                code = row.get("Edge Code", "")
                if not code:
                    continue
                session.run(
                    "MERGE (r:RelationshipType {code: $code}) "
                    "SET r.name = $name, r.from_entity = $from_e, "
                    "r.to_entity = $to_e, r.cardinality = $card, "
                    "r.functional_group = $group, r.notes = $notes",
                    code=str(code),
                    name=str(row.get("Edge Name", "")),
                    from_e=str(row.get("From Entity Code", "")),
                    to_e=str(row.get("To Entity Code", "")),
                    card=str(row.get("Cardinality", "")),
                    group=str(row.get("Functional Group", "")),
                    notes=str(row.get("Notes", "")),
                )
                count += 1
        return count

    # ------------------------------------------------------------------
    # Approval levels
    # ------------------------------------------------------------------
    def _load_approval_levels(
        self, xls: pd.ExcelFile, sheet: str, label: str
    ) -> int:
        if sheet not in xls.sheet_names:
            return 0
        df = xls.parse(sheet)
        count = 0
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
                    tmin=self._safe_num(row.get("Threshold_Min")),
                    tmax=str(row.get("Threshold_Max", "")),
                    role=str(row.get("Approver_Role", "")),
                    desc=str(row.get("Description", "")),
                )
                count += 1
        return count

    # ------------------------------------------------------------------
    # Keyed attributes (Supplier, Category)
    # ------------------------------------------------------------------
    def _load_keyed_attributes(
        self, xls: pd.ExcelFile, sheet: str, label: str, id_col: str
    ) -> int:
        if sheet not in xls.sheet_names:
            return 0
        df = xls.parse(sheet)
        count = 0
        with self._driver.session() as session:
            for _, row in df.iterrows():
                entity_id = row.get(id_col, "")
                attr = row.get("Attribute", "")
                if not entity_id or not attr:
                    continue
                name = row.get(f"{label}_Name", "")
                safe_attr = attr.replace(" ", "_").lower()
                session.run(
                    f"MERGE (n:{label} {{{id_col.lower()}: $eid}}) "
                    f"SET n.name = $name, n.{safe_attr} = $val",
                    eid=str(entity_id),
                    name=str(name),
                    val=str(row.get("Value", "")),
                )
                count += 1
        return count

    # ------------------------------------------------------------------
    # TPRM
    # ------------------------------------------------------------------
    def _load_tprm(self, xls: pd.ExcelFile) -> int:
        count = 0
        if "TPRM_Profile" in xls.sheet_names:
            df = xls.parse("TPRM_Profile")
            with self._driver.session() as session:
                for _, row in df.iterrows():
                    sid = row.get("Supplier_ID", "")
                    if not sid:
                        continue
                    session.run(
                        "MERGE (p:TPRMProfile {supplier_id: $sid}) "
                        "SET p.risk_tier = $tier, p.overall_score = $score, "
                        "p.last_assessed = $assessed, p.next_review = $review, "
                        "p.risk_owner = $owner",
                        sid=str(sid),
                        tier=str(row.get("Risk_Tier", "")),
                        score=self._safe_num(row.get("Overall_Score")),
                        assessed=str(row.get("Last_Assessed", "")),
                        review=str(row.get("Next_Review", "")),
                        owner=str(row.get("Risk_Owner", "")),
                    )
                    # Link to supplier
                    session.run(
                        "MATCH (s:Supplier {supplier_id: $sid}) "
                        "MATCH (p:TPRMProfile {supplier_id: $sid}) "
                        "MERGE (s)-[:HAS_PROFILE]->(p)",
                        sid=str(sid),
                    )
                    count += 1

        for sheet, label in [("TPRM_Domains", "TPRMDomain"),
                             ("TPRM_Controls", "TPRMControl"),
                             ("TPRM_Events", "TPRMEvent")]:
            if sheet not in xls.sheet_names:
                continue
            df = xls.parse(sheet)
            with self._driver.session() as session:
                for _, row in df.iterrows():
                    sid = row.get("Supplier_ID", "")
                    if not sid:
                        continue
                    props = {k.lower(): str(v) for k, v in row.items()
                             if k != "Supplier_ID" and pd.notna(v)}
                    props["supplier_id"] = str(sid)
                    session.run(
                        f"MERGE (n:{label} {{supplier_id: $sid, name: $name}}) SET n += $props",
                        sid=str(sid),
                        name=str(row.get("Domain", row.get("Control", row.get("Event", "")))),
                        props=props,
                    )
                    count += 1
        return count

    # ------------------------------------------------------------------
    # Risk scoring model
    # ------------------------------------------------------------------
    def _load_risk_model(self, xls: pd.ExcelFile) -> int:
        if "Risk_Scoring_Model" not in xls.sheet_names:
            return 0
        df = xls.parse("Risk_Scoring_Model")
        count = 0
        with self._driver.session() as session:
            for _, row in df.iterrows():
                comp = row.get("Component", "")
                if not comp:
                    continue
                session.run(
                    "MERGE (r:RiskComponent {name: $name}) "
                    "SET r.weight = $weight, r.example_score = $score, "
                    "r.weighted_contribution = $contrib",
                    name=str(comp),
                    weight=self._safe_num(row.get("Weight")),
                    score=self._safe_num(row.get("Example_Score")),
                    contrib=self._safe_num(row.get("Weighted_Contribution")),
                )
                count += 1
        return count

    # ------------------------------------------------------------------
    # Finance hierarchy
    # ------------------------------------------------------------------
    def _load_finance_hierarchy(self, xls: pd.ExcelFile) -> int:
        if "Finance_Budget_Hierarchy" not in xls.sheet_names:
            return 0
        df = xls.parse("Finance_Budget_Hierarchy")
        count = 0
        with self._driver.session() as session:
            for _, row in df.iterrows():
                node_id = row.get("finance_budget_hierarchy_id", "")
                if not node_id:
                    continue
                props = {}
                for col in df.columns:
                    val = row.get(col)
                    if pd.notna(val):
                        safe_key = col.lower().replace(" ", "_")
                        props[safe_key] = self._safe_num(val) if isinstance(val, (int, float)) else str(val)

                session.run(
                    "MERGE (n:FinanceBudgetHierarchy {node_id: $nid}) SET n += $props",
                    nid=str(node_id),
                    props=props,
                )

                # Parent relationship
                parent = row.get("parent_budget_id")
                if pd.notna(parent) and str(parent).strip():
                    session.run(
                        "MATCH (child:FinanceBudgetHierarchy {node_id: $child}) "
                        "MATCH (parent:FinanceBudgetHierarchy {node_id: $parent}) "
                        "MERGE (child)-[:ROLLS_UP_TO]->(parent)",
                        child=str(node_id),
                        parent=str(parent),
                    )
                count += 1
        return count

    # ------------------------------------------------------------------
    # Extended nodes and relationships
    # ------------------------------------------------------------------
    def _load_extended_nodes(self, xls: pd.ExcelFile) -> int:
        if "Extended_Nodes" not in xls.sheet_names:
            return 0
        df = xls.parse("Extended_Nodes")
        count = 0
        with self._driver.session() as session:
            for _, row in df.iterrows():
                node_type = row.get("Node_Type", "")
                name = row.get("Node_Name", "")
                if not node_type or not name:
                    continue
                attr_key = str(row.get("Attribute_Key", "")).lower().replace(" ", "_")
                attr_val = row.get("Attribute_Value", "")
                session.run(
                    f"MERGE (n:{node_type} {{name: $name}}) "
                    f"SET n.{attr_key} = $val, n.description = $desc",
                    name=str(name),
                    val=str(attr_val) if pd.notna(attr_val) else "",
                    desc=str(row.get("Description", "")),
                )
                count += 1
        return count

    def _load_extended_relationships(self, xls: pd.ExcelFile) -> int:
        if "Extended_Relationships" not in xls.sheet_names:
            return 0
        df = xls.parse("Extended_Relationships")
        count = 0
        with self._driver.session() as session:
            for _, row in df.iterrows():
                from_node = row.get("From_Node", "")
                rel = row.get("Relationship", "")
                to_node = row.get("To_Node", "")
                if not from_node or not rel or not to_node:
                    continue
                safe_rel = rel.upper().replace(" ", "_")
                session.run(
                    f"MATCH (a {{name: $from_name}}) "
                    f"MATCH (b {{name: $to_name}}) "
                    f"MERGE (a)-[:{safe_rel}]->(b)",
                    from_name=str(from_node),
                    to_name=str(to_node),
                )
                count += 1
        return count

    def _load_approval_relationships(self, xls: pd.ExcelFile) -> int:
        if "KG_Approval_Relationships" not in xls.sheet_names:
            return 0
        df = xls.parse("KG_Approval_Relationships")
        count = 0
        with self._driver.session() as session:
            for _, row in df.iterrows():
                from_n = row.get("From_Node", "")
                rel = row.get("Relationship", "")
                to_n = row.get("To_Node", "")
                if not from_n or not rel or not to_n:
                    continue
                safe_rel = rel.upper().replace(" ", "_")
                session.run(
                    f"MERGE (a:ApprovalNode {{name: $from_n}}) "
                    f"MERGE (b:ApprovalNode {{name: $to_n}}) "
                    f"MERGE (a)-[:{safe_rel} {{condition: $cond, description: $desc}}]->(b)",
                    from_n=str(from_n),
                    to_n=str(to_n),
                    cond=str(row.get("Condition", "")),
                    desc=str(row.get("Description", "")),
                )
                count += 1
        return count

    def _load_generic_relationships(self, xls: pd.ExcelFile, sheet: str) -> int:
        if sheet not in xls.sheet_names:
            return 0
        df = xls.parse(sheet)
        count = 0
        with self._driver.session() as session:
            for _, row in df.iterrows():
                from_n = row.get("From_Node", "")
                rel = row.get("Relationship", "")
                to_n = row.get("To_Node", "")
                if not from_n or not rel or not to_n:
                    continue
                safe_rel = rel.upper().replace(" ", "_")
                desc = str(row.get("Description", ""))
                cond = str(row.get("Condition", ""))
                session.run(
                    f"MERGE (a:KGNode {{name: $from_n}}) "
                    f"MERGE (b:KGNode {{name: $to_n}}) "
                    f"MERGE (a)-[:{safe_rel} {{description: $desc, condition: $cond}}]->(b)",
                    from_n=str(from_n),
                    to_n=str(to_n),
                    desc=desc,
                    cond=cond,
                )
                count += 1
        return count

    # ------------------------------------------------------------------
    # Live data sync from bp_ tables
    # ------------------------------------------------------------------
    def _sync_bp_suppliers(self) -> int:
        try:
            conn = self._agent_nick.get_db_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT supplier_id, supplier_name, trading_name, supplier_type, "
                        "registered_country, risk_score, default_currency, is_preferred_supplier "
                        "FROM proc.bp_supplier LIMIT 500"
                    )
                    rows = cur.fetchall()
                    cols = [d.name for d in cur.description]
            finally:
                conn.close()

            count = 0
            with self._driver.session() as session:
                for row in rows:
                    data = dict(zip(cols, row))
                    sid = data.get("supplier_id", "")
                    if not sid:
                        continue
                    props = {k: str(v) for k, v in data.items() if v is not None}
                    session.run(
                        "MERGE (s:Supplier {supplier_id: $sid}) SET s += $props",
                        sid=str(sid),
                        props=props,
                    )
                    count += 1
            return count
        except Exception:
            logger.debug("bp_supplier sync failed", exc_info=True)
            return 0

    def _sync_bp_data(
        self, table: str, label: str, pk_col: str
    ) -> int:
        try:
            conn = self._agent_nick.get_db_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT * FROM proc.{table} LIMIT 500")
                    rows = cur.fetchall()
                    cols = [d.name for d in cur.description]
            finally:
                conn.close()

            count = 0
            with self._driver.session() as session:
                for row in rows:
                    data = dict(zip(cols, row))
                    pk = data.get(pk_col, "")
                    if not pk:
                        continue
                    props = {}
                    for k, v in data.items():
                        if v is not None:
                            props[k] = str(v) if not isinstance(v, (int, float)) else v
                    session.run(
                        f"MERGE (n:{label} {{{pk_col}: $pk}}) SET n += $props",
                        pk=str(pk),
                        props=props,
                    )
                    # Link to supplier
                    supplier = data.get("supplier_id") or data.get("supplier_name")
                    if supplier:
                        session.run(
                            f"MATCH (n:{label} {{{pk_col}: $pk}}) "
                            "MERGE (s:Supplier {name: $sname}) "
                            f"MERGE (s)-[:HAS_{label.upper()}]->(n)",
                            pk=str(pk),
                            sname=str(supplier),
                        )
                    count += 1
            return count
        except Exception:
            logger.debug("bp_%s sync failed", table, exc_info=True)
            return 0

    @staticmethod
    def _safe_num(val) -> Any:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return str(val)

    def close(self):
        if self._driver:
            self._driver.close()
