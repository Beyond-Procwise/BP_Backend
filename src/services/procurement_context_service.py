"""Procurement Context Service — lifecycle tracking and proactive intelligence.

Determines where each document sits in the procurement lifecycle,
builds context briefs for agents, and detects anomalies proactively.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Procurement lifecycle stages in order
LIFECYCLE_STAGES = [
    "Need Identified",
    "RFQ Generated",
    "Quotes Received",
    "Quotes Evaluated",
    "Supplier Selected",
    "Negotiation",
    "PO Issued",
    "Invoice Received",
    "Invoice Matching",
    "Payment Approved",
    "Supplier Reviewed",
]

# Map document type + context to lifecycle stage
_STAGE_MAP = {
    ("Quote", False): "Quotes Received",
    ("Quote", True): "Quotes Evaluated",  # has_evaluation
    ("Purchase_Order", False): "PO Issued",
    ("Invoice", True): "Invoice Matching",  # has_po_link
    ("Invoice", False): "Invoice Received",
    ("Contract", False): "Supplier Selected",
}

# Anomaly detection thresholds
INVOICE_PO_TOLERANCE = 0.10  # 10% tolerance for invoice vs PO amount
QUOTE_EXPIRY_WARNING_DAYS = 7


class ProcurementContextService:
    """Tracks procurement lifecycle and provides proactive intelligence."""

    def __init__(self, agent_nick) -> None:
        self._agent_nick = agent_nick
        self._kgdrv = None  # lazy Neo4j driver for KG intelligence

    def _kg(self):
        """Lazy, cached Neo4j driver for KG intelligence. None if unreachable."""
        if self._kgdrv is None:
            try:
                import os
                from neo4j import GraphDatabase
                self._kgdrv = GraphDatabase.driver(
                    os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                    auth=(os.getenv("NEO4J_USERNAME", "neo4j"),
                          os.getenv("NEO4J_PASSWORD", "procwise2026")))
            except Exception:
                self._kgdrv = False  # unavailable
        return self._kgdrv or None

    def kg_supplier_intelligence(self, supplier_id) -> Dict[str, Any]:
        """Supplier history from the knowledge graph — invoice/PO/quote/contract
        counts and total invoiced USD — to inform supplier ranking, negotiation and
        opportunity reasoning. Fail-safe: returns {} if the KG is unavailable."""
        drv = self._kg()
        if drv is None or not supplier_id:
            return {}
        try:
            with drv.session() as s:
                # Pattern-comprehension counts (no cartesian blow-up); amount summed
                # via toFloat() since converted_amount_usd is stored as text in the KG.
                rec = s.run(
                    "MATCH (sup:Supplier {supplier_id:$sid}) RETURN "
                    "size([(sup)<-[:INVOICE_FROM_SUPPLIER]-(i) | i]) AS invoices, "
                    "size([(sup)<-[:PO_FROM_SUPPLIER]-(p) | p]) AS purchase_orders, "
                    "size([(sup)<-[:QUOTE_FROM_SUPPLIER]-(q) | q]) AS quotes, "
                    "size([(sup)-[:SUPPLIER_PARTY_TO_CONTRACT]->(c) | c]) AS contracts, "
                    "round(reduce(t=0.0, a IN [(sup)<-[:INVOICE_FROM_SUPPLIER]-(i) | "
                    "toFloat(i.converted_amount_usd)] | t + coalesce(a, 0.0))) AS invoiced_usd",
                    sid=str(supplier_id)).single()
            if not rec:
                return {}
            return {k: rec[k] for k in ("invoices", "purchase_orders", "quotes", "contracts", "invoiced_usd") if rec[k]}
        except Exception:
            return {}

    def kg_related_documents(self, doc_id, doc_type) -> List[Dict[str, str]]:
        """Related documents via the KG chain (Invoice<->PO<->Quote). Fail-safe []."""
        drv = self._kg()
        if drv is None or not doc_id:
            return []
        dt = (doc_type or "").lower()
        out: List[Dict[str, str]] = []
        try:
            with drv.session() as s:
                if "invoice" in dt:
                    for r in s.run(
                        "MATCH (i:Invoice {invoice_id:$id})-[:INVOICE_REFERENCES_PO]->(p:PurchaseOrder) "
                        "OPTIONAL MATCH (p)-[:PO_REFERENCES_QUOTE]->(q:Quote) "
                        "RETURN p.po_id AS po, collect(DISTINCT q.quote_id) AS quotes", id=str(doc_id)):
                        if r["po"]:
                            out.append({"type": "PurchaseOrder", "id": r["po"]})
                        out += [{"type": "Quote", "id": q} for q in (r["quotes"] or []) if q]
                elif "purchase" in dt or dt == "po":
                    for r in s.run(
                        "MATCH (p:PurchaseOrder {po_id:$id}) "
                        "OPTIONAL MATCH (p)-[:PO_REFERENCES_QUOTE]->(q:Quote) "
                        "OPTIONAL MATCH (p)<-[:INVOICE_REFERENCES_PO]-(i:Invoice) "
                        "RETURN collect(DISTINCT q.quote_id) AS quotes, collect(DISTINCT i.invoice_id) AS invs", id=str(doc_id)):
                        out += [{"type": "Quote", "id": q} for q in (r["quotes"] or []) if q]
                        out += [{"type": "Invoice", "id": v} for v in (r["invs"] or []) if v]
                elif "quote" in dt:
                    for r in s.run(
                        "MATCH (q:Quote {quote_id:$id})<-[:PO_REFERENCES_QUOTE]-(p:PurchaseOrder) "
                        "RETURN collect(DISTINCT p.po_id) AS pos", id=str(doc_id)):
                        out += [{"type": "PurchaseOrder", "id": v} for v in (r["pos"] or []) if v]
        except Exception:
            return []
        return out

    def determine_lifecycle_stage(
        self, doc_type: str, header: Dict[str, Any]
    ) -> str:
        """Determine where a document sits in the procurement lifecycle."""
        has_po = bool(header.get("po_id"))
        has_eval = bool(header.get("evaluation_score"))

        if doc_type == "Invoice" and has_po:
            return "Invoice Matching"
        if doc_type == "Invoice":
            return "Invoice Received"
        if doc_type == "Quote" and has_eval:
            return "Quotes Evaluated"
        if doc_type == "Quote":
            return "Quotes Received"
        if doc_type == "Purchase_Order":
            return "PO Issued"
        if doc_type == "Contract":
            return "Supplier Selected"
        return "Need Identified"

    def build_context_brief(
        self,
        doc_type: str,
        header: Dict[str, Any],
        patterns: Optional[List[Dict[str, Any]]] = None,
        related_docs: Optional[List[str]] = None,
        policies: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Build a procurement context brief for agent consumption."""
        stage = self.determine_lifecycle_stage(doc_type, header)

        # Document summary — key fields only, no raw data
        doc_summary = {}
        for key in ("invoice_id", "po_id", "quote_id", "contract_id",
                     "supplier_id", "supplier_name", "buyer_id",
                     "total_amount", "invoice_total_incl_tax",
                     "total_amount_incl_tax", "currency", "category"):
            val = header.get(key)
            if val:
                doc_summary[key] = str(val)

        # KG supplier intelligence (history, spend, contracts) — improves agent
        # reasoning (ranking/negotiation/opportunity). Fail-safe: {} if KG unavailable.
        supplier_intel = self.kg_supplier_intelligence(header.get("supplier_id"))

        return {
            "lifecycle_stage": stage,
            "document_type": doc_type,
            "document_summary": doc_summary,
            "related_documents": related_docs or [],
            "supplier_intelligence": supplier_intel,
            "patterns": [p.get("pattern_text", "") for p in (patterns or [])],
            "active_policies": policies or [],
            "next_expected_stage": self._next_stage(stage),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def detect_anomalies(
        self,
        header: Dict[str, Any],
        po_total: Optional[float] = None,
        quote_total: Optional[float] = None,
    ) -> List[str]:
        """Detect anomalies in the document vs related data."""
        anomalies: List[str] = []

        inv_total = self._to_float(
            header.get("invoice_total_incl_tax") or header.get("total_amount_incl_tax")
        )

        # Invoice vs PO amount check
        if po_total and inv_total and po_total > 0:
            deviation = abs(inv_total - po_total) / po_total
            if deviation > INVOICE_PO_TOLERANCE:
                direction = "exceeds" if inv_total > po_total else "below"
                anomalies.append(
                    f"Invoice total ({inv_total:.2f}) {direction} PO total "
                    f"({po_total:.2f}) by {deviation:.0%}"
                )

        # Invoice vs Quote check
        if quote_total and inv_total and quote_total > 0:
            deviation = abs(inv_total - quote_total) / quote_total
            if deviation > INVOICE_PO_TOLERANCE:
                anomalies.append(
                    f"Invoice total ({inv_total:.2f}) deviates from quote "
                    f"({quote_total:.2f}) by {deviation:.0%}"
                )

        # Missing critical fields
        if header.get("po_id") and not header.get("supplier_id"):
            anomalies.append("Invoice references PO but has no supplier identified")

        return anomalies

    def find_related_documents(
        self, doc_id: str, doc_type: str
    ) -> List[Dict[str, str]]:
        """Find related documents — authoritative source is the knowledge graph
        (Invoice<->PO<->Quote chain); falls back to the _trgt tables if the KG has
        nothing for this doc."""
        kg = self.kg_related_documents(doc_id, doc_type)
        if kg:
            return kg
        related: List[Dict[str, str]] = []
        try:
            conn = self._agent_nick.get_db_connection()
            with conn.cursor() as cur:
                if doc_type == "Invoice":
                    # Find linked PO
                    cur.execute(
                        "SELECT po_id FROM proc.bp_invoice_trgt WHERE invoice_id = %s AND po_id IS NOT NULL",
                        (doc_id,),
                    )
                    for r in cur.fetchall():
                        related.append({"type": "PurchaseOrder", "id": r[0]})
                        # Find quotes linked to this PO
                        cur.execute(
                            "SELECT DISTINCT quote_number FROM proc.bp_po_line_items_trgt "
                            "WHERE po_id = %s AND quote_number IS NOT NULL",
                            (r[0],),
                        )
                        for q in cur.fetchall():
                            related.append({"type": "Quote", "id": q[0]})

                elif doc_type == "Purchase_Order":
                    # Find linked quotes
                    cur.execute(
                        "SELECT DISTINCT quote_number FROM proc.bp_po_line_items_trgt "
                        "WHERE po_id = %s AND quote_number IS NOT NULL",
                        (doc_id,),
                    )
                    for r in cur.fetchall():
                        related.append({"type": "Quote", "id": r[0]})
                    # Find linked invoices
                    cur.execute(
                        "SELECT invoice_id FROM proc.bp_invoice_trgt WHERE po_id = %s",
                        (doc_id,),
                    )
                    for r in cur.fetchall():
                        related.append({"type": "Invoice", "id": r[0]})

                elif doc_type == "Quote":
                    # Find POs referencing this quote
                    cur.execute(
                        "SELECT DISTINCT po_id FROM proc.bp_po_line_items_trgt "
                        "WHERE quote_number = %s",
                        (doc_id,),
                    )
                    for r in cur.fetchall():
                        related.append({"type": "PurchaseOrder", "id": r[0]})

            conn.close()
        except Exception:
            logger.debug("Failed to find related documents", exc_info=True)

        return related

    def identify_opportunities(self) -> List[Dict[str, Any]]:
        """Proactive intelligence — identify actionable items."""
        opportunities: List[Dict[str, Any]] = []
        try:
            conn = self._agent_nick.get_db_connection()
            with conn.cursor() as cur:
                # POs without invoices (potential follow-up needed)
                cur.execute("""
                    SELECT p.po_id, p.supplier_name, p.order_date
                    FROM proc.bp_purchase_order_trgt p
                    LEFT JOIN proc.bp_invoice_trgt i ON i.po_id = p.po_id
                    WHERE i.invoice_id IS NULL
                    AND p.order_date < NOW() - INTERVAL '30 days'
                """)
                for r in cur.fetchall():
                    opportunities.append({
                        "type": "missing_invoice",
                        "message": f"PO {r[0]} from {r[1]} ({r[2]}) has no invoice after 30+ days",
                        "priority": "medium",
                    })

                # Quotes approaching expiry
                cur.execute("""
                    SELECT quote_id, supplier_id, validity_date
                    FROM proc.bp_quote
                    WHERE validity_date IS NOT NULL
                    AND validity_date BETWEEN NOW() AND NOW() + INTERVAL '7 days'
                """)
                for r in cur.fetchall():
                    opportunities.append({
                        "type": "expiring_quote",
                        "message": f"Quote {r[0]} from {r[1]} expires on {r[2]}",
                        "priority": "high",
                    })

            conn.close()
        except Exception:
            logger.debug("Failed to identify opportunities", exc_info=True)

        return opportunities

    @staticmethod
    def _next_stage(current: str) -> str:
        """Get the next expected procurement stage."""
        try:
            idx = LIFECYCLE_STAGES.index(current)
            if idx < len(LIFECYCLE_STAGES) - 1:
                return LIFECYCLE_STAGES[idx + 1]
        except ValueError:
            pass
        return ""

    @staticmethod
    def _to_float(val) -> float:
        try:
            return float(val) if val else 0.0
        except (ValueError, TypeError):
            return 0.0
