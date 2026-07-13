"""Real answers to questions about the corpus, read straight from the _trgt tables.

The ask bar could not answer "which suppliers do we buy from". Not because the model was
weak, but because nothing in that path ever queried the data: /workflows/ask went to Qdrant
for similar *text* and to a file of canned demo Q&A, and never once to proc.bp_supplier. So
the honest answer to a question about our own suppliers was assembled out of whatever the
vector store found lying around, and a demo fixture supplied names — Global Facilities,
BrightStage, TechCore — that exist nowhere in this database.

Vector search is the right tool for "what does our policy say about single-sourcing". It is
the wrong tool for "how much did we spend" — that is a SELECT, and it has an exact answer.
This module supplies the exact answers, as facts the model must ground its reply in.

The queries below are read-only, fixed (no interpolation of user text into SQL), bounded by
LIMIT, and read _trgt — the tier the product reads everywhere else. When a question matches
no intent here, we return None and the caller falls back to vector retrieval, which is what
should happen for genuinely open questions.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Intent -> the facts that answer it. Order matters: the first match wins, so the more
# specific patterns (findings, spend) sit above the broader ones (suppliers).
_INTENTS: List[tuple[str, re.Pattern]] = [
    ("policies", re.compile(r"\b(polic|rule|governance|complian|threshold|approval limit|mandate)\w*\b", re.I)),
    ("deals", re.compile(r"\b(deal|linked|quote to (po|invoice)|end[- ]to[- ]end|lifecycle)\w*\b", re.I)),
    ("findings", re.compile(r"\b(finding|discrepanc|exception|issue|problem|flag|over[- ]?bill|mismatch|risk)\w*\b", re.I)),
    ("spend", re.compile(r"\b(spend|spent|cost|total|invoiced|value|amount|money)\w*\b", re.I)),
    ("quotes", re.compile(r"\bquot\w*\b", re.I)),
    ("purchase_orders", re.compile(r"\b(purchase order|po|pos)\b", re.I)),
    ("invoices", re.compile(r"\binvoic\w*\b", re.I)),
    ("suppliers", re.compile(r"\b(supplier|vendor|who do we buy|buy from)\w*\b", re.I)),
]

_MAX_ROWS = 10


def detect_intent(query: str) -> Optional[str]:
    if not isinstance(query, str) or not query.strip():
        return None
    for name, pattern in _INTENTS:
        if pattern.search(query):
            return name
    return None


def _rows(cur, sql: str, limit: int = _MAX_ROWS) -> List[Dict[str, Any]]:
    cur.execute(sql, (limit,))
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _one(cur, sql: str) -> List[Dict[str, Any]]:
    """A single aggregate row — no LIMIT, because a total must not be truncated."""
    cur.execute(sql)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _fetch(cur, intent: str) -> Dict[str, Any]:
    if intent == "policies":
        # The governed policy set (proc.bp_policy) is what the agents are actually held to.
        # The ask bar used to answer policy questions from a demo fixture, so it described
        # rules this organisation had never adopted.
        return {
            # policy_status is a smallint flag, not a word: 1 = in force.
            "policies_in_force": _rows(cur, """
                SELECT policy_name, policy_type, policy_desc, version
                  FROM proc.bp_policy
                 WHERE policy_status IS NULL OR policy_status = 1
                 ORDER BY policy_type, policy_name
                 LIMIT %s"""),
        }

    if intent == "deals":
        return {
            "deals": _rows(cur, """
                SELECT deal_id, deal_name, supplier_name, deal_date, quote_count
                  FROM proc.bp_deal_overview
                 ORDER BY deal_date DESC NULLS LAST
                 LIMIT %s"""),
            "documents_in_deals": _rows(cur, """
                SELECT deal_id, doc_type, doc_number, doc_date
                  FROM proc.bp_deal_documents
                 ORDER BY deal_id, doc_date NULLS LAST
                 LIMIT %s"""),
        }

    if intent == "suppliers":
        return {
            "totals": _one(cur, """
                SELECT COUNT(DISTINCT i.supplier_id)::int AS suppliers_we_have_invoices_from,
                       (SELECT COUNT(*)::int FROM proc.bp_supplier) AS suppliers_on_record
                  FROM proc.bp_invoice_trgt i"""),
            "suppliers_we_buy_from": _rows(cur, """
                SELECT s.supplier_name,
                       COUNT(i.invoice_id)::int          AS invoices,
                       SUM(i.invoice_amount)::numeric    AS invoiced_amount,
                       MAX(i.currency)                   AS currency
                  FROM proc.bp_invoice_trgt i
                  JOIN proc.bp_supplier s ON s.supplier_id = i.supplier_id
                 GROUP BY s.supplier_name
                 ORDER BY invoices DESC, invoiced_amount DESC NULLS LAST
                 LIMIT %s"""),
        }

    if intent == "spend":
        return {
            "invoiced_spend_by_supplier": _rows(cur, """
                SELECT s.supplier_name,
                       SUM(i.invoice_amount)::numeric AS invoiced_amount,
                       MAX(i.currency)                AS currency
                  FROM proc.bp_invoice_trgt i
                  JOIN proc.bp_supplier s ON s.supplier_id = i.supplier_id
                 GROUP BY s.supplier_name
                 ORDER BY invoiced_amount DESC NULLS LAST
                 LIMIT %s"""),
            # The corpus is mixed-currency, so a single grand total would be a lie. Give the
            # model the split and let it say so.
            "totals_by_currency": _rows(cur, """
                SELECT currency,
                       COUNT(*)::int                  AS invoices,
                       SUM(invoice_amount)::numeric   AS invoiced_amount
                  FROM proc.bp_invoice_trgt
                 GROUP BY currency
                 ORDER BY invoiced_amount DESC NULLS LAST
                 LIMIT %s"""),
        }

    if intent == "findings":
        return {
            # The totals come first and are stated outright. The by-type list below is capped
            # at ten rows out of sixteen, and a model handed a truncated list will add it up
            # and present the sum as the total — it answered "647 unresolved cases" against a
            # real 685. Never make it do arithmetic it cannot check; give it the total.
            "totals": _one(cur, """
                SELECT COUNT(*)::int AS open_findings_total,
                       COUNT(*) FILTER (WHERE severity = 'critical')::int AS critical_total,
                       COUNT(DISTINCT issue_type)::int AS distinct_issue_types,
                       COUNT(DISTINCT doc_pk_candidate)::int AS documents_affected
                  FROM proc.bp_extraction_discrepancy
                 WHERE status = 'open'"""),
            "open_findings_by_type": _rows(cur, """
                SELECT issue_type,
                       COUNT(*)::int AS findings,
                       COUNT(*) FILTER (WHERE severity = 'critical')::int AS critical
                  FROM proc.bp_extraction_discrepancy
                 WHERE status = 'open'
                 GROUP BY issue_type
                 ORDER BY findings DESC
                 LIMIT %s"""),
            "largest_over_billing": _rows(cur, """
                SELECT doc_pk_candidate AS document,
                       issue_type,
                       severity,
                       notes,
                       NULLIF(computed_value, '')::numeric AS amount_over_po
                  FROM proc.bp_extraction_discrepancy
                 WHERE status = 'open' AND issue_type = 'amount_over_po'
                 ORDER BY NULLIF(computed_value, '')::numeric DESC NULLS LAST
                 LIMIT %s"""),
        }

    if intent == "invoices":
        return {
            "totals": _one(cur, """
                SELECT COUNT(*)::int AS invoices_total,
                       COUNT(*) FILTER (WHERE po_id IS NOT NULL)::int AS citing_a_po
                  FROM proc.bp_invoice_trgt"""),
            # The per-supplier count is stated, never left to be inferred. The list below is a
            # top-N sample, and asked "who has the most invoices" a model handed a sample will
            # count the rows in it — it answered "Coffee Bliss: 8 invoices" off a 10-row list
            # when the true count is 7. If a number can be counted in SQL, it is counted here.
            "invoice_count_by_supplier": _rows(cur, """
                SELECT s.supplier_name, COUNT(*)::int AS invoices
                  FROM proc.bp_invoice_trgt i
                  JOIN proc.bp_supplier s ON s.supplier_id = i.supplier_id
                 GROUP BY s.supplier_name
                 ORDER BY invoices DESC
                 LIMIT %s"""),
            "largest_invoices_sample": _rows(cur, """
                SELECT i.invoice_id, s.supplier_name, i.invoice_amount, i.currency,
                       i.invoice_date, i.po_id
                  FROM proc.bp_invoice_trgt i
                  LEFT JOIN proc.bp_supplier s ON s.supplier_id = i.supplier_id
                 ORDER BY i.invoice_amount DESC NULLS LAST
                 LIMIT %s"""),
        }

    if intent == "quotes":
        return {
            "quotes": _rows(cur, """
                SELECT q.quote_id, s.supplier_name, q.total_amount, q.currency, q.quote_date
                  FROM proc.bp_quote_trgt q
                  LEFT JOIN proc.bp_supplier s ON s.supplier_id = q.supplier_id
                 ORDER BY q.quote_date DESC NULLS LAST
                 LIMIT %s"""),
        }

    if intent == "purchase_orders":
        return {
            "purchase_orders": _rows(cur, """
                SELECT p.po_id, s.supplier_name, p.total_amount, p.currency, p.order_date
                  FROM proc.bp_purchase_order_trgt p
                  LEFT JOIN proc.bp_supplier s ON s.supplier_id = p.supplier_id
                 ORDER BY p.total_amount DESC NULLS LAST
                 LIMIT %s"""),
        }

    return {}


def fetch_graph(query: str, limit: int = 6) -> List[Dict[str, Any]]:
    """What the knowledge graph knows about the things named in the question.

    The tables above say *what* is true. The graph says how those things connect — which
    supplier sits on which deal, which document belongs to which process, which agent owns
    which step. That relational context is the part a SELECT cannot supply, and it is why an
    answer about a supplier can also mention the deal the supplier is on.

    A graph miss is not an error: not every question is about a modelled entity.
    """
    try:
        from services import platform_kg

        return platform_kg.describe(query, limit=limit) or []
    except Exception:
        logger.debug("Knowledge-graph lookup unavailable", exc_info=True)
        return []


def fetch_facts(agent_nick, query: str) -> Optional[Dict[str, Any]]:
    """Facts that answer `query`: the counted rows, plus how the graph relates them."""
    intent = detect_intent(query)
    facts: Dict[str, Any] = {}
    if intent:
        try:
            with agent_nick.get_db_connection() as conn:
                with conn.cursor() as cur:
                    facts = _fetch(cur, intent)
        except Exception:
            # A question is better answered from the vector store than not at all.
            logger.exception("Corpus fact lookup failed for intent %s", intent)
            facts = {}

    graph = fetch_graph(query)
    if graph:
        facts["knowledge_graph"] = graph

    if not any(facts.values()):
        return None
    return {"intent": intent or "graph", **facts}


def render_facts(facts: Dict[str, Any]) -> str:
    """The facts as compact text for the prompt. Values are printed exactly as stored."""
    lines: List[str] = []
    for key, rows in facts.items():
        if key == "intent" or not rows:
            continue
        lines.append(f"{key.replace('_', ' ')}:")
        for row in rows:
            parts = [
                f"{k}={v}" for k, v in row.items() if v is not None and str(v).strip() != ""
            ]
            lines.append("  - " + ", ".join(parts))
    return "\n".join(lines)
