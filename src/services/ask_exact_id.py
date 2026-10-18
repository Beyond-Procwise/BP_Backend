"""Exact-ID pre-route for /ask: a PO number, part number or supplier code is looked up, not searched for.

Vector search finds text that *resembles* the question; it does not find "PO-2025-0257". That PO is in the
database, yet the live answer to "what is the status of PO-2025-0257?" was that it "is not among them"
(test F1). A question that names an identifier has one exact answer, and it lives in Postgres.

``extract`` finds identifiers with regexes (no model). ``fetch`` reads the matching rows with parameterised
queries and returns them as text for the prompt, in the same "verified data" slot the corpus facts use. When
nothing matches, the caller carries on with normal retrieval, so an identifier that only appears in a document
is still found there.

Business fields only. A supplier's bank, tax and contact columns are never selected: this lookup answers
"what is this?", not "give me its payment details".
"""
from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# PO-2025-0257 (dashed, how it is written in documents) and PO000001 (compact, how the bulk of the table holds it).
_PO = re.compile(r"\bPO[-\s]?(\d{4})[-\s](\d{3,5})\b|\bPO[-\s]?(\d{5,8})\b", re.I)
_PART = re.compile(r"\bITM[-\s]?(\d{5,8})\b", re.I)
_SUPPLIER = re.compile(r"\bSUP-[A-Za-z0-9]{3,60}\b")
# Contract ids as held in bp_contracts / bp_contract_master: OF-2026-0117, FA-2026-0077, SA-2026-0310, C00003.
_CONTRACT = re.compile(r"\b(?:(?:OF|FA|SA|MSA|NDA|SOW)-\d{4}-\d{3,5}|C\d{5})\b")

MAX_PER_KIND = 3
MAX_LINES = 12


def extract(query: str) -> Dict[str, List[str]]:
    """{"po": [...], "part": [...], "supplier": [...]} - identifiers as stored, de-duplicated, in order of appearance."""
    text = query if isinstance(query, str) else ""
    po: List[str] = []
    for m in _PO.finditer(text):
        ident = f"PO-{m.group(1)}-{m.group(2)}" if m.group(1) else f"PO{m.group(3)}"
        if ident not in po:
            po.append(ident)
    part = list(dict.fromkeys(f"ITM{m.group(1)}" for m in _PART.finditer(text)))
    supplier = list(dict.fromkeys(m.group(0) for m in _SUPPLIER.finditer(text)))
    contract = list(dict.fromkeys(m.group(0) for m in _CONTRACT.finditer(text)))
    found = {"po": po[:MAX_PER_KIND], "part": part[:MAX_PER_KIND], "supplier": supplier[:MAX_PER_KIND], "contract": contract[:MAX_PER_KIND]}
    return {k: v for k, v in found.items() if v}


def _rows(cur: Any, sql: str, params: tuple) -> List[Dict[str, Any]]:
    cur.execute(sql, params)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


_PO_COLS = ("po_id, supplier_id, supplier_name, po_status, approval_status, order_date, expected_delivery_date, currency, "
            "total_amount, total_amount_incl_tax, payment_terms, incoterm, deal_name, contract_id, po_revision")
_LINE_COLS = "po_id, line_number, item_id, item_description, quantity, unit_of_measure, unit_price, line_total, currency"
_SUPPLIER_COLS = ("supplier_id, supplier_name, trading_name, supplier_type, registered_country, country, is_preferred_supplier, "
                  "risk_score, default_currency, incoterms, delivery_lead_time_days")


_CONTRACT_COLS = ("contract_id, contract_title, contract_type, supplier_id, contract_start_date, contract_end_date, term_months, currency, "
                  "total_contract_value, payment_terms, contract_lifecycle_status, auto_renew_flag, renewal_term")


def fetch(conn_factory: Callable[[], Any], ids: Dict[str, List[str]]) -> Dict[str, Any]:
    """Rows for the identifiers: {"records": [{"kind", "id", "rows": [...], "lines": [...]}], "missing": [...]}.

    ``conn_factory`` returns a context-managed DB connection (``agent_nick.get_db_connection``). Any database
    failure returns nothing found, so the question falls back to normal retrieval rather than failing.
    """
    out: Dict[str, Any] = {"records": [], "missing": []}
    if not ids:
        return out
    try:
        with conn_factory() as conn:
            with conn.cursor() as cur:
                for po_id in ids.get("po", []):
                    rows = _rows(cur, f"SELECT {_PO_COLS} FROM proc.bp_purchase_order_trgt WHERE upper(po_id) = upper(%s)", (po_id,))
                    if not rows:
                        out["missing"].append(po_id)
                        continue
                    lines = _rows(cur, f"SELECT {_LINE_COLS} FROM proc.bp_po_line_items_trgt WHERE upper(po_id) = upper(%s) "
                                       "ORDER BY line_number LIMIT %s", (po_id, MAX_LINES + 1))
                    out["records"].append({"kind": "purchase order", "id": po_id, "rows": rows, "lines": lines})
                for item_id in ids.get("part", []):
                    lines = _rows(cur, f"SELECT {_LINE_COLS} FROM proc.bp_po_line_items_trgt WHERE upper(item_id) = upper(%s) "
                                       "ORDER BY po_id, line_number LIMIT %s", (item_id, MAX_LINES + 1))
                    if not lines:
                        out["missing"].append(item_id)
                        continue
                    out["records"].append({"kind": "part", "id": item_id, "rows": [], "lines": lines})
                for contract_id in ids.get("contract", []):
                    rows = _rows(cur, f"SELECT {_CONTRACT_COLS} FROM proc.bp_contracts WHERE upper(contract_id) = upper(%s)", (contract_id,))
                    if not rows:
                        rows = _rows(cur, "SELECT contract_id, contract_title, contract_type, supplier_id, contract_start_date, contract_end_date, "
                                          "currency, total_contract_value, payment_terms, contract_lifecycle_status, auto_renew_flag, renewal_term "
                                          "FROM proc.bp_contract_master WHERE upper(contract_id) = upper(%s)", (contract_id,))
                    if not rows:
                        out["missing"].append(contract_id)
                        continue
                    out["records"].append({"kind": "contract", "id": contract_id, "rows": rows, "lines": []})
                for sup_id in ids.get("supplier", []):
                    rows = _rows(cur, f"SELECT {_SUPPLIER_COLS} FROM proc.bp_supplier WHERE supplier_id = %s", (sup_id,))
                    if not rows:
                        out["missing"].append(sup_id)
                        continue
                    out["records"].append({"kind": "supplier", "id": sup_id, "rows": rows, "lines": []})
    except Exception:
        logger.exception("exact-id lookup failed; falling back to retrieval")
        return {"records": [], "missing": []}
    return out


def _value(v: Any) -> str:
    return "" if v is None else str(v)


def render(found: Dict[str, Any]) -> str:
    """The records as compact text, values exactly as stored; empty string when nothing matched."""
    blocks: List[str] = []
    for rec in found.get("records", []):
        head = f"{rec['kind'].capitalize()} {rec['id']} (found in the records)"
        body: List[str] = []
        for row in rec["rows"]:
            body.append("; ".join(f"{k}={_value(v)}" for k, v in row.items() if v not in (None, "")))
        lines = rec["lines"]
        if lines:
            body.append(f"Line items ({'first %d shown' % MAX_LINES if len(lines) > MAX_LINES else len(lines)}):")
            for ln in lines[:MAX_LINES]:
                body.append("- " + "; ".join(f"{k}={_value(v)}" for k, v in ln.items() if v not in (None, "")))
        blocks.append(head + "\n" + "\n".join(body))
    if found.get("missing") and blocks:
        blocks.append("Not found in the records: " + ", ".join(found["missing"]))
    return "\n\n".join(blocks)
