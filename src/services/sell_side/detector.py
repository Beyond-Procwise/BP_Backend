"""Propose sell-side opportunities from what the distributor has STATED and what we sold.

Nothing is inferred. An account that won a quote for item A is offered item B when the
distributor's own relation rows say so (`bp_catalog_item_relation`: replaced_by,
upgrade_of, refill_of, accessory_of, requires). History is the account's won quote lines
only, which is what we invoiced them (history scope `our_invoices`); an account's spend
elsewhere is unknown here and is never read as "they buy nothing".
"""
from __future__ import annotations

import datetime as dt
import logging
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set, Tuple

from src.services.sell_side import opportunities
from src.services.sell_side._db import dict_cursor

logger = logging.getLogger(__name__)

DETECTOR_TYPE = "catalog_relation"
# a relation proposes its target to an account that bought its source
_TYPE_OF = {"replaced_by": "upgrade", "upgrade_of": "upgrade", "refill_of": "refill",
            "accessory_of": "cross_sell", "requires": "cross_sell"}
_WORDS = {"replaced_by": "replaced by", "upgrade_of": "upgraded by", "refill_of": "refilled by",
          "accessory_of": "has the accessory", "requires": "requires"}
_ENDED = ("end_of_life", "end_of_sale")

_SQL = """
SELECT q.account_id, bought.distributor_id, bought.distributor_sku, l.quantity,
       r.relation_id, r.relation_type, r.to_sku,
       t.catalog_item_id AS target_id,
       cur.catalog_item_id AS bought_id,
       COALESCE(cur.lifecycle_status, bought.lifecycle_status) AS lifecycle_status,
       COALESCE(cur.end_of_life_date, bought.end_of_life_date) AS end_of_life_date
  FROM proc.bp_sales_quote q
  JOIN proc.bp_sales_quote_outcome o ON o.sales_quote_id = q.sales_quote_id AND o.outcome = 'won'
  JOIN proc.bp_sales_quote_line l ON l.sales_quote_id = q.sales_quote_id
  JOIN proc.bp_catalog_item bought ON bought.catalog_item_id = l.catalog_item_id
  LEFT JOIN proc.bp_catalog_item cur ON cur.distributor_id = bought.distributor_id
       AND cur.distributor_sku = bought.distributor_sku AND cur.valid_to IS NULL
  JOIN proc.bp_catalog_item_relation r ON r.distributor_id = bought.distributor_id
       AND r.from_sku = bought.distributor_sku
  JOIN proc.bp_catalog_item t ON t.distributor_id = r.distributor_id
       AND t.distributor_sku = r.to_sku AND t.valid_to IS NULL AND t.list_price IS NOT NULL
 WHERE (%(acct)s::text IS NULL OR q.account_id = %(acct)s)
 ORDER BY q.account_id, q.quote_date, q.sales_quote_id, l.sales_quote_line_id
"""


def _ended(status: Optional[str], on: Optional[dt.date]) -> bool:
    return status in _ENDED or (on is not None and on <= dt.date.today())


def detect(conn: Any, *, account_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Create the opportunities the stated relations support; return them (with their
    justifications). Safe to re-run: an open or won opportunity for the same account,
    target and type blocks a new one."""
    cur = dict_cursor(conn)
    cur.execute(_SQL, {"acct": account_id})
    rows = [dict(r) for r in cur.fetchall()]

    bought: Set[Tuple[str, str, str]] = set()
    cur.execute("SELECT q.account_id, i.distributor_id, i.distributor_sku "
                "FROM proc.bp_sales_quote q "
                "JOIN proc.bp_sales_quote_outcome o ON o.sales_quote_id = q.sales_quote_id "
                "AND o.outcome = 'won' "
                "JOIN proc.bp_sales_quote_line l ON l.sales_quote_id = q.sales_quote_id "
                "JOIN proc.bp_catalog_item i ON i.catalog_item_id = l.catalog_item_id "
                "WHERE (%(a)s::text IS NULL OR q.account_id = %(a)s)", {"a": account_id})
    bought = {(r["account_id"], r["distributor_id"], r["distributor_sku"]) for r in cur.fetchall()}

    cur.execute("SELECT o.account_id, o.opportunity_type, i.distributor_id, i.distributor_sku "
                "FROM proc.bp_sales_opportunity o "
                "JOIN proc.bp_catalog_item i ON i.catalog_item_id = o.catalog_item_id "
                "WHERE o.outcome IN ('open', 'won') "
                "AND (%(a)s::text IS NULL OR o.account_id = %(a)s)", {"a": account_id})
    held = {(r["account_id"], r["opportunity_type"], r["distributor_id"], r["distributor_sku"])
            for r in cur.fetchall()}

    # one candidate per (account, target, type); the latest won quantity of the bought item
    # stands (rows are in quote order)
    found: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    for r in rows:
        kind = _TYPE_OF[r["relation_type"]]
        key = (r["account_id"], kind, r["distributor_id"], r["to_sku"])
        if (r["account_id"], r["distributor_id"], r["to_sku"]) in bought or key in held:
            continue
        c = found.setdefault(key, {"row": r, "reasons": set(), "gaps": {}, "ended": None})
        c["row"] = r
        c["reasons"].add(r["relation_type"])
        c["gaps"][r["relation_id"]] = r
        if kind == "upgrade" and _ended(r["lifecycle_status"], r["end_of_life_date"]):
            c["ended"] = r

    made: List[Dict[str, Any]] = []
    for (acct, kind, dist, _sku), c in found.items():
        r = c["row"]
        reasons = sorted(c["reasons"] | ({"end_of_life"} if c["ended"] else set()))
        try:
            o = opportunities.create_opportunity(
                conn, account_id=acct, opportunity_type=kind, catalog_item_id=r["target_id"],
                expected_quantity=Decimal(r["quantity"]) if r["quantity"] else None,
                detector_type=DETECTOR_TYPE, reason_codes=reasons)
        except ValueError as exc:      # e.g. a target the catalog cannot price
            logger.warning("opportunity for %s / %s skipped: %s", acct, r["to_sku"], exc)
            continue
        for g in c["gaps"].values():
            opportunities.add_justification(
                conn, o["sales_opportunity_id"], kind="coverage_gap",
                claim=f"Bought {g['distributor_sku']}; the distributor states it is "
                      f"{_WORDS[g['relation_type']]} {g['to_sku']}.",
                evidence_ref=f"relation:{g['relation_id']}")
        if c["ended"]:
            e = c["ended"]
            when = f"on {e['end_of_life_date']}" if e["end_of_life_date"] else "(no date given)"
            opportunities.add_justification(
                conn, o["sales_opportunity_id"], kind="end_of_life",
                claim=f"{e['distributor_sku']} reached end of life {when}.",
                evidence_ref=f"catalog_item:{e['bought_id']}")
        made.append(opportunities.get_opportunity(conn, o["sales_opportunity_id"]))
    return made
