"""The outbound quote -- the artifact this platform never had (spec §4.2).

Cost and list price are SNAPSHOTS copied onto the line at draft time and never
re-read. Lines are never edited: a changed quote is a new quote that supersedes
the old one, so what the customer was sent stays what they were sent.
"""
from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Any, Dict, List, Optional, Sequence

from src.services.sell_side._db import NotFound, StateConflict, dict_cursor
from src.services.sell_side.costing import cost_at
from src.services.sell_side.ladder import QUOTE_RUNG
from src.services.sell_side.money import iso_currency, price_line, total_quote

QUOTE_STATUSES = frozenset(
    {"draft", "in_review", "approved", "issued", "expired", "superseded"})
_SUPERSEDABLE = ("draft", "in_review", "approved", "issued")

_LINE_COLS = ("sales_quote_id", "line_no", "sales_opportunity_id", "catalog_item_id",
              "distributor_sku", "mpn", "item_description", "quantity", "unit_of_measure",
              "currency", "list_price_at_quote", "unit_price", "discount_pct", "line_total",
              "unit_cost", "cost_tier_applied", "line_margin", "line_margin_pct",
              "justification_id")


def create_draft(conn: Any, *, account_id: str, currency: str, valid_until: dt.date,
                 lines: Sequence[Dict[str, Any]], created_by: Optional[str],
                 contact_id: Optional[int] = None, quote_date: Optional[dt.date] = None,
                 supersedes_id: Optional[int] = None) -> Dict[str, Any]:
    currency = iso_currency(currency)
    quote_date = quote_date or dt.date.today()
    if valid_until < quote_date:
        raise ValueError("valid_until is before the quote date")
    if not lines:
        raise ValueError("a quote needs at least one line")

    cur = dict_cursor(conn)
    try:
        cur.execute("SELECT 1 FROM proc.bp_account WHERE account_id = %s", (account_id,))
        if cur.fetchone() is None:
            raise NotFound(f"account {account_id!r} does not exist")

        snapshots, priced = [], []
        for n, line in enumerate(lines, start=1):
            qty, price = Decimal(line["quantity"]), Decimal(line["unit_price"])
            if qty <= 0:
                raise ValueError(f"line {n}: quantity must be greater than zero")
            if price < 0:
                raise ValueError(f"line {n}: unit_price cannot be negative")
            c = cost_at(cur, int(line["catalog_item_id"]), qty)
            if not c.is_current:
                raise ValueError(f"line {n}: catalog item {c.catalog_item_id} is a closed "
                                 "version; quote the current one")
            if c.currency != currency:
                raise ValueError(f"line {n}: catalog item is priced in {c.currency}, the "
                                 f"quote is in {currency}; no FX conversion is performed")
            p = price_line(qty, price, c.unit_cost, c.list_price)
            priced.append(p)
            snapshots.append((line, n, qty, price, c, p))
        totals = total_quote(priced)

        if supersedes_id is not None:
            cur.execute("SELECT account_id, status FROM proc.bp_sales_quote "
                        "WHERE sales_quote_id = %s FOR UPDATE", (supersedes_id,))
            old = cur.fetchone()
            if old is None:
                raise NotFound(f"quote {supersedes_id} does not exist")
            if old["account_id"] != account_id:
                raise ValueError("a quote can only supersede one for the same account")
            if old["status"] not in _SUPERSEDABLE:
                raise StateConflict(f"quote {supersedes_id} is {old['status']} and cannot be superseded")
            cur.execute("UPDATE proc.bp_sales_quote SET status = 'superseded', "
                        "last_modified_date = now() WHERE sales_quote_id = %s", (supersedes_id,))

        cur.execute("SELECT nextval(pg_get_serial_sequence('proc.bp_sales_quote', "
                    "'sales_quote_id')) AS id")
        quote_id = cur.fetchone()["id"]
        phase, sub = QUOTE_RUNG["draft"]
        cur.execute(
            "INSERT INTO proc.bp_sales_quote (sales_quote_id, quote_ref, account_id, contact_id, "
            "currency, quote_date, valid_until, total_ex_tax, total_cost, total_margin, "
            "margin_pct, phase_id, subprocess_id, status, supersedes_id, created_by) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'draft', %s, %s)",
            (quote_id, f"SQ-{quote_date:%Y%m%d}-{quote_id:06d}", account_id, contact_id,
             currency, quote_date, valid_until, totals.total_ex_tax, totals.total_cost,
             totals.total_margin, totals.margin_pct, phase, sub, supersedes_id, created_by))
        for line, n, qty, price, c, p in snapshots:
            cur.execute(
                f"INSERT INTO proc.bp_sales_quote_line ({', '.join(_LINE_COLS)}) "
                f"VALUES ({', '.join(['%s'] * len(_LINE_COLS))})",
                (quote_id, n, line.get("sales_opportunity_id"), c.catalog_item_id,
                 c.distributor_sku, c.mpn, c.item_description, qty, c.unit_of_measure,
                 currency, c.list_price, price, p.discount_pct, p.line_total,
                 c.unit_cost, c.cost_tier_applied, p.line_margin, p.line_margin_pct,
                 line.get("justification_id")))
    except Exception:
        conn.rollback()
        raise
    conn.commit()
    return get_quote(conn, quote_id)


def get_quote(conn: Any, sales_quote_id: int) -> Dict[str, Any]:
    cur = dict_cursor(conn)
    cur.execute(
        "SELECT q.*, a.account_name, c.contact_name FROM proc.bp_sales_quote q "
        "JOIN proc.bp_account a ON a.account_id = q.account_id "
        "LEFT JOIN proc.bp_account_contact c ON c.contact_id = q.contact_id "
        "WHERE q.sales_quote_id = %s", (sales_quote_id,))
    row = cur.fetchone()
    if row is None:
        raise NotFound(f"quote {sales_quote_id} does not exist")
    out = dict(row)
    cur.execute("SELECT * FROM proc.bp_sales_quote_line WHERE sales_quote_id = %s "
                "ORDER BY line_no", (sales_quote_id,))
    out["lines"] = [dict(r) for r in cur.fetchall()]
    ids = [l["justification_id"] for l in out["lines"] if l["justification_id"]]
    if ids:
        cur.execute("SELECT * FROM proc.bp_sales_justification WHERE justification_id = ANY(%s) "
                    "ORDER BY justification_id", (ids,))
        out["justifications"] = [dict(r) for r in cur.fetchall()]
    else:
        out["justifications"] = []
    return out


def _transition(conn: Any, sales_quote_id: int, *, frm: str, to: str,
                check=None, sets: str = "", params: tuple = ()) -> Dict[str, Any]:
    cur = dict_cursor(conn)
    try:
        cur.execute("SELECT * FROM proc.bp_sales_quote WHERE sales_quote_id = %s FOR UPDATE",
                    (sales_quote_id,))
        row = cur.fetchone()
        if row is None:
            raise NotFound(f"quote {sales_quote_id} does not exist")
        if row["status"] != frm:
            raise StateConflict(f"quote {sales_quote_id} is {row['status']}, not {frm}")
        if check:
            check(row)
        phase, sub = QUOTE_RUNG[to]
        cur.execute(
            f"UPDATE proc.bp_sales_quote SET status = %s, phase_id = %s, subprocess_id = %s, "
            f"last_modified_date = now(){sets} WHERE sales_quote_id = %s",
            (to, phase, sub, *params, sales_quote_id))
    except Exception:
        conn.rollback()
        raise
    conn.commit()
    return get_quote(conn, sales_quote_id)


def submit(conn: Any, sales_quote_id: int, *, actor: Optional[str]) -> Dict[str, Any]:
    return _transition(conn, sales_quote_id, frm="draft", to="in_review")


def approve(conn: Any, sales_quote_id: int, *, approver: Optional[str]) -> Dict[str, Any]:
    def _check(row):
        if not (approver or "").strip():
            raise StateConflict("an approval needs an authenticated approver")
        if row["created_by"] and row["created_by"] == approver:
            raise StateConflict("nobody approves their own quote")
    return _transition(conn, sales_quote_id, frm="in_review", to="approved", check=_check,
                       sets=", approved_by = %s, approved_at = now()", params=(approver,))


def issue(conn: Any, sales_quote_id: int, *, actor: Optional[str]) -> Dict[str, Any]:
    def _check(row):
        if row["valid_until"] < dt.date.today():
            raise StateConflict(f"quote {sales_quote_id} has expired ({row['valid_until']})")
    return _transition(conn, sales_quote_id, frm="approved", to="issued", check=_check,
                       sets=", issued_at = now()")
