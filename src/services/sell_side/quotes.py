"""The outbound quote -- the artifact this platform never had (spec §4.2).

Cost and list price are SNAPSHOTS copied onto the line at draft time and never
re-read. Lines are never edited: a changed quote is a new quote that supersedes
the old one, so what the customer was sent stays what they were sent.

A quote is TWO records. The customer record (bp_sales_quote, bp_sales_quote_line)
is read by get_quote through a named column list that has no cost or margin in it.
What the quote costs us (bp_sales_quote_internal, bp_sales_quote_line_internal) is
read only by get_quote_internal. Nothing here selects * from a quote table.
"""
from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Any, Dict, List, Optional, Sequence

from src.services import agent_actions
from src.services.sell_side import numbering
from src.services.sell_side._db import NotFound, StateConflict, dict_cursor
from src.services.sell_side.costing import cost_at
from src.services.sell_side.ladder import QUOTE_RUNG
from src.services.sell_side.money import iso_currency, price_line, total_quote

QUOTE_STATUSES = frozenset(
    {"draft", "in_review", "approved", "issued", "expired", "superseded"})
_SUPERSEDABLE = ("draft", "in_review", "approved", "issued")

# The customer record. A column is readable through get_quote only if it is named here,
# so one added to the table later stays invisible until someone names it.
_QUOTE_COLS = ("sales_quote_id", "tenant_id", "quote_ref", "quote_number", "account_id",
               "contact_id",
               "currency", "quote_date", "valid_until", "total_ex_tax", "phase_id",
               "subprocess_id", "status", "supersedes_id", "approved_by", "approved_at",
               "issued_at", "created_by", "created_date", "last_modified_date")
# There is one tenant. Named, not NULL, so UNIQUE (tenant_id, quote_ref) is enforced.
TENANT = "default"
# The series a quote's number is taken from, and how the number is written. quote_ref
# (SQ-...) is the internal draft reference and is not this.
NUMBER_SERIES = "sales_quote"
NUMBER_PREFIX = "Q"
_LINE_COLS = ("sales_quote_id", "line_no", "sales_opportunity_id", "catalog_item_id",
              "distributor_sku", "mpn", "item_description", "quantity", "unit_of_measure",
              "currency", "list_price_at_quote", "unit_price", "discount_pct", "line_total",
              "justification_id")
_LINE_READ_COLS = ("sales_quote_line_id",) + _LINE_COLS + ("created_date",)

# The internal record.
_INTERNAL_QUOTE_COLS = ("total_cost", "total_margin", "margin_pct")
_INTERNAL_LINE_COLS = ("unit_cost", "cost_tier_applied", "line_margin", "line_margin_pct")

_AGENT = "SellSide"
# Approving, issuing and closing a quote cannot be taken back, so each is recorded or it
# does not happen. Drafting, submitting and superseding are recorded best-effort.
_MUST_RECORD = frozenset({"sales_quote.approved", "sales_quote.issued", "sales_quote.outcome"})


def record_event(conn: Any, event: str, quote: Dict[str, Any], *, actor: Optional[str],
                 **details: Any) -> None:
    """One lifecycle event on the audit spine, on the caller's transaction: the event and
    the change it describes commit together or not at all. Never carries cost or margin."""
    write = (agent_actions.record_action_or_fail if event in _MUST_RECORD
             else agent_actions.record_action)
    write(phase="sales_quote", action_type=event, conn=conn, agent=_AGENT,
          doc_type="sales_quote", doc_pk=str(quote["sales_quote_id"]),
          document_id=quote.get("quote_ref"),
          summary=f"{event} by {actor or 'unknown'}",
          details={"actor": actor, "sales_quote_id": quote["sales_quote_id"],
                   "quote_ref": quote.get("quote_ref"), **details})


def internal_totals(cur: Any, sales_quote_id: int) -> Dict[str, Any]:
    """Cost and margin for one quote. A quote with no internal record is refused, not read
    as "cost unknown": that would let a below-cost quote through on a missing row."""
    cur.execute(f"SELECT {', '.join(_INTERNAL_QUOTE_COLS)} FROM proc.bp_sales_quote_internal "
                "WHERE sales_quote_id = %s", (sales_quote_id,))
    row = cur.fetchone()
    if row is None:
        raise StateConflict(f"quote {sales_quote_id} has no internal cost record")
    return dict(row)


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
    for n, line in enumerate(lines, start=1):
        exponent = Decimal(line["unit_price"]).as_tuple().exponent
        if isinstance(exponent, int) and exponent < -4:
            raise ValueError(f"line {n}: unit_price has more than 4 decimal places")

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

            opp_id = line.get("sales_opportunity_id")
            just_id = line.get("justification_id")
            if opp_id is not None:
                cur.execute("SELECT account_id FROM proc.bp_sales_opportunity "
                            "WHERE sales_opportunity_id = %s", (opp_id,))
                orow = cur.fetchone()
                if orow is None:
                    raise ValueError(f"line {n}: opportunity {opp_id} does not exist")
                if orow["account_id"] != account_id:
                    raise ValueError(f"line {n}: opportunity {opp_id} belongs to a "
                                     "different account")
            if just_id is not None:
                cur.execute(
                    "SELECT j.sales_opportunity_id, o.account_id FROM "
                    "proc.bp_sales_justification j JOIN proc.bp_sales_opportunity o "
                    "ON o.sales_opportunity_id = j.sales_opportunity_id "
                    "WHERE j.justification_id = %s", (just_id,))
                jrow = cur.fetchone()
                if jrow is None:
                    raise ValueError(f"line {n}: justification {just_id} does not exist")
                if jrow["account_id"] != account_id:
                    raise ValueError(f"line {n}: justification {just_id} belongs to a "
                                     "different account")
                if opp_id is not None and jrow["sales_opportunity_id"] != opp_id:
                    raise ValueError(f"line {n}: justification {just_id} belongs to a "
                                     "different opportunity than the line's")

            c = cost_at(cur, int(line["catalog_item_id"]), qty)
            if not c.is_current:
                raise ValueError(f"line {n}: catalog item {c.catalog_item_id} is a closed "
                                 "version; quote the current one")
            if c.currency != currency:
                raise ValueError(f"line {n}: catalog item is priced in {c.currency}, the "
                                 f"quote is in {currency}; no FX conversion is performed")
            p = price_line(qty, price, c.unit_cost, c.list_price)
            for pct_name, pct_val in (("discount_pct", p.discount_pct),
                                      ("line_margin_pct", p.line_margin_pct)):
                if pct_val is not None and abs(pct_val) >= 1000:
                    raise ValueError(f"line {n}: {pct_name} {pct_val} is out of range")
            priced.append(p)
            snapshots.append((line, n, qty, price, c, p))
        totals = total_quote(priced)
        superseded = None

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
            cur.execute("SELECT 1 FROM proc.bp_sales_quote_outcome "
                        "WHERE sales_quote_id = %s", (supersedes_id,))
            if cur.fetchone() is not None:
                raise StateConflict(f"quote {supersedes_id} already has a recorded outcome "
                                    "and cannot be superseded")
            cur.execute("UPDATE proc.bp_sales_quote SET status = 'superseded', "
                        "last_modified_date = now() WHERE sales_quote_id = %s "
                        "RETURNING sales_quote_id, quote_ref", (supersedes_id,))
            superseded = dict(cur.fetchone())

        cur.execute("SELECT nextval(pg_get_serial_sequence('proc.bp_sales_quote', "
                    "'sales_quote_id')) AS id")
        quote_id = cur.fetchone()["id"]
        phase, sub = QUOTE_RUNG["draft"]
        quote_ref = f"SQ-{quote_date:%Y%m%d}-{quote_id:06d}"
        cur.execute(
            "INSERT INTO proc.bp_sales_quote (sales_quote_id, tenant_id, quote_ref, account_id, "
            "contact_id, currency, quote_date, valid_until, total_ex_tax, phase_id, "
            "subprocess_id, status, supersedes_id, created_by) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'draft', %s, %s)",
            (quote_id, TENANT, quote_ref, account_id, contact_id, currency, quote_date, valid_until,
             totals.total_ex_tax, phase, sub, supersedes_id, created_by))
        cur.execute(
            "INSERT INTO proc.bp_sales_quote_internal (sales_quote_id, total_cost, "
            "total_margin, margin_pct) VALUES (%s, %s, %s, %s)",
            (quote_id, totals.total_cost, totals.total_margin, totals.margin_pct))
        for line, n, qty, price, c, p in snapshots:
            cur.execute(
                f"INSERT INTO proc.bp_sales_quote_line ({', '.join(_LINE_COLS)}) "
                f"VALUES ({', '.join(['%s'] * len(_LINE_COLS))}) RETURNING sales_quote_line_id",
                (quote_id, n, line.get("sales_opportunity_id"), c.catalog_item_id,
                 c.distributor_sku, c.mpn, c.item_description, qty, c.unit_of_measure,
                 currency, c.list_price, price, p.discount_pct, p.line_total,
                 line.get("justification_id")))
            cur.execute(
                "INSERT INTO proc.bp_sales_quote_line_internal (sales_quote_line_id, "
                "sales_quote_id, unit_cost, cost_tier_applied, line_margin, line_margin_pct) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (cur.fetchone()["sales_quote_line_id"], quote_id, c.unit_cost,
                 c.cost_tier_applied, p.line_margin, p.line_margin_pct))
        new = {"sales_quote_id": quote_id, "quote_ref": quote_ref}
        if superseded is not None:
            record_event(conn, "sales_quote.superseded", superseded, actor=created_by,
                         superseded_by=quote_id)
        record_event(conn, "sales_quote.created", new, actor=created_by,
                     account_id=account_id, supersedes_id=supersedes_id)
    except Exception:
        conn.rollback()
        raise
    conn.commit()
    return get_quote(conn, quote_id)


def get_quote(conn: Any, sales_quote_id: int) -> Dict[str, Any]:
    """The customer record: no cost, no margin, and only the justifications marked safe
    for a customer. What the quote costs us is get_quote_internal."""
    cur = dict_cursor(conn)
    cur.execute(
        f"SELECT {', '.join('q.' + c for c in _QUOTE_COLS)}, a.account_name, c.contact_name "
        "FROM proc.bp_sales_quote q "
        "JOIN proc.bp_account a ON a.account_id = q.account_id "
        "LEFT JOIN proc.bp_account_contact c ON c.contact_id = q.contact_id "
        "WHERE q.sales_quote_id = %s", (sales_quote_id,))
    row = cur.fetchone()
    if row is None:
        raise NotFound(f"quote {sales_quote_id} does not exist")
    out = dict(row)
    cur.execute(f"SELECT {', '.join(_LINE_READ_COLS)} FROM proc.bp_sales_quote_line "
                "WHERE sales_quote_id = %s ORDER BY line_no", (sales_quote_id,))
    out["lines"] = [dict(r) for r in cur.fetchall()]
    out["justifications"] = _justifications(cur, out["lines"], customer_safe_only=True)
    return out


def _justifications(cur: Any, lines: Sequence[Dict[str, Any]], *,
                    customer_safe_only: bool) -> List[Dict[str, Any]]:
    ids = [l["justification_id"] for l in lines if l["justification_id"]]
    if not ids:
        return []
    cur.execute("SELECT * FROM proc.bp_sales_justification WHERE justification_id = ANY(%s) "
                + ("AND customer_safe IS TRUE " if customer_safe_only else "")
                + "ORDER BY justification_id", (ids,))
    return [dict(r) for r in cur.fetchall()]


def get_quote_internal(conn: Any, sales_quote_id: int) -> Dict[str, Any]:
    """What the quote costs us, keyed to the quote and to each line id, plus every
    justification including the ones a customer must not read. Behind margin.read."""
    cur = dict_cursor(conn)
    cur.execute("SELECT 1 FROM proc.bp_sales_quote WHERE sales_quote_id = %s", (sales_quote_id,))
    if cur.fetchone() is None:
        raise NotFound(f"quote {sales_quote_id} does not exist")
    out = {"sales_quote_id": sales_quote_id, **internal_totals(cur, sales_quote_id)}
    cur.execute(
        f"SELECT l.sales_quote_line_id, l.line_no, l.justification_id, "
        f"{', '.join('i.' + c for c in _INTERNAL_LINE_COLS)} "
        "FROM proc.bp_sales_quote_line l "
        "LEFT JOIN proc.bp_sales_quote_line_internal i "
        "ON i.sales_quote_line_id = l.sales_quote_line_id "
        "WHERE l.sales_quote_id = %s ORDER BY l.line_no", (sales_quote_id,))
    lines = [dict(r) for r in cur.fetchall()]
    out["justifications"] = _justifications(cur, lines, customer_safe_only=False)
    out["lines"] = [{k: v for k, v in l.items() if k != "justification_id"} for l in lines]
    return out


def _transition(conn: Any, sales_quote_id: int, *, frm: str, to: str, event: str,
                actor: Optional[str], check=None, sets: str = "",
                params: tuple = ()) -> Dict[str, Any]:
    cur = dict_cursor(conn)
    try:
        cur.execute(f"SELECT {', '.join(_QUOTE_COLS)} FROM proc.bp_sales_quote "
                    "WHERE sales_quote_id = %s FOR UPDATE", (sales_quote_id,))
        row = cur.fetchone()
        if row is None:
            raise NotFound(f"quote {sales_quote_id} does not exist")
        if row["status"] != frm:
            raise StateConflict(f"quote {sales_quote_id} is {row['status']}, not {frm}")
        # A check may return details for the event it is about to cause.
        extra = (check(row, cur) if check else None) or {}
        phase, sub = QUOTE_RUNG[to]
        cur.execute(
            f"UPDATE proc.bp_sales_quote SET status = %s, phase_id = %s, subprocess_id = %s, "
            f"last_modified_date = now(){sets} WHERE sales_quote_id = %s",
            (to, phase, sub, *params, sales_quote_id))
        record_event(conn, event, row, actor=actor, from_status=frm, to_status=to, **extra)
    except Exception:
        conn.rollback()
        raise
    conn.commit()
    return get_quote(conn, sales_quote_id)


def submit(conn: Any, sales_quote_id: int, *, actor: Optional[str]) -> Dict[str, Any]:
    return _transition(conn, sales_quote_id, frm="draft", to="in_review",
                       event="sales_quote.submitted", actor=actor)


def approve(conn: Any, sales_quote_id: int, *, approver: Optional[str]) -> Dict[str, Any]:
    def _check(row, cur):
        if not (approver or "").strip():
            raise StateConflict("an approval needs an authenticated approver")
        if row["created_by"] and row["created_by"] == approver:
            raise StateConflict("nobody approves their own quote")
        # A quote priced below its own cost is a loss the ladder labels
        # "margin-floor-tested" but nothing tested (capability audit 2026-10-08).
        # Unknown cost (NULL margin) is not a loss we can show, so it passes.
        margin = internal_totals(cur, sales_quote_id)["total_margin"]
        if margin is not None and margin < 0:
            raise StateConflict(
                f"quote {sales_quote_id} is priced below cost (margin "
                f"{margin}); it needs a deal-desk exception, "
                "not a routine approval")
    return _transition(conn, sales_quote_id, frm="in_review", to="approved", check=_check,
                       event="sales_quote.approved", actor=approver,
                       sets=", approved_by = %s, approved_at = now()", params=(approver,))


def issue(conn: Any, sales_quote_id: int, *, actor: Optional[str]) -> Dict[str, Any]:
    """Issue the quote and give it its number, in one transaction: the number is the next
    in the series with none missing, and an issue that fails gives it back."""
    def _check(row, cur):
        if row["valid_until"] < dt.date.today():
            raise StateConflict(f"quote {sales_quote_id} has expired ({row['valid_until']})")
        if row["quote_number"]:
            raise StateConflict(f"quote {sales_quote_id} already has the number "
                                f"{row['quote_number']}")
        number = numbering.formatted(NUMBER_PREFIX, *numbering.allocate(cur, NUMBER_SERIES))
        cur.execute("UPDATE proc.bp_sales_quote SET quote_number = %s "
                    "WHERE sales_quote_id = %s", (number, sales_quote_id))
        return {"quote_number": number}
    return _transition(conn, sales_quote_id, frm="approved", to="issued", check=_check,
                       event="sales_quote.issued", actor=actor,
                       sets=", issued_at = now()")
