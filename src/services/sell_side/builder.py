"""The Quote builder's quote: drafted on a deal, in the seller's own words and figures.

The same record as every other sell-side quote (bp_sales_quote), so review, approval,
numbering, the customer page, sending and the audit trail are the ones in quotes.py and
quote_send.py. What is different is how it is drafted:

  * It belongs to a deal -- or to none yet. The deal is set once, and one deal has one
    live quote. A quote with no deal can be saved and cannot be sent for review.
  * Its lines are free text with the seller's own cost, not catalog items, and a DRAFT
    is saved in place as many times as it takes. That does not weaken "what the customer
    was sent stays what they were sent": only a draft is editable, and a quote that has
    gone out is changed by `supersede`, which leaves it as it was and starts version n+1.
  * Nothing counts as free. A line with no cost, or a cost of zero, is uncosted, and the
    margin is taken over the costed lines only (money.total_builder).

Every read here returns cost and markup, so every caller of this module is behind
margin.read. The customer never sees this shape: the customer page still reads
quotes.get_quote, which has no cost in it.
"""
from __future__ import annotations

import datetime as dt
import json
from decimal import Decimal, InvalidOperation
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Sequence

import psycopg2

from src.services.sell_side import quotes
from src.services.sell_side._db import NotFound, StateConflict, dict_cursor
from src.services.sell_side.ladder import QUOTE_RUNG
from src.services.sell_side.money import builder_line, iso_currency, total_builder

LINE_TAGS = ("product", "service", "one-off")
TERM_KEYS = ("payment", "start", "lengthMonths", "renewal", "sla")
MAX_LINES = 200
MAX_HTML = 20_000

# Unset: the caller said nothing about the deal, which is not the same as saying "none".
UNSET: Any = object()

# What the screen calls each status. An issued quote is the one the customer holds; the
# screen's word for that is "sent" and it has no other. `backendStatus` carries ours.
_SCREEN_STATUS = {"draft": "draft", "in_review": "in-review", "approved": "approved",
                  "issued": "sent", "superseded": "superseded", "expired": "superseded"}

_COLS = ("sales_quote_id", "quote_ref", "quote_number", "deal_id", "version", "status",
         "account_id", "currency", "category", "quote_date", "valid_until", "prepared_by",
         "customer_name", "customer_contact", "scope_html", "assumptions_html", "terms",
         "conditions", "total_ex_tax", "phase_id", "subprocess_id", "supersedes_id",
         "created_by", "approved_by", "approved_at", "issued_at", "last_modified_date")


# --- rich text ---------------------------------------------------------------------

_ALLOWED_TAGS = frozenset({"p", "b", "strong", "i", "em", "u", "ul", "ol", "li", "br"})
_DROP_WITH_CONTENT = frozenset({"script", "style", "iframe", "object", "embed", "template"})


class _Rich(HTMLParser):
    """Scope and assumptions are printed on a page a customer opens, so what is kept is
    a closed list of tags with no attributes at all. Anything else loses its tag and
    keeps its words; a script or style loses both."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: List[str] = []
        self._dropping = 0

    def handle_starttag(self, tag, attrs):
        if tag in _DROP_WITH_CONTENT:
            self._dropping += 1
        elif not self._dropping and tag in _ALLOWED_TAGS:
            self.out.append(f"<{tag}>")

    def handle_startendtag(self, tag, attrs):
        if not self._dropping and tag in _ALLOWED_TAGS:
            self.out.append(f"<{tag}>")

    def handle_endtag(self, tag):
        if tag in _DROP_WITH_CONTENT:
            self._dropping = max(0, self._dropping - 1)
        elif not self._dropping and tag in _ALLOWED_TAGS and tag != "br":
            self.out.append(f"</{tag}>")

    def handle_data(self, data):
        if not self._dropping:
            self.out.append(data.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def clean_html(value: Optional[str], field: str) -> Optional[str]:
    if value is None:
        return None
    if len(value) > MAX_HTML:
        raise ValueError(f"{field} is longer than {MAX_HTML} characters")
    parser = _Rich()
    parser.feed(str(value))
    parser.close()
    return "".join(parser.out)


# --- validation --------------------------------------------------------------------

def _dec(value: Any, what: str, *, places: int = 4) -> Optional[Decimal]:
    if value is None or value == "":
        return None
    try:
        d = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError(f"{what}: {value!r} is not a number") from None
    if not d.is_finite():
        raise ValueError(f"{what}: {value!r} is not a number")
    exponent = d.as_tuple().exponent
    if isinstance(exponent, int) and exponent < -places:
        raise ValueError(f"{what} has more than {places} decimal places")
    return d


def _text(value: Any, what: str, limit: int = 300) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    if len(s) > limit:
        raise ValueError(f"{what} is longer than {limit} characters")
    return s or None


def _date(value: Any, what: str) -> Optional[dt.date]:
    if value is None or value == "":
        return None
    if isinstance(value, dt.date):
        return value
    try:
        return dt.date.fromisoformat(str(value))
    except ValueError:
        raise ValueError(f"{what}: {value!r} is not a date (YYYY-MM-DD)") from None


def _line(raw: Dict[str, Any], n: int) -> Dict[str, Any]:
    tag = raw.get("tag") or "service"
    if tag not in LINE_TAGS:
        raise ValueError(f"line {n}: tag {tag!r} is not one of {', '.join(LINE_TAGS)}")
    qty = _dec(raw.get("qty"), f"line {n}: qty")
    price = _dec(raw.get("unitPrice"), f"line {n}: unitPrice")
    cost = _dec(raw.get("cost"), f"line {n}: cost")
    markup = _dec(raw.get("markupPct"), f"line {n}: markupPct")
    discount = _dec(raw.get("discountPct"), f"line {n}: discountPct") or Decimal(0)
    if qty is None or qty <= 0:
        raise ValueError(f"line {n}: qty must be greater than zero")
    if price is None or price < 0:
        raise ValueError(f"line {n}: unitPrice is needed and cannot be negative")
    if cost is not None and cost < 0:
        raise ValueError(f"line {n}: cost cannot be negative")
    if not Decimal(0) <= discount <= Decimal(100):
        raise ValueError(f"line {n}: discountPct must be between 0 and 100")
    if markup is not None and abs(markup) >= Decimal(1_000_000):
        raise ValueError(f"line {n}: markupPct {markup} is out of range")
    return {"tag": tag, "description": _text(raw.get("description"), f"line {n}: description", 1000) or "",
            "qty": qty, "unit": _text(raw.get("unit"), f"line {n}: unit", 40),
            "unit_price": price, "cost": cost, "markup_pct": markup, "discount_pct": discount,
            "priced": builder_line(qty, price, discount, cost)}


def _clean(draft: Dict[str, Any]) -> Dict[str, Any]:
    """The draft as the screen holds it -> what is stored. Raises ValueError (HTTP 422)."""
    quote_date = _date(draft.get("issuedAt"), "issuedAt") or dt.date.today()
    valid_until = _date(draft.get("validUntil"), "validUntil")
    if valid_until is None:
        raise ValueError("validUntil is needed")
    if valid_until < quote_date:
        raise ValueError("validUntil is before the quote date")
    raw_lines = draft.get("lines") or []
    if len(raw_lines) > MAX_LINES:
        raise ValueError(f"a quote has at most {MAX_LINES} lines")
    terms_in = draft.get("terms") or {}
    if not isinstance(terms_in, dict):
        raise ValueError("terms must be an object")
    terms = {}
    for key in TERM_KEYS:
        value = terms_in.get(key)
        if key == "lengthMonths":
            months = _dec(value, "terms.lengthMonths", places=0)
            if months is not None and not 0 <= months <= 1200:
                raise ValueError("terms.lengthMonths is out of range")
            terms[key] = int(months) if months is not None else None
        else:
            terms[key] = _text(value, f"terms.{key}")
    customer = draft.get("customer") or {}
    if not isinstance(customer, dict):
        raise ValueError("customer must be an object")
    return {
        "currency": iso_currency(draft.get("currency")),
        "category": _text(draft.get("category"), "category"),
        "quote_date": quote_date, "valid_until": valid_until,
        "prepared_by": _text(draft.get("preparedBy"), "preparedBy"),
        "customer_name": _text(customer.get("name"), "customer.name"),
        "customer_contact": _text(customer.get("contact"), "customer.contact"),
        "scope_html": clean_html(draft.get("scopeHtml"), "scopeHtml"),
        "assumptions_html": clean_html(draft.get("assumptionsHtml"), "assumptionsHtml"),
        "terms": terms,
        "lines": [_line(l, n) for n, l in enumerate(raw_lines, start=1)],
    }


# --- the deal ----------------------------------------------------------------------

def _check_deal(cur: Any, deal_id: str, *, for_quote: Optional[int]) -> None:
    """A quote goes on a deal that exists, is still open, and has no live quote.

    "Open" is the one closing fact a deal carries here: it has been invoiced. That is
    the rule the Pipeline uses to call a deal Won. Lost and Cancelled are not recorded
    on a deal anywhere, so they cannot be refused here."""
    cur.execute("SELECT 1 FROM proc.bp_deal WHERE deal_id = %s", (deal_id,))
    if cur.fetchone() is None:
        raise NotFound(f"deal {deal_id!r} does not exist")
    cur.execute("SELECT COALESCE(SUM(invoice_count), 0) AS invoices "
                "FROM proc.bp_deal_overview WHERE deal_id = %s", (deal_id,))
    if (cur.fetchone() or {}).get("invoices"):
        raise StateConflict(f"deal {deal_id} is closed (it has been invoiced); "
                            "a quote goes out on an open deal")
    cur.execute("SELECT sales_quote_id FROM proc.bp_sales_quote WHERE deal_id = %s "
                "AND status <> 'superseded' AND sales_quote_id IS DISTINCT FROM %s",
                (deal_id, for_quote))
    held = cur.fetchone()
    if held is not None:
        raise StateConflict(f"deal {deal_id} already has quote {held['sales_quote_id']}; "
                            "open that one, or supersede it")


# --- writes ------------------------------------------------------------------------

def _write_lines(cur: Any, quote_id: int, currency: str, lines: Sequence[Dict[str, Any]]) -> None:
    cur.execute("DELETE FROM proc.bp_sales_quote_line WHERE sales_quote_id = %s", (quote_id,))
    for n, l in enumerate(lines, start=1):
        p = l["priced"]
        cur.execute(
            "INSERT INTO proc.bp_sales_quote_line (sales_quote_id, line_no, item_description, "
            "quantity, unit_of_measure, currency, unit_price, entered_discount_pct, line_total, "
            "line_tag) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
            "RETURNING sales_quote_line_id",
            (quote_id, n, l["description"], l["qty"], l["unit"], currency, l["unit_price"],
             l["discount_pct"], p.line_total, l["tag"]))
        cur.execute(
            "INSERT INTO proc.bp_sales_quote_line_internal (sales_quote_line_id, sales_quote_id, "
            "unit_cost, line_margin, line_margin_pct, cost_source, markup_pct) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (cur.fetchone()["sales_quote_line_id"], quote_id, l["cost"], p.line_margin,
             p.line_margin_pct, "entered" if l["cost"] is not None else None, l["markup_pct"]))
    t = total_builder([l["priced"] for l in lines])
    cur.execute("UPDATE proc.bp_sales_quote SET total_ex_tax = %s WHERE sales_quote_id = %s",
                (t.total_ex_tax, quote_id))
    cur.execute(
        "INSERT INTO proc.bp_sales_quote_internal (sales_quote_id, total_cost, total_margin, "
        "margin_pct, costed_lines, uncosted_lines) VALUES (%s, %s, %s, %s, %s, %s) "
        "ON CONFLICT (sales_quote_id) DO UPDATE SET total_cost = EXCLUDED.total_cost, "
        "total_margin = EXCLUDED.total_margin, margin_pct = EXCLUDED.margin_pct, "
        "costed_lines = EXCLUDED.costed_lines, uncosted_lines = EXCLUDED.uncosted_lines",
        (quote_id, t.total_cost, t.total_margin, t.margin_pct, t.costed_lines, t.uncosted_lines))


def _insert(cur: Any, d: Dict[str, Any], *, deal_id: Optional[str], created_by: Optional[str],
            version: int = 1, supersedes_id: Optional[int] = None,
            account_id: Optional[str] = None) -> Dict[str, Any]:
    cur.execute("SELECT nextval(pg_get_serial_sequence('proc.bp_sales_quote', "
                "'sales_quote_id')) AS id")
    quote_id = cur.fetchone()["id"]
    phase, sub = QUOTE_RUNG["draft"]
    quote_ref = f"SQ-{d['quote_date']:%Y%m%d}-{quote_id:06d}"
    cur.execute(
        "INSERT INTO proc.bp_sales_quote (sales_quote_id, tenant_id, quote_ref, account_id, "
        "deal_id, version, currency, quote_date, valid_until, category, prepared_by, "
        "customer_name, customer_contact, scope_html, assumptions_html, terms, phase_id, "
        "subprocess_id, status, supersedes_id, created_by) VALUES (%s, %s, %s, %s, %s, %s, %s, "
        "%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, 'draft', %s, %s)",
        (quote_id, quotes.TENANT, quote_ref, account_id, deal_id, version, d["currency"],
         d["quote_date"], d["valid_until"], d["category"], d["prepared_by"], d["customer_name"],
         d["customer_contact"], d["scope_html"], d["assumptions_html"], json.dumps(d["terms"]),
         phase, sub, supersedes_id, created_by))
    _write_lines(cur, quote_id, d["currency"], d["lines"])
    return {"sales_quote_id": quote_id, "quote_ref": quote_ref}


def _deal_taken(exc: Exception, deal_id: Optional[str]) -> Optional[StateConflict]:
    """Two people linking one deal at once: the index refuses the second, and this is
    that refusal in words."""
    if isinstance(exc, psycopg2.errors.UniqueViolation) and "ix_bp_sales_quote_deal_live" in str(exc):
        return StateConflict(f"deal {deal_id} already has a quote")
    return None


def create(conn: Any, *, deal_id: Optional[str], draft: Dict[str, Any],
           created_by: Optional[str]) -> Dict[str, Any]:
    d = _clean(draft)
    cur = dict_cursor(conn)
    try:
        if deal_id is not None:
            _check_deal(cur, deal_id, for_quote=None)
        new = _insert(cur, d, deal_id=deal_id, created_by=created_by)
        quotes.record_event(conn, "sales_quote.created", new, actor=created_by,
                            deal_id=deal_id, builder=True)
    except Exception as exc:
        conn.rollback()
        raise (_deal_taken(exc, deal_id) or exc) from None
    conn.commit()
    return get(conn, new["sales_quote_id"])


def _lock(cur: Any, quote_id: int) -> Dict[str, Any]:
    cur.execute(f"SELECT {', '.join(_COLS)} FROM proc.bp_sales_quote "
                "WHERE sales_quote_id = %s FOR UPDATE", (quote_id,))
    row = cur.fetchone()
    if row is None:
        raise NotFound(f"quote {quote_id} does not exist")
    return dict(row)


def _refuse_catalog_quote(cur: Any, quote_id: int) -> None:
    """A reseller quote's lines are cost snapshots from the price file. Saving it through
    the builder would replace those with typed figures, so it is refused."""
    cur.execute("SELECT 1 FROM proc.bp_sales_quote_line WHERE sales_quote_id = %s "
                "AND catalog_item_id IS NOT NULL LIMIT 1", (quote_id,))
    if cur.fetchone() is not None:
        raise StateConflict(f"quote {quote_id} is a catalog quote and is not edited in the "
                            "Quote builder")


def save(conn: Any, quote_id: int, *, draft: Dict[str, Any], actor: Optional[str],
         deal_id: Any = UNSET, revision: Optional[str] = None) -> Dict[str, Any]:
    """Save a draft in place. Only a draft is editable. `deal_id` links the quote to its
    deal, once. `revision` is the value the last read returned: when given, a save over
    somebody else's later save is refused instead of silently undoing it."""
    d = _clean(draft)
    cur = dict_cursor(conn)
    try:
        row = _lock(cur, quote_id)
        if row["status"] != "draft":
            raise StateConflict(f"quote {quote_id} is {row['status']}; only a draft is edited"
                                + (" (supersede it to change it)" if row["status"] == "issued" else ""))
        if revision is not None and revision != _revision(row):
            raise StateConflict(f"quote {quote_id} was saved by someone else since you opened "
                                "it; reload it before saving")
        _refuse_catalog_quote(cur, quote_id)
        linked = None
        if deal_id is not UNSET and deal_id != row["deal_id"]:
            if row["deal_id"] is not None:
                raise StateConflict(f"quote {quote_id} is already on deal {row['deal_id']}; "
                                    "a quote's deal is set once")
            if deal_id is not None:
                _check_deal(cur, deal_id, for_quote=quote_id)
                linked = deal_id
        cur.execute(
            "UPDATE proc.bp_sales_quote SET currency = %s, quote_date = %s, valid_until = %s, "
            "category = %s, prepared_by = %s, customer_name = %s, customer_contact = %s, "
            "scope_html = %s, assumptions_html = %s, terms = %s::jsonb, "
            "deal_id = COALESCE(%s, deal_id), last_modified_date = clock_timestamp() "
            "WHERE sales_quote_id = %s",
            (d["currency"], d["quote_date"], d["valid_until"], d["category"], d["prepared_by"],
             d["customer_name"], d["customer_contact"], d["scope_html"], d["assumptions_html"],
             json.dumps(d["terms"]), linked, quote_id))
        _write_lines(cur, quote_id, d["currency"], d["lines"])
        quotes.record_event(conn, "sales_quote.saved", row, actor=actor,
                            **({"linked_deal_id": linked} if linked else {}))
    except Exception as exc:
        conn.rollback()
        raise (_deal_taken(exc, deal_id if deal_id is not UNSET else None) or exc) from None
    conn.commit()
    return get(conn, quote_id)


def submit(conn: Any, quote_id: int, *, actor: Optional[str]) -> Dict[str, Any]:
    """Send for review. The quote must be something a reviewer can decide on: addressed
    to someone, with at least one line. It need not be on a deal yet -- that is asked
    before it is SENT (quote_send), which is the ruling of 2026-10-12."""
    cur = dict_cursor(conn)
    cur.execute("SELECT deal_id, account_id, customer_name, "
                "(SELECT count(*) FROM proc.bp_sales_quote_line l "
                " WHERE l.sales_quote_id = q.sales_quote_id) AS lines "
                "FROM proc.bp_sales_quote q WHERE sales_quote_id = %s", (quote_id,))
    row = cur.fetchone()
    conn.rollback()                      # a read; leave no transaction open for submit's
    if row is None:
        raise NotFound(f"quote {quote_id} does not exist")
    missing = [what for what, absent in (
        ("a customer", not (row["customer_name"] or row["account_id"])),
        ("at least one line", not row["lines"])) if absent]
    if missing:
        raise StateConflict(f"quote {quote_id} cannot go for review without "
                            + ", ".join(missing))
    quotes.submit(conn, quote_id, actor=actor)
    return get(conn, quote_id)


def supersede(conn: Any, quote_id: int, *, actor: Optional[str]) -> Dict[str, Any]:
    """Change a quote the customer already holds: it stays exactly as it was sent and is
    marked superseded, and a new draft, version n+1, starts as a copy of it. The new
    draft has no number and no conditions -- both belong to an approval it has not had."""
    cur = dict_cursor(conn)
    try:
        old = _lock(cur, quote_id)
        if old["status"] != "issued":
            raise StateConflict(f"quote {quote_id} is {old['status']}; only a quote that has "
                                "been sent is superseded (a draft is simply edited)")
        _refuse_catalog_quote(cur, quote_id)
        cur.execute("SELECT 1 FROM proc.bp_sales_quote_outcome WHERE sales_quote_id = %s",
                    (quote_id,))
        if cur.fetchone() is not None:
            raise StateConflict(f"quote {quote_id} already has a recorded outcome and cannot "
                                "be superseded")
        draft = _screen(cur, old)
        today = dt.date.today()
        draft["issuedAt"] = today
        if _date(draft["validUntil"], "validUntil") < today:
            draft["validUntil"] = today + (old["valid_until"] - old["quote_date"])
        cur.execute("UPDATE proc.bp_sales_quote SET status = 'superseded', "
                    "last_modified_date = clock_timestamp() WHERE sales_quote_id = %s",
                    (quote_id,))
        new = _insert(cur, _clean(draft), deal_id=old["deal_id"], created_by=actor,
                      version=old["version"] + 1, supersedes_id=quote_id,
                      account_id=old["account_id"])
        quotes.record_event(conn, "sales_quote.superseded", old, actor=actor,
                            superseded_by=new["sales_quote_id"])
        quotes.record_event(conn, "sales_quote.created", new, actor=actor,
                            deal_id=old["deal_id"], supersedes_id=quote_id, builder=True)
    except Exception:
        conn.rollback()
        raise
    conn.commit()
    return get(conn, new["sales_quote_id"])


# --- reads -------------------------------------------------------------------------

def _revision(row: Dict[str, Any]) -> str:
    return row["last_modified_date"].isoformat()


def _screen(cur: Any, row: Dict[str, Any]) -> Dict[str, Any]:
    """One quote in the shape the Quote builder holds (UI data/mockQuotes.js). Ids are
    strings; money and quantities are decimal strings by the time they leave the API."""
    quote_id = row["sales_quote_id"]
    cur.execute(
        "SELECT l.sales_quote_line_id, l.line_tag, l.item_description, l.quantity, "
        "l.unit_of_measure, l.unit_price, l.entered_discount_pct, l.line_total, "
        "i.unit_cost, i.markup_pct, i.cost_source "
        "FROM proc.bp_sales_quote_line l "
        "LEFT JOIN proc.bp_sales_quote_line_internal i "
        "ON i.sales_quote_line_id = l.sales_quote_line_id "
        "WHERE l.sales_quote_id = %s ORDER BY l.line_no", (quote_id,))
    lines = [{"id": str(l["sales_quote_line_id"]), "tag": l["line_tag"],
              "description": l["item_description"], "qty": l["quantity"],
              "unit": l["unit_of_measure"], "cost": l["unit_cost"],
              "markupPct": l["markup_pct"], "unitPrice": l["unit_price"],
              "discountPct": l["entered_discount_pct"] or Decimal(0),
              "lineTotal": l["line_total"], "costSource": l["cost_source"]}
             for l in cur.fetchall()]
    cur.execute("SELECT total_cost, total_margin, margin_pct, costed_lines, uncosted_lines "
                "FROM proc.bp_sales_quote_internal WHERE sales_quote_id = %s", (quote_id,))
    internal = dict(cur.fetchone() or {})
    return {
        "id": str(quote_id), "ref": row["quote_ref"], "dealId": row["deal_id"],
        "version": row["version"], "status": _SCREEN_STATUS.get(row["status"], row["status"]),
        "backendStatus": row["status"], "number": row["quote_number"],
        "currency": (row["currency"] or "").strip(), "category": row["category"],
        "issuedAt": row["quote_date"], "validUntil": row["valid_until"],
        "preparedBy": row["prepared_by"],
        "customer": {"name": row["customer_name"], "contact": row["customer_contact"],
                     "accountId": row["account_id"]},
        "scopeHtml": row["scope_html"] or "", "assumptionsHtml": row["assumptions_html"] or "",
        "lines": lines, "terms": row["terms"] or {}, "conditions": row["conditions"] or [],
        "totals": {"exTax": row["total_ex_tax"], "cost": internal.get("total_cost"),
                   "margin": internal.get("total_margin"),
                   # a fraction of costed revenue (0.4 = 40%), the sell side's convention
                   "marginFraction": internal.get("margin_pct"),
                   "costedLines": internal.get("costed_lines"),
                   "uncostedLines": internal.get("uncosted_lines")},
        "position": {"phaseId": row["phase_id"], "subprocessId": row["subprocess_id"]},
        "supersedesId": str(row["supersedes_id"]) if row["supersedes_id"] else None,
        "createdBy": row["created_by"], "approvedBy": row["approved_by"],
        "approvedAt": row["approved_at"], "sentAt": row["issued_at"],
        "revision": _revision(row),
    }


def _select(cur: Any, where: str, params: tuple) -> List[Dict[str, Any]]:
    cur.execute(f"SELECT {', '.join(_COLS)} FROM proc.bp_sales_quote WHERE {where}", params)
    return [_screen(cur, dict(r)) for r in cur.fetchall()]


def get(conn: Any, quote_id: int) -> Dict[str, Any]:
    found = _select(dict_cursor(conn), "sales_quote_id = %s", (quote_id,))
    if not found:
        raise NotFound(f"quote {quote_id} does not exist")
    return found[0]


def list_for_deal(conn: Any, deal_id: str) -> List[Dict[str, Any]]:
    """The deal's quotes, latest first: the live one, then what it superseded."""
    return _select(dict_cursor(conn),
                   "deal_id = %s ORDER BY version DESC, sales_quote_id DESC", (deal_id,))


def list_unlinked(conn: Any, created_by: Optional[str]) -> List[Dict[str, Any]]:
    """Drafts started before their deal existed. Only the drafter's own: an unlinked
    quote is on nobody's deal, so there is no deal team to share it with."""
    if not created_by:
        return []
    return _select(dict_cursor(conn),
                   "deal_id IS NULL AND account_id IS NULL AND created_by = %s "
                   "AND status <> 'superseded' ORDER BY last_modified_date DESC LIMIT 100",
                   (created_by,))


def review_queue(conn: Any) -> List[Dict[str, Any]]:
    """Every quote waiting for a reviewer, oldest first. A quote in review IS the pending
    approval; nothing is copied into a second table that could disagree with it."""
    return _select(dict_cursor(conn),
                   "status = 'in_review' ORDER BY last_modified_date, sales_quote_id", ())
