"""Quote templates: the shapes a seller starts a customer quote from.

A template is structure and nothing else -- which lines this kind of offer usually has,
what each is measured in, which terms need an answer. It carries no price, no cost, no
discount and no legal wording; `clean` refuses a body that tries to. So starting a
quote from a template can never put a figure on it that nobody gave.

`missing` is the other half: read against a quote, a template says what that quote
still needs. The quote_drafting agent uses it to tell the seller what to give next,
and it is worked out here, in code, not asked of the model.
"""
from __future__ import annotations

import datetime as dt
import json
import re
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Mapping, Optional

from src.services.sell_side._db import NotFound, dict_cursor
from src.services.sell_side.builder import LINE_TAGS, TERM_KEYS, clean_html

_KEY = re.compile(r"[a-z][a-z0-9_]{1,59}")
MAX_TEMPLATE_LINES = 30
VALIDITY_DAYS = 30          # the Quote builder's own blank quote runs for thirty days
_COLS = ("template_key", "template_name", "description", "body", "status",
         "last_modified_by", "last_modified_date")
TERM_LABELS = {"payment": "payment terms", "start": "a start date",
               "lengthMonths": "the length of the term in months",
               "renewal": "the renewal terms", "sla": "the service level"}


def _text(value: Any, what: str, limit: int, *, needed: bool = False) -> Optional[str]:
    s = str(value).strip() if value is not None else ""
    if len(s) > limit:
        raise ValueError(f"{what} is longer than {limit} characters")
    if needed and not s:
        raise ValueError(f"{what} is needed")
    return s or None


def clean(body: Any) -> Dict[str, Any]:
    """A template body as it is stored. Only the known fields are kept, so a price or a
    cost sent with a line is dropped here rather than carried onto every quote."""
    if not isinstance(body, dict):
        raise ValueError("a template body must be an object")
    raw_lines = body.get("lines") or []
    if not isinstance(raw_lines, list) or not raw_lines:
        raise ValueError("a template needs at least one line")
    if len(raw_lines) > MAX_TEMPLATE_LINES:
        raise ValueError(f"a template has at most {MAX_TEMPLATE_LINES} lines")
    lines, seen = [], set()
    for n, raw in enumerate(raw_lines, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"line {n} must be an object")
        key = str(raw.get("key") or "").strip()
        if not _KEY.fullmatch(key):
            raise ValueError(f"line {n}: key {key!r} must be lower-case letters, digits and "
                             "underscores, starting with a letter")
        if key in seen:
            raise ValueError(f"line {n}: key {key!r} is used twice")
        seen.add(key)
        tag = raw.get("tag") or "service"
        if tag not in LINE_TAGS:
            raise ValueError(f"line {n}: tag {tag!r} is not one of {', '.join(LINE_TAGS)}")
        lines.append({"key": key, "tag": tag,
                      "description": _text(raw.get("description"), f"line {n}: description",
                                           300, needed=True),
                      "unit": _text(raw.get("unit"), f"line {n}: unit", 40) or "unit"})
    fields = body.get("termFields") or []
    if not isinstance(fields, list) or any(f not in TERM_KEYS for f in fields):
        raise ValueError(f"termFields may only name {', '.join(TERM_KEYS)}")
    return {"category": _text(body.get("category"), "category", 300),
            "lines": lines,
            "termFields": [f for f in TERM_KEYS if f in fields],
            "assumptionsHtml": clean_html(body.get("assumptionsHtml"), "assumptionsHtml")}


def _row(row: Mapping[str, Any]) -> Dict[str, Any]:
    body = row["body"]
    if isinstance(body, str):
        body = json.loads(body)
    return {"key": row["template_key"], "name": row["template_name"],
            "description": row["description"], "status": row["status"], **body,
            "updatedBy": row["last_modified_by"], "updatedAt": row["last_modified_date"]}


def list_templates(conn: Any, *, include_retired: bool = False) -> List[Dict[str, Any]]:
    cur = dict_cursor(conn)
    cur.execute(f"SELECT {', '.join(_COLS)} FROM proc.bp_sales_quote_template "
                + ("" if include_retired else "WHERE status = 'active' ")
                + "ORDER BY template_name, template_key")
    return [_row(r) for r in cur.fetchall()]


def get(conn: Any, key: str) -> Dict[str, Any]:
    cur = dict_cursor(conn)
    cur.execute(f"SELECT {', '.join(_COLS)} FROM proc.bp_sales_quote_template "
                "WHERE template_key = %s", (key,))
    row = cur.fetchone()
    if row is None:
        raise NotFound(f"quote template {key!r} does not exist")
    return _row(row)


def find(conn: Any, key: Optional[str]) -> Optional[Dict[str, Any]]:
    """The template a quote was started from, if it still exists. Retired counts: a quote
    begun on a template is finished on it."""
    if not key:
        return None
    try:
        return get(conn, key)
    except NotFound:
        return None


def save(conn: Any, key: str, *, name: str, description: Optional[str], body: Any,
         status: str = "active", actor: Optional[str]) -> Dict[str, Any]:
    """Add a template or replace one. Quotes already started from it are untouched: a
    template is copied from, never linked to."""
    if not _KEY.fullmatch(key or ""):
        raise ValueError("a template key is lower-case letters, digits and underscores, "
                         "starting with a letter")
    if status not in ("active", "retired"):
        raise ValueError("status is active or retired")
    name = _text(name, "name", 120, needed=True)
    description = _text(description, "description", 500)
    cleaned = clean(body)
    cur = dict_cursor(conn)
    try:
        cur.execute(
            "INSERT INTO proc.bp_sales_quote_template (template_key, template_name, description, "
            "body, status, created_by, last_modified_by) VALUES (%s, %s, %s, %s::jsonb, %s, %s, %s) "
            "ON CONFLICT (template_key) DO UPDATE SET template_name = EXCLUDED.template_name, "
            "description = EXCLUDED.description, body = EXCLUDED.body, status = EXCLUDED.status, "
            "last_modified_by = EXCLUDED.last_modified_by, last_modified_date = now()",
            (key, name, description, json.dumps(cleaned), status, actor, actor))
    except Exception:
        conn.rollback()
        raise
    conn.commit()
    return get(conn, key)


def draft_from(template: Mapping[str, Any], *, currency: str,
               customer: Optional[Mapping[str, Any]] = None,
               today: Optional[dt.date] = None) -> Dict[str, Any]:
    """The blank quote this template starts: its lines, each at quantity 1 with NO price
    and NO cost, and nothing in the terms. Every figure is still to be given."""
    today = today or dt.date.today()
    return {
        "currency": currency, "category": template.get("category"),
        "issuedAt": today.isoformat(),
        "validUntil": (today + dt.timedelta(days=VALIDITY_DAYS)).isoformat(),
        "customer": dict(customer or {}),
        "scopeHtml": "", "assumptionsHtml": template.get("assumptionsHtml") or "",
        "lines": [{"tag": l["tag"], "description": l["description"], "qty": 1,
                   "unit": l["unit"], "cost": None, "markupPct": None, "unitPrice": 0,
                   "discountPct": 0} for l in template["lines"]],
        "terms": {},
    }


def _zero(value: Any) -> bool:
    try:
        return value is None or value == "" or Decimal(str(value)) == 0
    except (InvalidOperation, ValueError):
        return True


def missing(template: Optional[Mapping[str, Any]], quote: Mapping[str, Any]) -> List[Dict[str, str]]:
    """What this quote still needs before it is a finished quote of its kind: who it is
    for, a price on every line, and the terms its template asks for. In the order a
    seller would give them."""
    out: List[Dict[str, str]] = []

    def need(field: str, label: str) -> None:
        out.append({"field": field, "label": label})

    customer = quote.get("customer") or {}
    if not str(customer.get("name") or "").strip():
        need("customer.name", "who the quote is for")
    if not str(customer.get("contact") or "").strip():
        need("customer.contact", "a contact at the customer")
    for n, line in enumerate(quote.get("lines") or [], start=1):
        if isinstance(line, dict) and _zero(line.get("unitPrice")):
            name = str(line.get("description") or "").strip() or f"line {n}"
            need(f"lines.{line.get('id')}.unitPrice", f"a price for “{name[:60]}”")
    terms = quote.get("terms") or {}
    for field in (template or {}).get("termFields") or []:
        if terms.get(field) in (None, ""):
            need(f"terms.{field}", TERM_LABELS[field])
    return out
