"""What a customer may see of a quote (spec §4.3). THE control on cost and margin.

Three layers, deliberately redundant:
  1. an allowlist projection -- only named fields are copied, so a column added
     to a table later is invisible here until someone names it;
  2. a leak guard that walks the finished payload and raises on any internal
     field name, so naming one fails the render instead of shipping it;
  3. a leak guard on VALUES -- the finished output is searched for this quote's
     own cost and margin figures, whatever key or sentence they arrived under.
render_html re-runs the guards, so a view built by hand cannot bypass them.

The quote handed in here is the customer record (quotes.get_quote), which has no
cost in it. The internal record (quotes.get_quote_internal) is passed beside it
for one purpose: so layer 3 knows which figures to look for.
"""
from __future__ import annotations

import html
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Iterable, Iterator, Set

INTERNAL_FIELDS = frozenset({
    "unit_cost", "cost_tier_applied", "line_margin", "line_margin_pct",
    "total_cost", "total_margin", "margin_pct", "expected_cost", "expected_margin",
    "cost_price", "cost_basis", "customer_safe",
})
CUSTOMER_HEADER_FIELDS = frozenset({
    "quote_ref", "account_name", "contact_name", "currency", "quote_date",
    "valid_until", "total_ex_tax",
})
CUSTOMER_LINE_FIELDS = frozenset({
    "line_no", "distributor_sku", "mpn", "item_description", "quantity",
    "unit_of_measure", "currency", "list_price_at_quote", "unit_price",
    "discount_pct", "line_total",
})
CUSTOMER_JUSTIFICATION_FIELDS = frozenset({"kind", "claim"})
# Words a person typed. A number inside one of these is never "accounted for" (see
# assert_no_internal_values), so a cost figure written into a description is a leak.
FREE_TEXT_FIELDS = frozenset({"account_name", "contact_name", "item_description",
                              "unit_of_measure", "kind", "claim"})
# Searched for as figures. cost_tier_applied is a small quantity break (1, 10, 100) that
# would match ordinary text constantly and is not money; it is caught by name only.
_INTERNAL_MONEY = ("total_cost", "total_margin", "unit_cost", "line_margin")
_INTERNAL_PCT = ("margin_pct", "line_margin_pct")
_NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")

_READY = ("approved", "issued")


class InternalFieldLeak(RuntimeError):
    """An internal field reached a customer-facing payload. Never caught to recover."""


class NotCustomerReady(ValueError):
    """Only an approved or issued quote has a customer view."""


def _project(row: Dict[str, Any], allowed: Iterable[str]) -> Dict[str, Any]:
    return {k: row.get(k) for k in sorted(allowed)}


def _assert_no_internal(obj: Any, path: str = "$") -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in INTERNAL_FIELDS:
                raise InternalFieldLeak(f"{path}.{k} is internal and may not reach a customer")
            _assert_no_internal(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _assert_no_internal(v, f"{path}[{i}]")


def _canon(value: Any) -> Any:
    """A number as one comparable form: sign, trailing zeros and separators dropped, so
    16, 16.00, -16.0000 and "16.00" are the same figure. None when it is not a number."""
    try:
        d = abs(Decimal(str(value).replace(",", "")))
    except (InvalidOperation, ValueError):
        return None
    return d.normalize() if d.is_finite() else None


def _numbers(text: str) -> Counter:
    return Counter(n for n in (_canon(m) for m in _NUMBER.findall(text)) if n is not None)


def _flat(obj: Any) -> Iterator[str]:
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _flat(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _flat(v)
    elif obj is not None:
        yield str(obj)


def _internal_figures(internal: Dict[str, Any]) -> Dict[Any, str]:
    """Every form this quote's cost and margin could be printed in -> the field it is.
    A percentage is looked for as stored (0.2857) and as a person writes it (28.57, 28.6,
    29). Zero is not looked for: a bare 0 identifies nothing."""
    found: Dict[Any, str] = {}

    def add(value: Any, name: str) -> None:
        n = _canon(value)
        if n is not None and n != 0:
            found.setdefault(n, name)

    for row, where in [(internal, "")] + [
            (l, f"line {l.get('line_no')} ") for l in internal.get("lines") or []]:
        for name in _INTERNAL_MONEY:
            if row.get(name) is not None:
                v = Decimal(str(row[name]))
                add(v, where + name)
                add(v.quantize(Decimal("0.01")), where + name)
        for name in _INTERNAL_PCT:
            if row.get(name) is not None:
                v = Decimal(str(row[name]))
                add(v, where + name)
                for places in ("1", "0.1", "0.01"):
                    add((v * 100).quantize(Decimal(places)), where + name)
    return found


def assert_no_internal_values(rendered: str, quote: Dict[str, Any],
                              internal: Dict[str, Any]) -> None:
    """Search what the customer is about to be sent for this quote's cost and margin.

    A figure may appear only as often as the customer record's own structured fields
    (quantities, prices, totals, dates, references) account for it -- so a price that
    happens to equal a cost elsewhere still prints, once. Free text accounts for nothing.
    `quote` is the customer record as read from the database, not the view being
    rendered: an internal figure slipped under an allowed key is in the output but not in
    the record, and is caught. A coincidence in free text ("50mm", cost 50.00) is refused
    too; that is this check failing closed.
    """
    text = html.unescape(rendered)
    lowered = text.lower()
    for name in sorted(INTERNAL_FIELDS):
        for form in (name, name.replace("_", " ")):
            if re.search(rf"(?<![a-z]){re.escape(form)}(?![a-z])", lowered):
                raise InternalFieldLeak(
                    f"INTERNAL_FIELD_LEAK: the internal field name {name!r} appears in "
                    "what the customer would be sent")
    allowed: Counter = Counter()
    for k in CUSTOMER_HEADER_FIELDS - FREE_TEXT_FIELDS:
        allowed += _numbers(" ".join(_flat(quote.get(k))))
    for line in quote.get("lines") or []:
        for k in CUSTOMER_LINE_FIELDS - FREE_TEXT_FIELDS:
            allowed += _numbers(" ".join(_flat(line.get(k))))
    seen = _numbers(text)
    for figure, name in _internal_figures(internal).items():
        if seen[figure] > allowed[figure]:
            raise InternalFieldLeak(
                f"INTERNAL_FIELD_LEAK: {figure} ({name}) appears in what the customer "
                f"would be sent {seen[figure]} time(s); the quote's own figures account "
                f"for {allowed[figure]}")


def customer_view(quote: Dict[str, Any], internal: Dict[str, Any]) -> Dict[str, Any]:
    if quote.get("status") not in _READY:
        raise NotCustomerReady(f"quote is {quote.get('status')}; only approved or issued "
                               "quotes have a customer view")
    view = _project(quote, CUSTOMER_HEADER_FIELDS)
    view["lines"] = [_project(l, CUSTOMER_LINE_FIELDS) for l in quote.get("lines") or []]
    view["justifications"] = [
        _project(j, CUSTOMER_JUSTIFICATION_FIELDS)
        for j in quote.get("justifications") or [] if j.get("customer_safe") is True]
    _assert_no_internal(view)
    assert_no_internal_values(" ".join(_flat(view)), quote, internal)
    return view


def _e(value: Any) -> str:
    return "" if value is None else html.escape(str(value))


def render_html(view: Dict[str, Any], *, quote: Dict[str, Any],
                internal: Dict[str, Any]) -> str:
    """The page, checked. `quote` and `internal` are required: there is no way to ask for
    the HTML without the value check running over it."""
    _assert_no_internal(view)
    rows = "".join(
        f"<tr><td>{_e(l.get('line_no'))}</td><td>{_e(l.get('distributor_sku'))}</td>"
        f"<td>{_e(l.get('item_description'))}</td><td>{_e(l.get('quantity'))}</td>"
        f"<td>{_e(l.get('unit_of_measure'))}</td><td>{_e(l.get('unit_price'))}</td>"
        f"<td>{_e(l.get('line_total'))}</td></tr>"
        for l in view.get("lines") or [])
    notes = "".join(f"<li>{_e(j.get('claim'))}</li>" for j in view.get("justifications") or [])
    page = (
        f"<article class=\"sales-quote\"><h1>Quote {_e(view.get('quote_ref'))}</h1>"
        f"<p>For {_e(view.get('account_name'))}"
        f"{(' — ' + _e(view.get('contact_name'))) if view.get('contact_name') else ''}</p>"
        f"<p>Dated {_e(view.get('quote_date'))}, valid until {_e(view.get('valid_until'))}. "
        f"Prices in {_e(view.get('currency'))}, excluding tax.</p>"
        "<table><thead><tr><th>#</th><th>SKU</th><th>Description</th><th>Qty</th>"
        "<th>Unit</th><th>Unit price</th><th>Line total</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
        f"<p><strong>Total ex tax: {_e(view.get('total_ex_tax'))} {_e(view.get('currency'))}</strong></p>"
        f"{('<ul>' + notes + '</ul>') if notes else ''}</article>"
    )
    assert_no_internal_values(page, quote, internal)
    return page
