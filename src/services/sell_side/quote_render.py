"""What a customer may see of a quote (spec §4.3). THE control on cost and margin.

Two layers, deliberately redundant:
  1. an allowlist projection -- only named fields are copied, so a column added
     to a table later is invisible here until someone names it;
  2. a leak guard that walks the finished payload and raises on any internal
     field name, so naming one fails the render instead of shipping it.
render_html re-runs the guard, so a view built by hand cannot bypass it.
"""
from __future__ import annotations

import html
from typing import Any, Dict, Iterable

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


def customer_view(quote: Dict[str, Any]) -> Dict[str, Any]:
    if quote.get("status") not in _READY:
        raise NotCustomerReady(f"quote is {quote.get('status')}; only approved or issued "
                               "quotes have a customer view")
    view = _project(quote, CUSTOMER_HEADER_FIELDS)
    view["lines"] = [_project(l, CUSTOMER_LINE_FIELDS) for l in quote.get("lines") or []]
    view["justifications"] = [
        _project(j, CUSTOMER_JUSTIFICATION_FIELDS)
        for j in quote.get("justifications") or [] if j.get("customer_safe") is True]
    _assert_no_internal(view)
    return view


def _e(value: Any) -> str:
    return "" if value is None else html.escape(str(value))


def render_html(view: Dict[str, Any]) -> str:
    _assert_no_internal(view)
    rows = "".join(
        f"<tr><td>{_e(l.get('line_no'))}</td><td>{_e(l.get('distributor_sku'))}</td>"
        f"<td>{_e(l.get('item_description'))}</td><td>{_e(l.get('quantity'))}</td>"
        f"<td>{_e(l.get('unit_of_measure'))}</td><td>{_e(l.get('unit_price'))}</td>"
        f"<td>{_e(l.get('line_total'))}</td></tr>"
        for l in view.get("lines") or [])
    notes = "".join(f"<li>{_e(j.get('claim'))}</li>" for j in view.get("justifications") or [])
    return (
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
