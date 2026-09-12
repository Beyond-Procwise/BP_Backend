import datetime as dt
from decimal import Decimal as D

import pytest

from src.services.sell_side import quote_render as qr


def _quote(status="approved", safe=True):
    return {
        "sales_quote_id": 1, "quote_ref": "SQ-20260911-000001", "status": status,
        "account_name": "Acme <Ltd>", "contact_name": "Ann", "currency": "GBP",
        "quote_date": dt.date(2026, 9, 11), "valid_until": dt.date(2026, 10, 11),
        "total_ex_tax": D("56.00"), "total_cost": D("40.00"), "total_margin": D("16.00"),
        "margin_pct": D("0.2857"), "created_by": "sub-author",
        "lines": [{"line_no": 1, "distributor_sku": "IN-1", "mpn": None,
                   "item_description": "Widget & co", "quantity": D("4"),
                   "unit_of_measure": "EA", "currency": "GBP",
                   "list_price_at_quote": D("15.0000"), "unit_price": D("14.0000"),
                   "discount_pct": D("0.0667"), "line_total": D("56.00"),
                   "unit_cost": D("10.0000"), "cost_tier_applied": None,
                   "line_margin": D("16.00"), "line_margin_pct": D("0.2857")}],
        "justifications": [
            {"kind": "end_of_life", "claim": "EOS 2026-06-30", "customer_safe": True},
            {"kind": "price_gap", "claim": "We make 29% here", "customer_safe": safe},
        ],
    }


def _keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _keys(v)


def test_the_customer_view_carries_no_internal_field_anywhere():
    view = qr.customer_view(_quote())
    assert not set(_keys(view)) & qr.INTERNAL_FIELDS
    assert view["lines"][0]["unit_price"] == D("14.0000")


def test_an_internal_only_justification_never_leaves():
    view = qr.customer_view(_quote(safe=False))
    assert [j["claim"] for j in view["justifications"]] == ["EOS 2026-06-30"]


def test_widening_the_allowlist_to_a_cost_field_fails_the_render(monkeypatch):
    """Acceptance criterion 6: attempt to render cost and watch it fail."""
    monkeypatch.setattr(qr, "CUSTOMER_LINE_FIELDS", qr.CUSTOMER_LINE_FIELDS | {"unit_cost"})
    with pytest.raises(qr.InternalFieldLeak, match="unit_cost"):
        qr.customer_view(_quote())


def test_a_hand_built_view_with_margin_cannot_be_rendered_to_html():
    with pytest.raises(qr.InternalFieldLeak, match="total_margin"):
        qr.render_html({"quote_ref": "x", "total_margin": D("1"), "lines": []})


def test_an_unapproved_quote_has_no_customer_view():
    with pytest.raises(qr.NotCustomerReady):
        qr.customer_view(_quote(status="draft"))


def test_html_escapes_everything_and_shows_no_cost():
    html = qr.render_html(qr.customer_view(_quote()))
    assert "Acme &lt;Ltd&gt;" in html and "Widget &amp; co" in html
    assert "10.0000" not in html and "16.00" not in html


def test_the_allowlists_and_the_internal_set_are_disjoint():
    for allowed in (qr.CUSTOMER_HEADER_FIELDS, qr.CUSTOMER_LINE_FIELDS,
                    qr.CUSTOMER_JUSTIFICATION_FIELDS):
        assert not allowed & qr.INTERNAL_FIELDS
