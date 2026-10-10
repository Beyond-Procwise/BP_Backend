from decimal import Decimal as D

import pytest

from src.services.sell_side import quote_render as qr
from tests.sell_side.fixtures.synthetic_quotes import goods_quote, unknown_cost_quote


def _keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _keys(v)


def _html(quote, internal, view=None):
    return qr.render_html(view or qr.customer_view(quote, internal), quote=quote, internal=internal)


def test_the_customer_view_carries_no_internal_field_anywhere():
    view = qr.customer_view(*goods_quote())
    assert not set(_keys(view)) & qr.INTERNAL_FIELDS
    assert view["lines"][0]["unit_price"] == D("58.0000")


def test_an_internal_only_justification_never_leaves():
    quote, internal = goods_quote()
    quote["justifications"] = internal["justifications"]   # as if the reader had not filtered
    view = qr.customer_view(quote, internal)
    assert [j["claim"] for j in view["justifications"]] == ["Line 3 replaces a withdrawn part"]


def test_widening_the_allowlist_to_a_cost_field_fails_the_render(monkeypatch):
    """Acceptance criterion 6: attempt to render cost and watch it fail."""
    monkeypatch.setattr(qr, "CUSTOMER_LINE_FIELDS", qr.CUSTOMER_LINE_FIELDS | {"unit_cost"})
    with pytest.raises(qr.InternalFieldLeak, match="unit_cost"):
        qr.customer_view(*goods_quote())


def test_a_hand_built_view_with_margin_cannot_be_rendered_to_html():
    quote, internal = goods_quote()
    with pytest.raises(qr.InternalFieldLeak, match="total_margin"):
        qr.render_html({"quote_ref": "x", "total_margin": D("1"), "lines": []},
                       quote=quote, internal=internal)


def test_the_html_cannot_be_asked_for_without_the_value_check():
    with pytest.raises(TypeError):
        qr.render_html(qr.customer_view(*goods_quote()))


def test_an_unapproved_quote_has_no_customer_view():
    with pytest.raises(qr.NotCustomerReady):
        qr.customer_view(*goods_quote(status="draft"))


def test_html_escapes_everything_and_shows_no_cost():
    html = _html(*goods_quote())
    assert "Synthetic Fixture Co &lt;Ltd&gt;" in html and "bracket &amp; plate" in html
    for ours in ("1543.75", "633.25", "27.35", "420.00", "125.6667"):
        assert ours not in html


def test_the_allowlists_and_the_internal_set_are_disjoint():
    for allowed in (qr.CUSTOMER_HEADER_FIELDS, qr.CUSTOMER_LINE_FIELDS,
                    qr.CUSTOMER_JUSTIFICATION_FIELDS):
        assert not allowed & qr.INTERNAL_FIELDS


# --- the check on VALUES (QDA B1b) --------------------------------------------------
# The name guard above cannot see a figure. Each case below puts one of the quote's own
# cost or margin figures somewhere a name guard is blind to.

def test_a_price_that_equals_a_cost_on_another_line_still_renders():
    """41.20 is line 1's cost and line 2's price. The customer's own figure prints."""
    quote, internal = goods_quote()
    assert "41.2000" in _html(quote, internal)


def test_a_quote_with_an_unknown_cost_renders():
    assert "SQ-SYNTHETIC-B" in _html(*unknown_cost_quote())


@pytest.mark.parametrize("key, figure, named", [
    ("unit_price", D("27.3500"), "line 2 unit_cost"),      # cost under the price key
    ("line_total", D("420.00"), "line 1 line_margin"),     # margin under the total key
])
def test_an_internal_figure_under_an_allowed_line_key_is_caught(key, figure, named):
    quote, internal = goods_quote()
    view = qr.customer_view(quote, internal)
    view["lines"][1][key] = figure
    with pytest.raises(qr.InternalFieldLeak, match=named):
        _html(quote, internal, view)


def test_an_internal_figure_under_an_allowed_header_key_is_caught():
    quote, internal = goods_quote()
    view = qr.customer_view(quote, internal)
    view["total_ex_tax"] = internal["total_cost"]
    with pytest.raises(qr.InternalFieldLeak, match="total_cost"):
        _html(quote, internal, view)


@pytest.mark.parametrize("claim, named", [
    ("We clear 633.25 on this one", "total_margin"),
    ("we clear £1,543.75 less than that", "total_cost"),   # thousands separator
    ("that is 29% for us", "margin_pct"),                  # 0.2909 as a person writes it
    ("that is 29.09 per cent", "margin_pct"),
    ("0.2909 of it is ours", "margin_pct"),
    ("each one costs us 27.35", "line 2 unit_cost"),       # 27.3500 to two places
])
def test_an_internal_figure_inside_a_sentence_is_caught(claim, named):
    quote, internal = goods_quote()
    quote["justifications"] = [{"kind": "price_gap", "claim": claim, "customer_safe": True}]
    with pytest.raises(qr.InternalFieldLeak, match=named):
        qr.customer_view(quote, internal)


def test_a_cost_that_is_also_a_price_is_caught_the_second_time_it_appears():
    """41.20 may print once, as line 2's price. Written again in a sentence, it is ours."""
    quote, internal = goods_quote()
    quote["justifications"] = [{"kind": "price_gap", "claim": "the bracket is 41.20 to us",
                                "customer_safe": True}]
    with pytest.raises(qr.InternalFieldLeak, match="unit_cost"):
        qr.customer_view(quote, internal)


def test_a_description_that_happens_to_contain_a_cost_figure_is_refused():
    """Failing closed, on purpose (ruled 2026-10-10): the check cannot tell a coincidence
    from a leak, so it refuses and names the figure."""
    quote, internal = goods_quote()
    quote["lines"][2]["item_description"] = "Synthetic enclosure 144mm"
    with pytest.raises(qr.InternalFieldLeak, match="144.*line 3 line_margin"):
        qr.customer_view(quote, internal)


def test_an_internal_field_name_in_a_sentence_is_caught():
    quote, internal = goods_quote()
    quote["justifications"] = [{"kind": "price_gap", "claim": "see Unit Cost column",
                                "customer_safe": True}]
    with pytest.raises(qr.InternalFieldLeak, match="unit_cost"):
        qr.customer_view(quote, internal)
