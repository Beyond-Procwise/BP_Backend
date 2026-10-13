"""Quote templates: structure only, and what a quote of that kind still needs."""
import datetime as dt

import pytest

from src.services.sell_side import builder, templates as qt

BODY = {"category": None,
        "lines": [{"key": "licences", "tag": "product", "description": "Subscription licences",
                   "unit": "seat"},
                  {"key": "onboarding", "tag": "service", "description": "Onboarding",
                   "unit": "engagement"}],
        "termFields": ["start", "payment"], "assumptionsHtml": "<p><b>Assumes:</b> </p>"}
TEMPLATE = {"key": "software_subscription", "name": "Software subscription", **qt.clean(BODY)}


def test_a_template_keeps_no_price_cost_or_discount_even_if_one_is_sent():
    cleaned = qt.clean({**BODY, "lines": [{**BODY["lines"][0], "unitPrice": 200, "cost": 125,
                                           "discountPct": 10, "qty": 340}],
                        "marginFloorPct": 30})
    assert cleaned["lines"] == [{"key": "licences", "tag": "product",
                                 "description": "Subscription licences", "unit": "seat"}]
    assert set(cleaned) == {"category", "lines", "termFields", "assumptionsHtml"}


def test_term_fields_are_kept_in_the_quotes_own_order():
    assert qt.clean(BODY)["termFields"] == ["payment", "start"]


def test_what_a_template_prints_is_cleaned_like_any_other_rich_text():
    cleaned = qt.clean({**BODY, "assumptionsHtml": '<p onclick="x()">A</p><script>y()</script>'})
    assert cleaned["assumptionsHtml"] == "<p>A</p>"


@pytest.mark.parametrize("body, says", [
    ("not a body", "object"), ({**BODY, "lines": []}, "at least one line"),
    ({**BODY, "lines": [{"key": "Bad Key", "description": "x"}]}, "key"),
    ({**BODY, "lines": [BODY["lines"][0], BODY["lines"][0]]}, "twice"),
    ({**BODY, "lines": [{"key": "a1", "description": " "}]}, "description"),
    ({**BODY, "lines": [{"key": "a1", "description": "x", "tag": "gift"}]}, "tag"),
    ({**BODY, "termFields": ["payment", "price"]}, "termFields"),
])
def test_a_template_that_does_not_hold_together_is_refused(body, says):
    with pytest.raises(ValueError, match=says):
        qt.clean(body)


def test_a_quote_started_from_a_template_has_its_lines_and_not_one_figure():
    draft = qt.draft_from(TEMPLATE, currency="GBP", today=dt.date(2026, 10, 13),
                          customer={"name": "Kestrel Freight"})
    assert [(l["description"], l["unit"], l["tag"]) for l in draft["lines"]] == [
        ("Subscription licences", "seat", "product"), ("Onboarding", "engagement", "service")]
    assert all(l["unitPrice"] == 0 and l["cost"] is None and l["markupPct"] is None
               and l["discountPct"] == 0 and l["qty"] == 1 for l in draft["lines"])
    assert draft["terms"] == {} and draft["validUntil"] == "2026-11-12"
    assert draft["customer"] == {"name": "Kestrel Freight"}
    builder._clean(draft)             # it is a draft the builder accepts as it stands


def test_a_fresh_template_quote_is_missing_everything_a_seller_has_to_give():
    quote = {"customer": {}, "terms": {},
             "lines": [{"id": "1", "description": "Subscription licences", "unitPrice": "0.0000"},
                       {"id": "2", "description": "Onboarding", "unitPrice": 0}]}
    assert [m["field"] for m in qt.missing(TEMPLATE, quote)] == [
        "customer.name", "customer.contact", "lines.1.unitPrice", "lines.2.unitPrice",
        "terms.payment", "terms.start"]


def test_what_has_been_given_stops_being_asked_for():
    quote = {"customer": {"name": "Kestrel", "contact": "A. Reyes"},
             "terms": {"payment": "Net 30", "start": None},
             "lines": [{"id": "1", "description": "Subscription licences", "unitPrice": "200"},
                       {"id": "2", "description": "Onboarding", "unitPrice": None}]}
    assert qt.missing(TEMPLATE, quote) == [
        {"field": "lines.2.unitPrice", "label": "a price for “Onboarding”"},
        {"field": "terms.start", "label": "a start date"}]


def test_a_quote_with_no_template_is_still_asked_for_a_customer_and_prices():
    assert [m["field"] for m in qt.missing(None, {"lines": [{"id": "9", "unitPrice": 0}]})] == [
        "customer.name", "customer.contact", "lines.9.unitPrice"]
