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
    assert set(cleaned) == {"category", "lines", "termFields", "assumptionsHtml", "sections"}


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


# --- layout ------------------------------------------------------------------------

def test_a_template_with_no_layout_gets_the_standard_one():
    kinds = [x["kind"] for x in qt.clean(BODY)["sections"]]
    assert kinds == ["letterhead", "customer", "scope", "lines", "totals", "terms",
                     "assumptions", "conditions"]


def test_headings_are_the_sellers_and_a_layout_may_have_several_text_sections():
    layout = qt.clean_sections([
        {"key": "letterhead", "kind": "letterhead", "heading": "From", "logoId": 4},
        {"key": "deliverables", "kind": "text", "heading": "What you get", "html": "<p>x</p>"},
        {"key": "pricing", "kind": "lines", "heading": "Investment"},
        {"key": "timeline", "kind": "text", "heading": "Timeline"}], with_content=True)
    assert [(x["key"], x["heading"]) for x in layout] == [
        ("letterhead", "From"), ("deliverables", "What you get"), ("pricing", "Investment"),
        ("timeline", "Timeline")]
    assert layout[0]["logoId"] == 4 and layout[1]["html"] == "<p>x</p>" and layout[3]["html"] == ""


def test_a_template_layout_keeps_no_words_only_structure():
    layout = qt.clean_sections([{"key": "pricing", "kind": "lines", "heading": "Pricing"},
                                {"key": "notes", "kind": "text", "heading": "Notes",
                                 "html": "<p>last customer's notes</p>"}], with_content=False)
    assert "html" not in layout[1]


def test_what_is_written_in_a_section_is_cleaned():
    layout = qt.clean_sections([{"key": "pricing", "kind": "lines", "heading": "P"},
                                {"key": "notes", "kind": "text", "heading": "N",
                                 "html": '<p onclick="x()">ok</p><script>bad()</script>'}],
                               with_content=True)
    assert layout[1]["html"] == "<p>ok</p>"


@pytest.mark.parametrize("sections, says", [
    ([], "at least one"), ("layout", "list"),
    ([{"key": "a1", "kind": "banner", "heading": "x"}], "kind"),
    ([{"key": "a1", "kind": "text", "heading": " "}], "heading"),
    ([{"key": "A 1", "kind": "lines", "heading": "x"}], "key"),
    ([{"key": "p1", "kind": "lines", "heading": "x"}, {"key": "p1", "kind": "text", "heading": "y"}], "twice"),
    ([{"key": "p1", "kind": "lines", "heading": "x"}, {"key": "p2", "kind": "lines", "heading": "y"}], "one lines"),
    ([{"key": "t1", "kind": "text", "heading": "x"}], "needs its lines"),
    ([{"key": "p1", "kind": "lines", "heading": "x"},
      {"key": "lh", "kind": "letterhead", "heading": "x", "logoId": "7; drop"}], "logoId"),
])
def test_a_layout_that_does_not_hold_together_is_refused(sections, says):
    with pytest.raises(ValueError, match=says):
        qt.clean_sections(sections, with_content=True)


def test_an_empty_written_section_is_asked_for_by_its_heading():
    quote = {"customer": {"name": "K", "contact": "A"}, "terms": {}, "scopeHtml": "<p> </p>",
             "lines": [{"id": "1", "unitPrice": "5"}],
             "sections": [{"key": "scope", "kind": "scope", "heading": "Our understanding"},
                          {"key": "deliverables", "kind": "text", "heading": "Deliverables", "html": ""},
                          {"key": "timeline", "kind": "text", "heading": "Timeline", "html": "<p>Q1</p>"},
                          {"key": "pricing", "kind": "lines", "heading": "Pricing"}]}
    assert qt.missing(qt.BLANK, quote) == [
        {"field": "sections.scope", "label": "the “Our understanding” section"},
        {"field": "sections.deliverables", "label": "the “Deliverables” section"}]
