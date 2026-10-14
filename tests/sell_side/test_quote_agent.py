"""quote_drafting: it proposes and never writes, and it never states a figure it did not
read. The model is a stand-in here -- what is tested is everything around it."""
import datetime as dt
import json
from decimal import Decimal as D

import pytest

from src.services.sell_side import quote_agent as qa

TODAY = dt.date.today()
QUOTE = {
    "id": "12", "ref": "SQ-20261013-000012", "status": "draft",
    "validUntil": (TODAY + dt.timedelta(days=30)).isoformat(),
    "lines": [
        {"id": "l1", "tag": "product", "description": "Field service seats", "qty": "340",
         "cost": "125.0000", "markupPct": "60", "unitPrice": "200.0000", "discountPct": "4"},
        {"id": "l2", "tag": "service", "description": "Onboarding", "qty": "1",
         "cost": "12000", "markupPct": "50", "unitPrice": "18000", "discountPct": "0"},
        {"id": "l3", "tag": "one-off", "description": "Device provisioning", "qty": "11",
         "cost": None, "markupPct": None, "unitPrice": "950", "discountPct": "0"},
    ],
}
NO_LIMITS = {"discountCorridor": None, "marginFloorPct": None}
CORRIDOR_8 = {"discountCorridor": {"maxPct": 8.0}, "marginFloorPct": None}


class _Model:
    """Stands in for the model. ``sum`` is its answer to the separate sum turn."""

    def __init__(self, *intents, refused="", raw=None, sum=None):
        self.raw = raw if raw is not None else json.dumps({"refused": refused,
                                                           "intents": list(intents)})
        self.sum = sum
        self.prompts, self.kwargs, self.sum_prompts = [], [], []

    def __call__(self, prompt, **kw):
        if kw.get("format") == qa.SUM_SCHEMA:
            self.sum_prompts.append(prompt)
            return json.dumps({"expression": "", "label": "", "line_id": "", "into": "",
                               **(self.sum or {})})
        self.prompts.append(prompt)
        self.kwargs.append(kw)
        return self.raw


def _turn(model, message=None, quote=QUOTE, checks=NO_LIMITS, **kw):
    if message is None:
        # the instruction a seller would have typed to get these intents
        said = [i.get("percent") for i in json.loads(model.raw or "{}").get("intents", [])
                if isinstance(i, dict)] if (model.raw or "").startswith("{") else []
        message = "do it " + " ".join(f"{p}%" for p in said if p is not None)
    return qa.propose(quote, message, checks=checks, call=model,
                      prompt=lambda: "T {today} Q {quote} M {message}", **kw)


def test_a_markup_reprices_each_costed_line_from_its_own_cost_and_skips_the_rest():
    out = _turn(_Model({"op": "markup_all", "percent": 40}))
    assert out["refused"] is None
    assert out["change"] == {"lines": [
        {"id": "l1", "markupPct": D(40), "unitPrice": D("175.0000")},
        {"id": "l2", "markupPct": D(40), "unitPrice": D("16800.0000")}]}
    assert "40% markup on 2 lines" in out["summary"] and "1 with no cost is left" in out["summary"]
    assert out["reads"][0] == "quote SQ-20261013-000012"


def test_the_model_is_never_shown_a_cost_or_a_price():
    model = _Model({"op": "markup_all", "percent": 40})
    _turn(model, message="apply 40% markup to all items")
    sent = model.prompts[0]
    assert "apply 40% markup to all items" in sent and "Field service seats" in sent
    for figure in ("125", "12000", "200.0000", "18000", "950"):
        assert figure not in sent, f"{figure} went to the model"
    assert model.kwargs[0]["format"] == qa.SCHEMA and model.kwargs[0]["think"] is False


def test_the_answer_has_no_field_a_price_could_be_put_in():
    fields = set(qa.SCHEMA["properties"]["intents"]["items"]["properties"])
    assert fields == {"op", "line_id", "percent", "date", "description", "tag", "qty", "unit"}


def test_a_price_the_model_slips_in_anyway_goes_nowhere():
    out = _turn(_Model({"op": "markup_line", "line_id": "l1", "percent": 10,
                        "unitPrice": 1, "cost": 1, "unit_price": 1}))
    assert out["change"] == {"lines": [{"id": "l1", "markupPct": D(10),
                                        "unitPrice": D("137.5000")}]}


def test_a_line_with_no_cost_cannot_be_priced_from_a_markup():
    out = _turn(_Model({"op": "markup_line", "line_id": "l3", "percent": 40}))
    assert out["change"] is None and "has no cost" in out["refused"]
    assert out["summary"] == out["refused"]


def test_a_markup_on_a_quote_with_no_costs_is_refused_not_invented():
    bare = {**QUOTE, "lines": [QUOTE["lines"][2], {**QUOTE["lines"][0], "cost": "0"}]}
    out = _turn(_Model({"op": "markup_all", "percent": 40}), quote=bare)
    assert out["change"] is None and "nothing to mark up" in out["refused"]


def test_a_discount_inside_the_corridor_is_proposed_and_says_what_it_was_tested_against():
    out = _turn(_Model({"op": "discount_all", "percent": 5}), checks=CORRIDOR_8)
    assert out["change"] == {"lines": [{"id": i, "discountPct": D(5)} for i in ("l1", "l2", "l3")]}
    assert "discount corridor 8%" in out["reads"]


def test_a_discount_outside_the_corridor_is_refused():
    out = _turn(_Model({"op": "discount_line", "line_id": "l1", "percent": 12}),
                checks=CORRIDOR_8)
    assert out["change"] is None
    assert "12% discount is outside the 8% corridor" in out["refused"]


def test_with_no_corridor_set_a_discount_is_proposed_and_says_it_was_not_tested():
    out = _turn(_Model({"op": "discount_all", "percent": 30}))
    assert out["refused"] is None
    assert "discount corridor: not set, so not tested" in out["reads"]


def test_one_refused_intent_refuses_the_whole_turn():
    """Half a proposal is a proposal the seller did not ask for."""
    out = _turn(_Model({"op": "markup_all", "percent": 40},
                       {"op": "discount_all", "percent": 50}), checks=CORRIDOR_8)
    assert out["change"] is None and "outside" in out["refused"]


def test_validity_is_changed_to_a_real_date_that_has_not_passed():
    day = TODAY + dt.timedelta(days=45)
    out = _turn(_Model({"op": "valid_until", "date": day.isoformat()}), message="valid for 45 days")
    assert out["change"] == {"validUntil": day.isoformat()}
    for bad in ((TODAY - dt.timedelta(days=1)).isoformat(), "next Tuesday", None):
        assert _turn(_Model({"op": "valid_until", "date": bad}),
                     message="valid until whenever")["change"] is None


def test_a_validity_date_nobody_asked_for_is_dropped():
    """Seen against the real model: it "set" the date the quote already had."""
    day = (TODAY + dt.timedelta(days=30)).isoformat()
    out = _turn(_Model({"op": "markup_all", "percent": 40}, {"op": "valid_until", "date": day}),
                message="apply 40% markup")
    assert "validUntil" not in out["change"] and "valid" not in out["summary"].lower()


def test_a_new_line_arrives_with_no_price_and_no_cost():
    out = _turn(_Model({"op": "add_line", "description": "Training day", "qty": 2,
                        "unit": "day", "tag": "service"}))
    assert out["change"] == {"addLines": [{
        "tag": "service", "description": "Training day", "qty": D(2), "unit": "day",
        "cost": None, "markupPct": None, "unitPrice": D(0), "discountPct": D(0)}]}
    assert "no price yet" in out["summary"]


@pytest.mark.parametrize("intent", [
    {"op": "markup_all", "percent": -5}, {"op": "markup_all"}, {"op": "markup_all", "percent": "lots"},
    {"op": "discount_all", "percent": 101}, {"op": "discount_line", "line_id": "nope", "percent": 5},
    {"op": "set_price", "percent": 5}, {"op": "add_line", "description": "  "},
])
def test_an_intent_that_does_not_hold_together_is_refused(intent):
    out = _turn(_Model(intent))
    assert out["change"] is None and out["refused"]


def test_when_the_model_names_no_change_the_turn_says_so_in_our_words_not_its_own():
    out = _turn(_Model(refused="Sure! I set every price to £99 for you."))
    assert out["change"] is None and out["refused"] == out["summary"]
    assert "99" not in out["summary"]


def test_a_quote_that_is_not_a_draft_gets_no_proposal_and_no_model_call():
    model = _Model({"op": "markup_all", "percent": 40})
    out = _turn(model, quote={**QUOTE, "status": "approved"})
    assert out["change"] is None and "no longer a draft" in out["refused"]
    assert model.prompts == []


@pytest.mark.parametrize("raw", ["", None, "not json", "[1,2]", '{"intents": "all of them"}',
                                 '{"intents": ["markup"]}'])
def test_a_model_that_is_silent_or_unreadable_is_unavailable_not_a_guess(raw):
    model = _Model()
    model.raw = raw
    with pytest.raises(qa.AgentUnavailable):
        _turn(model)


def test_a_busy_model_is_not_called():
    model = _Model({"op": "markup_all", "percent": 40})
    with pytest.raises(qa.AgentUnavailable, match="busy"):
        _turn(model, busy=lambda: True)
    assert model.prompts == []


def test_an_empty_or_enormous_instruction_is_refused_before_anything_runs():
    for message in ("  ", "x" * (qa.MAX_MESSAGE + 1)):
        with pytest.raises(ValueError):
            _turn(_Model(), message=message)


def test_text_in_the_instruction_cannot_pull_the_quote_into_itself():
    """The prompt is filled in one pass: '{quote}' typed by the seller stays as typed."""
    model = _Model()
    _turn(model, message="please {quote} and {today}")
    assert model.prompts[0].endswith("M please {quote} and {today}")


def test_the_session_is_the_quotes_own_until_the_caller_carries_one():
    assert _turn(_Model({"op": "markup_all", "percent": 1}))["session_id"] == "12"
    assert _turn(_Model({"op": "markup_all", "percent": 1}),
                 session_id="s-9")["session_id"] == "s-9"


@pytest.mark.parametrize("message, found", [
    ("apply 40% markup to all items", [D(40)]), ("12.5 % off, then 3 percent", [D("12.5"), D(3)]),
    ("5 per cent", [D(5)]), ("set the seat price to £150", []), ("340 seats at 200", []),
    ("valid for 60 days", []),
])
def test_only_a_number_typed_as_a_percentage_is_a_percentage(message, found):
    assert qa.stated_percents(message) == found


def test_an_amount_of_money_is_not_turned_into_a_markup():
    """Seen against the real model: "set the seat price to £150" came back as a 150%
    markup. The seller typed no percentage, so none is proposed."""
    out = _turn(_Model({"op": "markup_line", "line_id": "l1", "percent": 150}),
                message="set the seat price to £150")
    assert out["change"] is None and "will not guess" in out["refused"]


def test_a_percentage_the_seller_did_not_type_is_not_proposed():
    out = _turn(_Model({"op": "discount_all", "percent": 7}), message="give them 5% off")
    assert out["change"] is None and out["refused"]


# --- filling a template ------------------------------------------------------------

TEMPLATE = {"key": "software_subscription", "name": "Software subscription",
            "description": "Licences for a period.", "termFields": ["payment", "start", "lengthMonths"],
            "lines": []}
BLANK = {"id": "30", "ref": "SQ-30", "status": "draft", "customer": {}, "terms": {},
         "validUntil": (TODAY + dt.timedelta(days=30)).isoformat(),
         "lines": [{"id": "s", "tag": "product", "description": "Subscription licences",
                    "qty": "1", "cost": None, "unitPrice": "0", "discountPct": "0"},
                   {"id": "o", "tag": "service", "description": "Onboarding and set-up",
                    "qty": "1", "cost": None, "unitPrice": "0", "discountPct": "0"}]}
SAID = "For Kestrel Freight (A. Reyes). 340 seats at £200, cost 125. Onboarding 18k. Net 30, 12 months, start 1 November."


def _fill(model, message=SAID, quote=BLANK, checks=NO_LIMITS):
    return qa.propose(quote, message, checks=checks, call=model, template=TEMPLATE,
                      prompt=lambda: "{template} {quote} {message}",
                      sum_prompt=lambda: "SUM {quote} {message}")


def test_a_template_is_filled_from_the_sellers_words_and_says_what_is_left():
    out = _fill(_Model(
        {"op": "set_customer", "name": "Kestrel Freight", "contact": "A. Reyes"},
        {"op": "set_line", "line_id": "s", "qty": 340, "unit_price": 200, "cost": 125},
        {"op": "set_line", "line_id": "o", "unit_price": 18000},
        {"op": "set_term", "field": "payment", "text": "net 30"},
        {"op": "set_term", "field": "lengthMonths", "number": 12}))
    assert out["refused"] is None
    assert out["change"] == {
        "customer": {"name": "Kestrel Freight", "contact": "A. Reyes"},
        "terms": {"payment": "Net 30", "lengthMonths": 12},
        "lines": [{"id": "s", "qty": D(340), "unitPrice": D(200), "cost": D(125),
                   "markupPct": D("60.0")},
                  {"id": "o", "unitPrice": D(18000)}]}
    assert out["missing"] == [{"field": "terms.start", "label": "a start date"}]
    assert out["summary"].endswith("Still needed: a start date.")
    assert "the “Software subscription” template" in out["reads"]


def test_the_template_turn_uses_the_wider_answer_and_still_sends_no_stored_figure():
    model = _Model({"op": "set_line", "line_id": "s", "qty": 340})
    priced = {**BLANK, "lines": [{**BLANK["lines"][0], "cost": "77.7700", "unitPrice": "999.9900"}]}
    _fill(model, quote=priced)
    assert model.kwargs[0]["format"] == qa.FILL_SCHEMA
    assert "Software subscription" in model.prompts[0]
    assert "77.77" not in model.prompts[0] and "999.99" not in model.prompts[0]


@pytest.mark.parametrize("intent, says", [
    ({"op": "set_line", "line_id": "s", "unit_price": 210}, "a price of 210"),
    ({"op": "set_line", "line_id": "s", "qty": 350}, "a quantity of 350"),
    ({"op": "set_line", "line_id": "o", "cost": 9000}, "a cost of 9000"),
    ({"op": "set_line", "line_id": "s", "percent": 5}, "a discount of 5%"),
    ({"op": "set_customer", "name": "Kestrel Logistics"}, "Kestrel Logistics"),
    ({"op": "set_customer", "contact": "B. Smith"}, "B. Smith"),
    ({"op": "set_term", "field": "payment", "text": "Net 60"}, "Net 60"),
    ({"op": "set_term", "field": "lengthMonths", "number": 24}, "a length in months of 24"),
    ({"op": "set_term", "field": "lengthMonths", "number": 1.5}, "a length in months"),
    ({"op": "set_term", "field": "warranty", "text": "Net 30"}, "a term this quote does not have"),
    ({"op": "set_line", "line_id": "zz", "qty": 340}, "a line I could not match"),
])
def test_a_value_the_seller_did_not_give_is_left_out_and_named(intent, says):
    """The rest of what they gave is still filled in; the invented value never is."""
    out = _fill(_Model({"op": "set_line", "line_id": "o", "unit_price": 18000}, intent),
                message=SAID + " 1.5")
    assert out["change"] == {"lines": [{"id": "o", "unitPrice": D(18000)}]}
    assert "I left out" in out["summary"] and says in out["summary"]


def test_a_turn_of_nothing_but_invented_values_proposes_nothing():
    """Seen against the real model: asked to "fill in sensible prices", it offered 500."""
    out = _fill(_Model({"op": "set_line", "line_id": "s", "unit_price": 500},
                       {"op": "set_line", "line_id": "o", "unit_price": 1500}),
                message="Just make it competitive and fill in sensible prices.")
    assert out["change"] is None
    assert "a price of 500" in out["refused"] and "a price of 1500" in out["refused"]


def test_a_start_date_is_read_from_the_sellers_words_not_the_models():
    """Seen against the real model: it answered "1 November" where a date was asked for."""
    year = TODAY.year + (TODAY > dt.date(TODAY.year, 11, 1))
    for said in ("1 November", f"{year}-11-01", "2031-02-30", None):
        out = _fill(_Model({"op": "set_term", "field": "start", "date": said}))
        assert out["change"] == {"terms": {"start": f"{year}-11-01"}}, said
    out = _fill(_Model({"op": "set_term", "field": "start", "date": "2027-03-09"}),
                message="start some time in spring")
    assert out["change"] is None and "a start date" in out["refused"]


def test_between_two_typed_dates_the_models_choice_must_be_one_of_them():
    said = "starts 1 November 2030, first review 1 December 2030"
    out = _fill(_Model({"op": "set_term", "field": "start", "date": "2030-11-01"}), message=said)
    assert out["change"] == {"terms": {"start": "2030-11-01"}}
    out = _fill(_Model({"op": "set_term", "field": "start", "date": "2030-10-01"}), message=said)
    assert out["change"] is None


@pytest.mark.parametrize("message, found", [
    ("1 November 2030", [dt.date(2030, 11, 1)]), ("3rd of Jan 2031", [dt.date(2031, 1, 3)]),
    ("2030-11-01", [dt.date(2030, 11, 1)]), ("31 February 2030", []), ("12 months", []),
    ("May 2030", []),
])
def test_only_a_day_and_a_month_is_a_date(message, found):
    assert qa.typed_dates(message) == found


def test_a_zero_from_the_model_means_it_had_nothing_to_give():
    out = _fill(_Model({"op": "set_line", "line_id": "s", "qty": 340, "unit_price": 0, "cost": 0}))
    assert out["change"] == {"lines": [{"id": "s", "qty": D(340)}]}
    assert "left out" not in out["summary"]


def test_the_sellers_own_spelling_is_what_is_kept():
    out = _fill(_Model({"op": "set_customer", "name": "KESTREL   FREIGHT"}))
    assert out["change"] == {"customer": {"name": "Kestrel Freight"}}


def test_a_filled_discount_is_still_tested_against_the_corridor():
    out = _fill(_Model({"op": "set_line", "line_id": "s", "percent": 15}),
                message="340 seats with 15% off", checks=CORRIDOR_8)
    assert out["change"] is None and "outside the 8% corridor" in out["refused"]


def test_a_price_on_a_line_that_already_has_a_cost_gets_the_markup_the_two_imply():
    costed = {**BLANK, "lines": [{**BLANK["lines"][0], "cost": "100"}]}
    out = _fill(_Model({"op": "set_line", "line_id": "s", "unit_price": 140}),
                message="seats are 140 each", quote=costed)
    assert out["change"]["lines"] == [{"id": "s", "unitPrice": D(140), "markupPct": D("40.0")}]


def test_without_a_template_nothing_is_filled_in():
    """A quote that was not started from a template keeps the narrow agent: it is given
    the narrow answer to fill, and a fill it sends anyway is refused."""
    model = _Model({"op": "set_line", "line_id": "l1", "unit_price": 200})
    out = _turn(model, message="seats at 200")
    assert model.kwargs[0]["format"] == qa.SCHEMA
    assert out["change"] is None and "not something I can change" in out["refused"]


def test_a_turn_that_fills_nothing_says_so_and_still_lists_what_is_missing():
    out = _fill(_Model({"op": "set_line", "line_id": "s"}), message="hello")
    assert out["change"] is None and "Nothing in that" in out["refused"]
    assert [m["field"] for m in out["missing"]][:2] == ["customer.name", "customer.contact"]


@pytest.mark.parametrize("message, found", [
    ("340 seats at £200", [D(340), D(200)]), ("18k and 1.5m", [D(18000), D(18), D(1500000), D("1.5")]),
    ("£18,000.50", [D("18000.50")]), ("no figures here", []), ("SLA-B v2", []),
    ("two days of training", [D(2)]), ("someone to attend", []),
])
def test_only_a_number_the_seller_typed_is_a_number(message, found):
    assert qa.stated_numbers(message) == found


def test_a_cost_is_taken_only_when_the_seller_was_talking_about_cost():
    """Seen against the real model: "Monthly fee is 4,200" came back as price AND cost."""
    out = _fill(_Model({"op": "set_line", "line_id": "s", "unit_price": 4200, "cost": 4200}),
                message="Monthly fee is 4,200")
    assert out["change"]["lines"] == [{"id": "s", "unitPrice": D(4200)}]
    out = _fill(_Model({"op": "set_line", "line_id": "s", "unit_price": 200, "cost": 125}),
                message="200 each, they cost us 125")
    assert out["change"]["lines"][0]["cost"] == D(125)


def test_a_cost_with_no_price_yet_does_not_get_a_markup_of_minus_a_hundred():
    out = _fill(_Model({"op": "set_line", "line_id": "s", "cost": 125}), message="they cost us 125")
    assert out["change"]["lines"] == [{"id": "s", "cost": D(125)}]


def test_a_discount_is_not_filed_as_a_term():
    """Seen against the real model: "5% off" came back as the payment terms."""
    out = _fill(_Model({"op": "set_line", "line_id": "s", "qty": 200},
                       {"op": "set_term", "field": "payment", "text": "5% off"}),
                message="200 units with 5% off")
    assert out["change"] == {"lines": [{"id": "s", "qty": D(200)}]}
    assert "reads as a discount" in out["summary"]


# --- sections ----------------------------------------------------------------------

LAID_OUT = {**BLANK, "scopeHtml": "", "sections": [
    {"key": "letterhead", "kind": "letterhead", "heading": "From", "logoId": None},
    {"key": "scope", "kind": "scope", "heading": "Our understanding"},
    {"key": "deliverables", "kind": "text", "heading": "Deliverables", "html": ""},
    {"key": "pricing", "kind": "lines", "heading": "Investment"},
    {"key": "terms", "kind": "terms", "heading": "Terms"}]}
WORDS = "Deliverables are a configured platform & two training days. Scope: 340 seats across 11 sites."


def test_sections_are_recognised_by_the_heading_the_page_shows():
    found = qa.section_words(
        "The deliverables are a configured platform and two training days. Our understanding "
        "is that Kestrel run 11 depots. What does that come to?", LAID_OUT["sections"])
    assert found == [
        {"op": "set_section", "section_key": "deliverables",
         "text": "a configured platform and two training days."},
        {"op": "set_section", "section_key": "scope", "text": "Kestrel run 11 depots."}]


@pytest.mark.parametrize("message, key, words", [
    ("Deliverables: platform; training", "deliverables", "platform; training"),
    ("DELIVERABLES - a platform", "deliverables", "a platform"),
    ("340 seats. Scope: eleven sites.", "scope", "eleven sites."),
    ("Deliverables will be a platform.\nAnything else?", "deliverables", "a platform."),
])
def test_ways_a_seller_names_a_section(message, key, words):
    assert qa.section_words(message, LAID_OUT["sections"]) == [
        {"op": "set_section", "section_key": key, "text": words}]


@pytest.mark.parametrize("message", [
    "340 seats at 200", "We discussed the deliverables are late", "Pricing: 200 each",
    "Investment is high", "Terms: Net 30", "Deliverables:", "Deliverables: ?",
])
def test_what_is_not_a_written_section_is_not_filed_as_one(message):
    """Mid-sentence mentions, the pricing and terms sections, and a heading with nothing
    after it."""
    assert qa.section_words(message, LAID_OUT["sections"]) == []


def test_the_sellers_words_are_filed_under_the_section_they_belong_to():
    out = _fill(_Model(), message=WORDS, quote=LAID_OUT)
    assert out["change"] == {
        "sections": [{"key": "deliverables",
                      "html": "<p>a configured platform &amp; two training days.</p>"}],
        "scopeHtml": "<p>340 seats across 11 sites.</p>"}
    assert "Put your words under “Deliverables”." in out["summary"]
    assert "Put your words under “Our understanding”." in out["summary"]
    left = [m["field"] for m in out["missing"]]
    assert "sections.deliverables" not in left and "sections.scope" not in left


def test_the_model_does_not_get_to_write_a_section():
    """Whatever it offers for one is dropped: a section holds the seller's words, found in
    their message by code, or nothing."""
    out = _fill(_Model({"op": "set_section", "section_key": "deliverables",
                        "text": "A best-in-class platform delivering transformational value"}),
                message="Write a compelling deliverables section for me.", quote=LAID_OUT)
    assert out["change"] is None and out["refused"]


def test_markup_in_the_sellers_words_is_text_not_html():
    out = _fill(_Model(), message="deliverables: <script>x()</script> works", quote=LAID_OUT)
    assert out["change"]["sections"][0]["html"] == "<p>&lt;script&gt;x()&lt;/script&gt; works</p>"


# --- calculations ------------------------------------------------------------------

PRICED = {**BLANK, "lines": [
    {"id": "s", "tag": "product", "description": "Seats", "qty": "340", "cost": "125",
     "unitPrice": "200", "discountPct": "4"},
    {"id": "o", "tag": "service", "description": "Onboarding", "qty": "1", "cost": "12000",
     "unitPrice": "18000", "discountPct": "0"},
    {"id": "d", "tag": "one-off", "description": "Devices", "qty": "11", "cost": None,
     "unitPrice": "950", "discountPct": "0"}]}


def test_a_total_is_worked_out_in_code_and_shown_with_its_working():
    out = _fill(_Model(), message="what's the total?", quote=PRICED)
    assert out["refused"] is None and out["change"] is None, "a sum changes nothing"
    assert out["calculations"] == [{
        "label": "Quote total, before tax",
        "working": "65,280.00 + 18,000.00 + 10,450.00 = 93,730.00", "result": D("93730.00")}]
    assert "Check these figures" in out["summary"]


def test_margin_is_over_the_costed_lines_and_says_what_it_left_out():
    out = _fill(_Model(), message="what margin are we making?", quote=PRICED)
    calc = out["calculations"][0]
    assert calc["result"] == D("28780.00") and "1 with no cost left out" in calc["label"]
    assert calc["working"] == "83,280.00 − 54,500.00 = 28,780.00, which is 34.6% of that revenue"


def test_a_margin_with_no_costs_is_not_invented():
    out = _fill(_Model(), message="margin?", quote=BLANK)
    assert out["calculations"] == [] and "no line has a cost" in out["refused"]


def test_tax_needs_a_rate_the_seller_gave():
    out = _fill(_Model(), message="total with 20% VAT", quote=PRICED)
    assert out["calculations"][0]["working"] == "93,730.00 + 18,746.00 tax = 112,476.00"
    out = _fill(_Model({"op": "calculate", "calc": "with_tax", "percent": 20}),
                message="total with VAT", quote=PRICED)
    assert out["calculations"] == [] and "give the tax rate" in out["refused"]


def test_a_total_asked_for_in_the_same_breath_includes_what_was_just_filled_in():
    out = _fill(_Model({"op": "set_line", "line_id": "s", "qty": 340, "unit_price": 200}),
                message="340 seats at 200, what does that come to?")
    assert out["change"]["lines"] == [{"id": "s", "qty": D(340), "unitPrice": D(200)}]
    assert out["calculations"][0]["result"] == D("68000.00")
    assert "Check these figures" in out["summary"]


@pytest.mark.parametrize("expression, said, working, result", [
    ("12000 / 340", "spread 12000 over 340 seats", "12000 ÷ 340 = 35.2941", D("35.2941")),
    ("200 * (1 - 4%)", "200 less 4%", "200 × (1 - 4%) = 192", D("192.0000")),
    ("4,200 x 24", "4,200 a month for 24 months", "4200 × 24 = 100800", D("100800.0000")),
    ("18k + 2 * 950", "18k plus two days at 950", "18000 + 2 × 950 = 19900", D("19900.0000")),
    ("-(5 - 20)", "20 minus 5", "- (5 - 20) = 15", D("15.0000")),
])
def test_a_sum_on_the_sellers_own_numbers(expression, said, working, result):
    out = _fill(_Model(sum={"expression": expression, "label": "Worked out"}),
                message="work out " + said)
    assert out["calculations"] == [{"label": "Worked out", "working": working, "result": result}]


@pytest.mark.parametrize("expression, says", [
    ("12000 / 350", "a sum using 350, which you did not give me"),
    ("__import__('os').system('id')", "a sum I could not read"),
    ("12000 ** 340", "a sum I could not read"),
    ("12000 / (340 - 340)", "a sum I could not read"),
    ("(12000 / 340", "a sum I could not read"),
    ("price * 2", "a sum I could not read"),
    ("", "a sum I could not read"),
])
def test_a_sum_that_is_not_plain_arithmetic_on_typed_numbers_is_left_out(expression, says):
    out = _fill(_Model(sum={"expression": expression}), message="spread 12000 over 340 seats")
    assert out["calculations"] == [] and out["change"] is None
    assert (says in out["refused"]) if expression else out["refused"]


def test_a_result_is_put_on_a_line_only_as_a_proposal_marked_as_worked_out():
    out = _fill(_Model(sum={"expression": "12000 / 340", "label": "Price per seat",
                            "line_id": "s", "into": "unit_price"}),
                message="spread 12000 over 340 seats and use that as the seat price")
    assert out["change"] == {"lines": [{"id": "s", "unitPrice": D("35.2941")}]}
    assert out["calculations"][0]["into"] == {"lineId": "s", "field": "unitPrice"}
    assert "Price per seat: 12000 ÷ 340 = 35.2941." in out["summary"]
    assert "Check these figures" in out["summary"]


def test_a_worked_out_cost_still_needs_the_seller_to_be_talking_about_cost():
    answer = {"expression": "12000 / 340", "line_id": "s", "into": "cost"}
    assert _fill(_Model(sum=answer), message="work out 12000 over 340")["change"] is None
    out = _fill(_Model(sum=answer), message="it costs us 12000 for 340, work out the cost per seat")
    assert out["change"] == {"lines": [{"id": "s", "cost": D("35.2941")}]}


def test_there_is_no_calculator_without_the_wider_answer():
    out = _turn(_Model({"op": "calculate", "calc": "quote_total"}), message="total?")
    assert out["change"] is None and "calculations" in out and out["calculations"] == []


def test_the_quotes_own_sums_are_asked_for_in_words_not_offered_by_the_model():
    """Seen against the real model: it volunteered a total, a margin and a tax figure on
    a message that asked for none."""
    out = _fill(_Model({"op": "calculate", "calc": "quote_total"},
                       {"op": "calculate", "calc": "margin"},
                       {"op": "calculate", "calc": "with_tax", "percent": 20},
                       {"op": "set_customer", "name": "Kestrel Freight"}),
                message="It's for Kestrel Freight.", quote=PRICED)
    assert out["calculations"] == [] and out["change"] == {"customer": {"name": "Kestrel Freight"}}


@pytest.mark.parametrize("message, asked", [
    ("what's the total?", ["quote_total"]), ("what does that come to", ["quote_total"]),
    ("total with 20% VAT", ["with_tax"]), ("what margin are we making, and the total?", ["quote_total", "margin"]),
    ("give them a 40% margin", []), ("340 seats at 200", []),
])
def test_which_sums_a_message_asks_for(message, asked):
    assert [i["calc"] for i in qa.asked_sums(message)] == asked


def test_a_price_to_be_worked_out_is_not_also_filed_as_it_stands():
    """Seen against the real model: "a budget of 12000 across 340 seats, work out the price
    per seat" came back as 340 seats at 12,000 each."""
    said = "We've got a budget of 12000 for the licences across 340 seats, work out the price per seat and use it."
    out = _fill(_Model({"op": "set_line", "line_id": "s", "qty": 340, "unit_price": 12000}), message=said)
    assert out["change"] == {"lines": [{"id": "s", "qty": D(340)}]}
    assert "asked for it to be worked out" in out["summary"]
    out = _fill(_Model({"op": "set_line", "line_id": "s", "qty": 340},
                       sum={"expression": "12000 / 340", "label": "Price per seat",
                            "line_id": "s", "into": "unit_price"}), message=said)
    assert out["change"] == {"lines": [{"id": "s", "qty": D(340), "unitPrice": D("35.2941")}]}


def test_the_sum_turn_is_only_taken_when_a_sum_is_asked_for_and_there_are_numbers_for_it():
    for message in ("340 seats at 200", "what's the total?", "work out a fair price", "hello"):
        model = _Model(sum={"expression": "340 * 200"})
        out = _fill(model, message=message, quote=PRICED)
        assert model.sum_prompts == [], message
        assert not any("340 × 200" in c["working"] for c in out["calculations"])
    model = _Model(sum={"expression": "340 * 200", "label": "Seats"})
    out = _fill(model, message="what is 340 seats at 200?")
    assert len(model.sum_prompts) == 1 and "999" not in model.sum_prompts[0]
    assert out["calculations"][-1]["working"] == "340 × 200 = 68000"


def test_the_model_cannot_slip_a_sum_into_the_fill_turn():
    out = _fill(_Model({"op": "calculate", "calc": "expression", "expression": "340 * 200",
                        "line_id": "o", "into": "unit_price"}), message="340 seats at 200")
    assert out["change"] is None and out["calculations"] == []


def test_a_sentence_of_prose_is_not_a_contact_or_a_term():
    """Seen against the real model: the scope came back as the contact, and half the
    timeline as the service level."""
    said = ("Our understanding is that Kestrel run 11 depots and want one system for all of "
            "them. Timeline: kick-off in January, go-live by the end of March. It's for Kestrel Freight.")
    out = _fill(_Model(
        {"op": "set_customer", "name": "Kestrel Freight",
         "contact": "Kestrel run 11 depots and want one system for all of them"},
        {"op": "set_term", "field": "sla", "text": "go-live by the end of March"}),
        message=said, quote={**LAID_OUT, "sections": LAID_OUT["sections"] + [
            {"key": "timeline", "kind": "text", "heading": "Timeline", "html": ""}]})
    assert out["change"]["customer"] == {"name": "Kestrel Freight"}
    assert "terms" not in out["change"]
    assert out["summary"].count("reads as part of a section") == 2


def test_a_figure_inside_a_written_section_is_prose_not_a_quantity():
    """Seen against the real model: "Kestrel run 11 depots" in the scope came back as 11
    licences, and "Kestrel" as the contact."""
    said = "Our understanding is that Kestrel run 11 depots. It's for Kestrel Freight."
    out = _fill(_Model({"op": "set_line", "line_id": "s", "qty": 11},
                       {"op": "set_customer", "name": "Kestrel Freight", "contact": "Kestrel"}),
                message=said, quote=LAID_OUT)
    assert "lines" not in out["change"]
    assert out["change"]["customer"] == {"name": "Kestrel Freight"}
    # the same figure typed OUTSIDE a section is the seller's instruction
    out = _fill(_Model({"op": "set_line", "line_id": "s", "qty": 11}),
                message="11 licences. Our understanding is that Kestrel run 11 depots.", quote=LAID_OUT)
    assert out["change"]["lines"] == [{"id": "s", "qty": D(11)}]


def test_a_long_list_of_what_was_left_out_is_cut_short():
    out = _fill(_Model(*[{"op": "set_line", "line_id": "s", "unit_price": 1000 + n} for n in range(7)]),
                message="fill in sensible prices")
    assert "a price of 1003; 3 more:" in out["refused"]


def test_the_model_cannot_file_the_sellers_words_under_a_section_they_did_not_name():
    """Even words that ARE in the message: which section they belong under is read from
    the heading the seller used, and they used none."""
    out = _fill(_Model({"op": "set_section", "section_key": "deliverables",
                        "text": "340 seats at 200"}),
                message="340 seats at 200", quote=LAID_OUT)
    assert out["change"] is None


def test_only_a_section_the_seller_writes_can_be_given_words():
    change, did, _reads, _sums = qa._work_out(
        [{"op": "set_section", "section_key": "pricing", "text": "340 seats"}],
        LAID_OUT, NO_LIMITS, "pricing: 340 seats", filling=True)
    assert change == {} and "could not match to a section" in did[0]


def test_a_tax_rate_has_to_be_a_rate():
    out = _fill(_Model(), message="total with 150% VAT", quote=PRICED)
    assert out["calculations"] == [] and "give the tax rate" in out["refused"]
