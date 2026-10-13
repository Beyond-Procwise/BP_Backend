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
    def __init__(self, *intents, refused="", raw=None):
        self.raw = raw if raw is not None else json.dumps({"refused": refused,
                                                           "intents": list(intents)})
        self.prompts, self.kwargs = [], []

    def __call__(self, prompt, **kw):
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
    out = _turn(_Model({"op": "valid_until", "date": day.isoformat()}))
    assert out["change"] == {"validUntil": day.isoformat()}
    for bad in ((TODAY - dt.timedelta(days=1)).isoformat(), "next Tuesday", None):
        assert _turn(_Model({"op": "valid_until", "date": bad}))["change"] is None


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
