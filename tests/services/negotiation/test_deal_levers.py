from src.services.negotiation.deal_levers import LEVER_LABELS, read_levers

# The two scenarios from the 2026-10-15 lever audit.
SCENARIO_A = "Our rate for the 3-month haulage lane is 945 per load. We can't move on price."
SCENARIO_B = ("For the 5-year renewal we need exclusivity, CPI+2% indexation and "
              "service credits capped at 5%. Net 45 payment is fine.")


def _by_lever(text):
    return {lv["lever"]: lv for lv in read_levers(text)}


def test_price_only_reply_raises_no_levers():
    assert read_levers(SCENARIO_A) == []


def test_multi_year_renewal_lists_exactly_the_levers_the_supplier_raised():
    got = _by_lever(SCENARIO_B)
    assert set(got) == {"term", "renewal_termination", "exclusivity",
                        "indexation", "service_levels", "payment_terms"}


def test_each_figure_is_the_one_written_beside_that_lever():
    got = _by_lever(SCENARIO_B)
    assert got["term"]["figure"] == "5-year"
    assert got["indexation"]["figure"] == "CPI+2%"
    assert got["service_levels"]["figure"] == "capped at 5%"
    assert got["payment_terms"]["figure"] == "Net 45"


def test_no_figure_is_invented_when_none_is_written():
    got = _by_lever(SCENARIO_B)
    assert got["exclusivity"]["figure"] is None
    assert got["renewal_termination"]["figure"] is None


def test_phrase_and_figure_are_quoted_verbatim_from_the_reply():
    for lv in read_levers(SCENARIO_B):
        assert lv["phrase"] in SCENARIO_B
        if lv["figure"] is not None:
            assert lv["figure"] in lv["phrase"]
        assert lv["label"] == LEVER_LABELS[lv["lever"]]


def test_quoted_copy_of_our_own_email_is_not_read():
    text = ("Thanks, we can hold the price.\n\n"
            "On Tue, 14 Oct 2026 at 10:00, Buyer <b@x.com> wrote:\n"
            "> Could you offer a 24-month term and Net 15 payment?")
    assert read_levers(text) == []


def test_warranty_duration_is_warranty_not_term():
    got = _by_lever("We will include a 2-year warranty on all units.")
    assert set(got) == {"warranty"}
    assert got["warranty"]["figure"] == "2-year"


def test_lead_time_and_notice_period_figures():
    got = _by_lever("Lead time is 6 weeks from order. Either party may terminate "
                    "on 90 days' notice.")
    assert got["lead_time"]["figure"] == "6 weeks"
    assert got["renewal_termination"]["figure"] == "90 days"


def test_volume_minimum_order():
    got = _by_lever("Pricing assumes a minimum order of 500 units per quarter.")
    assert got["volume"]["figure"] == "500 units"


def test_each_lever_listed_once_in_order_of_first_mention():
    text = "Net 30 applies. We also need exclusivity. Net 30 is standard for us."
    assert [lv["lever"] for lv in read_levers(text)] == ["payment_terms", "exclusivity"]


def test_empty_or_missing_text():
    assert read_levers(None) == []
    assert read_levers("") == []
