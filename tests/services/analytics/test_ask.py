"""The seam: an analytic question leaves the ask path and comes back answered.

Everything before this file is a layer nothing called. This is where the four
of them meet a real question — "what are the top 10 suppliers by spend?" — and
where the decision is made that this particular question is not a search.

Three things are decided here and asserted:

  * **Which questions leave the old path.** Only a supplier spend ranking, and
    only when the flag is on. Everything else — a count, a policy question, a
    spend total with no ranking in it — goes down the path it always did.
  * **What comes back.** The rendered answer, no invented follow-up questions,
    and the next steps as structured data rather than as sentences to re-ask.
  * **That a failure here costs the reader nothing.** A database that will not
    answer, or a fetch that raises, falls back to the old path rather than
    turning a working ask bar into an error.
"""

from datetime import date, datetime, timezone
from decimal import Decimal
import json

import pytest

from src.services.analytics.ask import (
    SpendData,
    analytic_answer,
    is_supplier_spend_ranking,
    requested_top_n,
)
from src.services.analytics.currency import DisplayCurrency
from src.services.analytics.next_steps import ALL_DATA
from src.services.analytics.supplier_spend import SupplierSpendRow

TODAY = date(2026, 9, 7)
RATES = {"USD": Decimal("1.0"), "GBP": Decimal("0.739722"), "INR": Decimal("94.547339")}
FETCHED = datetime(2026, 9, 7, 9, 53, tzinfo=timezone.utc)


def _row(name, amount, currency="GBP"):
    return SupplierSpendRow(supplier_id=name.lower().replace(" ", "-"), supplier_name=name,
                            currency=currency, amount=Decimal(str(amount)), invoices=1)


ROWS = [_row("Kestrel Supplies 8", "1200000"), _row("Blackwood Group 10", "900000"),
        _row("Harbourline Trading 13", "400000")]


def _fetch(period, prior):
    return SpendData(rows=ROWS, prior_rows=[_row("Kestrel Supplies 8", "1000000")],
                     supplier_count=3, invoice_count=120, available_data=ALL_DATA)


def _display(target="GBP"):
    return DisplayCurrency(target=target, rates=RATES, fetched_at=FETCHED)


def _answer(query="what are the top 10 suppliers by spend?", **kw):
    params = dict(enabled=True, fetch=_fetch, writer=lambda prompt: None, today=TODAY,
                  display=_display(), audit=lambda **f: None)
    params.update(kw)
    return analytic_answer(query, **params)


class TestWhichQuestionsLeaveTheOldPath:
    @pytest.mark.parametrize("query", [
        "what are the top 10 suppliers by spend?",
        "Top 5 vendors by invoiced spend",
        "who are our biggest suppliers?",
        "show me the largest suppliers by spend this year",
        "rank our suppliers by spend",
    ])
    def test_a_supplier_ranking_is_answered_here(self, query):
        assert is_supplier_spend_ranking(query) is True

    @pytest.mark.parametrize("query", [
        "how many suppliers do we have?",          # a count, not a ranking
        "what did we spend last quarter?",         # a total, not a ranking
        "which invoices have discrepancies?",
        "what is our payment terms policy?",
        "top 10 invoices by value",                # ranked, but not suppliers
        "",
    ])
    def test_everything_else_stays_where_it_was(self, query):
        assert is_supplier_spend_ranking(query) is False

    def test_the_reader_gets_the_number_of_rows_they_asked_for(self):
        assert requested_top_n("top 5 suppliers by spend") == 5
        assert requested_top_n("what are the top 10 suppliers by spend?") == 10

    def test_a_ranking_with_no_number_in_it_gets_the_default(self):
        assert requested_top_n("who are our biggest suppliers?") == 10

    def test_an_absurd_number_is_not_taken_literally(self):
        # "top 5000 suppliers" is not a table anyone reads, and it is one query
        # away from a page that never renders.
        assert requested_top_n("top 5000 suppliers by spend") == 50


class TestTheFlag:
    def test_nothing_leaves_the_old_path_while_the_flag_is_off(self):
        assert _answer(enabled=False) is None

    def test_a_question_this_layer_cannot_answer_is_not_taken(self):
        assert _answer(query="what is our payment terms policy?") is None


class TestWhatComesBack:
    def test_the_answer_is_the_rendered_analytic_answer(self):
        payload = _answer()
        assert payload["answer"].startswith("<section")
        assert 'class="agent-answer"' in payload["answer"]
        assert "Invoiced spend · FY26 YTD" in payload["answer"]

    def test_no_follow_up_questions_are_invented(self):
        # The old path asked a model for three questions and put them under
        # every answer. Next steps replace them, and an empty list is the
        # honest value while the client still renders the old chips.
        assert _answer()["follow_ups"] == []

    def test_the_next_steps_travel_as_data_a_click_can_dispatch(self):
        steps = _answer()["next_steps"]
        assert steps and len(steps) <= 2
        assert steps[0]["action_id"].startswith("analytic.")
        assert steps[0]["entity_refs"]

    def test_the_whole_answer_travels_so_the_client_can_read_the_figures(self):
        payload = _answer()
        assert payload["analytic_answer"]["scope"]["currency"] == "GBP"
        assert payload["analytic_answer"]["table"]["rows"][0]["supplier"] == "Kestrel Supplies 8"

    def test_the_shape_is_the_one_the_ask_endpoint_already_returns(self):
        assert set(_answer()) >= {"answer", "follow_ups", "retrieved_documents"}

    def test_the_reader_sees_the_currency_they_chose(self):
        payload = _answer(display=_display("USD"))
        assert payload["analytic_answer"]["scope"]["currency"] == "USD"
        assert "$" in payload["answer"]

    def test_last_year_is_compared_when_the_data_reaches_back(self):
        # The fetch supplies a prior-year row for Kestrel, so its column and the
        # delta are part of the answer rather than a promise for later.
        assert "vs FY25 YTD" in _answer()["answer"]


class TestAChipThatDispatches:
    """A next step arrives back as an action id, never as a sentence to re-parse.

    "Review supplier concentration (35.4%)" is a label, not a question. Sent
    back as text it would have to be guessed at; sent back as
    analytic.supplier_concentration it is the answer the chip promised.
    """

    def test_an_action_is_answered_even_though_its_label_is_not_a_question(self):
        payload = _answer(query="Review supplier concentration (35.4%)",
                          action_id="analytic.supplier_concentration")
        assert payload is not None
        assert payload["analytic_answer"]["provenance"]["query_ref"] == "supplier_concentration/v1"

    def test_concentration_shows_the_running_share(self):
        payload = _answer(action_id="analytic.supplier_concentration")
        columns = [c["key"] for c in payload["analytic_answer"]["table"]["columns"]]
        assert "cumulative" in columns

    def test_the_trend_action_orders_by_movement(self):
        payload = _answer(action_id="analytic.supplier_spend_trend")
        rows = payload["analytic_answer"]["table"]["rows"]
        # Kestrel is the biggest supplier and grew 20%; Blackwood and
        # Harbourline have no prior spend at all in this fixture.
        assert rows[0]["supplier"] == "Kestrel Supplies 8"
        assert payload["analytic_answer"]["provenance"]["query_ref"] == "supplier_spend_trend/v1"

    def test_an_action_this_layer_does_not_serve_is_declined(self):
        # Better the old path than a chip that quietly answers a different
        # question from the one it offered.
        assert _answer(action_id="analytic.spend_composition") is None

    def test_a_rung_this_layer_cannot_answer_yet_is_not_offered(self):
        # Above the trend rung the ladder reaches composition, contract
        # coverage, relationship owner and risk exposure. None of them has an
        # answer behind it, and a chip that dispatches into nothing is the dead
        # end this whole mechanism exists to avoid.
        assert _answer(action_id="analytic.supplier_spend_trend")["next_steps"] == []

    def test_the_next_step_from_here_is_the_next_rung_not_this_one(self):
        payload = _answer(action_id="analytic.supplier_concentration")
        assert "analytic.supplier_concentration" not in [
            step["action_id"] for step in payload["next_steps"]]


class TestTheHeadline:
    def test_a_grounded_sentence_from_the_writer_becomes_the_headline(self):
        written = "Kestrel Supplies 8 holds 48.0% of invoiced spend."
        payload = _answer(writer=lambda prompt: json.dumps({"text": written}))
        assert written in payload["answer"]

    def test_a_writer_that_says_nothing_leaves_the_templated_headline(self):
        payload = _answer(writer=lambda prompt: None)
        assert "Kestrel Supplies 8 is the largest supplier" in payload["answer"]


class TestWhenSomethingBreaks:
    def test_a_database_that_will_not_answer_falls_back_to_the_old_path(self):
        def _raises(period, prior):
            raise RuntimeError("bp_invoice_trgt is unreachable")

        assert _answer(fetch=_raises) is None

    def test_a_period_with_no_spend_in_it_is_still_an_answer(self):
        empty = SpendData(rows=[], prior_rows=[], supplier_count=0, invoice_count=0,
                          available_data=frozenset())
        payload = _answer(fetch=lambda period, prior: empty)
        assert "No invoiced spend in FY26 YTD" in payload["answer"]
        assert payload["next_steps"] == []
