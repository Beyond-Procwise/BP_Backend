"""The two answers a next-step chip leads to.

A chip that cannot dispatch anywhere is a dead end drawn to look like a
control, so "Review supplier concentration" and "Compare against last year"
have to lead to answers that are actually different from the ranking they were
offered under. Both are the same measurement seen a different way, which is why
they are lenses on one builder rather than three builders:

  * **Concentration** keeps the spend order and adds the running share, because
    concentration is read cumulatively — the top three hold this much, the top
    ten that much — not off any single row.
  * **Trend** re-orders the same suppliers by how much they moved and shows
    what they were, because the largest supplier and the fastest-moving one are
    rarely the same name, and the second question is not answered by the first
    table sorted the same way.

Each answer says which rung it stands on, so the ladder advances rather than
offering the reader the screen they are already looking at.
"""

from datetime import date, datetime, timezone
from decimal import Decimal

import pytest

from src.services.analytics.currency import DisplayCurrency
from src.services.analytics.models import AnomalyCode, FactCode
from src.services.analytics.next_steps import ALL_DATA, AllowAll, select_next_steps
from src.services.analytics.period import Period
from src.services.analytics.supplier_spend import (
    Lens,
    SupplierSpendRow,
    build_supplier_spend_ranking,
)

RATES = {"USD": Decimal("1.0"), "GBP": Decimal("0.739722")}
FETCHED = datetime(2026, 9, 7, 9, 53, tzinfo=timezone.utc)
PERIOD = Period(date(2026, 4, 1), date(2026, 9, 7), "FY26 YTD")
PRIOR = Period(date(2025, 4, 1), date(2025, 9, 7), "FY25 YTD")


def _row(name, amount, currency="GBP"):
    return SupplierSpendRow(supplier_id=name.lower().replace(" ", "-"), supplier_name=name,
                            currency=currency, amount=Decimal(str(amount)), invoices=1)


NOW = [_row("Kestrel", "1000000"), _row("Blackwood", "600000"), _row("Harbourline", "400000")]
THEN = [_row("Kestrel", "900000"), _row("Blackwood", "200000"), _row("Harbourline", "500000")]


def _answer(lens, rows=None, prior_rows=THEN, **kw):
    params = dict(
        rows=rows if rows is not None else NOW,
        prior_rows=prior_rows, prior_period=PRIOR if prior_rows is not None else None,
        display=DisplayCurrency(target="GBP", rates=RATES, fetched_at=FETCHED),
        period=PERIOD, population_count=3, invoice_count=30,
        answer_id="a-1", refreshed_at="2026-09-07T20:48:00Z",
        concentration_threshold_pct=Decimal("30"), lens=lens,
    )
    params.update(kw)
    return build_supplier_spend_ranking(**params)


def _column(answer, key):
    return next((c for c in answer.table.columns if c.key == key), None)


class TestTheConcentrationLens:
    def test_the_running_share_is_shown_because_that_is_what_concentration_is(self):
        answer = _answer(Lens.CONCENTRATION)
        assert _column(answer, "cumulative") is not None
        assert [row["cumulative"] for row in answer.table.rows] == [
            Decimal(50), Decimal(80), Decimal(100)]

    def test_the_order_is_still_the_order_of_spend(self):
        answer = _answer(Lens.CONCENTRATION)
        assert [row["supplier"] for row in answer.table.rows] == [
            "Kestrel", "Blackwood", "Harbourline"]

    def test_the_headline_is_about_the_group_not_the_leader(self):
        answer = _answer(Lens.CONCENTRATION)
        assert "top 3" in answer.headline.text
        assert answer.headline.text.startswith("The top")

    def test_the_answer_says_which_question_it_answers(self):
        assert _answer(Lens.CONCENTRATION).provenance.query_ref == "supplier_concentration/v1"


class TestTheTrendLens:
    def test_the_suppliers_are_ordered_by_how_much_they_moved(self):
        # Blackwood tripled on a small base; Kestrel is the biggest supplier and
        # barely moved; Harbourline fell. Size does not decide this table.
        answer = _answer(Lens.TREND)
        assert [row["supplier"] for row in answer.table.rows] == [
            "Blackwood", "Kestrel", "Harbourline"]

    def test_what_they_were_is_shown_beside_what_they_are(self):
        answer = _answer(Lens.TREND)
        prior = _column(answer, "prior")
        assert prior is not None and PRIOR.label in prior.label
        assert answer.table.rows[0]["prior"] == Decimal("200000")

    def test_the_population_it_ranks_is_stated_rather_than_implied(self):
        # The movers are chosen from the top suppliers by spend, not from all
        # 354 — otherwise a supplier that went from £40 to £400 leads the table.
        answer = _answer(Lens.TREND, top_n=2)
        assert any("by spend" in f for f in answer.scope.filters_applied)
        assert len(answer.table.rows) == 2

    def test_the_headline_names_the_mover_and_the_book(self):
        answer = _answer(Lens.TREND)
        assert "Blackwood" in answer.headline.text
        assert "+200.0%" in answer.headline.text   # 200K -> 600K
        assert "+25.0%" in answer.headline.text    # 1.6M -> 2.0M across the book

    def test_the_book_is_described_as_the_book_and_not_as_the_table(self):
        # The book delta is measured across every supplier, not across the ten
        # on screen, and the sentence has to say which.
        assert "all 3 suppliers" in _answer(Lens.TREND).headline.text


    def test_only_suppliers_whose_change_was_measured_are_listed(self):
        # Live, nine of the ten rows had no prior-year spend at all, so a table
        # asking "what moved" answered it with eight blanks. A supplier that
        # was never measured has not moved; it is simply unknown, and it does
        # not belong in a movement table.
        answer = _answer(Lens.TREND, prior_rows=[_row("Kestrel", "900000")])
        assert [row["supplier"] for row in answer.table.rows] == ["Kestrel"]

    def test_no_bar_is_drawn_against_spend_in_a_movement_table(self):
        # The bar measures the primary column, and the primary column here is
        # not what the table is ordered by: a bar showing the biggest supplier
        # at the bottom of a movement table reads as a mistake.
        assert _column(_answer(Lens.TREND), "spend").is_primary is False

    def test_a_period_with_nothing_to_compare_to_says_so(self):
        answer = _answer(Lens.TREND, prior_rows=[])
        assert any(a.code is AnomalyCode.MISSING_PERIOD_DATA for a in answer.anomalies)
        assert "no change" in answer.headline.text.lower() or \
               "no comparable" in answer.headline.text.lower()

    def test_the_answer_says_which_question_it_answers(self):
        assert _answer(Lens.TREND).provenance.query_ref == "supplier_spend_trend/v1"


class TestAChangeMeasuredFromNothing:
    """Live: "Windrose Services 14 moved most, +345261.1%" — on £26.72 of spend
    the year before. The arithmetic is right and the finding is not: a
    percentage taken off a base that small says nothing about the supplier, and
    it takes the top of a table meant to show what actually moved."""

    NOW = NOW + [_row("Featherstone", "50000")]
    THEN = THEN + [_row("Featherstone", "10")]

    def _answer(self):
        return _answer(Lens.TREND, rows=self.NOW, prior_rows=self.THEN)

    def test_it_does_not_lead_the_table(self):
        assert self._answer().table.rows[0]["supplier"] == "Blackwood"

    def test_it_is_not_the_supplier_the_headline_names(self):
        assert "Featherstone" not in self._answer().headline.text

    def test_the_row_is_marked_so_the_reader_knows_why(self):
        row = next(r for r in self._answer().table.rows if r["supplier"] == "Featherstone")
        assert "NEGLIGIBLE_BASE" in row["_flags"]

    def test_the_base_it_was_measured_from_is_a_fact(self):
        fact = next(f for f in self._answer().facts if f.code is FactCode.NEGLIGIBLE_BASE)
        assert fact.entity == "Featherstone"
        assert fact.display == "£10"


class TestTheLadderKnowsWhereItIs:
    def _steps(self, answer):
        return select_next_steps(answer, persona="default", entitlements=AllowAll(),
                                 available_data=ALL_DATA)

    def test_a_concentration_answer_does_not_offer_concentration_again(self):
        steps = self._steps(_answer(Lens.CONCENTRATION))
        assert "analytic.supplier_concentration" not in [s.action_id for s in steps]
        assert steps[0].action_id == "analytic.supplier_spend_trend"

    def test_a_trend_answer_moves_on_up_the_ladder(self):
        steps = self._steps(_answer(Lens.TREND))
        assert [s.rung for s in steps][:1] == ["composition"]
