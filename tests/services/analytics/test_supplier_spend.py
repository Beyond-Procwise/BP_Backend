"""Top suppliers by spend — the answer, decided before any model sees it.

The defect this replaces: `corpus_facts._fetch("spend")` ranks suppliers on raw
`invoice_amount` with the currency carried alongside, so a supplier billing in
rupees outranks one billing in sterling by a factor of about 128 for no reason
other than the denomination. Every one of the top five in the live corpus is
INR. Harbourline's 102,142,166.94 INR is under £0.8M, and no amount of better
prose fixes a table sorted that way.

So the ranking here is computed on converted values, in the currency the reader
selected on screen, and everything the headline could say is derived as a fact
first. The model's turn comes later and it may only rephrase these.
"""

from datetime import date, datetime, timezone
from decimal import Decimal

import pytest

from src.services.analytics.currency import NATIVE, DisplayCurrency
from src.services.analytics.models import (
    AnomalyCode,
    Confidence,
    CurrencyBasis,
    FactCode,
)
from src.services.analytics.period import Period
from src.services.analytics.supplier_spend import (
    SupplierSpendRow,
    build_supplier_spend_ranking,
)

RATES = {"USD": Decimal("1.0"), "GBP": Decimal("0.739722"),
         "EUR": Decimal("0.861072"), "INR": Decimal("94.547339")}
FETCHED = datetime(2026, 9, 7, 9, 53, tzinfo=timezone.utc)
PERIOD = Period(date(2026, 4, 1), date(2026, 9, 7), "FY26 YTD")


def _display(**kw) -> DisplayCurrency:
    return DisplayCurrency(target=kw.pop("target", "GBP"), rates=RATES,
                           fetched_at=FETCHED, **kw)


def _row(name, amount, currency="GBP", invoices=1, supplier_id=None) -> SupplierSpendRow:
    return SupplierSpendRow(
        supplier_id=supplier_id or name.lower().replace(" ", "-"),
        supplier_name=name, currency=currency, amount=Decimal(str(amount)),
        invoices=invoices,
    )


# Two sterling suppliers above the rupee giant once it is converted, so the
# ordering under test cannot be produced by accident.
CORPUS = [
    _row("Harbourline Trading 13", "102142166.94", "INR", invoices=140),
    _row("Kestrel Supplies 8", "1200000.00", "GBP", invoices=60),
    _row("Blackwood Group 10", "900000.00", "GBP", invoices=40),
    _row("Ironbridge Services 11", "500000.00", "EUR", invoices=20),
]


def _build(rows=None, **kw):
    params = dict(
        rows=rows if rows is not None else CORPUS,
        display=_display(),
        period=PERIOD,
        population_count=4,
        invoice_count=260,
        answer_id="a-1",
        refreshed_at="2026-09-07T20:48:00Z",
    )
    params.update(kw)
    return build_supplier_spend_ranking(**params)


class TestTheRankingIsOnConvertedValues:
    def test_the_rupee_giant_no_longer_tops_the_table(self):
        answer = _build()
        assert [r["supplier"] for r in answer.table.rows][:2] == [
            "Kestrel Supplies 8", "Blackwood Group 10",
        ]

    def test_the_rupee_supplier_still_appears_at_its_real_size(self):
        answer = _build()
        harbourline = next(r for r in answer.table.rows
                           if r["supplier"] == "Harbourline Trading 13")
        assert Decimal("799000") < harbourline["spend"] < Decimal("800000")

    def test_a_supplier_billing_in_two_currencies_is_summed_once_per_currency(self):
        rows = [_row("Split Ltd", "100", "GBP", supplier_id="split"),
                _row("Split Ltd", "100", "USD", supplier_id="split")]
        answer = _build(rows=rows, population_count=1, invoice_count=2)
        # 100 GBP untouched plus 100 USD at 0.739722.
        assert answer.table.rows[0]["spend"] == Decimal("173.9722")

    def test_only_the_top_n_are_listed(self):
        answer = _build(top_n=2)
        assert len(answer.table.rows) == 2

    def test_the_totals_row_is_the_whole_population_not_the_visible_rows(self):
        # A total that only adds up what is on screen is the "647 unresolved
        # cases against a real 685" failure in another costume.
        answer = _build(top_n=2)
        assert answer.table.totals["spend"] > sum(r["spend"] for r in answer.table.rows)


class TestScope:
    def test_scope_states_the_measure_period_population_and_currency(self):
        scope = _build().scope
        assert scope.measure.value == "INVOICED"
        assert scope.period_label == "FY26 YTD"
        assert scope.population.count == 4
        assert scope.currency == "GBP"
        assert scope.currency_basis is CurrencyBasis.CONVERTED

    def test_scope_carries_the_rate_note_so_a_converted_figure_is_never_bare(self):
        assert "rates as of" in _build().scope.line()

    def test_provenance_says_what_was_read_in_the_words_the_product_uses(self):
        # Not the table it came from: that name travels to a browser as a
        # dictionary key, and the answer beside it was replaced wholesale by
        # the output-safety gate the first time the rendered line carried one.
        provenance = _build().provenance
        assert provenance.source_counts["invoices"] == 260
        assert "bp_invoice_trgt" not in str(provenance.source_counts)
        assert provenance.query_ref


class TestFacts:
    def _codes(self, answer):
        return {f.code for f in answer.facts}

    def test_the_leader_share_and_the_top_n_share_are_both_computed(self):
        codes = self._codes(_build())
        assert FactCode.TOP_1_SHARE in codes
        assert FactCode.TOP_N_SHARE_OF_TOTAL in codes

    def test_the_leader_share_is_measured_against_the_whole_population(self):
        answer = _build(top_n=2)
        share = next(f for f in answer.facts if f.code is FactCode.TOP_1_SHARE)
        # Kestrel's 1.2M of the 3,328,678 converted total (36.05%), not of the
        # 2.1M on screen — which would have read as 57%.
        assert Decimal("35") < share.value < Decimal("37")

    def test_the_gap_to_second_place_is_stated_as_a_ratio(self):
        ratio = next(f for f in _build().facts
                     if f.code is FactCode.TOP_1_TO_TOP_2_RATIO)
        assert ratio.value == Decimal("1.3")

    def test_concentration_is_measured_on_the_top_n_not_on_the_leader_alone(self):
        # A top-1 threshold is the wrong instrument for this corpus: across
        # 3,510 suppliers the leader holds 1.4%, so a 20% top-1 rule would never
        # fire however concentrated the book actually was. The breach is the
        # share the visible group holds between them.
        answer = _build(top_n=2, concentration_threshold_pct=Decimal("30"))
        breach = next(f for f in answer.facts
                      if f.code is FactCode.CONCENTRATION_THRESHOLD_BREACHED)
        assert Decimal("62") < breach.value < Decimal("64")
        assert breach.unit == "2"
        # No single supplier owns a group finding, so none is named.
        assert breach.entity is None

    def test_no_breach_when_the_top_n_share_is_under_the_threshold(self):
        answer = _build(top_n=1, concentration_threshold_pct=Decimal("90"))
        assert FactCode.CONCENTRATION_THRESHOLD_BREACHED not in self._codes(answer)

    def test_a_supplier_billing_in_more_than_one_currency_is_flagged(self):
        rows = [_row("Split Ltd", "100", "GBP", supplier_id="split"),
                _row("Split Ltd", "100", "USD", supplier_id="split")]
        answer = _build(rows=rows, population_count=1, invoice_count=2)
        mismatch = next(f for f in answer.facts if f.code is FactCode.CURRENCY_MISMATCH)
        assert mismatch.entity == "Split Ltd"
        assert answer.table.rows[0]["_flags"] == ["CURRENCY_MISMATCH"]

    def test_a_period_delta_is_computed_per_row_when_a_prior_window_is_given(self):
        prior = [_row("Kestrel Supplies 8", "1000000.00", "GBP")]
        answer = _build(prior_rows=prior,
                        prior_period=Period(date(2025, 4, 1), date(2025, 9, 7), "FY25 YTD"))
        delta = next(f for f in answer.facts
                     if f.code is FactCode.PERIOD_DELTA and f.entity == "Kestrel Supplies 8")
        assert delta.value == Decimal("20.0")

    def test_no_prior_window_means_no_delta_facts_rather_than_zeroes(self):
        assert FactCode.PERIOD_DELTA not in self._codes(_build())


class TestAnomalies:
    def _codes(self, answer):
        return {a.code for a in answer.anomalies}

    def test_a_currency_with_no_rate_is_reported_not_silently_dropped(self):
        rows = CORPUS + [_row("Sahel Freight", "5000000", "XOF")]
        answer = _build(rows=rows, population_count=5)
        anomaly = next(a for a in answer.anomalies
                       if a.code is AnomalyCode.UNCONVERTED_CURRENCY)
        assert "XOF" in anomaly.text

    def test_an_unconvertible_supplier_is_kept_out_of_the_ranking(self):
        rows = CORPUS + [_row("Sahel Freight", "5000000", "XOF")]
        answer = _build(rows=rows, population_count=5)
        assert "Sahel Freight" not in [r["supplier"] for r in answer.table.rows]

    def test_a_manual_rate_is_declared_and_costs_the_answer_its_assertion(self):
        answer = _build(display=_display(manual={"GBP": Decimal("0.50")}))
        assert AnomalyCode.MANUAL_FX_RATE in self._codes(answer)
        assert answer.scope.currency_basis is CurrencyBasis.MANUAL
        assert answer.headline.confidence is not Confidence.ASSERTED

    def test_the_worst_anomaly_is_first(self):
        rows = CORPUS + [_row("Sahel Freight", "5000000", "XOF")]
        answer = _build(rows=rows, population_count=5,
                        display=_display(manual={"GBP": Decimal("0.50")}))
        assert answer.anomalies[0].code is AnomalyCode.UNCONVERTED_CURRENCY


class TestReportingAsBilled:
    """Native mode: the reader asked for figures as billed, so there is no
    common currency and therefore no single league table. Presenting one
    anyway is the defect this work exists to end."""

    def _native(self, **kw):
        return _build(display=DisplayCurrency(target=NATIVE, rates=RATES,
                                              fetched_at=FETCHED), **kw)

    def test_a_mixed_ranking_is_declared_not_comparable(self):
        assert AnomalyCode.RANKING_NOT_COMPARABLE in {
            a.code for a in self._native().anomalies}

    def test_the_rank_restarts_within_each_currency(self):
        rows = self._native().table.rows
        firsts = [r for r in rows if r["rank"] == 1]
        assert {r["currency"] for r in firsts} == {"GBP", "EUR", "INR"}

    def test_the_currency_is_a_column_of_its_own(self):
        assert "currency" in [c.key for c in self._native().table.columns]

    def test_no_cross_currency_share_is_claimed(self):
        codes = {f.code for f in self._native().facts}
        assert FactCode.TOP_1_SHARE not in codes
        assert FactCode.TOP_N_SHARE_OF_TOTAL not in codes

    def test_scope_says_the_figures_are_as_billed(self):
        assert self._native().scope.currency_basis is CurrencyBasis.NATIVE


class TestTheHeadlineIsTemplatedUntilAModelEarnsIt:
    def test_a_negligible_gap_to_second_place_is_not_narrated_as_a_lead(self):
        # Live: Kestrel £288.4K against Meridian £274.9K is a ratio of 1.0, and
        # "1.0 times the next" is not a finding — it is a sentence pretending to
        # be one. The fact is still produced; the headline just does not use it.
        rows = [_row("Kestrel Supplies 12", "288400"), _row("Meridian Services 2", "274900")]
        answer = _build(rows=rows, population_count=2)
        assert "times the next" not in answer.headline.text

    def test_a_real_lead_is_narrated(self):
        assert "times the next" in _build().headline.text


    def test_every_number_in_the_templated_headline_comes_off_the_payload(self):
        # The same check the insight writer's output will face in PR 5. The
        # template has to pass it too, or the fallback would be held to a
        # weaker standard than the model it replaces.
        answer = _build()
        assert answer.unquoted_numbers(answer.headline.text) == set()

    def test_an_empty_corpus_says_so_rather_than_ranking_nothing(self):
        answer = _build(rows=[], population_count=0, invoice_count=0)
        assert answer.table.rows == []
        assert "no" in answer.headline.text.lower()
