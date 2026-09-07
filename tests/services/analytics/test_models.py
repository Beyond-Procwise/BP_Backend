"""The answer contract. What a valid analytic answer is, enforced by the type.

Two properties matter more than the rest and are asserted hardest:

  * **Scope is never empty.** A ranking with no period, no measure and no stated
    currency is not an answer — it is a table someone has to interrogate before
    they can trust it. The type refuses to build one.
  * **Facts carry their own rendered form.** The insight writer is shown facts,
    never rows, and is allowed to use no number that is not already in them. That
    check is only possible if each fact carries the exact text a sentence may
    quote, so ``display`` is built here by the shared formatter rather than left
    to the model.

Field names are snake_case, as every other payload this API returns is
(``follow_ups``, ``retrieved_documents``).
"""

from decimal import Decimal

import pytest
from pydantic import ValidationError

from src.services.analytics.models import (
    AnalyticAnswer,
    Anomaly,
    AnomalyCode,
    Column,
    Confidence,
    ColumnType,
    CurrencyBasis,
    Fact,
    FactCode,
    Headline,
    NextStep,
    Population,
    Provenance,
    Scope,
    Severity,
    Table,
)


def _scope(**kw) -> Scope:
    defaults = dict(
        measure="INVOICED",
        period_start="2026-04-01",
        period_end="2026-09-07",
        period_label="FY26 YTD",
        population=Population(entity="SUPPLIER", count=3510),
        currency="GBP",
        currency_basis=CurrencyBasis.CONVERTED,
        rate_note="rates as of 07 Sep 2026, 09:53 (live)",
    )
    defaults.update(kw)
    return Scope(**defaults)


def _answer(**kw) -> AnalyticAnswer:
    defaults = dict(
        answer_id="a-1",
        scope=_scope(),
        headline=Headline(text="Harbourline holds 23.4% of invoiced spend."),
        table=Table(
            columns=[Column(key="supplier", label="Supplier", type=ColumnType.TEXT),
                     Column(key="spend", label="Invoiced spend", type=ColumnType.MONEY)],
            rows=[{"supplier": "Harbourline Trading 13", "spend": Decimal("799412.10")}],
        ),
        provenance=Provenance(source_counts={"proc.bp_invoice_trgt": 12408},
                              refreshed_at="2026-09-07T20:48:00Z",
                              query_ref="supplier_spend_ranking/v1"),
    )
    defaults.update(kw)
    return AnalyticAnswer(**defaults)


class TestScopeIsMandatory:
    def test_an_answer_without_scope_cannot_be_built(self):
        with pytest.raises(ValidationError):
            AnalyticAnswer(
                answer_id="a-1",
                headline=Headline(text="x"),
                table=Table(columns=[], rows=[]),
                provenance=Provenance(source_counts={}, refreshed_at="2026-09-07T20:48:00Z",
                                      query_ref="q"),
            )

    def test_scope_refuses_a_blank_period_label(self):
        with pytest.raises(ValidationError):
            _scope(period_label="   ")

    def test_scope_refuses_a_blank_currency_unless_reporting_natively(self):
        with pytest.raises(ValidationError):
            _scope(currency="", currency_basis=CurrencyBasis.CONVERTED)

    def test_native_reporting_needs_no_single_currency(self):
        scope = _scope(currency="", currency_basis=CurrencyBasis.NATIVE)
        assert scope.currency == ""

    def test_the_scope_line_reads_as_one_sentence_of_provenance(self):
        assert _scope().line() == (
            "Invoiced spend · FY26 YTD (1 Apr–7 Sep 2026) · 3,510 suppliers · GBP "
            "· rates as of 07 Sep 2026, 09:53 (live)"
        )

    def test_the_scope_line_names_the_filters_that_were_applied(self):
        scope = _scope(filters_applied=["excludes credit notes"])
        assert scope.line().endswith("· excludes credit notes")


class TestFactsCarryTheirRenderedForm:
    def test_a_money_fact_renders_through_the_shared_formatter(self):
        fact = Fact(code=FactCode.TOP_1_SHARE, entity="Harbourline Trading 13",
                    value=Decimal("799412.10"), type=ColumnType.MONEY, currency="GBP")
        assert fact.display == "£799.4K"

    def test_a_percentage_fact_renders_as_a_percentage(self):
        fact = Fact(code=FactCode.TOP_N_SHARE_OF_TOTAL, value=Decimal("23.42"),
                    type=ColumnType.PCT)
        assert fact.display == "23.4%"

    def test_the_numeric_tokens_a_sentence_may_quote_come_off_the_facts(self):
        answer = _answer(facts=[
            Fact(code=FactCode.TOP_1_SHARE, value=Decimal("23.42"), type=ColumnType.PCT),
            Fact(code=FactCode.TOP_1_TO_TOP_2_RATIO, value=Decimal("1.8"),
                 type=ColumnType.TEXT),
        ])
        assert answer.quotable_numbers() == {"23.4", "1.8"}


class TestNextSteps:
    def test_two_steps_are_allowed(self):
        answer = _answer(next_steps=[
            NextStep(action_id="normalise_currency", label="Normalise INR spend",
                     rung="scope", reason="UNCONVERTED_CURRENCY"),
            NextStep(action_id="concentration", label="Check supplier concentration",
                     rung="concentration", reason="ladder"),
        ])
        assert len(answer.next_steps) == 2

    def test_a_third_step_is_refused_rather_than_silently_trimmed(self):
        # Trimming would hide a selection bug behind a plausible answer.
        with pytest.raises(ValidationError):
            _answer(next_steps=[
                NextStep(action_id="a", label="Do a", rung="scope", reason="r"),
                NextStep(action_id="b", label="Do b", rung="concentration", reason="r"),
                NextStep(action_id="c", label="Do c", rung="period_trend", reason="r"),
            ])

    def test_no_steps_is_a_valid_answer(self):
        assert _answer().next_steps == []

    def test_a_step_carries_the_ids_a_click_dispatches_with(self):
        step = NextStep(action_id="concentration", label="Flag concentration risk",
                        rung="concentration", reason="threshold breached",
                        entity_refs=["SUP-13"])
        assert step.entity_refs == ["SUP-13"]


class TestHeadline:
    def test_confidence_defaults_to_asserted(self):
        assert Headline(text="x").confidence is Confidence.ASSERTED

    def test_a_manual_rate_answer_is_not_asserted(self):
        # A what-if rate is the user's number, not the record's.
        headline = Headline(text="x", confidence=Confidence.UNASSESSED)
        assert headline.confidence is Confidence.UNASSESSED


class TestAnomalies:
    def test_anomalies_sort_highest_severity_first(self):
        answer = _answer(anomalies=[
            Anomaly(code=AnomalyCode.MISSING_PERIOD_DATA, severity=Severity.LOW,
                    text="No invoices dated before 2023."),
            Anomaly(code=AnomalyCode.UNCONVERTED_CURRENCY, severity=Severity.HIGH,
                    text="2 currencies have no rate and are excluded."),
        ])
        assert answer.anomalies[0].code is AnomalyCode.UNCONVERTED_CURRENCY

    def test_a_native_ranking_is_marked_not_comparable(self):
        # Ranking across unconverted currencies is the defect this work exists
        # to end; the code for it has to exist in the contract.
        assert AnomalyCode.RANKING_NOT_COMPARABLE.value == "RANKING_NOT_COMPARABLE"
