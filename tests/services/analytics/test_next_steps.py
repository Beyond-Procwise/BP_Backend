"""What to offer next, decided by rule rather than invented by a model.

Today the three chips under an answer are LLM-written questions. They are not
grounded in the data, they carry no entity, and clicking one re-submits a
sentence that has to be parsed back into an intent — which is where the subject
of the question gets lost. Worse, a model asked for "three follow-ups" will
always produce three, whether or not there is anything worth asking.

This engine is pure and table-driven: same answer in, same steps out, no model.
It returns at most two, and it returns none rather than pad.
"""

from datetime import date, datetime, timezone
from decimal import Decimal

import pytest

from src.services.analytics.currency import NATIVE, DisplayCurrency
from src.services.analytics.models import AnomalyCode, FactCode
from src.services.analytics.next_steps import (
    ALL_DATA,
    LADDER,
    VERBS,
    AllowAll,
    AllowList,
    select_next_steps,
)
from src.services.analytics.period import Period
from src.services.analytics.supplier_spend import SupplierSpendRow, build_supplier_spend_ranking

RATES = {"USD": Decimal("1.0"), "GBP": Decimal("0.739722"), "INR": Decimal("94.547339")}
FETCHED = datetime(2026, 9, 7, 9, 53, tzinfo=timezone.utc)
PERIOD = Period(date(2026, 4, 1), date(2026, 9, 7), "FY26 YTD")


def _row(name, amount, currency="GBP"):
    return SupplierSpendRow(supplier_id=name.lower().replace(" ", "-"), supplier_name=name,
                            currency=currency, amount=Decimal(str(amount)), invoices=1)


def _answer(rows=None, display=None, **kw):
    params = dict(
        rows=rows if rows is not None else [_row("Kestrel Supplies 8", "1200000"),
                                            _row("Blackwood Group 10", "900000")],
        display=display or DisplayCurrency(target="GBP", rates=RATES, fetched_at=FETCHED),
        period=PERIOD, population_count=2, invoice_count=100,
        answer_id="a-1", refreshed_at="2026-09-07T20:48:00Z",
        concentration_threshold_pct=Decimal("200"),  # off unless a test wants it
    )
    params.update(kw)
    return build_supplier_spend_ranking(**params)


def _steps(answer, **kw):
    params = dict(persona="default", entitlements=AllowAll(), available_data=ALL_DATA)
    params.update(kw)
    return select_next_steps(answer, **params)


class TestTheCatalogueIsWellFormed:
    def test_every_label_template_starts_with_a_catalogue_verb(self):
        for rung in LADDER:
            assert rung.label_template.split()[0] in VERBS, rung.name

    def test_no_label_template_runs_past_six_words(self):
        for rung in LADDER:
            assert len(rung.label_template.split()) <= 6, rung.name

    def test_the_ladder_opens_on_scope_which_is_the_answer_itself(self):
        assert LADDER[0].name == "scope"


class TestAnomaliesComeFirst:
    def test_an_unconvertible_currency_outranks_the_ladder(self):
        answer = _answer(rows=[_row("Kestrel Supplies 8", "1200000"),
                               _row("Sahel Freight", "5000000", "XOF")])
        first = _steps(answer)[0]
        assert first.reason == "anomaly:UNCONVERTED_CURRENCY"
        assert "XOF" in first.label

    def test_reporting_as_billed_offers_a_currency_before_anything_else(self):
        answer = _answer(display=DisplayCurrency(target=NATIVE, rates=RATES, fetched_at=FETCHED))
        assert _steps(answer)[0].reason == "anomaly:RANKING_NOT_COMPARABLE"

    def test_only_the_worst_anomaly_earns_a_step(self):
        # Two anomalies must not consume both slots and crowd out the ladder.
        answer = _answer(
            rows=[_row("Kestrel Supplies 8", "1200000"), _row("Sahel Freight", "5000000", "XOF")],
            display=DisplayCurrency(target="GBP", rates=RATES, fetched_at=FETCHED,
                                    manual={"GBP": Decimal("0.5")}),
        )
        steps = _steps(answer)
        assert [s.reason for s in steps].count("anomaly:UNCONVERTED_CURRENCY") == 1
        assert not any(s.reason == "anomaly:MANUAL_FX_RATE" for s in steps)

    def test_the_step_carries_the_suppliers_it_is_about(self):
        answer = _answer(rows=[_row("Kestrel Supplies 8", "1200000"),
                               _row("Sahel Freight", "5000000", "XOF")])
        assert _steps(answer)[0].entity_refs == ["sahel-freight"]


class TestFactTriggeredPromotion:
    def test_a_concentration_breach_promotes_that_rung_with_its_value_in_the_label(self):
        answer = _answer(concentration_threshold_pct=Decimal("30"))
        step = next(s for s in _steps(answer) if s.rung == "concentration")
        assert step.reason == "fact:CONCENTRATION_THRESHOLD_BREACHED"
        assert "%" in step.label

    def test_without_a_breach_concentration_is_still_the_next_rung_but_unpromoted(self):
        step = next(s for s in _steps(_answer()) if s.rung == "concentration")
        assert step.reason.startswith("ladder:")
        assert "%" not in step.label


class TestTheLadder:
    def test_the_default_persona_walks_the_ladder_in_order(self):
        assert [s.rung for s in _steps(_answer())] == ["concentration", "period_trend"]

    def test_a_rung_more_than_one_ahead_is_never_offered(self):
        # risk_exposure sits at the far end; a persona asking for it from a
        # scope-level answer must not vault the reader over everything between.
        steps = _steps(_answer(), persona="risk")
        assert "risk_exposure" not in [s.rung for s in steps]

    def test_a_persona_reorders_the_rungs_it_can_reach(self):
        finance = [s.rung for s in _steps(_answer(), persona="finance")]
        assert finance and finance[0] == "period_trend"

    def test_an_unknown_persona_falls_back_to_the_default_order(self):
        assert [s.rung for s in _steps(_answer(), persona="nobody")] == \
               [s.rung for s in _steps(_answer())]


class TestGates:
    def test_a_step_the_tenant_cannot_run_is_dropped_not_errored(self):
        steps = _steps(_answer(), entitlements=AllowList({"analytic.supplier_concentration"}))
        assert [s.rung for s in steps] == ["concentration"]

    def test_no_entitlement_service_means_no_steps_rather_than_all_of_them(self):
        # Fail closed. An absent gate is not an open gate.
        assert _steps(_answer(), entitlements=None) == []

    def test_a_rung_whose_data_is_absent_is_dropped(self):
        # bp_invoice_trgt.contract_id is filled on 0 of 12,408 rows, so nothing
        # links spend to the 3,051 contracts on record: coverage cannot be
        # computed, and offering it would be a dead end.
        steps = _steps(_answer(), available_data=ALL_DATA - {"period_comparison"})
        assert "period_trend" not in [s.rung for s in steps]

    def test_dropping_one_rung_lets_the_next_one_through(self):
        # An unavailable rung is not "skipped" — it was never eligible — so the
        # one behind it moves up rather than the ladder stalling.
        steps = _steps(_answer(), available_data=ALL_DATA - {"period_comparison"})
        assert [s.rung for s in steps] == ["concentration", "composition"]


class TestTheCap:
    def test_never_more_than_two(self):
        answer = _answer(rows=[_row("Kestrel Supplies 8", "1200000"),
                               _row("Sahel Freight", "5000000", "XOF")],
                         concentration_threshold_pct=Decimal("30"))
        assert len(_steps(answer)) <= 2

    def test_nothing_qualifying_returns_nothing_rather_than_padding(self):
        assert _steps(_answer(), available_data=set()) == []

    def test_an_empty_answer_offers_nothing_to_do_with_it(self):
        assert _steps(_answer(rows=[], population_count=0)) == []

    def test_the_same_action_is_never_offered_twice(self):
        answer = _answer(concentration_threshold_pct=Decimal("30"))
        steps = _steps(answer)
        assert len({s.action_id for s in steps}) == len(steps)


class TestAClickDispatchesWithoutReparsing:
    def test_every_step_names_an_action_and_the_entities_it_acts_on(self):
        for step in _steps(_answer()):
            assert step.action_id
            assert isinstance(step.entity_refs, list)

    def test_the_concentration_step_carries_the_suppliers_on_screen(self):
        step = next(s for s in _steps(_answer()) if s.rung == "concentration")
        assert step.entity_refs == ["kestrel-supplies-8", "blackwood-group-10"]
