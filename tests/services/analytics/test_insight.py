"""The one sentence a model is allowed to write, and the gate it passes first.

Everything else in an analytic answer is decided by arithmetic. The headline is
the single place a model earns its keep — it reads the facts and says which one
matters — and it is also the only place a model can put a number into a
customer's answer. So it is held to a rule that is checkable rather than
merely instructed: it may use no figure that is not already in the payload, and
a sentence that breaks the rule is thrown away for the templated headline the
answer already shipped with.

The stub here stands in for the model, and only for the model. Everything the
gate does — the counting, the grounding check, the fallback and the audit row —
is the real code.
"""

from datetime import date, datetime, timezone
from decimal import Decimal
import json

import pytest

from src.services.analytics.currency import DisplayCurrency
from src.services.analytics.models import Anomaly, AnomalyCode, Severity
from src.services.analytics.insight import (
    INSIGHT_SCHEMA,
    Rejection,
    build_prompt,
    validate_insight,
    write_insight,
)
from src.services.analytics.period import Period
from src.services.analytics.supplier_spend import SupplierSpendRow, build_supplier_spend_ranking

RATES = {"USD": Decimal("1.0"), "GBP": Decimal("0.739722"), "INR": Decimal("94.547339")}
FETCHED = datetime(2026, 9, 7, 9, 53, tzinfo=timezone.utc)
PERIOD = Period(date(2026, 4, 1), date(2026, 9, 7), "FY26 YTD")


def _row(name, amount, currency="GBP"):
    return SupplierSpendRow(supplier_id=name.lower().replace(" ", "-"), supplier_name=name,
                            currency=currency, amount=Decimal(str(amount)), invoices=1)


def _answer(**kw):
    params = dict(
        rows=[_row("Kestrel Supplies 8", "1200000"), _row("Blackwood Group 10", "900000")],
        display=DisplayCurrency(target="GBP", rates=RATES, fetched_at=FETCHED),
        period=PERIOD, population_count=2, invoice_count=100,
        answer_id="a-1", refreshed_at="2026-09-07T20:48:00Z",
        concentration_threshold_pct=Decimal("200"),
    )
    params.update(kw)
    return build_supplier_spend_ranking(**params)


GROUNDED = "Kestrel Supplies 8 holds 57.1% of invoiced spend. Blackwood Group 10 follows."


def _says(text):
    """A model that returns exactly this, in the shape the schema constrains it to."""
    return lambda prompt: json.dumps({"text": text})


class _Spy:
    """The audit spine, watched rather than written to."""

    def __init__(self):
        self.rows = []

    def __call__(self, **fields):
        self.rows.append(fields)


class TestWhatTheWriterIsShown:
    def test_the_writer_is_shown_the_facts_and_never_the_rows(self):
        answer = _answer()
        prompt = build_prompt(answer, persona="cpo")
        assert "TOP_1_SHARE" in prompt
        assert "57.1%" in prompt
        # The rows are where the unrounded figures live. A model shown those is
        # a model formatting money again, which is the defect this layer ends.
        assert "1200000" not in prompt
        assert "entity_ref" not in prompt

    def test_the_writer_is_told_what_the_answer_covers(self):
        assert _answer().scope.line() in build_prompt(_answer(), persona="cpo")

    def test_the_persona_reaches_the_writer(self):
        assert "cpo" in build_prompt(_answer(), persona="cpo").lower()

    def test_the_schema_admits_one_field_only(self):
        assert INSIGHT_SCHEMA["required"] == ["text"]
        assert set(INSIGHT_SCHEMA["properties"]) == {"text"}


class TestTheGate:
    def test_a_grounded_sentence_passes(self):
        assert validate_insight(_answer(), GROUNDED) is None

    def test_a_figure_that_is_not_in_the_payload_is_refused(self):
        rejection = validate_insight(_answer(), "Kestrel Supplies 8 holds £9.9M of spend.")
        assert isinstance(rejection, Rejection)
        assert rejection.reason == "ungrounded_number"
        assert "9.9" in rejection.detail

    def test_a_number_inside_a_supplier_name_licences_nothing(self):
        # "Kestrel Supplies 8" is a name, not a figure. Naming it must not let
        # the same sentence claim "8% of spend" and pass.
        assert validate_insight(_answer(), "Kestrel Supplies 8 takes 8.4% of spend.") is not None

    def test_a_third_sentence_is_refused(self):
        text = "Kestrel Supplies 8 leads. Blackwood Group 10 follows. Spend is concentrated."
        assert validate_insight(_answer(), text).reason == "too_many_sentences"

    def test_a_decimal_does_not_count_as_the_end_of_a_sentence(self):
        assert validate_insight(_answer(), "Kestrel Supplies 8 holds 57.1% of spend.") is None

    def test_a_hedge_is_refused(self):
        assert validate_insight(_answer(), "Spend may be concentrated.").reason == "hedged"

    def test_a_recommendation_is_refused(self):
        text = "Kestrel Supplies 8 holds 57.1% of spend; you should renegotiate."
        assert validate_insight(_answer(), text).reason == "recommendation"

    def test_a_judgement_the_facts_do_not_carry_is_refused(self):
        # Live, the model wrote "...breaching the concentration threshold and
        # indicating elevated supply risk." Nothing measured says the risk is
        # elevated, or that there is risk at all. A claim nobody can check is
        # the same defect as a figure nobody can check.
        text = ("The top 2 suppliers hold 100.0% of invoiced spend, "
                "indicating elevated supply risk.")
        assert validate_insight(_answer(), text).reason == "unsupported_claim"

    def test_an_adjective_standing_in_for_a_measurement_is_refused(self):
        assert validate_insight(
            _answer(), "Kestrel Supplies 8 holds a significant share of spend."
        ).reason == "unsupported_claim"

    def test_a_word_the_answer_itself_uses_is_not_a_judgement(self):
        # The caveats under the table are written by this system, not by the
        # model, so a sentence repeating one is quoting the answer rather than
        # editorialising over it. Nothing in the corpus produces a risk caveat
        # yet — the risk_exposure rung has no data behind it — so the answer
        # here is given one, which is the shape that rung will take.
        answer = _answer()
        answer = answer.model_copy(update={"anomalies": [Anomaly(
            code=AnomalyCode.MISSING_PERIOD_DATA, severity=Severity.LOW,
            text="Two suppliers carry an open risk finding.")]})
        assert validate_insight(answer, "Two suppliers carry an open risk finding.") is None

    def test_a_claim_about_suppliers_not_in_the_table_is_refused(self):
        # Live, from the top-5 answer: "Copperleaf Works 3 drove a 362.0%
        # increase in invoiced spend, the largest period-over-period growth
        # among all suppliers." The figure was real and the superlative was
        # not: growth was measured for the five suppliers shown, out of 354.
        answer = _answer(population_count=354)
        text = ("Kestrel Supplies 8 holds 57.1% of invoiced spend, "
                "the largest share among all suppliers.")
        assert validate_insight(answer, text).reason == "overreaching_claim"

    def test_the_same_claim_stands_when_every_supplier_is_in_the_table(self):
        # Two suppliers, both listed: "any other supplier" is checkable from
        # the table itself.
        text = "Kestrel Supplies 8 holds 57.1%, more than any other supplier."
        assert validate_insight(_answer(), text) is None

    def test_an_empty_sentence_is_refused(self):
        assert validate_insight(_answer(), "   ").reason == "empty"


class TestWhatTheReaderEndsUpWith:
    def test_a_sentence_that_passes_becomes_the_headline(self):
        answer = _answer()
        written = write_insight(answer, persona="cpo", generate=_says(GROUNDED), audit=_Spy())
        assert written.headline.text == GROUNDED

    def test_a_sentence_that_fails_leaves_the_templated_headline_standing(self):
        answer = _answer()
        written = write_insight(answer, persona="cpo",
                                generate=_says("Spend hit £9.9M."), audit=_Spy())
        assert written.headline.text == answer.headline.text
        assert "9.9" not in written.headline.text

    def test_the_rest_of_the_answer_is_untouched(self):
        answer = _answer()
        written = write_insight(answer, persona="cpo", generate=_says(GROUNDED), audit=_Spy())
        assert written.table == answer.table
        assert written.facts == answer.facts
        assert written.answer_id == answer.answer_id

    def test_a_model_that_says_nothing_leaves_the_headline_standing(self):
        answer = _answer()
        written = write_insight(answer, persona="cpo", generate=lambda p: None, audit=_Spy())
        assert written.headline.text == answer.headline.text

    def test_output_that_is_not_the_agreed_shape_leaves_the_headline_standing(self):
        answer = _answer()
        written = write_insight(answer, persona="cpo",
                                generate=lambda p: "Kestrel leads.", audit=_Spy())
        assert written.headline.text == answer.headline.text

    def test_a_model_that_raises_does_not_take_the_answer_down_with_it(self):
        def _fails(prompt):
            raise RuntimeError("ollama is down")

        answer = _answer()
        written = write_insight(answer, persona="cpo", generate=_fails, audit=_Spy())
        assert written.headline.text == answer.headline.text


class TestWhenThePlatformIsDegraded:
    """A headline is optional; a 45-second wait for one is not acceptable.

    Live: with the card too full to hold the model whole, Ollama runs it half
    on the CPU and the sentence does not arrive inside the writer's bound. Every
    ask then paid 45 seconds to end up with the templated headline it already
    had. So while the card is refusing, the model is not asked at all — the
    answer is the same, and it arrives in a second.
    """

    def test_the_model_is_not_asked_while_the_card_is_refusing(self):
        from src.services import ollama_client

        asked = []
        ollama_client.note_layout_rejection("full")
        try:
            written = write_insight(_answer(), persona="cpo",
                                    generate=lambda p: asked.append(p) or _says(GROUNDED)(p),
                                    audit=_Spy())
            assert asked == []
            assert written.headline.text == _answer().headline.text
        finally:
            ollama_client.clear_layout_rejection()

    def test_the_skip_is_recorded_rather_than_silent(self):
        from src.services import ollama_client

        spy = _Spy()
        ollama_client.note_layout_rejection("full")
        try:
            write_insight(_answer(), persona="cpo", generate=_says(GROUNDED), audit=spy)
        finally:
            ollama_client.clear_layout_rejection()
        details = spy.rows[0]["details"]
        assert details["reason"] == "platform_degraded"
        assert spy.rows[0]["status"] == "rejected"

    def test_a_caller_may_insist(self):
        # The skip is a default, not a law: a caller whose writer is not the
        # local model — a batch job, a test, a hosted model — turns it off and
        # the sentence is written as usual.
        from src.services import ollama_client

        ollama_client.note_layout_rejection("full")
        try:
            written = write_insight(_answer(), persona="cpo", generate=_says(GROUNDED),
                                    audit=_Spy(), skip_when_degraded=False)
            assert written.headline.text == GROUNDED
        finally:
            ollama_client.clear_layout_rejection()


class TestTheAuditSpine:
    def test_an_accepted_sentence_is_recorded_with_what_produced_it(self):
        spy = _Spy()
        write_insight(_answer(), persona="cpo", generate=_says(GROUNDED), audit=spy)
        assert len(spy.rows) == 1
        row = spy.rows[0]
        assert row["status"] == "ok"
        assert row["trace_id"] == "a-1"
        assert row["summary"] == GROUNDED
        details = json.loads(row["details"]) if isinstance(row["details"], str) else row["details"]
        assert details["persona"] == "cpo"
        assert len(details["prompt_sha256"]) == 64

    def test_a_refusal_is_recorded_with_its_reason_and_the_sentence_refused(self):
        spy = _Spy()
        write_insight(_answer(), persona="cpo", generate=_says("Spend hit £9.9M."), audit=spy)
        row = spy.rows[0]
        assert row["status"] == "rejected"
        details = json.loads(row["details"]) if isinstance(row["details"], str) else row["details"]
        assert details["reason"] == "ungrounded_number"
        assert details["text"] == "Spend hit £9.9M."

    def test_an_audit_that_will_not_write_does_not_cost_the_reader_the_answer(self):
        def _broken(**fields):
            raise RuntimeError("no such table")

        written = write_insight(_answer(), persona="cpo", generate=_says(GROUNDED), audit=_broken)
        assert written.headline.text == GROUNDED
