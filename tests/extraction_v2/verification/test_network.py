"""Tests for the verification network and built-in rules."""
from decimal import Decimal
from datetime import date

from services.extraction_v2.verification.network import Rule, run_verification
from services.extraction_v2.verification.rules.math import (
    INVOICE_TOTAL_MATH, PO_TOTAL_MATH, QUOTE_TOTAL_MATH,
    TAX_NOT_EQUAL_SUBTOTAL_INVOICE, TAX_PERCENT_PLAUSIBLE,
    ALL_MATH_RULES,
)
from services.extraction_v2.verification.rules.dates import (
    INVOICE_DATE_BEFORE_DUE, ALL_DATE_RULES,
)


class TestNetworkPlumbing:
    def test_no_rules_no_demotions(self):
        outcome = run_verification({"x": 1}, [])
        assert outcome.rule_results == []
        assert not outcome.demoted_fields
        assert not outcome.abstained_fields

    def test_skips_rule_with_missing_field(self):
        rule = Rule(
            name="needs_x_y", fields=("x", "y"),
            check=lambda v: True, on_fail="demote",
        )
        outcome = run_verification({"x": 1}, [rule])
        # rule was skipped (y missing) — no result, no demotion
        assert outcome.rule_results == []
        assert not outcome.demoted_fields

    def test_passing_rule_does_not_demote(self):
        rule = Rule(name="r", fields=("a",), check=lambda v: True, on_fail="demote")
        outcome = run_verification({"a": 1}, [rule])
        assert len(outcome.rule_results) == 1
        assert outcome.rule_results[0].passed
        assert not outcome.demoted_fields

    def test_failing_rule_demote(self):
        rule = Rule(name="r", fields=("a", "b"), check=lambda v: False,
                    on_fail="demote", why="never passes")
        outcome = run_verification({"a": 1, "b": 2}, [rule])
        assert outcome.demoted_fields == {"a", "b"}

    def test_failing_rule_abstain(self):
        rule = Rule(name="r", fields=("a",), check=lambda v: False,
                    on_fail="abstain", why="bad")
        outcome = run_verification({"a": 1}, [rule])
        assert outcome.abstained_fields == {"a"}

    def test_rule_exception_treated_as_fail(self):
        rule = Rule(name="r", fields=("a",),
                    check=lambda v: 1/0, on_fail="demote")
        outcome = run_verification({"a": 1}, [rule])
        assert outcome.rule_results[0].passed is False


class TestMathRules:
    def test_invoice_total_math_passes(self):
        v = {"invoice_amount": Decimal("100"),
             "tax_amount":     Decimal("20"),
             "invoice_total_incl_tax": Decimal("120")}
        outcome = run_verification(v, [INVOICE_TOTAL_MATH])
        assert not outcome.demoted_fields

    def test_invoice_total_math_fails(self):
        v = {"invoice_amount": Decimal("100"),
             "tax_amount":     Decimal("20"),
             "invoice_total_incl_tax": Decimal("130")}   # off by 10
        outcome = run_verification(v, [INVOICE_TOTAL_MATH])
        assert outcome.demoted_fields == set(INVOICE_TOTAL_MATH.fields)

    def test_tax_equals_subtotal_abstains(self):
        v = {"invoice_amount": Decimal("675"), "tax_amount": Decimal("675")}
        outcome = run_verification(v, [TAX_NOT_EQUAL_SUBTOTAL_INVOICE])
        assert outcome.abstained_fields == {"invoice_amount", "tax_amount"}

    def test_tax_percent_implausible_abstains(self):
        v = {"tax_percent": Decimal("100")}
        outcome = run_verification(v, [TAX_PERCENT_PLAUSIBLE])
        assert outcome.abstained_fields == {"tax_percent"}

    def test_tax_percent_zero_passes(self):
        v = {"tax_percent": Decimal("0")}
        outcome = run_verification(v, [TAX_PERCENT_PLAUSIBLE])
        assert not outcome.abstained_fields


class TestDateRules:
    def test_invoice_date_before_due_passes(self):
        v = {"invoice_date": date(2024, 1, 1), "due_date": date(2024, 2, 1)}
        outcome = run_verification(v, [INVOICE_DATE_BEFORE_DUE])
        assert not outcome.demoted_fields

    def test_invoice_date_after_due_fails(self):
        v = {"invoice_date": date(2024, 3, 1), "due_date": date(2024, 2, 1)}
        outcome = run_verification(v, [INVOICE_DATE_BEFORE_DUE])
        assert outcome.demoted_fields == {"invoice_date", "due_date"}


class TestAllRulesIntegration:
    def test_all_math_and_date_rules_load(self):
        # Smoke: all bundled rules instantiate without error
        assert len(ALL_MATH_RULES) >= 5
        assert len(ALL_DATE_RULES) >= 3
