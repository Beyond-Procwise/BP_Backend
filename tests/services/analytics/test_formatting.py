"""One formatter, and it must agree with the one already on screen.

The UI has formatted money since July in src/lib/format/currency.js
(`formatCompactCurrency`). Rendering the analytic answer server-side does not
retire that function — Procurement Home and the SpendIQ dashboard still call it
for every figure they draw themselves. So there are now two formatters in the
product, and the only acceptable relationship between them is byte-for-byte
agreement: a supplier's spend must not read £1.1M in the answer and £1,100,000
on the dashboard tile beside it.

Every case below is lifted from the JS module's own documented behaviour, so
this file is the parity contract. If someone changes one side, this goes red.
"""

import pytest

from src.services.analytics.formatting import (
    EMPTY_AMOUNT,
    format_delta,
    format_int,
    format_money,
    format_pct,
)


class TestFormatMoney:
    def test_below_one_thousand_renders_in_full_with_no_suffix(self):
        assert format_money(639, "GBP") == "£639"

    def test_thousands_scale_to_k_with_one_decimal(self):
        assert format_money(15_000, "GBP") == "£15.0K"

    def test_millions_scale_to_m_with_one_decimal(self):
        assert format_money(1_100_000, "GBP") == "£1.1M"

    def test_pennies_never_survive_onto_a_hundred_million_figure(self):
        # The defect that started this: £102,142,166.94 printed to the penny.
        assert format_money(102_142_166.94, "GBP") == "£102.1M"

    def test_rollover_promotes_to_the_next_tier_rather_than_printing_1000K(self):
        # 999,950 / 1e3 rounds to 1000.0 — four integer digits. Promote to M.
        assert format_money(999_950, "GBP") == "£1.0M"

    def test_a_missing_amount_is_an_em_dash_never_a_zero(self):
        assert format_money(None, "GBP") == EMPTY_AMOUNT
        assert format_money(float("nan"), "GBP") == EMPTY_AMOUNT
        assert format_money(float("inf"), "GBP") == EMPTY_AMOUNT

    def test_negative_amounts_carry_the_sign_before_the_symbol(self):
        assert format_money(-1_100_000, "GBP") == "-£1.1M"

    @pytest.mark.parametrize(
        "currency,expected",
        [("GBP", "£1.1M"), ("USD", "$1.1M"), ("EUR", "€1.1M"), ("JPY", "¥1.1M"),
         ("NZD", "NZ$1.1M"), ("AUD", "A$1.1M")],
    )
    def test_known_symbols_match_the_client(self, currency, expected):
        assert format_money(1_100_000, currency) == expected

    def test_an_unsymbolled_currency_is_spelled_out_rather_than_given_a_pound_sign(self):
        # A ₹ amount stamped with £ is a different number, not a formatting nicety.
        assert format_money(1_100_000, "INR") == "INR 1.1M"

    def test_no_currency_renders_the_number_bare(self):
        assert format_money(1_100_000, None) == "1.1M"
        assert format_money(1_100_000, "") == "1.1M"

    def test_grouping_is_en_gb(self):
        assert format_money(999, "GBP") == "£999"
        assert format_money(-999, "GBP") == "-£999"


class TestFormatPct:
    def test_one_decimal_by_default(self):
        assert format_pct(23.4) == "23.4%"

    def test_rounds_rather_than_truncates(self):
        assert format_pct(23.45) == "23.5%"

    def test_missing_is_an_em_dash(self):
        assert format_pct(None) == EMPTY_AMOUNT


class TestFormatInt:
    def test_thousands_are_grouped(self):
        assert format_int(2005) == "2,005"

    def test_missing_is_an_em_dash(self):
        assert format_int(None) == EMPTY_AMOUNT


class TestFormatDelta:
    def test_a_rise_carries_an_explicit_plus(self):
        assert format_delta(12.4) == "+12.4%"

    def test_a_fall_carries_a_minus(self):
        assert format_delta(-8.0) == "-8.0%"

    def test_no_change_is_signed_zero_not_bare_zero(self):
        assert format_delta(0) == "+0.0%"

    def test_missing_is_an_em_dash(self):
        assert format_delta(None) == EMPTY_AMOUNT
