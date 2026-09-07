"""The answer reports in the currency the user picked on screen.

Display currency is not a server setting. It is a control on Procurement Home's
top bar (`CurrencyChip.jsx`), backed by a shared controller
(`src/lib/currency/displayCurrency.js`) that the SpendIQ dashboard reads too —
so the whole app already agrees on one currency, and an analytic answer that
picked its own would be the one surface disagreeing.

That controller enforces three rules, each of which was a real way to report a
wrong number. This module is the server-side half and must enforce the same
three, because both now convert the same figures:

  1. Convert each currency ONCE from its own native amount. Going via a stored
     USD scalar lost ~5.5% on every sterling figure.
  2. A currency with no rate is EXCLUDED and counted, never assumed 1:1.
  3. A manual override announces itself, so a what-if cannot be mistaken for a
     live-rate figure.
"""

from datetime import datetime, timezone
from decimal import Decimal

from src.services.analytics.currency import NATIVE, DisplayCurrency

# USD-quoted, exactly as proc.bp_fx_rates stores them and GET /fx/rates serves
# them: units of that currency per 1 USD. These are the live values.
RATES = {"USD": Decimal("1.0"), "GBP": Decimal("0.739722"),
         "EUR": Decimal("0.861072"), "INR": Decimal("94.547339"),
         "AED": Decimal("3.6725")}

FETCHED = datetime(2026, 9, 7, 9, 53, tzinfo=timezone.utc)


def _gbp(**kw) -> DisplayCurrency:
    return DisplayCurrency(target="GBP", rates=RATES, fetched_at=FETCHED, **kw)


class TestConversion:
    def test_converting_a_currency_to_itself_is_an_exact_identity(self):
        # Not "close to": the figure must come back untouched, which is only
        # true if the conversion is per-currency rather than via a USD scalar.
        assert _gbp().convert_amount(Decimal("38200824.98"), "GBP") == Decimal("38200824.98")

    def test_a_foreign_amount_converts_through_the_usd_quote(self):
        # 102,142,166.94 INR / 94.547339 * 0.739722
        converted = _gbp().convert_amount(Decimal("102142166.94"), "INR")
        assert converted is not None
        assert Decimal("799000") < converted < Decimal("800000")

    def test_the_inr_top_of_the_table_falls_below_the_gbp_one_once_converted(self):
        # The whole reason this exists. Ranked on raw amounts, Harbourline's
        # 102.1M INR beat every sterling supplier in the corpus; converted, it
        # is under £0.8M and the ranking is a different ranking.
        display = _gbp()
        harbourline = display.convert_amount(Decimal("102142166.94"), "INR")
        sterling = display.convert_amount(Decimal("1200000.00"), "GBP")
        assert harbourline < sterling

    def test_usd_is_one_usd_even_with_no_rate_table(self):
        # Its definition, not a rate we are guessing — and it must hold with the
        # table absent, or a manual GBP rate could not convert a USD amount.
        display = DisplayCurrency(target="USD", rates={}, fetched_at=None)
        assert display.convert_amount(Decimal("100"), "USD") == Decimal("100")

    def test_a_currency_with_no_rate_converts_to_nothing_rather_than_one_to_one(self):
        assert _gbp().convert_amount(Decimal("100"), "XOF") is None

    def test_a_non_positive_rate_is_not_a_rate(self):
        display = DisplayCurrency(target="GBP", rates={"GBP": Decimal("0.74"), "ZWL": Decimal("0")},
                                  fetched_at=FETCHED)
        assert display.convert_amount(Decimal("100"), "ZWL") is None


class TestNativeMode:
    def test_native_converts_nothing(self):
        display = DisplayCurrency(target=NATIVE, rates=RATES, fetched_at=FETCHED)
        assert display.is_native is True
        assert display.convert_amount(Decimal("100"), "INR") is None

    def test_native_totals_are_refused_not_zeroed(self):
        display = DisplayCurrency(target=NATIVE, rates=RATES, fetched_at=FETCHED)
        result = display.total([(Decimal("100"), "INR"), (Decimal("50"), "GBP")])
        assert result.value is None
        assert result.excluded == 2


class TestTotals:
    def test_a_mixed_total_sums_each_currency_from_its_own_amount(self):
        result = _gbp().total([(Decimal("100"), "USD"), (Decimal("100"), "GBP")])
        # 100 USD -> 73.9722 GBP, plus 100 GBP untouched.
        assert result.value == Decimal("173.9722")
        assert result.excluded == 0

    def test_an_unconvertible_row_is_excluded_and_named(self):
        result = _gbp().total([(Decimal("100"), "GBP"), (Decimal("100"), "XOF")])
        assert result.value == Decimal("100")
        assert result.excluded == 1
        assert result.excluded_currencies == ("XOF",)

    def test_nothing_convertible_yields_no_figure_at_all(self):
        # The caller must drop the figure. Rendering it as zero would state that
        # this organisation spent nothing.
        result = _gbp().total([(Decimal("100"), "XOF")])
        assert result.value is None
        assert result.excluded == 1


class TestManualOverrides:
    def test_a_manual_rate_wins_over_the_live_one(self):
        display = _gbp(manual={"GBP": Decimal("0.50")})
        assert display.convert_amount(Decimal("100"), "USD") == Decimal("50.00")

    def test_a_manual_rate_announces_itself(self):
        display = _gbp(manual={"GBP": Decimal("0.50")})
        assert display.is_manual("GBP") is True
        assert "manual rate" in display.rate_note()
        assert "1 USD = 0.50 GBP" in display.rate_note()

    def test_a_live_note_carries_the_time_and_whether_it_is_stale(self):
        assert _gbp().rate_note() == "rates as of 07 Sep 2026, 09:53 (live)"
        assert "(stale)" in _gbp(stale=True).rate_note()

    def test_no_rates_at_all_says_so(self):
        display = DisplayCurrency(target="GBP", rates={}, fetched_at=None)
        assert display.rate_note() == "rates unavailable"
        assert display.rates_unavailable is True
