"""The period an answer covers, resolved before anything is counted.

Scope is mandatory in the contract, and the period is the half of it a reader
checks first. It is computed here from a clock, never described by a model, and
never left implicit — "spend" with no period is the question, not the answer.

The fiscal year starts in April by default (the convention this organisation
reports on) and is settable, because it is a business fact and not a law.
"""

from datetime import date

from src.services.analytics.period import Period, fiscal_year_to_date, prior_year_equivalent


class TestFiscalYearToDate:
    def test_a_september_date_sits_in_the_fiscal_year_that_opened_in_april(self):
        period = fiscal_year_to_date(date(2026, 9, 7))
        assert period.start == date(2026, 4, 1)
        assert period.end == date(2026, 9, 7)
        assert period.label == "FY26 YTD"

    def test_a_january_date_belongs_to_the_fiscal_year_that_opened_last_april(self):
        period = fiscal_year_to_date(date(2026, 1, 15))
        assert period.start == date(2025, 4, 1)
        assert period.label == "FY25 YTD"

    def test_the_first_day_of_the_fiscal_year_opens_it_rather_than_closing_the_last(self):
        period = fiscal_year_to_date(date(2026, 4, 1))
        assert period.start == date(2026, 4, 1)
        assert period.label == "FY26 YTD"

    def test_a_calendar_year_organisation_can_say_so(self):
        period = fiscal_year_to_date(date(2026, 9, 7), fy_start_month=1)
        assert period.start == date(2026, 1, 1)
        assert period.label == "FY26 YTD"


class TestPriorYearEquivalent:
    def test_the_comparison_window_is_the_same_span_one_year_back(self):
        prior = prior_year_equivalent(Period(date(2026, 4, 1), date(2026, 9, 7), "FY26 YTD"))
        assert prior.start == date(2025, 4, 1)
        assert prior.end == date(2025, 9, 7)
        assert prior.label == "FY25 YTD"

    def test_a_leap_day_end_falls_back_to_the_28th_rather_than_erroring(self):
        prior = prior_year_equivalent(Period(date(2024, 3, 1), date(2024, 2, 29), "x"))
        assert prior.end == date(2023, 2, 28)
