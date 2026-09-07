"""Reading the rows the ranking is computed from.

Two properties are worth pinning here, because both have gone wrong in this
codebase before:

  * **Grouped by supplier AND currency, never pre-summed.** A supplier's GBP and
    USD invoices must arrive as separate rows, or the conversion cannot follow
    rule 1 of the display-currency contract — convert each currency once, from
    its own native amount.
  * **No LIMIT on the aggregate.** ``corpus_facts`` caps its lists at ten rows
    deliberately, and a share computed against a ten-row sample would state that
    the top supplier holds a third of all spend when it holds a fraction of that.
    The denominators here are the population, so the aggregate is not truncated.
"""

from datetime import date
from decimal import Decimal

from src.services.analytics.period import Period
from src.services.analytics.repository import (
    fetch_population,
    fetch_supplier_spend,
)

PERIOD = Period(date(2026, 4, 1), date(2026, 9, 7), "FY26 YTD")


class FakeCursor:
    """Records what it was asked, returns what it was primed with."""

    def __init__(self, rows, description):
        self._rows = rows
        self.description = [(name,) for name in description]
        self.executed = []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0]


class TestFetchSupplierSpend:
    def _cursor(self):
        return FakeCursor(
            rows=[("s-1", "Split Ltd", "GBP", Decimal("100"), 2),
                  ("s-1", "Split Ltd", "USD", Decimal("100"), 1)],
            description=["supplier_id", "supplier_name", "currency", "amount", "invoices"],
        )

    def test_each_currency_arrives_as_its_own_row(self):
        rows = fetch_supplier_spend(self._cursor(), PERIOD)
        assert [(r.supplier_id, r.currency, r.amount) for r in rows] == [
            ("s-1", "GBP", Decimal("100")), ("s-1", "USD", Decimal("100"))]

    def test_the_period_is_bound_as_parameters_never_interpolated(self):
        cursor = self._cursor()
        fetch_supplier_spend(cursor, PERIOD)
        sql, params = cursor.executed[0]
        assert params == (PERIOD.start, PERIOD.end)
        assert "2026" not in sql

    def test_the_aggregate_is_not_truncated(self):
        cursor = self._cursor()
        fetch_supplier_spend(cursor, PERIOD)
        sql, _ = cursor.executed[0]
        assert "LIMIT" not in sql.upper()

    def test_the_grouping_keeps_currency(self):
        cursor = self._cursor()
        fetch_supplier_spend(cursor, PERIOD)
        sql, _ = cursor.executed[0]
        assert "GROUP BY" in sql.upper()
        assert "currency" in sql


class TestFetchPopulation:
    def test_the_population_is_counted_not_inferred_from_the_rows(self):
        # A count taken by tallying a top-N list is how "647 unresolved cases"
        # was reported against a real 685.
        cursor = FakeCursor(rows=[(3510, 12408)], description=["suppliers", "invoices"])
        suppliers, invoices = fetch_population(cursor, PERIOD)
        assert (suppliers, invoices) == (3510, 12408)

    def test_the_period_is_bound_as_parameters(self):
        cursor = FakeCursor(rows=[(0, 0)], description=["suppliers", "invoices"])
        fetch_population(cursor, PERIOD)
        assert cursor.executed[0][1] == (PERIOD.start, PERIOD.end)
