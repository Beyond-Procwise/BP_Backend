"""A countable entity must arrive with its count, never as a sample to be counted.

Asked "how many purchase orders are in the system", the ask path answered "There
are 10 purchase orders" -- the length of the top-N sample it was handed. The true
figure is 5,042. Quotes answered 10 against a true 21,054. Invoices answered
12,408 correctly, because its fact query carries COUNT(*).

The principle is already written down beside the invoices query: "If a number can
be counted in SQL, it is counted here." These tests apply it to the rest.
"""
import pytest

from src.services import corpus_facts


class FakeCursor:
    """Records the SQL it is asked to run and returns one plausible row."""

    def __init__(self):
        self.statements: list[str] = []
        self.description = [("col",)]

    def execute(self, sql, params=None):
        self.statements.append(" ".join(sql.split()))

    def fetchall(self):
        return [(1,)]


@pytest.mark.parametrize("intent", ["purchase_orders", "quotes", "invoices"])
def test_a_countable_entity_is_counted_in_sql_not_inferred_from_a_sample(intent):
    cur = FakeCursor()
    facts = corpus_facts._fetch(cur, intent)

    assert facts, f"{intent} returned no facts at all"
    counted = [s for s in cur.statements if "COUNT(*)" in s.upper()]
    assert counted, (
        f"{intent} hands the model only sampled rows, so 'how many' is answered "
        f"by counting the sample. Statements run: {cur.statements}"
    )


@pytest.mark.parametrize("intent", ["purchase_orders", "quotes", "invoices"])
def test_the_total_is_not_truncated_by_a_limit(intent):
    cur = FakeCursor()
    corpus_facts._fetch(cur, intent)

    # Only a GRAND total must be limit-free. A per-supplier count under a top-N
    # is a legitimate ranking, not a truncated total, so it is named by alias.
    for sql in cur.statements:
        if "_TOTAL" in sql.upper() and "COUNT(*)" in sql.upper():
            assert "LIMIT" not in sql.upper(), (
                f"{intent}: a total computed under a LIMIT is not a total: {sql}"
            )
