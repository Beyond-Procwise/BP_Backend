"""If a person keeps telling us this supplier bills in CAD, stop asking.

This is the concrete payoff of the verdict loop: the fourth "$" invoice from a Canadian
supplier resolves itself instead of stopping for review, because three people already
answered the question.
"""
from src.services.extraction_feedback.supplier_currency import (
    default_for, learned_currency, MIN_AGREEMENTS,
)


def _row(supplier="SUP-A", value="CAD", verdict="corrected"):
    return {"supplier_id": supplier, "corrected_value": value, "verdict": verdict}


def test_three_people_saying_CAD_settles_it():
    assert learned_currency([_row()] * MIN_AGREEMENTS) == {"SUP-A": "CAD"}


def test_two_is_not_enough():
    assert learned_currency([_row()] * (MIN_AGREEMENTS - 1)) == {}


def test_a_supplier_people_disagree_about_is_left_alone():
    # Genuinely ambiguous, or the supplier really does bill in both. Either way, guessing
    # is exactly what this whole feature exists to stop.
    rows = [_row(value="CAD")] * MIN_AGREEMENTS + [_row(value="USD")] * MIN_AGREEMENTS
    assert learned_currency(rows) == {}


def test_a_clear_majority_settles_it_even_with_one_dissenter():
    rows = [_row(value="CAD")] * 5 + [_row(value="USD")]
    assert learned_currency(rows) == {"SUP-A": "CAD"}


def test_only_corrections_teach_it():
    # A confirmation says the value we already had was right; it does not tell us what this
    # supplier's currency IS when we had nothing.
    assert learned_currency([_row(verdict="confirmed")] * MIN_AGREEMENTS) == {}


def test_suppliers_are_learned_independently():
    rows = [_row(supplier="SUP-A", value="CAD")] * MIN_AGREEMENTS + \
           [_row(supplier="SUP-B", value="SGD")] * MIN_AGREEMENTS
    assert learned_currency(rows) == {"SUP-A": "CAD", "SUP-B": "SGD"}


def test_a_blank_correction_teaches_nothing():
    assert learned_currency([_row(value=None)] * MIN_AGREEMENTS) == {}


# ---------------------------------------------------------------------------
# default_for: only human-earned confidence answers
# ---------------------------------------------------------------------------

class _Cur:
    """Records every query, and answers the supplier-master lookup with `master`."""

    def __init__(self, master="USD"):
        self.sqls: list[str] = []
        self._master = master
        self.description = [("default_currency",)]

    def execute(self, sql, params=None):
        self.sqls.append(sql)

    def fetchone(self):
        return (self._master,)

    def fetchall(self):
        return []


def test_an_unlearned_supplier_gets_no_answer_even_though_the_master_has_one():
    """proc.bp_supplier.default_currency is set on 5,000 of 5,027 suppliers, 481 of them
    to a dollar currency. If it answered here, every bare-"$" document from any of those
    suppliers would auto-resolve instead of stopping for a person — the pipeline getting
    bolder on its own, off evidence no human ever gave. None sends it to the Action page,
    which is where an unsettled currency belongs."""
    cur = _Cur(master="USD")
    assert default_for(cur, "SUP-A", learned={}) is None
    assert not any("bp_supplier " in s or "default_currency" in s for s in cur.sqls), (
        "the supplier master must not even be consulted for a resolution"
    )


def test_a_learned_supplier_gets_the_answer_people_taught_us():
    assert default_for(_Cur(master="USD"), "SUP-A", learned={"SUP-A": "CAD"}) == "CAD"


def test_the_master_lookup_is_still_reachable_for_a_caller_that_asks_for_it():
    # Kept as a SUGGESTION path (e.g. showing a reviewer what the vendor record says).
    # It is off by default and the extraction pipeline never turns it on.
    assert default_for(_Cur(master="SGD"), "SUP-A", learned={},
                       allow_supplier_master=True) == "SGD"


def test_no_supplier_means_no_answer():
    assert default_for(_Cur(), "", learned={"": "CAD"}) is None
