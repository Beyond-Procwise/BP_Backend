"""An end date worked out from a stated duration, and marked as derived.

Measured on bp_testdb 2026-10-07: framework FA-2026-0042 (GBP 750,000) has no
contract_end_date, because clause 2 says

    This Framework Agreement shall commence on 5 January 2026 and shall continue
    for a period of thirtysix (36) months unless terminated earlier ...

and no line anywhere prints an end date. The reader only knew written-out dates, so
the contract never reached the renewals buckets. Its sibling FA-2026-0077 prints
"End Date: 4 January 2029" and extracts fine.

The rule: a duration stated in the same sentence as the start gives
start + N months - 1 day, and the value is recorded as DERIVED (its own
pattern_name in the field's provenance), never as a printed date.

    CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest \
        tests/services/extraction/test_contract_term_duration.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.extraction.engineered.contract_dates import (  # noqa: E402
    DERIVED_END_PATTERN, date_candidates, decide_date_correction, read_dates,
)

REAL_CLAUSE = (
    "## 2. TERM\n\nThis Framework Agreement shall commence on 5 January 2026 and "
    "shall continue for a period of thirtysix (36) months unless terminated earlier "
    "in accordance with clause 9.\n\n## 3. CHARGES\n\nThe rates set out in Schedule 1 "
    "are fixed for the first twelve (12) months.\n"
)


def test_the_real_clause_gives_the_end_date_of_its_sibling():
    d = read_dates(REAL_CLAUSE)
    assert (d.start, d.end) == ("2026-01-05", "2029-01-04"), d
    assert d.end_derived is True


@pytest.mark.parametrize("clause,end", [
    ("commence on 5 January 2026 and shall continue for a period of 36 months", "2029-01-04"),
    ("commence on 5 January 2026 and continue for 36 months", "2029-01-04"),
    ("commence on 5 January 2026 and shall continue for a period of thirty-six (36) months", "2029-01-04"),
    ("commence on 5 January 2026 and shall continue for a period of thirty six months", "2029-01-04"),
    ("commence on 5 January 2026 and shall continue for three (3) years", "2029-01-04"),
    ("commence on 5 January 2026 and shall continue for a period of 2 years", "2028-01-04"),
    ("commence on 1 March 2026 and shall remain in force for twelve (12) months", "2027-02-28"),
    ("commence on 5 January 2026 and shall continue for an initial term of 24 months", "2028-01-04"),
])
def test_each_duration_wording_is_read(clause, end):
    d = read_dates(f"This Agreement shall {clause}.")
    assert d.end == end, (clause, d)
    assert d.end_derived is True


def test_a_printed_end_date_beats_a_duration():
    text = ("Effective Date: 5 January 2026 End Date: 4 January 2029. This Agreement "
            "shall commence on 5 January 2026 and shall continue for 12 months.")
    d = read_dates(text)
    assert d.end == "2029-01-04"
    assert d.end_derived is False


def test_a_printed_end_date_is_not_marked_derived():
    d = read_dates("Effective Date: 5 January 2026 End Date: 4 January 2029")
    assert d.end_derived is False


@pytest.mark.parametrize("sentence", [
    # A duration that is not the term.
    "Payment shall be made within 30 days of receipt of a valid invoice.",
    "The rates are fixed for the first twelve (12) months.",
    "Either party may terminate on 3 months notice.",
    "This Agreement shall commence on 5 January 2026. A notice period of 3 months applies.",
    "This Agreement shall commence on 5 January 2026 and renews for successive periods of 12 months.",
    # No start to count from.
    "This Agreement shall continue for a period of 36 months.",
    # Words and digits disagree: refuse rather than pick one.
    "This Agreement shall commence on 5 January 2026 and shall continue for thirty (36) months.",
    # A number we cannot read.
    "This Agreement shall commence on 5 January 2026 and shall continue for several months.",
    # Absurd.
    "This Agreement shall commence on 5 January 2026 and shall continue for 9999 years.",
    # Month lengths differ, so "one month from the 31st" has no single answer.
    "This Agreement shall commence on 31 January 2026 and shall continue for 1 month.",
])
def test_a_duration_that_is_not_the_term_gives_no_end(sentence):
    d = read_dates(sentence)
    assert d.end is None, (sentence, d)
    assert d.end_derived is False


def test_the_derived_candidate_is_marked_in_its_provenance_name():
    by_field = {c.field: c for c in date_candidates(REAL_CLAUSE)}
    end = by_field["contract_end_date"]
    assert end.value == "2029-01-04"
    assert end.pattern_name == DERIVED_END_PATTERN
    # The start is a printed date and is NOT marked derived.
    assert by_field["contract_start_date"].pattern_name != DERIVED_END_PATTERN
    # The evidence is the document's own words, so the grounding gate is not handed
    # a string the page does not contain.
    assert end.span.text in REAL_CLAUSE
    assert "36" in end.span.text


def test_a_printed_end_candidate_keeps_the_ordinary_pattern_name():
    by_field = {c.field: c for c in date_candidates("Effective Date: 5 January 2026 End Date: 4 January 2029")}
    assert by_field["contract_end_date"].pattern_name != DERIVED_END_PATTERN


def test_the_backfill_fills_a_null_end_and_says_it_is_derived():
    d = decide_date_correction(full_text=REAL_CLAUSE, stored_start="2026-01-05",
                               stored_end=None, provenance_source=None)
    assert d.end == "2029-01-04" and d.changed is True
    assert d.end_derived is True


def test_the_backfill_does_not_overwrite_a_human_confirmed_end():
    d = decide_date_correction(full_text=REAL_CLAUSE, stored_start="2026-01-05",
                               stored_end="2028-06-30", provenance_source="human")
    assert d.end == "2028-06-30" and d.changed is False


def test_a_derived_end_never_replaces_a_stored_end():
    """A stored value may have been grounded by the context layer or confirmed by
    a person; arithmetic on a duration is weaker evidence than either."""
    d = decide_date_correction(full_text=REAL_CLAUSE, stored_start="2026-01-05",
                               stored_end="2029-01-05", provenance_source="regex")
    assert d.end == "2029-01-05" and d.end_derived is False
