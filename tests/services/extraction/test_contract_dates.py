"""When a contract starts and ends, read from its own words.

Measured on bp_sqldb 2026-10-04: the real Marketing Agreement promoted nowhere
because `contract_start_date` was NULL and blocking. The document states it three
times over:

    ... is entered into on June 12, 2025 ( ' the Effective Date') by and between ...
    ... (hereinafter referred to as the "Effective Date"). It will end on December 12, 2025
    Name: John Smith  Signature: ______  Date: June 12, 2025

and NOTHING read any of them. `contract_start_date` and `contract_end_date` declare
eight and eight canonical_labels between them in
extraction_schemas/contract.yaml and have **no patterns at all** -- exactly the
structural hole `supplier_id` had. The only path was the context layer, which
returned nothing, so the finding read "context_layer (AgentNick) could not ground a
value in the document". There was no provenance entry for the field: nothing ever
produced a candidate.

NOT a parsing problem: the runtime binder reads every one of these shapes
(`parse_date("June 12, 2025") -> 2025-06-12`, via dateparser 1.4.0 in `.venv`).
The four `TestIsoDate` failures in the suite are the two-venv trap -- `venv` has no
dateparser -- not a defect in the date code.

Offline and model-free.
    CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest \
        tests/services/extraction/test_contract_dates.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.extraction.engineered.contract_dates import (  # noqa: E402
    date_candidates, read_dates,
)

#: The real document's own wording, as the parser renders it -- including the
#: spaced-out quotes docling leaves behind ("( ' the Effective Date')").
REAL = (
    "This Marketing Agreement (hereinafter referred to as the ' Agreement') is "
    "entered into on June 12, 2025 ( ' the Effective Date') by and between BrightWave "
    "Digital Ltd. (hereinafter referred to as the ' Client') with an address of 123 "
    "Innovation Park, London, UK and NexaSpark Marketing Ltd.\n\n"
    "TERM\n\nThis Agreement shall be effective on the date of signing this Agreement "
    "(hereinafter referred to as the “Effective Date”). It will end on "
    "December 12, 2025.\n"
)


def test_the_real_contract_start_date_is_read():
    """The one that blocked promotion on bp_sqldb."""
    d = read_dates(REAL)
    assert d.start == "2025-06-12", d


def test_the_real_contract_end_date_is_read():
    d = read_dates(REAL)
    assert d.end == "2025-12-12", d


def test_a_label_without_a_date_is_not_a_date():
    """THE guard the real document demands: 'shall be effective on the date of
    signing this Agreement (hereinafter referred to as the "Effective Date")' names
    the label twice and states no date at all."""
    text = ("This Agreement shall be effective on the date of signing this Agreement "
            "(hereinafter referred to as the “Effective Date”).\n")
    d = read_dates(text)
    assert d.start is None, d


@pytest.mark.parametrize("sentence,iso", [
    ("Effective Date: 5 January 2026", "2026-01-05"),
    ("Start Date: 2026-01-05", "2026-01-05"),
    ("Commencement Date: 5 January 2026", "2026-01-05"),
    ("Contract Start Date: 5 January 2026", "2026-01-05"),
    ("This Agreement is dated 5 January 2026.", "2026-01-05"),
    ("This Agreement is made on 5 January 2026.", "2026-01-05"),
    ("This Agreement is entered into on 5 January 2026.", "2026-01-05"),
    ("The Services shall commence on 5 January 2026.", "2026-01-05"),
    ("This Agreement is effective from 5 January 2026.", "2026-01-05"),
    ("This Agreement is effective as of 5 January 2026.", "2026-01-05"),
])
def test_each_start_shape_is_read(sentence, iso):
    assert read_dates(sentence).start == iso, sentence


@pytest.mark.parametrize("sentence,iso", [
    ("End Date: 4 January 2029", "2029-01-04"),
    ("Expiry Date: 4 January 2029", "2029-01-04"),
    ("Valid Until: 4 January 2029", "2029-01-04"),
    ("Term End: 4 January 2029", "2029-01-04"),
    ("It will end on 4 January 2029.", "2029-01-04"),
    ("This Agreement shall expire on 4 January 2029.", "2029-01-04"),
    ("The term shall terminate on 4 January 2029.", "2029-01-04"),
])
def test_each_end_shape_is_read(sentence, iso):
    assert read_dates(sentence).end == iso, sentence


def test_the_parsers_one_line_header_is_read():
    """The shape that caught the signature reader: a whole header on one line.
    The end-date label must not swallow the start date's value, or vice versa."""
    text = ("## FRAMEWORK AGREEMENT\n\nFramework Agreement No. FA-2026-0077 "
            "Buyer: BrightWave Digital Ltd. Supplier: NexaSpark Marketing Ltd. "
            "Effective Date: 5 January 2026 End Date: 4 January 2029\n")
    d = read_dates(text)
    assert (d.start, d.end) == ("2026-01-05", "2029-01-04"), d


def test_a_payment_term_is_not_a_date():
    """'within 30 days' is a duration. Nothing here states a date."""
    d = read_dates("Payment shall be made within 30 days of receipt of a valid invoice.\n")
    assert d.start is None and d.end is None, d


def test_a_bare_date_label_is_ignored():
    """A signature block's 'Date:' sits beside every field and means the day
    somebody signed, not the term. Reading it as the contract start is how a
    renewal gets the wrong anniversary."""
    d = read_dates("SIGNATURES\nName: John Smith Signature: ____ Date: June 12, 2025\n")
    assert d.start is None, d


def test_a_term_in_the_wrong_order_yields_neither():
    """A start after its end is a misread, not two facts."""
    d = read_dates("Effective Date: 4 January 2029 End Date: 5 January 2026\n")
    assert d.start is None and d.end is None, d


def test_a_date_that_does_not_parse_is_not_emitted():
    d = read_dates("Effective Date: the first Tuesday after Michaelmas\n")
    assert d.start is None, d


def test_an_unlabelled_date_in_prose_is_not_the_start():
    """The corpus is full of dates. Only a stated start is a start."""
    d = read_dates("The Parties met on 3 February 2026 to discuss renewal.\n")
    assert d.start is None and d.end is None, d


def test_a_label_wins_over_prose():
    text = ("Effective Date: 5 January 2026\n"
            "This Agreement is entered into on 1 March 2026.\n")
    assert read_dates(text).start == "2026-01-05"


def test_candidates_carry_the_schemas_field_names_and_iso_values():
    by_field = {c.field: c.value for c in date_candidates(REAL)}
    assert by_field == {"contract_start_date": "2025-06-12",
                        "contract_end_date": "2025-12-12"}, by_field


def test_the_candidate_keeps_the_literal_text_as_its_evidence():
    """The value is ISO for the column; the span must stay byte-exact source, or
    the grounding gate is being handed something the page does not contain."""
    for c in date_candidates(REAL):
        assert c.span.text in REAL, c
        assert c.source == "date", c


def test_nothing_is_emitted_when_the_document_states_no_dates():
    assert date_candidates("AGREEMENT\nGoverned by the laws of England.\n") == []


def test_the_date_fields_are_barred_from_the_entity_sweep_for_a_contract():
    """They are GPE/none-typed rather than PERSON, so the sweep was never the
    problem here -- but the reader has to be the answer, and a later schema edit
    that adds ner_type_check to a date field must not reopen the hole."""
    from src.services.extraction.dispatch import _contract_party_candidates
    cands, _barred = _contract_party_candidates("contract", REAL)
    by_field = {c.field: c.value for c in cands}
    assert by_field.get("contract_start_date") == "2025-06-12", by_field


# ---------------------------------------------------------------------------
# The reader converts dates ITSELF, on purpose. The first version called the
# pipeline's parse_date (dateparser underneath), which is installed in .venv --
# what the server runs -- and NOT in venv, what pytest runs. So it produced
# nothing at all under test while working in production: the two-venv trap that
# hid the supplier bug for months, pointed at my own tests. These assert the
# conversion directly, so they mean the same thing in both environments.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,iso", [
    ("2026-01-05", "2026-01-05"),
    ("5 January 2026", "2026-01-05"),
    ("5th January, 2026", "2026-01-05"),
    ("June 12, 2025", "2025-06-12"),
    ("Jun 12 2025", "2025-06-12"),
    ("Sept 1, 2026", "2026-09-01"),
    ("12/06/2025", "2025-06-12"),
    ("12.06.2025", "2025-06-12"),
    ("12/06/25", "2025-06-12"),
])
def test_every_accepted_shape_converts_without_dateparser(raw, iso):
    from src.services.extraction.engineered.contract_dates import _iso
    assert _iso(raw) == iso, raw


def test_the_numeric_form_is_read_day_first():
    """A choice, not a guess: it matches what dateparser already answers for this
    UK corpus, so the pipeline cannot change its mind about a date depending on
    which reader saw it."""
    from src.services.extraction.engineered.contract_dates import _iso
    assert _iso("05/01/2026") == "2026-01-05"


@pytest.mark.parametrize("raw", ["31 February 2026", "2026-02-31", "32/01/2026",
                                 "Februbry 5, 2026", "the first Tuesday"])
def test_a_non_date_converts_to_nothing(raw):
    from src.services.extraction.engineered.contract_dates import _iso
    assert _iso(raw) is None, raw


def test_the_reader_agrees_with_the_pipelines_own_binder():
    """Dependency-free must not mean divergent: where dateparser IS available,
    both must give the same answer, or a value's meaning depends on who read it.
    Skips in the test venv, which has no dateparser -- and that skip is the point:
    it is why the tests above do not go through it."""
    try:
        from src.services.extraction_v2.parsers.dates import parse_date
        if parse_date("June 12, 2025") is None:
            pytest.skip("dateparser not installed in this venv")
    except Exception:
        pytest.skip("dateparser not installed in this venv")
    from src.services.extraction.engineered.contract_dates import _iso
    for raw in ("June 12, 2025", "5 January 2026", "2026-01-05", "12/06/2025"):
        assert _iso(raw) == str(parse_date(raw)), raw


# ---------------------------------------------------------------------------
# The backfill decision. Dates differ from the parties in one way that matters:
# the entity sweep never produced them (no patterns, no NER type), so there is no
# wrong value of the sweep's to clear -- only an absent one to fill.
# ---------------------------------------------------------------------------

def _decide(**kw):
    from src.services.extraction.engineered.contract_dates import decide_date_correction
    return decide_date_correction(**kw)


def test_a_stored_row_with_no_dates_is_filled_from_the_document():
    d = _decide(full_text=REAL, stored_start=None, stored_end=None,
                provenance_source=None)
    assert (d.start, d.end) == ("2025-06-12", "2025-12-12"), d
    assert d.changed is True


def test_a_row_already_holding_the_documents_dates_is_left_alone():
    d = _decide(full_text=REAL, stored_start="2025-06-12", stored_end="2025-12-12",
                provenance_source="date")
    assert d.changed is False, d


def test_a_human_confirmed_date_is_never_touched():
    d = _decide(full_text=REAL, stored_start="2024-01-01", stored_end=None,
                provenance_source="hitl")
    assert d.changed is False and d.start == "2024-01-01", d


def test_a_date_the_reader_cannot_find_is_not_cleared():
    """The context layer may legitimately have grounded a date this reader's
    shapes do not cover. Absence of a read is not evidence of a wrong value --
    and unlike the parties, there is no broken sweep here to overrule."""
    d = _decide(full_text="AGREEMENT\nGoverned by English law.\n",
                stored_start="2026-01-05", stored_end=None, provenance_source="context_layer")
    assert d.changed is False and d.start == "2026-01-05", d


def test_a_row_with_no_stored_text_is_left_alone():
    d = _decide(full_text="", stored_start=None, stored_end=None, provenance_source=None)
    assert d.changed is False, d
