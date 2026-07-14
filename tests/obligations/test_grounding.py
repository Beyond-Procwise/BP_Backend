"""The grounding guard is this feature's only defence against an invented obligation.

Every test here is a fabrication that must NOT be persisted as fact.
"""
from src.services.obligations.grounding import is_quote_grounded

# Real text from the DWP PS1 procurement contract (gov.uk).
CONTRACT = """
27.4. Time of delivery shall be of the essence and if the Contractor fails to deliver
the Goods within the time promised or specified in the Specification, the Authority may
release itself from any obligation to accept and pay for the Goods.

27.7. The risk in any over delivered Goods shall remain with the Contractor.

28.1. Subject to condition 27.5, risk in the Goods shall pass to the Authority on delivery.
"""


def test_real_clause_is_grounded():
    assert is_quote_grounded(
        "The risk in any over delivered Goods shall remain with the Contractor.", CONTRACT
    )


def test_real_clause_survives_formatting_drift():
    """Docling/OCR reflow whitespace and case. That must not un-ground a true quote."""
    assert is_quote_grounded(
        "the risk in ANY  over   delivered goods shall remain with the contractor", CONTRACT
    )


def test_fabricated_clause_is_blocked():
    """A wholly invented obligation, citing clauses that DO exist in the document.

    This exact sentence passes extraction_v3's field guard, because its digit
    signature (271272) occurs in the document's digit stream.
    """
    assert not is_quote_grounded(
        "The Contractor shall indemnify the Authority in full under conditions 27.1 "
        "and 27.2 for all consequential loss.",
        CONTRACT,
    )


def test_bare_clause_number_is_blocked():
    """The model reaches for this when asked for a quote. It proves nothing."""
    assert not is_quote_grounded("27.4", CONTRACT)


def test_elided_quote_is_blocked():
    """An ellipsis means the model paraphrased. The sentence was never in the document."""
    assert not is_quote_grounded(
        "Where the Goods... fail to be delivered on the due date, the Authority may terminate.",
        CONTRACT,
    )


def test_empty_quote_is_blocked():
    assert not is_quote_grounded("", CONTRACT)


def test_missing_document_blocks_rather_than_allows():
    """The field guard allows when it cannot verify. For obligations that is backwards:
    no document means no proof, and no proof means no obligation."""
    assert not is_quote_grounded("The risk shall remain with the Contractor at all times.", "")
