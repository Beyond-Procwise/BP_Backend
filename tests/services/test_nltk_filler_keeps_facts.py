"""The filler-sentence filter must never delete a sentence that carries a fact.

`_FILLER_PREFIXES` listed "based on", and `postprocess` drops any sentence whose
prefix matches. But "Based on the records, you have 19 invoices" is not filler —
it is the answer, and "Based on..." is exactly how a grounded model introduces
its evidence. The filter was therefore deleting the best-grounded sentence in the
reply, on every Ask request. Users saw "This total includes 7 from X and 7 from
Y..." with the total itself silently removed.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import pytest

from services.nltk_pipeline import NLTKProcessor


@pytest.fixture(scope="module")
def proc() -> NLTKProcessor:
    return NLTKProcessor()


def test_the_grounded_answer_sentence_survives(proc):
    """The exact regression: the sentence with the number must not vanish."""
    text = (
        "Happy to help! Based on the verified records in your extracted documents, "
        "you have exactly 19 invoices in total. This total includes 7 invoices from "
        "Veruca Organic Nuts."
    )
    out = proc.postprocess(text)
    assert "19 invoices" in out, (
        "the sentence carrying the answer was deleted as 'filler' — it began with "
        f"'Based on'. Got: {out!r}"
    )


def test_a_sentence_with_a_figure_is_never_dropped_as_filler(proc):
    """Insurance against the other prefixes eating facts too."""
    text = "As an AI, I found 42 purchase orders worth GBP 101,120.00."
    out = proc.postprocess(text)
    assert "42" in out and "101,120" in out


def test_genuine_filler_with_no_facts_is_still_removed(proc):
    text = "Sure, I can help. The contract expires in March."
    out = proc.postprocess(text)
    assert "The contract expires in March." in out
    assert "Sure, I can help" not in out


def test_empty_input_is_handled(proc):
    assert proc.postprocess("") == ""
