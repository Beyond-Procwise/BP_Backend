"""Grounding a supplier's sentence, without opening the digit hole.

The obligations guard requires 8+ words because a contract sentence is never
shorter. A supplier reply legitimately is: "We can offer 94,000.00 GBP." is five
words and is the single most important sentence in the thread. The floor has to be
a parameter, not a constant -- and lowering it must not reintroduce
extraction_v3.is_value_grounded's digit-signature fallback, which passes wholly
invented sentences.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from src.services.obligations.grounding import is_quote_grounded

REPLY = (
    "Thank you for the proposal. We can offer 94,000.00 GBP with 45 day payment "
    "terms and a 14 day lead time."
)


def test_short_supplier_sentence_grounds_with_a_lower_floor():
    assert is_quote_grounded("We can offer 94,000.00 GBP.", REPLY, min_words=4) is False
    # The trailing full stop is not in the source; the sentence itself is.
    assert is_quote_grounded("We can offer 94,000.00 GBP", REPLY, min_words=4) is True


def test_default_floor_is_unchanged_for_contract_callers():
    eight_words = "The Contractor shall indemnify the Authority in full"
    assert is_quote_grounded(eight_words, eight_words) is True
    assert is_quote_grounded("shall indemnify the Authority", eight_words) is False


def test_a_fabricated_sentence_never_grounds():
    # Digits present in the source, sentence not. This is the exact failure mode the
    # obligations guard exists to prevent -- it must survive the new parameter.
    assert is_quote_grounded(
        "We will absorb the 94,000.00 GBP increase entirely", REPLY, min_words=4
    ) is False


def test_empty_source_never_grounds():
    assert is_quote_grounded("We can offer 94,000.00 GBP", "", min_words=4) is False
