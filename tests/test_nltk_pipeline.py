import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.nltk_pipeline import NLTKProcessor


@pytest.fixture(scope="module")
def nltk_processor():
    processor = NLTKProcessor()
    if not processor.available:
        pytest.skip("NLTK resources unavailable in test environment")
    return processor


def test_preprocess_extracts_keywords_and_phrases(nltk_processor):
    text = "We need guidance on handling late payment charges for supplier invoices."
    features = nltk_processor.preprocess(text)
    assert features.keywords, "Expected keywords to be extracted"
    assert any("payment" in keyword for keyword in features.keywords)
    assert any("supplier" in phrase.lower() for phrase in features.key_phrases)


def test_postprocess_strips_filler_and_adds_no_tone_sentence(nltk_processor):
    """Filler goes; the answer starts on the answer.

    This test used to require the opposite of its second assertion — that a
    negative sentiment score prepend "I understand this situation ... may feel
    frustrating". That sentence was pasted onto every reply the sentiment model
    scored below -0.2, which on procurement questions (late fees, disputes,
    overspend) is most of them. Tone is the model's to set from the question in
    front of it, not something to bolt on from a VADER score.
    """
    draft = "Thanks for flagging this. Here's what I can confirm. late fees apply for overdue invoices"
    cleaned = nltk_processor.postprocess(draft, sentiment={"compound": -0.6})
    assert "thanks for flagging" not in cleaned.lower()
    assert "i understand this situation" not in cleaned.lower()
    assert cleaned.lower().startswith("late fees apply")
