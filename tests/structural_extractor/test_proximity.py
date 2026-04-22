from src.services.structural_extractor.discovery.proximity import inferred_label, arithmetic_fit
from src.services.structural_extractor.discovery.type_entities import Candidate
from src.services.structural_extractor.parsing.model import Token, BBox


def _tok(text, x, y, order):
    return Token(text=text, anchor=BBox(1, x, y, x + 50, y + 10), order=order)


def test_inferred_label_from_same_line_left():
    tokens = [
        _tok("Invoice", 0, 100, 0),
        _tok("Date:", 60, 100, 1),
        _tok("01/07/2022", 120, 100, 2),
    ]
    cand = Candidate(text="01/07/2022", tokens=[tokens[2]], parsed_value=None)
    label = inferred_label(cand, tokens)
    assert "Date" in label


def test_arithmetic_fit_reconciles():
    assert arithmetic_fit(8333.0, 1666.60, 9999.60) == 1.0
    assert arithmetic_fit(8333.0, 1666.60, 10000.0) == 0.0
