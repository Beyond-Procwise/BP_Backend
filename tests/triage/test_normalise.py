from decimal import Decimal as D

from src.services.triage.normalise import (
    norm_text, similarity, terms_days, to_confidence, to_decimal)


def test_to_decimal():
    assert to_decimal("1,234.50") == D("1234.50")
    assert to_decimal(12) == D("12")
    assert to_decimal("") is None
    assert to_decimal(None) is None
    assert to_decimal("n/a") is None


def test_to_confidence_reads_both_scales():
    assert to_confidence(D("88.89")) == 0.8889
    assert to_confidence(0.5) == 0.5
    assert to_confidence(None) is None


def test_norm_text():
    assert norm_text("  Steel-Bolts, M8 ") == "steel bolts m8"
    assert norm_text(None) == ""


def test_similarity():
    assert similarity("Widget large", "widget  LARGE") == 1.0
    assert similarity("Steel bolts M8", "Office chair") < 0.4
    assert similarity("", "x") == 0.0


def test_terms_days():
    assert terms_days("30 days — due 30 Jul 2025") == 30
    assert terms_days("Net 30") == 30
    assert terms_days("60 days") == 60
    assert terms_days("on receipt") is None
    assert terms_days(None) is None
