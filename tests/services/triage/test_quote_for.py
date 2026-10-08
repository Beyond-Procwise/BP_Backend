"""A PO's quote reference names a bid; a bare number is its latest version (2026-10-08)."""
from src.services.triage.link import quote_for
from src.services.triage.model import Doc


def _q(*ids):
    return {i: Doc(doc_id=i, doc_type="quote") for i in ids}


QUOTES = _q("APX-PS-5512", "APX-PS-5512 (V2)", "APX-PS-5512 (V3 (BAFO))", "ORB-Q-6612 (V3)")


def test_a_bare_number_is_the_latest_version_not_v1():
    assert quote_for("APX-PS-5512", QUOTES).doc_id == "APX-PS-5512 (V3 (BAFO))"


def test_a_named_version_is_that_version():
    assert quote_for("APX-PS-5512 (V2)", QUOTES).doc_id == "APX-PS-5512 (V2)"


def test_a_named_version_matches_whatever_words_follow_the_number():
    assert quote_for("APX-PS-5512 (V3)", QUOTES).doc_id == "APX-PS-5512 (V3 (BAFO))"
    assert quote_for("ORB-Q-6612 (V3 (BAFO))", QUOTES).doc_id == "ORB-Q-6612 (V3)"


def test_an_unknown_reference_links_nothing():
    assert quote_for("ZZZ-1", QUOTES) is None
    assert quote_for(None, QUOTES) is None
