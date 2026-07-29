"""A quote revision must keep its own identity.

Three revisions of one quote were uploaded. Every source document states its
version explicitly:

    Quote ref:  CPS-Q-3380  (V1)
    Quote ref:  CPS-Q-3380  (V2)
    Quote ref:  CPS-Q-3380  (V3 (BAFO))

The context layer's prompt says "Output the raw token", so whether the version
marker belongs to the identifier was left to the model. It kept "(V2)" and dropped
"(V1)" and "(V3 (BAFO))" — so V1 and V3 shared one primary key, collided on
persist, and last-write-wins picked arbitrarily. Nine uploaded quotes became five
rows, and two suppliers' best-and-final offers were overwritten by their opening
bids (Orbis kept V1 at GBP 3,371,910; its BAFO of GBP 1,096,000, the figure the PO
was actually raised against, was lost).

The persisted convention already exists and the gateway parses it:
    regexp_replace(quote_id, '\\s*\\(V.*\\)$', '')  AS base_quote
    COALESCE((regexp_match(quote_id, '\\(V(\\d+)'))[1]::int, 1) AS version
These tests pin the extraction side to that same convention, derived
deterministically from the document text rather than from model judgement.
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from src.services.extraction.context_layer import canonical_quote_revision


CLEARPATH = "Quote ref:  CPS-Q-3380  (V3 (BAFO))\nTotal 994,000.00"


@pytest.mark.parametrize("extracted,text,expected", [
    # The model dropped the marker; the document still states it.
    ("CPS-Q-3380", "Quote ref:  CPS-Q-3380  (V1)", "CPS-Q-3380"),
    ("CPS-Q-3380", "Quote ref:  CPS-Q-3380  (V2)", "CPS-Q-3380 (V2)"),
    ("CPS-Q-3380", CLEARPATH, "CPS-Q-3380 (V3)"),
    # The model kept it — already canonical, must not be doubled up.
    ("CPS-Q-3380 (V2)", "Quote ref:  CPS-Q-3380  (V2)", "CPS-Q-3380 (V2)"),
    # Nested parenthetical is what defeated the previous handling.
    ("ORB-Q-6612", "Quote ref:  ORB-Q-6612  (V3 (BAFO))", "ORB-Q-6612 (V3)"),
])
def test_revision_is_recovered_from_the_document(extracted, text, expected):
    assert canonical_quote_revision(extracted, text) == expected


def test_version_one_carries_no_suffix():
    """The gateway reads a bare id as version 1; adding "(V1)" would create a
    second identity for the same quote."""
    assert canonical_quote_revision("CPS-Q-3380", "Quote ref: CPS-Q-3380 (V1)") == "CPS-Q-3380"


def test_a_quote_with_no_version_marker_is_untouched():
    assert canonical_quote_revision("QUT30746", "Quote number: QUT30746") == "QUT30746"


def test_a_version_belonging_to_another_quote_is_ignored():
    """The marker must be adjacent to THIS quote's reference, not anywhere on the
    page — a covering letter mentioning another quote must not retag this one."""
    text = "Quote ref: ABC-1 (V2)\nSupersedes quote XYZ-9 (V4)"
    assert canonical_quote_revision("XYZ-9", text) == "XYZ-9 (V4)"
    assert canonical_quote_revision("ABC-1", text) == "ABC-1 (V2)"


def test_missing_inputs_are_returned_unchanged():
    assert canonical_quote_revision(None, "anything") is None
    assert canonical_quote_revision("", "anything") == ""
    assert canonical_quote_revision("CPS-Q-3380", "") == "CPS-Q-3380"
    assert canonical_quote_revision("CPS-Q-3380", None) == "CPS-Q-3380"


def test_double_digit_and_spaced_markers():
    assert canonical_quote_revision("Q-1", "Q-1 (V10)") == "Q-1 (V10)"
    assert canonical_quote_revision("Q-1", "Q-1 ( v2 )") == "Q-1 (V2)"
    assert canonical_quote_revision("Q-1", "Q-1 Rev 3") == "Q-1 (V3)"


def test_the_result_matches_what_the_gateway_parses():
    """Round-trip against the gateway's own base/version regexes."""
    import re
    canon = canonical_quote_revision("ORB-Q-6612", "Quote ref: ORB-Q-6612 (V3 (BAFO))")
    base = re.sub(r"\s*\(V.*\)$", "", canon)
    version = re.search(r"\(V(\d+)", canon)
    assert base == "ORB-Q-6612"
    assert version and int(version.group(1)) == 3
