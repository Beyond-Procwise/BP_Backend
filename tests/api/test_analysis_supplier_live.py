"""A saved analysis names the supplier of every document it holds.

    PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest \
        tests/api/test_analysis_supplier_live.py

The saved-analysis document list carried type, reference, file and outcome but
no supplier, so the report's "How these documents connect" map showed
"Supplier not identified" on every bid -- even though each quote's supplier is
printed on it and sits in proc.bp_deal_documents. Test Deal TESTDEAL2026072901
(three competing SaaS quotes) is the case that surfaced it.
"""

from __future__ import annotations

import os

import pytest

from api.routers import analysis as mod

_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in (
    "1", "true", "yes", "on")
pytestmark = pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")

_ANALYSIS = "f38392b8-20a7-4cb1-b9ea-a224849e142d"

# As printed on the quotes themselves (checked against the .xlsx files).
_EXPECTED = {
    "CPS-Q-3380": "ClearPath Systems Ltd",
    "NXF-2024-441": "NexusFlow Platform Ltd",
    "ORB-Q-6612": "Orbis Platform Solutions Ltd",
}


def test_every_quote_in_the_analysis_names_its_supplier():
    body = mod.get_analysis(_ANALYSIS)
    quotes = [d for d in body["documents"] if d["doc_type"] == "quote"]
    assert quotes, "the fixture analysis has no quotes"
    for d in quotes:
        base = d["doc_pk"].split(" (")[0]
        assert d.get("supplier_name") == _EXPECTED[base], d
