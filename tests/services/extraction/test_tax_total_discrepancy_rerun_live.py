"""Re-reading a document whose totals don't reconcile must not abort promotion.

    PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest \
        tests/services/extraction/test_tax_total_discrepancy_rerun_live.py

_check_tax_total_consistency INSERTed its finding with no ON CONFLICT. The open
finding for (doc_type, doc_pk, issue_type, field_name) is unique
(ix_bp_extraction_discrepancy_open_key), so the second read of the same quote --
Orbis ORB-Q-6612 on Test Deal TESTDEAL2026072901 -- raised UniqueViolation and
the whole promotion failed, leaving the stale first read in _trgt. A re-read
refreshes the open finding instead, as persistence.write_discrepancies does.

Runs inside a transaction that is rolled back: nothing is left behind.
"""

from __future__ import annotations

import os
import uuid

import pytest

from src.services.db import get_conn
from src.services.extraction.promotion import _check_tax_total_consistency

_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in (
    "1", "true", "yes", "on")
pytestmark = pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")


def test_a_second_read_refreshes_the_open_finding_instead_of_failing():
    pk = f"TEST-RERUN-{uuid.uuid4().hex[:8]}"
    # net + VAT = 1,339,200, but the stored total says 1,000,000.
    row = {"quote_id": pk, "source_file": "test.xlsx", "total_amount": "1116000",
           "tax_amount": "223200", "total_amount_incl_tax": "1000000"}
    with get_conn() as conn:
        conn.autocommit = False
        try:
            cur = conn.cursor()
            assert _check_tax_total_consistency(cur, "quote", 1, dict(row)) >= 1
            row["total_amount_incl_tax"] = "1200000"
            _check_tax_total_consistency(cur, "quote", 2, dict(row))
            cur.execute(
                "SELECT raw_id, computed_value FROM proc.bp_extraction_discrepancy "
                " WHERE doc_pk_candidate = %s AND issue_type = 'sum_mismatch'", (pk,))
            rows = cur.fetchall()
            assert len(rows) == 1, rows
            assert rows[0][0] == 2 and rows[0][1].startswith("1200000"), rows
        finally:
            conn.rollback()
