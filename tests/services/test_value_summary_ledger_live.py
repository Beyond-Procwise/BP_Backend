"""build_value_summary reads a real ledger row, inside a rolled-back transaction."""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in ("1", "true", "yes", "on")
pytestmark = [pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1"),
              pytest.mark.integration]


@pytest.fixture()
def conn():
    from src.services.db import get_conn
    with get_conn() as c:
        c.autocommit = False
        try:
            yield c
        finally:
            c.rollback()


def test_a_recovered_claim_reaches_the_summary(conn):
    from src.services import value_ledger as vl, value_summary_service as vss
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO proc.bp_extraction_discrepancy (doc_type, source_file, doc_pk_candidate, "
        "field_name, raw_value, computed_value, issue_type, severity, status, blocks_promotion) "
        "VALUES ('invoice','probe',%s,%s,'900.00','+900.00','duplicate_invoice','warning','open',false) "
        "RETURNING discrepancy_id", (f"PROBE-{uuid.uuid4().hex[:8]}", f"p_{uuid.uuid4().hex[:8]}"))
    did = cur.fetchone()[0]
    before = vss.build_value_summary(conn=conn)
    vl.record_finding_outcome(did, "claimed", "900", "GBP", actor="pytest-value-ledger", conn=conn)
    vl.settle_claim(did, "recovered", "900", "GBP", actor="pytest-value-ledger",
                    evidence_ref="CN-PROBE", conn=conn)
    after = vss.build_value_summary(conn=conn)
    assert round(after["recovered_gbp"] - before["recovered_gbp"], 2) == 900.0
    assert after["sources"]["ledger"] == "ok"
