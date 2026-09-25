"""Ledger writes against the real database, each inside a transaction that is rolled back.
Spec §4, §7."""
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

ACTOR = "pytest-value-ledger"


@pytest.fixture()
def conn():
    from src.services.db import get_conn
    with get_conn() as c:
        c.autocommit = False
        try:
            yield c
        finally:
            c.rollback()


def _finding(cur, status="open", issue_type="duplicate_invoice", raw="500.00") -> int:
    cur.execute(
        "INSERT INTO proc.bp_extraction_discrepancy (doc_type, source_file, doc_pk_candidate, "
        "field_name, raw_value, computed_value, issue_type, severity, status, blocks_promotion) "
        "VALUES ('invoice', 'probe', %s, %s, %s, %s, %s, 'warning', %s, false) "
        "RETURNING discrepancy_id",
        (f"PROBE-{uuid.uuid4().hex[:8]}", f"probe_{uuid.uuid4().hex[:8]}", raw, "+" + raw,
         issue_type, status))
    return cur.fetchone()[0]


def _rows(cur, did):
    cur.execute("SELECT outcome_type, amount, recorded_by FROM proc.bp_value_outcome "
                "WHERE source_type='finding' AND source_id=%s ORDER BY outcome_id", (str(did),))
    return cur.fetchall()


def test_claim_closes_the_finding_and_writes_one_row(conn):
    from src.services import value_ledger as vl
    cur = conn.cursor()
    did = _finding(cur)
    out = vl.record_finding_outcome(did, "claimed", "500", "GBP", actor=ACTOR, conn=conn)
    assert out["state"] == "claimed"
    cur.execute("SELECT status, resolved_by FROM proc.bp_extraction_discrepancy "
                "WHERE discrepancy_id=%s", (did,))
    assert cur.fetchone() == ("resolved", ACTOR)
    assert [r[0] for r in _rows(cur, did)] == ["claimed"]


def test_accepting_the_charge_closes_it_and_records_no_money(conn):
    from src.services import value_ledger as vl
    cur = conn.cursor()
    did = _finding(cur)
    out = vl.record_finding_outcome(did, "accepted", None, None, actor=ACTOR, conn=conn)
    assert out == {"outcome_id": None, "state": "accepted", "amount_gbp": None}
    assert _rows(cur, did) == []


def test_second_outcome_on_same_finding_is_refused(conn):
    from src.services import value_ledger as vl
    cur = conn.cursor()
    did = _finding(cur)
    vl.record_finding_outcome(did, "avoided", "500", "GBP", actor=ACTOR, conn=conn)
    with pytest.raises(vl.LedgerError) as e:
        vl.record_finding_outcome(did, "claimed", "500", "GBP", actor=ACTOR, conn=conn)
    assert e.value.code == "finding_already_moved"
    assert len(_rows(cur, did)) == 1


def test_a_non_money_finding_cannot_record_money(conn):
    from src.services import value_ledger as vl
    cur = conn.cursor()
    did = _finding(cur, issue_type="line_missing_amount")
    with pytest.raises(vl.LedgerError) as e:
        vl.record_finding_outcome(did, "claimed", "10", "GBP", actor=ACTOR, conn=conn)
    assert e.value.code == "not_a_money_finding"


def test_settle_needs_an_open_claim(conn):
    from src.services import value_ledger as vl
    cur = conn.cursor()
    did = _finding(cur)
    with pytest.raises(vl.LedgerError) as e:
        vl.settle_claim(did, "recovered", "500", "GBP", actor=ACTOR, evidence_ref="CN-1", conn=conn)
    assert e.value.code == "no_open_claim"


def test_claim_then_partial_credit(conn):
    from src.services import value_ledger as vl
    cur = conn.cursor()
    did = _finding(cur)
    vl.record_finding_outcome(did, "claimed", "500", "GBP", actor=ACTOR, conn=conn)
    out = vl.settle_claim(did, "recovered", "320", "GBP", actor=ACTOR,
                          evidence_ref="CN-77", conn=conn)
    assert out["state"] == "recovered"
    assert [(r[0], str(r[1])) for r in _rows(cur, did)] == [("claimed", "500.00"),
                                                            ("recovered", "320.00")]
    with pytest.raises(vl.LedgerError):          # settled: no second settle
        vl.settle_claim(did, "claim_dropped", actor=ACTOR, conn=conn)


def test_a_failed_audit_write_leaves_the_finding_open(conn, monkeypatch):
    from src.services import value_ledger as vl
    from src.services.agent_actions import AuditWriteError
    cur = conn.cursor()
    did = _finding(cur)
    cur.execute("SAVEPOINT before_write")

    def _boom(**kw):
        raise AuditWriteError("audit down")
    monkeypatch.setattr(vl, "record_action_or_fail", _boom)
    with pytest.raises(AuditWriteError):
        vl.record_finding_outcome(did, "claimed", "500", "GBP", actor=ACTOR, conn=conn)
    cur.execute("ROLLBACK TO SAVEPOINT before_write")
    cur.execute("SELECT status FROM proc.bp_extraction_discrepancy WHERE discrepancy_id=%s", (did,))
    assert cur.fetchone()[0] == "open"
    assert _rows(cur, did) == []


def test_correction_supersedes_and_becomes_the_state(conn):
    from src.services import value_ledger as vl
    cur = conn.cursor()
    did = _finding(cur)
    first = vl.record_finding_outcome(did, "avoided", "500", "GBP", actor=ACTOR, conn=conn)
    fixed = vl.correct_outcome(first["outcome_id"], "450", "GBP", actor=ACTOR,
                               note="invoice was 450 net", conn=conn)
    hist = vl.finding_outcomes(did, conn=conn)
    assert hist["state"] == "avoided"
    assert hist["history"][-1]["outcome_id"] == fixed["outcome_id"]
    assert str(hist["history"][-1]["amount"]) == "450.00"


def test_prefill_offers_the_findings_own_figure(conn):
    from src.services import value_ledger as vl
    cur = conn.cursor()
    did = _finding(cur, raw="1234.50")
    pre = vl.finding_outcomes(did, conn=conn)["prefill"]
    assert pre["is_money"] is True
    assert pre["amount"] == "1234.50"
