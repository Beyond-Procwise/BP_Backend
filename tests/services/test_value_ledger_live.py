"""Ledger writes against the real database, each inside a transaction that is rolled back.
Spec §4, §7."""
from __future__ import annotations

import os
import sys
import uuid
from contextlib import contextmanager
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


def test_audit_write_failure_raises_auditwriteerror_on_a_caller_owned_conn(conn, monkeypatch):
    """AuditWriteError propagates out of record_finding_outcome when the audit write
    fails, on a caller-owned conn. This does NOT exercise _in_tx's own rollback path
    (conn= is given, so _in_tx just calls the closure) -- the finding-stays-open
    assertion here is only true because the test itself rolls back to its own
    savepoint below. See test_audit_write_failure_on_the_private_path_rolls_back_
    everything for a test that actually exercises _in_tx's rollback."""
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


def test_audit_write_failure_on_the_private_path_rolls_back_everything(conn, monkeypatch):
    """R5: exercise _in_tx's OWN rollback, not the test's. record_finding_outcome is
    called WITHOUT conn, so _in_tx opens its "private" connection via vl.get_conn().
    We monkeypatch vl.get_conn to hand back a thin wrapper over the test's own conn
    (so everything still happens inside this test's rolled-back transaction), whose
    .autocommit setter is a no-op, .commit() is a no-op, and .rollback() runs
    ROLLBACK TO SAVEPOINT -- i.e. exactly what _in_tx calls on failure. If _in_tx's
    except-block rollback were ever removed, this test would go red (see the report
    for the RED run)."""
    from src.services import value_ledger as vl
    from src.services.agent_actions import AuditWriteError
    cur = conn.cursor()
    did = _finding(cur)
    cur.execute("SAVEPOINT before_private_write")

    class _ConnWrapper:
        """Stands in for vl.get_conn()'s private connection, but every operation
        actually runs on the test's own `conn` so it stays inside the outer
        rolled-back transaction."""

        def cursor(self):
            return conn.cursor()

        @property
        def autocommit(self):
            return conn.autocommit

        @autocommit.setter
        def autocommit(self, value):
            pass  # _in_tx sets this; the real conn is already autocommit=False

        def commit(self):
            pass  # never reached on this path (AuditWriteError raises first)

        def rollback(self):
            conn.cursor().execute("ROLLBACK TO SAVEPOINT before_private_write")

    @contextmanager
    def _fake_get_conn():
        yield _ConnWrapper()

    def _boom(**kw):
        raise AuditWriteError("audit down")

    monkeypatch.setattr(vl, "get_conn", _fake_get_conn)
    monkeypatch.setattr(vl, "record_action_or_fail", _boom)

    with pytest.raises(AuditWriteError):
        vl.record_finding_outcome(did, "claimed", "500", "GBP", actor=ACTOR)  # no conn=

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


def test_correcting_a_superseded_claim_after_settlement_is_refused(conn):
    """R4: a claim that has since been settled recovered is no longer the finding's
    current state. Correcting it must not resurrect 'claimed' and must not allow a
    second settle_claim to double-count recovered money."""
    from src.services import value_ledger as vl
    cur = conn.cursor()
    did = _finding(cur)
    claim = vl.record_finding_outcome(did, "claimed", "500", "GBP", actor=ACTOR, conn=conn)
    vl.settle_claim(did, "recovered", "320", "GBP", actor=ACTOR, evidence_ref="CN-1", conn=conn)
    with pytest.raises(vl.LedgerError) as e:
        vl.correct_outcome(claim["outcome_id"], "480", "GBP", actor=ACTOR,
                           note="claim was actually 480", conn=conn)
    assert e.value.code == "not_current"
    recovered_rows = [r for r in _rows(cur, did) if r[0] == "recovered"]
    assert len(recovered_rows) == 1
    # and the already-settled claim can't be settled a second time either
    with pytest.raises(vl.LedgerError):
        vl.settle_claim(did, "recovered", "1", "GBP", actor=ACTOR, evidence_ref="CN-2", conn=conn)


def test_correcting_the_current_recovered_row_still_works(conn):
    """R4: the settle_claim outcome IS the current state, so correcting it is allowed."""
    from src.services import value_ledger as vl
    cur = conn.cursor()
    did = _finding(cur)
    vl.record_finding_outcome(did, "claimed", "500", "GBP", actor=ACTOR, conn=conn)
    recovered = vl.settle_claim(did, "recovered", "320", "GBP", actor=ACTOR,
                                evidence_ref="CN-1", conn=conn)
    fixed = vl.correct_outcome(recovered["outcome_id"], "300", "GBP", actor=ACTOR,
                               note="credit note was actually 300", evidence_ref="CN-1",
                               conn=conn)
    hist = vl.finding_outcomes(did, conn=conn)
    assert hist["state"] == "recovered"
    assert hist["history"][-1]["outcome_id"] == fixed["outcome_id"]
    assert str(hist["history"][-1]["amount"]) == "300.00"


def test_prefill_offers_the_findings_own_figure(conn):
    from src.services import value_ledger as vl
    cur = conn.cursor()
    did = _finding(cur, raw="1234.50")
    pre = vl.finding_outcomes(did, conn=conn)["prefill"]
    assert pre["is_money"] is True
    assert pre["amount"] == "1234.50"


def test_triage_prefill_matches_the_action_centres_own_figure(conn):
    """R16 (2026-09-25): the prefill for a triage-sourced finding must read the SAME
    leading-£ figure the Action Centre shows for it (bp_detection_finding.delta), not
    one arbitrary bp_triage_result line. Read-only proof against a real, live example
    (discrepancy 7295 / finding 8477, INV006000-1 against PO006000): the Action Centre
    shows £226.78; before this fix the prefill offered £56.60 (one line's own exposure).
    Skips rather than fails if the corpus has moved on and 7295 is no longer open."""
    cur = conn.cursor()
    cur.execute("SELECT status FROM proc.bp_extraction_discrepancy WHERE discrepancy_id = 7295")
    row = cur.fetchone()
    if not row or row[0] != "open":
        pytest.skip("discrepancy 7295 no longer exists or is no longer open")
    from src.services import value_ledger as vl
    pre = vl.finding_outcomes(7295, conn=conn)["prefill"]
    assert pre["amount"] == "226.78"
    assert pre["currency"] == "GBP"


# --------------------------------------------------------------------------
# Final-review fixes (R17, R18, R20, minors)
# --------------------------------------------------------------------------

def test_accepting_the_charge_sets_it_aside_and_drops_it_from_found(conn):
    """R17: "accept the charge" is the gateway's dismiss -- status 'ignored' ('set aside'),
    which the summary excludes. 'resolved' would leave the money in Value found and sync
    the detection finding as resolved instead of accepted_risk."""
    from src.services import value_ledger as vl, value_summary_service as vss
    cur = conn.cursor()
    # A GBP invoice nothing else has a finding on (and whose PO carries no PO-level
    # finding), so the probe is priced, live, and moves the total by exactly its amount.
    cur.execute("""
        SELECT i.invoice_id FROM proc.bp_invoice_trgt i
         WHERE i.currency = 'GBP'
           AND NOT EXISTS (SELECT 1 FROM proc.bp_extraction_discrepancy d
                            WHERE d.doc_pk_candidate IN (i.invoice_id, i.po_id))
         LIMIT 1""")
    invoice = cur.fetchone()
    if not invoice:
        pytest.skip("no finding-free GBP invoice in this corpus")
    cur.execute(
        "INSERT INTO proc.bp_extraction_discrepancy (doc_type, source_file, doc_pk_candidate, "
        "field_name, raw_value, computed_value, issue_type, severity, status, blocks_promotion) "
        "VALUES ('invoice', 'probe', %s, %s, '500.00', '+500.00', 'duplicate_invoice', "
        "'warning', 'open', false) RETURNING discrepancy_id",
        (invoice[0], f"probe_{uuid.uuid4().hex[:8]}"))
    did = cur.fetchone()[0]
    before = vss.build_value_summary(conn=conn)
    mine = [f for f in before["findings"] if f["id"] == f"disc:{did}"]
    assert mine and mine[0]["amount_gbp"] == 500.0 and mine[0]["superseded_by"] is None
    vl.record_finding_outcome(did, "accepted", None, None, actor=ACTOR, conn=conn)
    cur.execute("SELECT status, resolved_by FROM proc.bp_extraction_discrepancy "
                "WHERE discrepancy_id=%s", (did,))
    assert cur.fetchone() == ("ignored", ACTOR)
    after = vss.build_value_summary(conn=conn)
    assert not any(f["id"] == f"disc:{did}" for f in after["findings"])
    assert round(before["verified_found_gbp"] - after["verified_found_gbp"], 2) == 500.0
    assert _rows(cur, did) == []


def _superseded_pair(cur):
    """Two probe findings on the same document: dedupe() keeps one and supersedes the
    other under it. Returns (superseded_id, live_id)."""
    from src.services import value_summary_service as vss
    doc = f"PROBE-{uuid.uuid4().hex[:8]}"
    ids = []
    for _ in range(2):
        cur.execute(
            "INSERT INTO proc.bp_extraction_discrepancy (doc_type, source_file, "
            "doc_pk_candidate, field_name, raw_value, computed_value, issue_type, severity, "
            "status, blocks_promotion) VALUES ('invoice', 'probe', %s, %s, '500.00', "
            "'+500.00', 'duplicate_invoice', 'warning', 'open', false) "
            "RETURNING discrepancy_id", (doc, f"probe_{uuid.uuid4().hex[:8]}"))
        ids.append(cur.fetchone()[0])
    by = {i: vss.superseded_by_for(i, cur.connection) for i in ids}
    hidden = [i for i, s in by.items() if s]
    assert len(hidden) == 1, by
    live = [i for i in ids if i != hidden[0]][0]
    assert by[hidden[0]] == f"disc:{live}"
    return hidden[0], live


@pytest.mark.parametrize("outcome", ["avoided", "claimed"])
def test_money_on_a_superseded_finding_is_refused(conn, outcome):
    """R18: a superseded finding's money is already counted under the live finding.
    Stopping or claiming it here too would count the same money twice."""
    from src.services import value_ledger as vl
    cur = conn.cursor()
    hidden, live = _superseded_pair(cur)
    with pytest.raises(vl.LedgerError) as e:
        vl.record_finding_outcome(hidden, outcome, "500", "GBP", actor=ACTOR, conn=conn)
    assert e.value.code == "superseded"
    assert f"disc:{live}" in str(e.value)
    cur.execute("SELECT status FROM proc.bp_extraction_discrepancy WHERE discrepancy_id=%s",
                (hidden,))
    assert cur.fetchone()[0] == "open"
    assert _rows(cur, hidden) == []
    # the live finding still takes the outcome
    assert vl.record_finding_outcome(live, outcome, "500", "GBP", actor=ACTOR,
                                     conn=conn)["state"] == outcome


def test_accepting_a_superseded_finding_is_allowed(conn):
    """R18: accepting records no money, so it cannot double-count."""
    from src.services import value_ledger as vl
    cur = conn.cursor()
    hidden, _ = _superseded_pair(cur)
    assert vl.record_finding_outcome(hidden, "accepted", None, None, actor=ACTOR,
                                     conn=conn)["state"] == "accepted"


def test_triage_prefill_never_falls_back_to_a_unit_count(conn):
    """R20(a): a triage finding's raw/expected values are quantities or unit prices, not
    money. With no £ figure on its detection finding, the buyer types the amount."""
    from src.services import value_ledger as vl
    cur = conn.cursor()
    did = _finding(cur, issue_type="quantity_invoiced_above_po", raw="12")
    pre = vl.finding_outcomes(did, conn=conn)["prefill"]
    assert pre["is_money"] is True
    assert pre["amount"] is None and pre["currency"] is None


def test_realising_a_rejected_opportunity_says_it_has_moved_on(conn):
    from src.services import value_ledger as vl
    cur = conn.cursor()
    oid = f"probe-opp-{uuid.uuid4().hex[:12]}"
    cur.execute("INSERT INTO proc.bp_opportunity (opportunity_id, opportunity_ref_id, stage) "
                "VALUES (%s, %s, 'rejected')", (oid, oid))
    with pytest.raises(vl.LedgerError) as e:
        vl.realise_opportunity(oid, "100", "GBP", actor=ACTOR, conn=conn)
    assert e.value.code == "opportunity_already_moved"
