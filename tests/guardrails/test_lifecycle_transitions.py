"""Findings and opportunities move only along declared edges, and the database enforces it.

Before deploy/sql/2026-09-17_bp_lifecycle_transitions.sql either table accepted any
status from any status:

  * a realised opportunity could be sent back to identified, erasing the savings claim;
  * a finding one person resolved could be silently re-resolved by a second person,
    overwriting who closed it and with what value;
  * automatic writers overruled people -- the PO reconciler resolved findings a person
    had deliberately ignored, and the miner sync could mark a realised opportunity
    rejected.

The guard is a trigger, not a Python check, because the Node gateway writes finding
statuses to the same table with no from-state check of its own. The allowed moves live
in ONE place, proc.bp_lifecycle_transition, which the trigger and the Python writers
both read.

Nothing real is touched. Every test inserts its own probe rows and rolls back.
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in (
    "1", "true", "yes", "on")
pytestmark = [
    pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1"),
    pytest.mark.integration,
]

# The SQLSTATE the trigger raises, so callers can tell a refused move from any other
# failure. Class "BP" is not used by Postgres.
REFUSED = "BP409"


@pytest.fixture()
def conn():
    """get_conn() is AUTOCOMMIT, on which rollback is a no-op -- switched off here, or
    every probe below would be a real write."""
    from src.services.db import get_conn

    with get_conn() as c:
        c.autocommit = False
        try:
            yield c
        finally:
            c.rollback()


def _finding(cur, status="open", issue_type="lifecycle_probe", raw_value=None) -> int:
    cur.execute(
        "INSERT INTO proc.bp_extraction_discrepancy "
        "(doc_type, source_file, doc_pk_candidate, field_name, issue_type, severity, "
        " status, raw_value) "
        "VALUES ('invoice', 'lifecycle-probe', %s, 'po_id', %s, 'info', %s, %s) "
        "RETURNING discrepancy_id",
        # Unique per probe: at most one OPEN finding per (doc, issue, field) is allowed.
        (f"lifecycle-probe-{uuid.uuid4().hex[:12]}", issue_type, status, raw_value),
    )
    return cur.fetchone()[0]


def _opportunity(cur, stage="identified") -> str:
    oid = f"lifecycle-probe-{uuid.uuid4().hex[:12]}"
    cur.execute(
        "INSERT INTO proc.bp_opportunity (opportunity_id, opportunity_ref_id, stage) "
        "VALUES (%s, %s, %s)",
        (oid, oid, stage),
    )
    return oid


def _refused(conn, sql, params):
    """Run one UPDATE inside a savepoint; return the refusal, or None if it went through."""
    import psycopg2

    cur = conn.cursor()
    cur.execute("SAVEPOINT probe")
    try:
        cur.execute(sql, params)
    except psycopg2.Error as exc:
        cur.execute("ROLLBACK TO SAVEPOINT probe")
        if exc.pgcode != REFUSED:
            raise  # refused for some other reason -- that is a broken test, not a pass
        return exc
    cur.execute("RELEASE SAVEPOINT probe")
    return None


# --------------------------------------------------------------------------- opportunity

_OPP_ALLOWED = [
    ("identified", "negotiation"), ("identified", "agreed"), ("identified", "realised"),
    ("identified", "closed"), ("identified", "rejected"),
    ("negotiation", "agreed"), ("negotiation", "realised"),
    ("negotiation", "closed"), ("negotiation", "rejected"),
    ("agreed", "realised"), ("agreed", "closed"), ("agreed", "rejected"),
]
_OPP_REFUSED = [
    ("realised", "identified"), ("realised", "agreed"), ("realised", "rejected"),
    ("realised", "closed"),
    ("agreed", "negotiation"), ("negotiation", "identified"),
    ("closed", "identified"), ("closed", "negotiation"),
    ("rejected", "identified"), ("rejected", "closed"),
]


@pytest.mark.parametrize("frm,to", _OPP_ALLOWED)
def test_an_opportunity_moves_forward(conn, frm, to):
    cur = conn.cursor()
    oid = _opportunity(cur, frm)
    assert _refused(conn, "UPDATE proc.bp_opportunity SET stage=%s WHERE opportunity_id=%s",
                    (to, oid)) is None


@pytest.mark.parametrize("frm,to", _OPP_REFUSED)
def test_an_opportunity_never_moves_back_or_out_of_a_final_stage(conn, frm, to):
    cur = conn.cursor()
    oid = _opportunity(cur, frm)
    exc = _refused(conn, "UPDATE proc.bp_opportunity SET stage=%s WHERE opportunity_id=%s",
                   (to, oid))
    assert exc is not None, f"opportunity moved {frm} -> {to}"
    assert frm in str(exc) and to in str(exc), f"refusal does not say what moved: {exc}"


def test_a_realised_opportunity_can_still_have_its_savings_edited(conn):
    """A same-stage write is not a move; the savings figure is corrected at realised."""
    cur = conn.cursor()
    oid = _opportunity(cur, "realised")
    assert _refused(conn, "UPDATE proc.bp_opportunity SET stage='realised', "
                          "realised_savings_gbp=123.45 WHERE opportunity_id=%s", (oid,)) is None


# --------------------------------------------------------------------------- finding

_FINDING_ALLOWED = [
    ("open", "resolved"), ("open", "ignored"), ("open", "superseded"),
    ("resolved", "open"), ("ignored", "open"),
]
_FINDING_REFUSED = [
    ("resolved", "ignored"), ("ignored", "resolved"),
    ("superseded", "open"), ("superseded", "resolved"),
    ("resolved", "superseded"),
]


@pytest.mark.parametrize("frm,to", _FINDING_ALLOWED)
def test_a_finding_moves_along_a_declared_edge(conn, frm, to):
    cur = conn.cursor()
    fid = _finding(cur, frm)
    assert _refused(conn, "UPDATE proc.bp_extraction_discrepancy SET status=%s "
                          "WHERE discrepancy_id=%s", (to, fid)) is None


@pytest.mark.parametrize("frm,to", _FINDING_REFUSED)
def test_a_closed_finding_is_reopened_before_it_is_closed_differently(conn, frm, to):
    cur = conn.cursor()
    fid = _finding(cur, frm)
    exc = _refused(conn, "UPDATE proc.bp_extraction_discrepancy SET status=%s "
                         "WHERE discrepancy_id=%s", (to, fid))
    assert exc is not None, f"finding moved {frm} -> {to}"


def test_a_resolved_finding_cannot_be_resolved_again_by_someone_else(conn):
    """The stale double-click: the second person's resolution silently replaced the first."""
    cur = conn.cursor()
    fid = _finding(cur, "open")
    cur.execute("UPDATE proc.bp_extraction_discrepancy SET status='resolved', "
                "resolved_by='first', resolution_action='apply_value', resolved_value='100' "
                "WHERE discrepancy_id=%s", (fid,))
    exc = _refused(conn, "UPDATE proc.bp_extraction_discrepancy SET status='resolved', "
                         "resolved_by='second', resolution_action='dismiss', resolved_value=NULL "
                         "WHERE discrepancy_id=%s", (fid,))
    assert exc is not None, "a resolved finding was re-resolved over the first person's decision"
    cur.execute("SELECT resolved_by FROM proc.bp_extraction_discrepancy "
                "WHERE discrepancy_id=%s", (fid,))
    assert cur.fetchone()[0] == "first"


def test_an_open_finding_can_be_flagged_again(conn):
    """flag/hold/escalate write open -> open with a note; that is not a move."""
    cur = conn.cursor()
    fid = _finding(cur, "open")
    assert _refused(conn, "UPDATE proc.bp_extraction_discrepancy SET status='open', "
                          "notes='flagged' WHERE discrepancy_id=%s", (fid,)) is None


def test_a_closed_finding_still_accepts_writes_that_are_not_its_resolution(conn):
    """Stamping a sent query or a promotion outcome is not re-resolving it."""
    cur = conn.cursor()
    fid = _finding(cur, "resolved")
    assert _refused(conn, "UPDATE proc.bp_extraction_discrepancy SET query_sent_at=now(), "
                          "notes='stamped' WHERE discrepancy_id=%s", (fid,)) is None


# --------------------------------------------------------------------------- the rules, read once

def test_can_apply_reads_the_same_rules_the_trigger_enforces(conn):
    from src.services.lifecycle import can_apply

    cur = conn.cursor()
    for frm, to in _OPP_ALLOWED:
        assert can_apply(cur, "opportunity", frm, to), (frm, to)
    for frm, to in _OPP_REFUSED:
        assert not can_apply(cur, "opportunity", frm, to), (frm, to)
    for frm, to in _FINDING_ALLOWED:
        assert can_apply(cur, "finding", frm, to), (frm, to)
    for frm, to in _FINDING_REFUSED:
        assert not can_apply(cur, "finding", frm, to), (frm, to)
    assert can_apply(cur, "opportunity", "realised", "realised")


# --------------------------------------------------------------------------- automatic writers

def test_the_po_reconciler_leaves_a_deliberately_ignored_finding_alone(conn):
    """A person set this aside. The PO arriving later is not a reason to overrule them,
    and with the guard in place, trying would abort the whole reconcile statement."""
    from src.services.session_postprocess import reconcile_po_discrepancies

    cur = conn.cursor()
    cur.execute("SELECT po_id FROM proc.bp_purchase_order_trgt WHERE po_id IS NOT NULL LIMIT 1")
    row = cur.fetchone()
    if row is None:
        pytest.skip("no purchase order in this database to cite")
    ignored = _finding(cur, "ignored", issue_type="po_not_found", raw_value=row[0])
    still_open = _finding(cur, "open", issue_type="po_not_found", raw_value=row[0])

    reconcile_po_discrepancies(cur)

    cur.execute("SELECT discrepancy_id, status FROM proc.bp_extraction_discrepancy "
                "WHERE discrepancy_id = ANY(%s)", ([ignored, still_open],))
    status = dict(cur.fetchall())
    assert status[ignored] == "ignored"
    assert status[still_open] == "resolved"


def test_rejection_feedback_does_not_unrealise_an_opportunity(conn):
    from src.services import opportunity_store

    cur = conn.cursor()
    realised = _opportunity(cur, "realised")
    live = _opportunity(cur, "negotiation")
    for oid in (realised, live):
        cur.execute("INSERT INTO proc.opportunity_feedback (opportunity_id, status) "
                    "VALUES (%s, 'rejected')", (oid,))

    opportunity_store._fold_in_rejections(cur)

    cur.execute("SELECT opportunity_id, stage FROM proc.bp_opportunity "
                "WHERE opportunity_id = ANY(%s)", ([realised, live],))
    stage = dict(cur.fetchall())
    assert stage[realised] == "realised"
    assert stage[live] == "rejected"


def test_a_rejected_redetection_does_not_unrealise_an_opportunity(conn):
    from src.services.opportunity_store import upsert_opportunity

    cur = conn.cursor()
    oid = _opportunity(cur, "realised")
    upsert_opportunity(cur, {"opportunity_id": oid, "opportunity_ref_id": oid,
                             "is_rejected": True})
    cur.execute("SELECT stage FROM proc.bp_opportunity WHERE opportunity_id=%s", (oid,))
    assert cur.fetchone()[0] == "realised"


def test_set_stage_refuses_a_backward_move_by_name(conn):
    from src.services.lifecycle import IllegalTransition
    from src.services.opportunity_store import set_stage

    cur = conn.cursor()
    oid = _opportunity(cur, "realised")
    with pytest.raises(IllegalTransition) as exc:
        set_stage(oid, "identified", conn=conn)
    assert "realised" in str(exc.value) and "identified" in str(exc.value)
