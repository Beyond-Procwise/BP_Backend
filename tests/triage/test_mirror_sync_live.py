"""Two-way decision sync between a triage finding and its Action Centre mirror row.

Each test makes a real finding + mirror with the writer, then replays one gateway path as
the plain SQL UPDATE that path runs, and checks the other side followed straight away --
no write_batch in between. Run with PROCWISE_TEST_LIVE_DB=1; every synthetic row
(TRIAGE-TEST-*) is removed afterwards.
"""
import os
import uuid
from types import SimpleNamespace

import pytest

from src.services.db import get_conn
from tests.triage.test_writer_live import _currency_deal, _findings, _mirrors, _write

pytestmark = pytest.mark.skipif(os.environ.get("PROCWISE_TEST_LIVE_DB") != "1",
                                reason="needs PROCWISE_TEST_LIVE_DB=1")


@pytest.fixture
def ctx():
    deal_id = f"TRIAGE-TEST-{uuid.uuid4().hex[:8]}"
    runs = []
    with get_conn() as conn:
        yield SimpleNamespace(conn=conn, deal_id=deal_id, runs=runs)
        cur = conn.cursor()
        cur.execute("DELETE FROM proc.bp_detection_finding WHERE deal_id = %s", (deal_id,))
        cur.execute("DELETE FROM proc.bp_extraction_discrepancy "
                    "WHERE source_file IN ('triage:' || %s, 'test:' || %s)", (deal_id, deal_id))
        cur.execute("DELETE FROM proc.bp_triage_finding WHERE deal_id = %s", (deal_id,))
        cur.execute("DELETE FROM proc.bp_triage_result WHERE deal_id = %s", (deal_id,))
        cur.execute("DELETE FROM proc.bp_triage_deal_state WHERE deal_id = %s", (deal_id,))
        for run_id in runs:
            cur.execute("DELETE FROM proc.bp_triage_run WHERE run_id = %s", (run_id,))


@pytest.fixture
def pair(ctx):
    """One S1 currency finding and its mirror: (finding_id, mirror_id)."""
    _write(ctx, _currency_deal(ctx))
    (fid, *_rest), = _findings(ctx)
    (mirror,) = _mirrors(ctx)
    return fid, mirror[0]


def _finding(ctx, fid):
    cur = ctx.conn.cursor()
    cur.execute("""SELECT status, lifecycle_status, resolved_by, resolved_at IS NOT NULL
                     FROM proc.bp_detection_finding WHERE finding_id = %s""", (fid,))
    return cur.fetchone()


def _mirror(ctx, mid):
    cur = ctx.conn.cursor()
    cur.execute("""SELECT status, resolved_by, resolved_at IS NOT NULL
                     FROM proc.bp_extraction_discrepancy WHERE discrepancy_id = %s""", (mid,))
    return cur.fetchone()


# The gateway's writes, as SQL.
def _detection_service_resolve(ctx, fid, status, lifecycle, by):
    ctx.conn.cursor().execute(
        """UPDATE proc.bp_detection_finding SET status = %s, lifecycle_status = %s,
                  resolved_by = %s, resolved_at = now() WHERE finding_id = %s""",
        (status, lifecycle, by, fid))


def _pipeline_patch(ctx, fid, lifecycle):
    status = {"open": "open", "resolved": "resolved", "accepted_risk": "ignored"}[lifecycle]
    ctx.conn.cursor().execute(
        """UPDATE proc.bp_detection_finding SET lifecycle_status = %s, status = %s,
                  resolved_at = CASE WHEN %s = 'open' THEN NULL ELSE now() END
            WHERE finding_id = %s""", (lifecycle, status, status, fid))


def _spendiq_resolve(ctx, mid, status, by):
    ctx.conn.cursor().execute(
        """UPDATE proc.bp_extraction_discrepancy SET status = %s, resolved_by = %s,
                  resolved_at = CASE WHEN %s = 'open' THEN NULL ELSE now() END
            WHERE discrepancy_id = %s""", (status, by, status, mid))


def test_a_detection_service_resolve_closes_the_mirror(ctx, pair):
    fid, mid = pair
    _detection_service_resolve(ctx, fid, "resolved", "resolved", "buyer@x")
    assert _mirror(ctx, mid) == ("resolved", "buyer@x", True)


def test_a_pipeline_patch_ignore_closes_the_mirror_as_detection_finding(ctx, pair):
    fid, mid = pair
    _pipeline_patch(ctx, fid, "accepted_risk")
    assert _finding(ctx, fid)[2] is None                 # the PATCH never sets resolved_by
    assert _mirror(ctx, mid) == ("ignored", "detection-finding", True)


def test_reopening_the_finding_reopens_the_mirror(ctx, pair):
    fid, mid = pair
    _detection_service_resolve(ctx, fid, "resolved", "resolved", "buyer@x")
    assert _mirror(ctx, mid)[0] == "resolved"            # it closed, so the reopen is real
    _pipeline_patch(ctx, fid, "open")
    status, _by, resolved_at_set = _mirror(ctx, mid)
    assert (status, resolved_at_set) == ("open", False)


def test_a_spendiq_ignore_reaches_the_finding_without_a_triage_run(ctx, pair):
    fid, mid = pair
    _spendiq_resolve(ctx, mid, "ignored", "ac@x")
    assert _finding(ctx, fid) == ("ignored", "accepted_risk", "ac@x", True)


def test_a_spendiq_flag_reopens_the_finding(ctx, pair):
    fid, mid = pair
    _spendiq_resolve(ctx, mid, "ignored", "ac@x")
    assert _finding(ctx, fid)[0] == "ignored"            # it closed, so the reopen is real
    _spendiq_resolve(ctx, mid, "open", "ac@x")
    status, lifecycle, _by, resolved_at_set = _finding(ctx, fid)
    assert (status, lifecycle, resolved_at_set) == ("open", "open", False)


def test_decisions_bounce_between_both_sides_without_ping_pong(ctx, pair):
    """Each trigger's echo finds the other side already moved, so it writes nothing;
    a loop would end in 'stack depth limit exceeded'."""
    fid, mid = pair
    steps = [
        lambda: _detection_service_resolve(ctx, fid, "resolved", "resolved", "buyer@x"),
        lambda: _spendiq_resolve(ctx, mid, "open", "ac@x"),
        lambda: _spendiq_resolve(ctx, mid, "ignored", "ac@x"),
        lambda: _pipeline_patch(ctx, fid, "open"),
        lambda: _pipeline_patch(ctx, fid, "resolved"),
    ]
    for step in steps:
        step()
        assert _finding(ctx, fid)[0] == _mirror(ctx, mid)[0]
    assert (_finding(ctx, fid)[:2], _mirror(ctx, mid)[0]) == (("resolved", "resolved"),
                                                              "resolved")


def test_a_non_triage_discrepancy_is_left_alone(ctx, pair):
    fid, mid = pair
    cur = ctx.conn.cursor()
    cur.execute(
        """INSERT INTO proc.bp_extraction_discrepancy
               (doc_type, source_file, doc_pk_candidate, field_name, issue_type, severity,
                status, blocks_promotion)
           VALUES ('invoice', %s, %s, 'probe', 'triage_sync_probe', 'warning', 'open', false)
           RETURNING discrepancy_id""", (f"test:{ctx.deal_id}", ctx.deal_id))
    probe = cur.fetchone()[0]
    _spendiq_resolve(ctx, probe, "ignored", "ac@x")
    assert _mirror(ctx, probe) == ("ignored", "ac@x", True)
    assert _finding(ctx, fid) == ("open", "open", None, False)
    assert _mirror(ctx, mid) == ("open", None, False)


def test_the_engine_still_supersedes_both_sides(ctx, pair):
    fid, mid = pair
    _run_id, counts = _write(ctx, _currency_deal(ctx, currency="GBP"))   # problem fixed
    assert counts["superseded"] == 1
    assert _finding(ctx, fid)[:2] == ("superseded", "resolved")
    assert _mirror(ctx, mid)[0] == "superseded"
