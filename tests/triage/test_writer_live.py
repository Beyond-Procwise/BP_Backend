"""Writer against the real database. Run with PROCWISE_TEST_LIVE_DB=1.

Uses synthetic deal ids (TRIAGE-TEST-*) and removes every row it made.
"""
import os
import uuid
from types import SimpleNamespace

import pytest

from src.services.db import get_conn
from src.services.triage import writer
from tests.triage.helpers import deal, inv, line, make_cfg, pipeline, po

pytestmark = pytest.mark.skipif(os.environ.get("PROCWISE_TEST_LIVE_DB") != "1",
                                reason="needs PROCWISE_TEST_LIVE_DB=1")
CFG = make_cfg()


@pytest.fixture
def ctx():
    deal_id = f"TRIAGE-TEST-{uuid.uuid4().hex[:8]}"
    runs = []
    with get_conn() as conn:
        yield SimpleNamespace(conn=conn, deal_id=deal_id, runs=runs)
        cur = conn.cursor()
        cur.execute("DELETE FROM proc.bp_detection_finding WHERE deal_id = %s", (deal_id,))
        cur.execute("DELETE FROM proc.bp_triage_finding WHERE deal_id = %s", (deal_id,))
        cur.execute("DELETE FROM proc.bp_triage_result WHERE deal_id = %s", (deal_id,))
        for run_id in runs:
            cur.execute("DELETE FROM proc.bp_triage_run WHERE run_id = %s", (run_id,))


def _write(ctx, ds):
    run_id = writer.start_run(ctx.conn, "single", CFG)
    ctx.runs.append(run_id)
    counts = writer.write_batch(ctx.conn, run_id, [pipeline(ds, CFG)])
    return run_id, counts


def _findings(ctx):
    cur = ctx.conn.cursor()
    cur.execute("""SELECT finding_id, rule_id, severity, status, lifecycle_status,
                          blocks_promotion, pipeline_record_id
                     FROM proc.bp_detection_finding WHERE deal_id = %s ORDER BY finding_id""",
                (ctx.deal_id,))
    return cur.fetchall()


def _currency_deal(ctx, currency="EUR"):
    return deal(po(), inv(currency=currency), deal_id=ctx.deal_id)


def test_first_write_inserts_s1_and_audits_everything(ctx):
    ds = _currency_deal(ctx)
    run_id, counts = _write(ctx, ds)
    rows = _findings(ctx)
    assert counts["inserted"] == 1 and len(rows) == 1
    _fid, rule, sev, status, life, blocks, record = rows[0]
    assert (rule, sev, status, life, blocks, record) == (
        "currency", "critical", "open", "open", True, ctx.deal_id)
    cur = ctx.conn.cursor()
    cur.execute("SELECT count(*) FROM proc.bp_triage_result WHERE run_id = %s", (run_id,))
    assert cur.fetchone()[0] == counts["audit_rows"] == len(pipeline(ds, CFG).results)


def test_rerun_is_idempotent(ctx):
    _write(ctx, _currency_deal(ctx))
    first = _findings(ctx)
    _run_id, counts = _write(ctx, _currency_deal(ctx))
    assert counts["inserted"] == 0 and counts["updated"] == 1
    assert _findings(ctx) == first


def test_fixed_problem_is_superseded(ctx):
    _write(ctx, _currency_deal(ctx))
    _run_id, counts = _write(ctx, _currency_deal(ctx, currency="GBP"))
    assert counts["superseded"] == 1
    (_fid, _rule, _sev, status, life, _b, _r), = _findings(ctx)
    assert (status, life) == ("superseded", "resolved")


def test_resolved_finding_is_not_reopened(ctx):
    _write(ctx, _currency_deal(ctx))
    ctx.conn.cursor().execute(
        """UPDATE proc.bp_detection_finding SET status='resolved', lifecycle_status='resolved',
                  resolved_by='tester' WHERE deal_id = %s""", (ctx.deal_id,))
    _run_id, counts = _write(ctx, _currency_deal(ctx))
    rows = _findings(ctx)
    assert counts["unchanged"] == 1 and len(rows) == 1 and rows[0][3] == "resolved"


def test_severity_rise_opens_a_new_finding(ctx):
    def price_deal(price):
        return deal(po(lines=[line(1, qty="600", price="30.00")]),
                    inv(lines=[line(1, qty="300", price=price)]), deal_id=ctx.deal_id)

    _write(ctx, price_deal("30.50"))                   # exposure £150 -> S2
    (fid, rule, sev, *_rest), = _findings(ctx)
    assert (rule, sev) == ("unit_price", "warning")
    ctx.conn.cursor().execute(
        "UPDATE proc.bp_detection_finding SET status='ignored', lifecycle_status='accepted_risk' "
        "WHERE finding_id = %s", (fid,))
    _run_id, counts = _write(ctx, price_deal("32.00"))  # exposure £600 -> S1
    rows = _findings(ctx)
    assert counts["reopened"] == 1 and len(rows) == 2
    assert rows[0][3] == "ignored" and rows[1][2] == "critical" and rows[1][3] == "open"


def test_rollback_removes_only_untouched_findings(ctx):
    ds = deal(po(), inv("INV-1", currency="EUR"), inv("INV-2", currency="USD", po_id="PO-1"),
              deal_id=ctx.deal_id)
    run_id, _counts = _write(ctx, ds)
    fids = [r[0] for r in _findings(ctx) if r[1] == "currency"]
    ctx.conn.cursor().execute(
        "UPDATE proc.bp_detection_finding SET owner='buyer@example.com' WHERE finding_id = %s",
        (fids[0],))
    result = writer.rollback_run(ctx.conn, run_id)
    remaining = [r[0] for r in _findings(ctx)]
    assert fids[0] in remaining and fids[1] not in remaining
    assert result["findings_kept"] >= 1 and result["audit_rows_removed"] > 0
