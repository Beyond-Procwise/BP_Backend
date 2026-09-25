"""R18(2): "saved" is one figure on every surface.

The exec summary's "Saved (GBP)" (SQL), ledger_totals over the same window (the drawer's
rules) and the weekly digest's "saved this week" must agree, on the real database, inside
a transaction that is rolled back. The live ledger is not assumed empty: the real rows
already in the window are part of all three figures.

    PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest \
        tests/services/rga/test_saved_is_one_figure_live.py
"""
from __future__ import annotations

import os
import re
import uuid
from contextlib import contextmanager
from datetime import datetime, time, timedelta, timezone
from decimal import Decimal

import pytest

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


def _probe_finding(cur) -> int:
    cur.execute(
        "INSERT INTO proc.bp_extraction_discrepancy (doc_type, source_file, doc_pk_candidate, "
        "field_name, raw_value, computed_value, issue_type, severity, status, blocks_promotion) "
        "VALUES ('invoice', 'probe', %s, %s, '500.00', '+500.00', 'duplicate_invoice', "
        "'warning', 'open', false) RETURNING discrepancy_id",
        (f"PROBE-{uuid.uuid4().hex[:8]}", f"probe_{uuid.uuid4().hex[:8]}"))
    return cur.fetchone()[0]


def _probe_opportunity(cur) -> str:
    oid = f"probe-opp-{uuid.uuid4().hex[:12]}"
    cur.execute("INSERT INTO proc.bp_opportunity (opportunity_id, opportunity_ref_id, stage) "
                "VALUES (%s, %s, 'identified')", (oid, oid))
    return oid


def test_exec_summary_ledger_totals_and_digest_state_the_same_saved_figure(conn, monkeypatch):
    import src.services.rga  # noqa: F401  registers the builders
    from src.services import value_digest as vd, value_ledger as vl
    from src.services import value_summary_service as vss
    from src.services.rga.builders import exec_procurement_summary as ex
    from src.services.rga.factpack import build_fact_pack

    cur = conn.cursor()
    cur.execute("SELECT current_date")
    today = cur.fetchone()[0]
    now = datetime.combine(today, time(12), tzinfo=timezone.utc)
    since, until = vd.saved_window(now)

    # recovered after a claim; a corrected avoided figure; a realised saving; an open
    # claim (in progress, not saved); an avoided figure dated before the window.
    a = _probe_finding(cur)
    vl.record_finding_outcome(a, "claimed", "500", "GBP", actor=ACTOR, conn=conn)
    vl.settle_claim(a, "recovered", "320", "GBP", actor=ACTOR, evidence_ref="CN-P", conn=conn)
    b = _probe_finding(cur)
    first = vl.record_finding_outcome(b, "avoided", "100", "GBP", actor=ACTOR, conn=conn)
    vl.correct_outcome(first["outcome_id"], "90", "GBP", actor=ACTOR, note="probe", conn=conn)
    c_ = _probe_finding(cur)
    vl.record_finding_outcome(c_, "claimed", "70", "GBP", actor=ACTOR, conn=conn)
    d = _probe_finding(cur)
    vl.record_finding_outcome(d, "avoided", "40", "GBP", actor=ACTOR,
                              valid_from=(since - timedelta(days=1)).isoformat(), conn=conn)
    opp = _probe_opportunity(cur)
    probes = {("finding", str(x)) for x in (a, b, c_, d)} | {("opportunity", opp)}
    vl.realise_opportunity(opp, "25", "GBP", actor=ACTOR, conn=conn)
    # A later outcome of another kind on the same opportunity: its CURRENT state is no
    # longer a saving, so no surface may count the earlier realised row.
    cur.execute(
        "INSERT INTO proc.bp_value_outcome (source_type, source_id, outcome_type, amount, "
        "currency, amount_gbp, recorded_by, valid_from) "
        "VALUES ('opportunity', %s, 'terms_improved', 1, 'GBP', 1, %s, current_date)",
        (opp, ACTOR))

    # --- ledger_totals over the window (the drawer's rules) ---------------------
    ledger_rows = vss._load_ledger(conn.cursor())
    expected = vss.ledger_totals(ledger_rows, since=since, until=until)["saved_gbp"]

    # --- the exec summary, its queries run on this (rolled-back) connection -----
    def _fetch_here(sql, params):
        c = conn.cursor()
        c.execute(sql, params)
        return c.fetchall()
    monkeypatch.setattr(ex, "_fetch", _fetch_here)
    pack = build_fact_pack("exec_procurement_summary",
                           scope={"period_label": "probe week", "currency": "GBP",
                                  "period_start": since.isoformat(),
                                  "period_end": until.isoformat()},
                           as_of=until.isoformat(), emit_audit=False)
    exec_saved = pack.fact_by_label("Saved (GBP)").value

    # --- the digest ------------------------------------------------------------
    summary = vss.build_value_summary(conn=conn)
    summary["ledger_rows"] = ledger_rows
    body = vd.compose_digest(summary, now)["body"]
    digest_saved = Decimal(re.search(r"£([0-9,]+\.[0-9]{2}) saved this week", body)
                           .group(1).replace(",", ""))

    assert Decimal(str(expected)).quantize(Decimal("0.01")) == \
        Decimal(exec_saved).quantize(Decimal("0.01")) == digest_saved
    assert vd.saved_this_week(ledger_rows, now) == expected
    # the probes' own contribution: 320 recovered + 90 avoided (not 100, not 25, not 40,
    # not the open 70 claim) -- the real rows already in the window make up the rest.
    real = [r for r in ledger_rows if (r["source_type"], r["source_id"]) not in probes]
    assert round(expected - vss.ledger_totals(real, since=since, until=until)["saved_gbp"],
                 2) == 410.0
