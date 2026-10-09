"""The value-at-risk queries behind the upload summary, against the real schema.

session_postprocess._session_billed_value / _session_uplift_value are SQL over six
tables (the session's raw documents, the _trgt/_stg currency, the triage figure). A
unit test on dicts cannot catch a wrong join, so this one writes a small tagged upload
into the database and reads it back through the real functions.

Everything happens inside ONE transaction on a private connection that is rolled
back: no other session ever sees the rows, nothing is left behind, and the only
trigger side effect that leaves the database (pg_notify on process_monitor and on a
resolved finding) is delivered only on commit. get_conn() is autocommit, which is why
this test opens its own connection rather than using it.

Needs PROCWISE_TEST_LIVE_DB=1 (and .env loaded).
"""
from __future__ import annotations

import json
import os
import uuid

import pytest

_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in ("1", "true", "yes", "on")
pytestmark = pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")


@pytest.fixture()
def cur():
    import psycopg2
    from src.services.db import _pg_dsn
    conn = psycopg2.connect(_pg_dsn())
    conn.autocommit = False
    try:
        yield conn.cursor()
    finally:
        conn.rollback()
        conn.close()


def _monitor(cur, session_id):
    cur.execute("insert into proc.process_monitor (session_id) values (%s) returning id",
                (session_id,))
    return cur.fetchone()[0]


def _raw(cur, table, monitor_id, pk):
    cur.execute(f"insert into proc.{table} (process_monitor_id, doc_pk_candidate, source_file, "
                "raw_payload, pipeline_version) values (%s, %s, %s, %s, 'test')",
                (monitor_id, pk, f"{pk}.pdf", json.dumps({})))


def _finding(cur, doc_type, pk, issue_type, *, computed=None, status="open"):
    cur.execute("insert into proc.bp_extraction_discrepancy (doc_type, source_file, "
                "doc_pk_candidate, field_name, issue_type, severity, status, computed_value) "
                "values (%s, %s, %s, 'probe', %s, 'critical', %s, %s) returning discrepancy_id",
                (doc_type, f"{pk}.pdf", pk, issue_type, status, computed))
    return cur.fetchone()[0]


@pytest.fixture()
def upload(cur):
    """One upload: 4 invoices, 1 PO, 3 bid families, plus a document from elsewhere."""
    t = uuid.uuid4().hex[:8].upper()
    sid, deal = f"ses-PROBE-{t}", f"PROBE-DEAL-{t}"
    k = {n: f"PROBE-{t}-{n}" for n in ("PO", "INV-A", "INV-B", "INV-C", "INV-D", "OUT", "Q", "Q2", "Q3")}
    m = _monitor(cur, sid)
    for n in ("INV-A", "INV-B", "INV-C", "INV-D"):
        _raw(cur, "bp_invoice_raw", m, k[n])
    _raw(cur, "bp_purchase_order_raw", m, k["PO"])
    for q in (f"{k['Q']}", f"{k['Q']} (V2)", k["Q2"], k["Q3"], f"{k['Q3']} (V2)"):
        _raw(cur, "bp_quote_raw", m, q)
        cur.execute("insert into proc.bp_quote_stg (quote_id, currency) values (%s, 'GBP')", (q,))

    # currency + PO linkage the value query joins through
    cur.execute("insert into proc.bp_purchase_order_trgt (po_id, currency, deal_id) "
                "values (%s, 'GBP', %s)", (k["PO"], deal))
    for n, po, ccy in (("INV-A", k["PO"], "GBP"), ("INV-B", f"{k['PO']}-OTHER", "GBP"),
                       ("INV-C", f"{k['PO']}-THIRD", None), ("INV-D", f"{k['PO']}-FOURTH", "GBP")):
        cur.execute("insert into proc.bp_invoice_trgt (invoice_id, po_id, currency, deal_id) "
                    "values (%s, %s, %s, %s)", (k[n], po, ccy, deal))

    # PO-level triage finding: its figure is the detection finding's £ delta
    po_level = _finding(cur, "purchase_order", k["PO"], "invoices_exceed_po_total")
    cur.execute("insert into proc.bp_detection_finding (category, severity, delta) "
                "values ('invoices_exceed_po_total', 'critical', '£20,000.00') returning finding_id")
    fid = cur.fetchone()[0]
    cur.execute("insert into proc.bp_triage_finding (fingerprint, finding_id, deal_id, "
                "first_run_id, last_run_id, last_severity, mirror_id) "
                "values (%s, %s, %s, %s, %s, 'critical', %s)",
                (f"probe-{t}", fid, deal, str(uuid.uuid4()), str(uuid.uuid4()), po_level))

    # the same £20,000 seen on the invoice line: superseded under the PO, not added
    _finding(cur, "invoice", k["INV-A"], "line_amount_over_po", computed="+20000.00")
    # a possible duplicate: counted
    _finding(cur, "invoice", k["INV-B"], "duplicate_invoice", computed="+1003.81")
    # already resolved: not counted. On its own invoice: Value Found counts one money
    # finding per document (dedupe), so beside another finding it would vanish anyway
    # and this line would check nothing.
    _finding(cur, "invoice", k["INV-D"], "amount_over_po", computed="+777.00", status="resolved")
    # no currency on the document: not valued, never £0
    _finding(cur, "invoice", k["INV-C"], "amount_over_po", computed="+50.00")
    # a document from another upload: not this session's money
    _finding(cur, "invoice", k["OUT"], "duplicate_invoice", computed="+9999.00")

    # uplift: only each bid's latest round counts; the figure is the largest bid's
    _finding(cur, "quote", k["Q"], "uplift_above_stated", computed="58124.80")        # V1: superseded by V2
    _finding(cur, "quote", f"{k['Q']} (V2)", "uplift_above_stated", computed="52938.60")
    _finding(cur, "quote", k["Q2"], "uplift_above_stated", computed="48327.80")
    _finding(cur, "quote", k["Q3"], "uplift_above_stated", computed="99999.00")      # V2 clears it
    return sid


def test_billed_value_is_this_uploads_open_money_counted_once(cur, upload, monkeypatch):
    from src.services import session_postprocess as sp
    from src.services import value_summary_service as vs

    def no_fx():
        raise AssertionError("every probe amount is GBP or unlabelled: no FX fetch needed")
    monkeypatch.setattr(vs, "_get_rates", no_fx)

    billed, unvalued = sp._session_billed_value(cur, upload)
    assert billed == {"invoices_exceed_po_total": 20000.0, "duplicate_invoice": 1003.81}
    assert unvalued == {"amount_over_po": 1}


def test_uplift_is_the_largest_latest_round_never_a_sum(cur, upload):
    from src.services import session_postprocess as sp
    assert sp._session_uplift_value(cur, upload) == (52938.6, 0)


def test_the_summary_facts_carry_both_figures(cur, upload):
    from src.services import session_postprocess as sp
    var = sp._session_value_at_risk(cur, upload)
    assert var == {"billed_gbp": {"invoices_exceed_po_total": 20000.0,
                                  "duplicate_invoice": 1003.81},
                   "uplift_up_to_gbp": 52938.6,
                   "unvalued": {"amount_over_po": 1}}


def test_a_session_with_no_documents_has_no_value(cur):
    from src.services import session_postprocess as sp
    assert sp._session_value_at_risk(cur, f"ses-PROBE-EMPTY-{uuid.uuid4().hex[:6]}") == {
        "billed_gbp": {}, "uplift_up_to_gbp": None, "unvalued": {}}
