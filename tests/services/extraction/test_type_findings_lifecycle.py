"""Lifecycle of the document-type findings: keyed per document, closed when
the document stops raising them, and kept out of two existing counts.

Probe rows use doc_pk 'PROBE-TYPE-*' / 'process_monitor:-9xxx'. The
proc.bp_agent_actions rows written alongside are append-only and PERMANENT:
they are test residue, not real agent activity.
"""
from __future__ import annotations

import inspect
import os
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.concepts.vocabulary import SEED_VOCABULARY as V  # noqa: E402
from src.services.extraction import persistence  # noqa: E402
from src.services.extraction.type_resolver import (  # noqa: E402
    resolve_document_type, type_resolution_discrepancies,
)

_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in ("1", "true", "yes", "on")
live = pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")

INVOICE_PAGE = "TAX INVOICE\nInvoice No: INV-1\nAmount due: 10.00\n"


# ---- no database needed -------------------------------------------------

def test_the_issue_types_the_builder_raises_are_the_ones_the_consumers_exclude():
    raised = set()
    for declared, page in (("doctype.quote", INVOICE_PAGE), ("doctype.sow", "INVOICE / QUOTE\n"),
                           ("doctype.invoice", "Dear Sir\n"), ("doctype.sow", INVOICE_PAGE)):
        r = resolve_document_type(declared_concept=declared, full_text=page, vocabulary=V)
        raised |= {d.issue_type for d in type_resolution_discrepancies(r)}
    assert raised == set(persistence.TYPE_FINDING_ISSUE_TYPES)


def test_unknown_document_type_is_unreachable_with_a_declared_concept():
    """Why that issue type was withdrawn: dispatch queues nothing when nothing
    was declared, and with a declared concept status is always 'matched'."""
    for page in ("Dear Sir\n", INVOICE_PAGE, "INVOICE / QUOTE\n", ""):
        r = resolve_document_type(declared_concept="doctype.sow", full_text=page, vocabulary=V)
        assert r.status != "unknown", page
    assert "unknown_document_type" not in persistence.TYPE_FINDING_ISSUE_TYPES


def test_benchmark_flagging_query_excludes_every_type_finding_without_a_database():
    """NON-LIVE guard (most runs have no PROCWISE_TEST_LIVE_DB): the SQL the
    function sends must carry a NOT IN list naming every type finding. Source-level,
    so the behavioural live test below is the stronger guard; both are kept."""
    from src.services import benchmark_live

    class Cur:
        sql = ""
        def execute(self, sql, params=None): Cur.sql = sql
        def fetchall(self): return []
    benchmark_live.load_flagged_documents(Cur())
    flat = " ".join(
        line.split("--")[0] for line in Cur.sql.splitlines()).lower()  # comments don't count
    assert "and issue_type not in (" in flat
    for t in persistence.TYPE_FINDING_ISSUE_TYPES:
        assert f"'{t}'" in flat, t


def test_session_warning_count_mentions_every_type_finding():
    """SOURCE-LEVEL only (the session query needs raw/process_monitor rows to
    run behaviourally); the benchmark filter has a behavioural live test below."""
    from src.services import session_postprocess
    src = inspect.getsource(session_postprocess._session_facts)
    for t in persistence.TYPE_FINDING_ISSUE_TYPES:
        assert f"'{t}'" in src, t


def test_a_pkless_document_is_keyed_by_its_monitor_row_and_visibly_not_a_pk():
    k = persistence.type_finding_doc_key
    assert k("INV-1", 5, "a.pdf") == "INV-1"
    assert k(None, 5, "a.pdf") == "process_monitor:5"
    assert k("", 6, "a.pdf") == "process_monitor:6"
    assert k(None, None, "a.pdf") == "file:a.pdf"
    assert k(None, None, None) is None
    assert k(None, 5, "a.pdf") != k(None, 6, "a.pdf")


# ---- live database ------------------------------------------------------

@pytest.fixture()
def pks():
    from src.services.db import get_conn
    tag = uuid.uuid4().hex[:8]
    keys = [f"PROBE-TYPE-{tag}-A", f"process_monitor:-9{tag[:4].replace('a','1')}1",
            f"process_monitor:-9{tag[:4].replace('a','1')}2"]
    yield keys
    with get_conn() as c:
        cur = c.cursor()
        # decision rows this test wrote for its own probe findings (by id, then the
        # findings); if a trigger ever makes bp_decision append-only they stay as residue.
        try:
            cur.execute("DELETE FROM proc.bp_decision WHERE subject_type = 'finding' "
                        "AND subject_id IN (SELECT discrepancy_id::text FROM "
                        "proc.bp_extraction_discrepancy WHERE doc_pk_candidate = ANY(%s))", (keys,))
        except Exception:
            pass
        cur.execute(
            "DELETE FROM proc.bp_extraction_discrepancy WHERE doc_pk_candidate = ANY(%s)", (keys,))


def _write(pk, declared, page, raw_id=1, source_file=None):
    r = resolve_document_type(declared_concept=declared, full_text=page, vocabulary=V)
    items = type_resolution_discrepancies(r)
    persistence.write_discrepancies(
        doc_type="invoice", raw_id=raw_id, source_file=source_file or f"{pk}.pdf",
        doc_pk_candidate=pk, discrepancies=items)
    return {d.issue_type for d in items}


def _rows(pk):
    from src.services.db import get_conn
    with get_conn() as c:
        cur = c.cursor()
        cur.execute("SELECT issue_type, status, resolution_action, resolved_by, resolved_at IS NOT NULL "
                    "FROM proc.bp_extraction_discrepancy WHERE doc_pk_candidate=%s ORDER BY issue_type", (pk,))
        return cur.fetchall()


def _clear(pk, current, other_doc_keys=(), source_file=None):
    return persistence.resolve_stale_type_findings(
        doc_type="invoice", doc_pk_candidate=pk, current_issue_types=current,
        other_doc_keys=other_doc_keys, source_file=source_file)


@live
def test_a_finding_the_document_no_longer_raises_is_closed_by_convention(pks):
    pk = pks[0]
    now = _write(pk, "doctype.quote", INVOICE_PAGE)
    assert _rows(pk) == [("document_type_disagreement", "open", None, None, False)]
    assert _clear(pk, now) == 0 and _rows(pk)[0][1] == "open", "still raised -> stays open"
    agreeing = _write(pk, "doctype.invoice", INVOICE_PAGE, raw_id=2)
    assert agreeing == set()
    assert _clear(pk, agreeing) == 1
    assert _rows(pk) == [("document_type_disagreement", "resolved", "dismiss",
                          persistence.TYPE_FINDING_RESOLVER, True)]


@live
def test_a_finding_that_changed_kind_closes_the_old_one_only(pks):
    pk = pks[0]
    _write(pk, "doctype.quote", INVOICE_PAGE)
    now = _write(pk, "doctype.sow", "INVOICE / QUOTE\n", raw_id=2)
    assert now == {"unresolved_document_type"}
    assert _clear(pk, now) == 1
    assert _rows(pk) == [("document_type_disagreement", "resolved", "dismiss",
                          persistence.TYPE_FINDING_RESOLVER, True),
                         ("unresolved_document_type", "open", None, None, False)]


@live
def test_a_person_s_ignored_finding_is_not_reopened_or_rewritten(pks):
    from src.services.db import get_conn
    pk = pks[0]
    _write(pk, "doctype.quote", INVOICE_PAGE)
    with get_conn() as c:
        c.cursor().execute("UPDATE proc.bp_extraction_discrepancy SET status='ignored' "
                           "WHERE doc_pk_candidate=%s", (pk,))
    assert _clear(pk, set()) == 0
    assert _rows(pk)[0][1] == "ignored"


@live
def test_clearing_one_document_leaves_every_other_document_alone(pks):
    a, b = pks[0], pks[1]
    _write(a, "doctype.quote", INVOICE_PAGE)
    _write(b, "doctype.quote", INVOICE_PAGE)
    assert _clear(a, set()) == 1
    assert _rows(b)[0][1] == "open"


@live
def test_two_pkless_documents_each_keep_their_own_row(pks):
    _, one, two = pks
    _write(one, "doctype.quote", INVOICE_PAGE, raw_id=1)
    _write(two, "doctype.quote", INVOICE_PAGE, raw_id=2)
    assert len(_rows(one)) == 1 and len(_rows(two)) == 1


def _first_id(pk):
    from src.services.db import get_conn
    with get_conn() as c:
        cur = c.cursor()
        cur.execute("SELECT discrepancy_id FROM proc.bp_extraction_discrepancy "
                    "WHERE doc_pk_candidate=%s", (pk,))
        return cur.fetchone()[0]


@live
def test_benchmark_flagging_behaviour_ignores_type_findings_but_counts_real_ones(pks):
    """Behavioural: real rows through the real write path, real query."""
    from src.services import benchmark_live
    from src.services.db import get_conn
    from src.services.extraction.persistence import Discrepancy
    typed, real = pks[0], f"{pks[0]}-REAL"
    _write(typed, "doctype.quote", INVOICE_PAGE)
    persistence.write_discrepancies(
        doc_type="invoice", raw_id=1, source_file="probe.pdf", doc_pk_candidate=real,
        discrepancies=[Discrepancy(field_name="invoice_amount", issue_type="amount_over_po",
                                   severity="warning", blocks_promotion=False)])
    try:
        with get_conn() as c:
            flagged = benchmark_live.load_flagged_documents(c.cursor())
        assert f"inv:{real}" in flagged, "control: a real finding must still flag"
        assert f"inv:{typed}" not in flagged
    finally:
        with get_conn() as c:
            c.cursor().execute("DELETE FROM proc.bp_extraction_discrepancy WHERE doc_pk_candidate=%s", (real,))


@live
def test_a_row_the_decision_engine_escalated_is_never_auto_closed(pks):
    """Drives the REAL DecisionEngine.execute('escalate'), so the
    subject_type/subject_id the closer's anti-join depends on is written by the
    code that owns the convention. A rename there breaks this test instead of
    silently disabling the guard (a mirrored INSERT here would not)."""
    from contextlib import contextmanager
    from src.engines.decision_engine import DecisionEngine
    from src.services.db import get_conn

    class _Nick:
        @contextmanager
        def get_db_connection(self):
            with get_conn() as c:
                c.autocommit = False
                yield c

    pk = pks[0]
    _write(pk, "doctype.quote", INVOICE_PAGE)
    did = _first_id(pk)
    out = DecisionEngine(_Nick()).execute(str(did), "escalate", user_id="probe-user")
    assert out["applied"] and out["new_status"] == "open"
    assert out["decision_id"], "the engine must have written its decision row"
    assert _rows(pk)[0][1:4] == ("open", None, None), "an escalation leaves no row-level trace"
    assert _clear(pk, set()) == 0
    assert _rows(pk)[0][1] == "open"


@live
def test_a_row_the_gateway_flagged_is_never_auto_closed(pks):
    """The Node gateway's resolveDiscrepancy maps flag (and unknown verbs) to
    status open / resolution_action NULL, writes no bp_decision row, and sets
    resolved_by."""
    from src.services.db import get_conn
    pk = pks[0]
    _write(pk, "doctype.quote", INVOICE_PAGE)
    with get_conn() as c:
        c.cursor().execute("UPDATE proc.bp_extraction_discrepancy SET resolved_by = 'gateway-user' "
                           "WHERE doc_pk_candidate=%s", (pk,))
    assert _clear(pk, set()) == 0
    assert _rows(pk)[0][1] == "open"


@live
def test_a_pk_that_was_lost_on_the_later_read_still_closes_the_earlier_row(pks):
    """Run 1 extracted a pk and filed under it; run 2 extracted none, so it has
    only the monitor key and no way to name the pk. The file is the shared fact."""
    pk, later_key = pks[0], pks[1]
    _write(pk, "doctype.quote", INVOICE_PAGE, source_file="same_doc.pdf")
    assert _clear(later_key, set(), source_file="same_doc.pdf") == 1
    assert _rows(pk)[0][1] == "resolved"


@live
def test_a_pk_lost_read_that_still_disagrees_replaces_the_old_row(pks):
    pk, later_key = pks[0], pks[1]
    _write(pk, "doctype.quote", INVOICE_PAGE, source_file="same_doc.pdf")
    now = _write(later_key, "doctype.quote", INVOICE_PAGE, raw_id=2, source_file="same_doc.pdf")
    assert _clear(later_key, now, source_file="same_doc.pdf") == 1
    assert _rows(pk)[0][1] == "resolved" and _rows(later_key)[0][1] == "open"


@live
def test_a_different_document_is_not_closed_by_source_file(pks):
    pk, other = pks[0], pks[1]
    _write(pk, "doctype.quote", INVOICE_PAGE, source_file="doc_one.pdf")
    assert _clear(other, set(), source_file="doc_two.pdf") == 0
    assert _rows(pk)[0][1] == "open"


@live
def test_a_row_with_a_query_sent_is_never_auto_closed(pks):
    from src.services.db import get_conn
    pk = pks[0]
    _write(pk, "doctype.quote", INVOICE_PAGE)
    with get_conn() as c:
        c.cursor().execute("UPDATE proc.bp_extraction_discrepancy SET query_sent_at = now() "
                           "WHERE doc_pk_candidate=%s", (pk,))
    assert _clear(pk, set()) == 0
    assert _rows(pk)[0][1] == "open"


@live
def test_a_stranded_row_under_another_key_form_is_closed_too(pks):
    """Run 1 had no pk (filed under the monitor key); run 2 extracted one."""
    pk, stranded = pks[0], pks[1]
    _write(stranded, "doctype.quote", INVOICE_PAGE)
    now = _write(pk, "doctype.quote", INVOICE_PAGE, raw_id=2)
    assert _clear(pk, now, other_doc_keys=[stranded]) == 1
    assert _rows(stranded)[0][1] == "resolved"
    assert _rows(pk)[0][1] == "open", "the current run's own row stays open"


def test_every_key_form_of_one_document_is_listed():
    assert persistence.type_finding_doc_keys("INV-1", 5, "a.pdf") == [
        "INV-1", "process_monitor:5", "file:a.pdf"]
    assert persistence.type_finding_doc_keys(None, 5, None) == ["process_monitor:5"]
