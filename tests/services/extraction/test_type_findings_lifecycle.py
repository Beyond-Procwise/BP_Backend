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
                           (None, "Dear Sir\n")):
        r = resolve_document_type(declared_concept=declared, full_text=page, vocabulary=V)
        raised |= {d.issue_type for d in type_resolution_discrepancies(r)}
    assert raised == set(persistence.TYPE_FINDING_ISSUE_TYPES)


def test_benchmark_flagging_excludes_every_type_finding():
    from src.services import benchmark_live

    class Cur:
        sql = ""
        def execute(self, sql, params=None): Cur.sql = sql
        def fetchall(self): return []
    benchmark_live.load_flagged_documents(Cur())
    for t in persistence.TYPE_FINDING_ISSUE_TYPES:
        assert f"'{t}'" in Cur.sql, t


def test_session_warning_count_excludes_every_type_finding():
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
        c.cursor().execute(
            "DELETE FROM proc.bp_extraction_discrepancy WHERE doc_pk_candidate = ANY(%s)", (keys,))


def _write(pk, declared, page, raw_id=1):
    r = resolve_document_type(declared_concept=declared, full_text=page, vocabulary=V)
    items = type_resolution_discrepancies(r)
    persistence.write_discrepancies(
        doc_type="invoice", raw_id=raw_id, source_file=f"{pk}.pdf",
        doc_pk_candidate=pk, discrepancies=items)
    return {d.issue_type for d in items}


def _rows(pk):
    from src.services.db import get_conn
    with get_conn() as c:
        cur = c.cursor()
        cur.execute("SELECT issue_type, status, resolution_action, resolved_by, resolved_at IS NOT NULL "
                    "FROM proc.bp_extraction_discrepancy WHERE doc_pk_candidate=%s ORDER BY issue_type", (pk,))
        return cur.fetchall()


def _clear(pk, current):
    return persistence.resolve_stale_type_findings(
        doc_type="invoice", doc_pk_candidate=pk, current_issue_types=current)


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
