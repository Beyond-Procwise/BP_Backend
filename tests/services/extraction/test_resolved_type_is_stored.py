"""A classification nothing records is a classification no maths can use.

dispatch.py records the resolved structure "never acted on": a log line, and on
disagreement a review item. These tests pin it to a column, and pin that the
column survives promotion and refreshes on a re-read.

Live-only. Run with:
    set -a && . ./.env && set +a
    CUDA_VISIBLE_DEVICES="" PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest \
        tests/services/extraction/test_resolved_type_is_stored.py -v
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.extraction import persistence                      # noqa: E402

_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in ("1", "true", "yes", "on")
pytestmark = pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")

_COLS = ("resolved_doc_type", "resolved_role", "type_agreement")


@pytest.fixture()
def cleanup():
    """Remove only the rows this test made, by its own unique contract_id."""
    made: list[str] = []
    yield made
    from src.services.db import get_conn
    with get_conn() as conn:
        cur = conn.cursor()
        for contract_id in made:
            cur.execute("DELETE FROM proc.bp_contracts WHERE contract_id = %s", (contract_id,))
            cur.execute("DELETE FROM proc.bp_contract_raw WHERE contract_id = %s", (contract_id,))


def _write(contract_id, *, resolved="doctype.sow", role="role.master",
           agreement="refined", source_file=None):
    return persistence.write_raw(
        doc_type="contract",
        file_path=source_file or f"documents/contract/{contract_id}.pdf",
        process_monitor_id=None,
        trace_id=uuid.uuid4(),
        pipeline_version="test",
        columns={"contract_id": contract_id, "supplier_id": "S-TEST"},
        parser_snapshot={"full_text": "STATEMENT OF WORK\n"},
        promotion_status="pending",
        resolved_doc_type=resolved,
        resolved_role=role,
        type_agreement=agreement,
    )


def _read(table, contract_id):
    from src.services.db import get_conn
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            f"SELECT {', '.join(_COLS)} FROM proc.{table} WHERE contract_id = %s "
            f"ORDER BY {'raw_id DESC' if table == 'bp_contract_raw' else 'contract_id'} LIMIT 1",
            (contract_id,),
        )
        row = cur.fetchone()
    return dict(zip(_COLS, row)) if row else None


def test_both_tables_have_the_three_columns():
    from src.services.db import get_conn
    with get_conn() as conn:
        cur = conn.cursor()
        for table in ("bp_contract_raw", "bp_contracts"):
            cur.execute(
                """SELECT column_name FROM information_schema.columns
                    WHERE table_schema='proc' AND table_name=%s AND column_name = ANY(%s)""",
                (table, list(_COLS)),
            )
            got = sorted(r[0] for r in cur.fetchall())
            assert got == sorted(_COLS), f"{table} is missing columns: {got}"


def test_the_resolved_structure_reaches_the_raw_row(cleanup):
    cid = f"CT-{uuid.uuid4().hex[:10].upper()}"
    cleanup.append(cid)
    _write(cid)
    assert _read("bp_contract_raw", cid) == {
        "resolved_doc_type": "doctype.sow",
        "resolved_role": "role.master",
        "type_agreement": "refined",
    }


def test_write_raw_stores_null_when_given_none(cleanup):
    """No fabrication: an unrecognised page records no structure.

    The tempting fallback -- store the declared type when the page is silent --
    would make every document look as though it had confirmed its own category.
    """
    cid = f"CT-{uuid.uuid4().hex[:10].upper()}"
    cleanup.append(cid)
    _write(cid, resolved=None, role=None, agreement="declared_only")
    assert _read("bp_contract_raw", cid) == {
        "resolved_doc_type": None, "resolved_role": None,
        "type_agreement": "declared_only",
    }


def test_the_structure_survives_promotion(cleanup):
    cid = f"CT-{uuid.uuid4().hex[:10].upper()}"
    cleanup.append(cid)
    raw_id = _write(cid, resolved="doctype.master_agreement", role="role.master",
                    agreement="refined")
    from src.services.extraction import promotion
    promotion.promote(raw_id, "contract")
    assert _read("bp_contracts", cid) == {
        "resolved_doc_type": "doctype.master_agreement",
        "resolved_role": "role.master",
        "type_agreement": "refined",
    }


def test_a_reread_refreshes_the_stored_structure(cleanup):
    """Review Focus 3: _stg updating while the promoted row stays stale is an
    existing, measured failure in this product. A re-read that corrects the
    structure must correct it everywhere, or the maths links on a stale answer.
    """
    cid = f"CT-{uuid.uuid4().hex[:10].upper()}"
    cleanup.append(cid)
    from src.services.extraction import promotion

    first = _write(cid, resolved="doctype.contract_unspecified", agreement="agreed")
    promotion.promote(first, "contract")
    assert _read("bp_contracts", cid)["resolved_doc_type"] == "doctype.contract_unspecified"

    second = _write(cid, resolved="doctype.sow", agreement="refined")
    promotion.promote(second, "contract")
    assert _read("bp_contracts", cid) == {
        "resolved_doc_type": "doctype.sow",
        "resolved_role": "role.master",
        "type_agreement": "refined",
    }, "the promoted row kept the first read's structure"


def test_write_raw_gates_the_columns_on_the_contract_doc_type():
    """Source-level pin; the live behaviour is the next test."""
    import inspect
    src = inspect.getsource(persistence.write_raw)
    assert 'doc_type == "contract"' in src, (
        "write_raw must gate the three columns on the contract doc_type"
    )


def test_an_invoice_write_with_resolved_arguments_does_not_break():
    """Only bp_contract_raw has these columns, so write_raw must not send them
    to an invoice, quote or purchase order -- that would be an UndefinedColumn
    error on the live ingestion path for three of the four pipelines.
    """
    cid = f"IV-{uuid.uuid4().hex[:10].upper()}"
    raw_id = persistence.write_raw(
        doc_type="invoice",
        file_path=f"documents/invoice/{cid}.pdf",
        process_monitor_id=None,
        trace_id=uuid.uuid4(),
        pipeline_version="test",
        columns={"invoice_id": cid},
        parser_snapshot={"full_text": "TAX INVOICE\n"},
        promotion_status="pending",
        resolved_doc_type="doctype.invoice",
        resolved_role="role.transaction",
        type_agreement="agreed",
    )
    try:
        assert raw_id
    finally:
        from src.services.db import get_conn
        with get_conn() as conn:
            conn.cursor().execute("DELETE FROM proc.bp_invoice_raw WHERE raw_id = %s", (raw_id,))


def test_a_promoted_raw_row_survives_promotion_marked_promoted(cleanup):
    """promote() keeps the _raw row (permanent retention tier); the earlier read
    of a re-read is therefore still there, marked promoted, not deleted."""
    cid = f"CT-{uuid.uuid4().hex[:10].upper()}"
    cleanup.append(cid)
    from src.services.extraction import promotion
    raw_id = _write(cid)
    promotion.promote(raw_id, "contract")
    from src.services.db import get_conn
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT promotion_status FROM proc.bp_contract_raw WHERE raw_id=%s",
                    (raw_id,))
        row = cur.fetchone()
    assert row and row[0] == "promoted"


# --- dispatch: what it hands to write_raw -----------------------------------

class _Stop(Exception):
    """Raised by the spy so dispatch halts right after write_raw."""


def _dispatch_kwargs(tmp_path, text, declared_concept, *, resolver_raises=False):
    """Run the real dispatch_document up to write_raw and return write_raw's kwargs."""
    from unittest.mock import patch
    from src.services.extraction import dispatch
    from src.services.extraction_v3.schemas.parsed_document import ParsedDocument
    f = tmp_path / "doc.pdf"
    parsed = ParsedDocument(source_path=str(f), file_format="pdf-native", pages=[],
                            full_text=text, parser_backend="test", parser_confidence=1.0)
    seen = {}

    def spy(**kw):
        seen.update(kw)
        raise _Stop()

    ctx = [patch.object(dispatch.persistence, "write_raw", side_effect=spy),
           patch.object(dispatch, "parse_document", return_value=parsed)]
    if resolver_raises:
        ctx.append(patch("src.services.extraction.type_resolver.resolve_document_type",
                         side_effect=RuntimeError("boom")))
    for c in ctx:
        c.start()
    try:
        with pytest.raises(_Stop):
            dispatch.dispatch_document(
                process_monitor_id=None, file_path=str(f), doc_type="contract",
                declared_concept=declared_concept,
            )
    finally:
        for c in reversed(ctx):
            c.stop()
    assert seen, "write_raw was never reached; the fixture proves nothing"
    return seen


def _live_vocab():
    from src.services.concepts.vocabulary import ensure_vocabulary
    vocab = ensure_vocabulary()
    assert vocab.source.startswith("bp_concept@"), (
        f"vocabulary came from {vocab.source!r}, not the database"
    )
    return vocab


_SOW_TEXT = ("STATEMENT OF WORK\nThis Statement of Work is issued under the Master "
             "Services Agreement between Acme Ltd and Globex Ltd.\n")


def test_dispatch_hands_write_raw_the_resolved_structure(tmp_path):
    vocab = _live_vocab()
    kw = _dispatch_kwargs(tmp_path, _SOW_TEXT, "doctype.contract_unspecified")
    assert kw["resolved_doc_type"] == "doctype.sow"
    assert kw["resolved_role"] == vocab.document_types["doctype.sow"].role
    assert kw["resolved_role"], "role must be looked up, not left empty"
    assert kw["type_agreement"] == "refined"


def test_dispatch_stores_null_for_a_silent_page_never_the_declared_type(tmp_path):
    _live_vocab()
    kw = _dispatch_kwargs(tmp_path, "lorem ipsum dolor sit amet\n",
                          "doctype.master_agreement")
    assert kw["resolved_doc_type"] is None
    assert kw["resolved_role"] is None
    assert kw["type_agreement"] == "declared_only"


def test_dispatch_tolerates_a_resolver_failure_with_nulls(tmp_path):
    kw = _dispatch_kwargs(tmp_path, _SOW_TEXT, "doctype.sow", resolver_raises=True)
    assert kw["resolved_doc_type"] is None
    assert kw["resolved_role"] is None
    assert kw["type_agreement"] is None
