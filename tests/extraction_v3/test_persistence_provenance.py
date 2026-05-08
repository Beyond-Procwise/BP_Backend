"""Tests for single-transaction persistence with provenance writes.

C9 contract: persist() writes the header row, line items, and one provenance
row per committed field all in a single transaction. Any failure rolls back
the entire write set — a failed provenance write must not leave orphan data.
"""
import pytest
from src.services.extraction_v3.persistence import persist
from src.services.extraction_v3.schemas.result import (
    ExtractionResult,
    CommittedField,
    ResidualField,
)
from src.services.db import get_conn


def _conn_query(sql: str, params=None) -> list[tuple]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params or ())
        return cur.fetchall()


def _cleanup(doc_type: str, doc_pk: str) -> None:
    with get_conn() as conn:
        conn.autocommit = False
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM proc.bp_extraction_provenance_v3 WHERE doc_pk = %s",
                    (doc_pk,),
                )
                if doc_type == "invoice":
                    cur.execute(
                        "DELETE FROM proc.bp_invoice_line_items WHERE invoice_id = %s",
                        (doc_pk,),
                    )
                    cur.execute(
                        "DELETE FROM proc.bp_invoice WHERE invoice_id = %s",
                        (doc_pk,),
                    )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.autocommit = True


@pytest.fixture
def cleanup_test_invoice():
    pk = "INV-V3-TEST-001"
    yield pk
    _cleanup("invoice", pk)


def test_persist_writes_header_and_provenance(cleanup_test_invoice):
    """Header row and two provenance rows are written for two committed fields."""
    pk = cleanup_test_invoice
    result = ExtractionResult(
        doc_type="invoice",
        doc_pk=pk,
        committed=[
            CommittedField(
                field_path="invoice_id",
                value=pk,
                page=0,
                bbox=(0, 0, 1, 1),
                evidence_text=pk,
                model="layoutlmv3",
                model_confidence=0.9,
                judge_actions=[],
                final_confidence=0.9,
            ),
            CommittedField(
                field_path="invoice_amount",
                value="123.45",
                page=0,
                bbox=(10, 20, 30, 40),
                evidence_text="123.45",
                model="qa_roberta",
                model_confidence=0.7,
                judge_actions=["tiebreaker"],
                final_confidence=0.7,
            ),
        ],
        residuals=[],
        judge_calls=1,
        pipeline_version="v3.1.0",
    )
    persist(result)

    inv_rows = _conn_query(
        "SELECT invoice_id, invoice_amount FROM proc.bp_invoice WHERE invoice_id = %s",
        (pk,),
    )
    assert inv_rows, "header row was not written"
    assert inv_rows[0][0] == pk

    prov_count = _conn_query(
        "SELECT count(*) FROM proc.bp_extraction_provenance_v3 WHERE doc_pk = %s",
        (pk,),
    )[0][0]
    # One provenance row per committed field
    assert prov_count == 2


def test_persist_writes_line_items_and_their_provenance(cleanup_test_invoice):
    """Line item rows and provenance rows are written for all committed fields."""
    pk = cleanup_test_invoice
    result = ExtractionResult(
        doc_type="invoice",
        doc_pk=pk,
        committed=[
            CommittedField(
                field_path="invoice_id",
                value=pk,
                page=0,
                bbox=(0, 0, 1, 1),
                evidence_text=pk,
                model="layoutlmv3",
                model_confidence=0.9,
                judge_actions=[],
                final_confidence=0.9,
            ),
            CommittedField(
                field_path="line_items[0].item_description",
                value="widget",
                page=0,
                bbox=(0, 0, 1, 1),
                evidence_text="widget",
                model="table_transformer",
                model_confidence=0.85,
                judge_actions=[],
                final_confidence=0.85,
            ),
            CommittedField(
                field_path="line_items[0].line_amount",
                value="100.00",
                page=0,
                bbox=(0, 0, 1, 1),
                evidence_text="100.00",
                model="table_transformer",
                model_confidence=0.85,
                judge_actions=[],
                final_confidence=0.85,
            ),
        ],
        residuals=[],
        judge_calls=0,
        pipeline_version="v3.1.0",
    )
    persist(result)

    line_rows = _conn_query(
        "SELECT invoice_line_id, item_description, line_amount"
        " FROM proc.bp_invoice_line_items WHERE invoice_id = %s",
        (pk,),
    )
    assert len(line_rows) == 1
    assert line_rows[0][1] == "widget"

    prov_count = _conn_query(
        "SELECT count(*) FROM proc.bp_extraction_provenance_v3 WHERE doc_pk = %s",
        (pk,),
    )[0][0]
    assert prov_count == 3


def test_persist_raises_on_missing_doc_pk():
    """persist() raises ValueError immediately when doc_pk is None."""
    result = ExtractionResult(
        doc_type="invoice",
        doc_pk=None,
        committed=[],
        residuals=[],
        judge_calls=0,
        pipeline_version="v3.1.0",
    )
    with pytest.raises(ValueError, match="doc_pk"):
        persist(result)


def test_persist_provenance_failure_rolls_back_data(cleanup_test_invoice, monkeypatch):
    """If the provenance INSERT fails, the header INSERT must be rolled back."""
    pk = cleanup_test_invoice

    import src.services.extraction_v3.persistence as persist_mod

    def boom(*args, **kwargs):
        raise RuntimeError("simulated provenance failure")

    monkeypatch.setattr(persist_mod, "_build_provenance_inserts", boom)

    result = ExtractionResult(
        doc_type="invoice",
        doc_pk=pk,
        committed=[
            CommittedField(
                field_path="invoice_id",
                value=pk,
                page=0,
                bbox=(0, 0, 1, 1),
                evidence_text=pk,
                model="layoutlmv3",
                model_confidence=0.9,
                judge_actions=[],
                final_confidence=0.9,
            ),
        ],
        residuals=[],
        judge_calls=0,
        pipeline_version="v3.1.0",
    )
    with pytest.raises(RuntimeError, match="simulated"):
        persist(result)

    # The bp_invoice row must NOT be present — the whole transaction rolled back.
    rows = _conn_query(
        "SELECT * FROM proc.bp_invoice WHERE invoice_id = %s", (pk,)
    )
    assert rows == [], f"row still present after rollback: {rows}"
