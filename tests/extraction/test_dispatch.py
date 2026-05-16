"""End-to-end integration test for the new dispatch.

Runs a real invoice fixture through the new single-flow pipeline and
asserts the row landed in proc.bp_invoice_raw with the expected fields.

Run against live bp_sqldb.
"""
from __future__ import annotations

import sys
from pathlib import Path

import psycopg2
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import Settings  # noqa: E402
from src.services.extraction.dispatch import dispatch_document  # noqa: E402
from src.services.extraction.pattern_registry import clear_cache  # noqa: E402

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "tests" / "extraction_v3" / "fixtures" / "invoices" / "INV-001-clean.pdf"
)


def _conn():
    s = Settings()
    return psycopg2.connect(
        host=s.db_host, dbname=s.db_name, user=s.db_user,
        password=s.db_password, port=s.db_port,
    )


@pytest.fixture(autouse=True)
def _reset_cache():
    clear_cache()


def test_dispatch_writes_raw_and_provenance():
    result = dispatch_document(
        process_monitor_id=None,
        file_path=str(FIXTURE),
        doc_type="invoice",
    )
    assert result["status"] in ("pending", "discrepancy")
    assert result["raw_id"] is not None
    raw_id = result["raw_id"]

    with _conn() as c:
        cur = c.cursor()
        cur.execute(
            """SELECT invoice_id, supplier_name, invoice_amount, currency,
                       parser_snapshot IS NOT NULL, trace_id IS NOT NULL,
                       promotion_status
                  FROM proc.bp_invoice_raw WHERE raw_id = %s""",
            (raw_id,),
        )
        row = cur.fetchone()
        assert row is not None
        inv_id, sup_name, inv_amt, currency, has_snap, has_trace, status = row
        # Expected fixture values
        assert inv_id == "0526", f"invoice_id wrong: {inv_id}"
        assert inv_amt is not None
        assert float(inv_amt) == 9085.00, f"invoice_amount wrong: {inv_amt}"
        assert has_snap is True
        assert has_trace is True
        # supplier_name still NULL (L1 alone can't find it — L2 NER will)
        # currency still NULL (no L1 pattern yet)

        # provenance rows should exist for committed fields
        cur.execute(
            "SELECT COUNT(*) FROM proc.bp_extraction_provenance_v3 WHERE doc_pk=%s",
            (inv_id,),
        )
        n_prov = cur.fetchone()[0]
        assert n_prov >= 2, f"expected provenance for invoice_id+amount, got {n_prov}"

        # Cleanup
        cur.execute("DELETE FROM proc.bp_extraction_provenance_v3 WHERE doc_pk=%s", (inv_id,))
        cur.execute("DELETE FROM proc.bp_extraction_discrepancy WHERE raw_id=%s", (raw_id,))
        cur.execute("DELETE FROM proc.bp_invoice_raw WHERE raw_id=%s", (raw_id,))
        c.commit()
