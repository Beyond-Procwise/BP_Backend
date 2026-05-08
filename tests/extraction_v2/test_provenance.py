"""Tests for the per-field extraction provenance writer."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.services.extraction_v2.provenance import (  # noqa: E402
    record_extraction_provenance, record_field_provenance,
)


class _FakeCursor:
    def __init__(self, log: list):
        self._log = log

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def execute(self, sql, params=None):
        norm = " ".join(sql.split()).lower()
        self._log.append((norm, params))


class _FakeConn:
    def __init__(self, log: list):
        self._log = log

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def cursor(self):
        return _FakeCursor(self._log)

    def commit(self):
        self._log.append(("commit", None))


def _factory():
    log: list = []

    def get_conn():
        return _FakeConn(log)

    return get_conn, log


def test_record_field_provenance_inserts_one_row():
    get_conn, log = _factory()
    record_field_provenance(
        get_conn,
        record_id="INV1", doc_type="Invoice",
        field_name="supplier_name", value="ACME Co",
        source="template", confidence=0.95,
    )
    inserts = [op for (op, _) in log if "insert into proc.bp_extraction_provenance" in op]
    assert len(inserts) == 1
    commits = [op for (op, _) in log if op == "commit"]
    assert len(commits) == 1
    # Insert params: (parent_table, parent_pk, field_name, source,
    #                 anchor_ref, derivation_trace, confidence)
    params = [p for (op, p) in log if "insert into" in op][0]
    assert params[0] == "bp_invoice"  # doc_type → parent_table mapping
    assert params[1] == "INV1"
    assert params[2] == "supplier_name"
    assert params[3] == "template"


def test_record_field_provenance_coerces_unknown_source_to_llm():
    get_conn, log = _factory()
    record_field_provenance(
        get_conn,
        record_id="INV2", doc_type="Invoice",
        field_name="supplier_name", value="X",
        source="bogus_source",
    )
    insert_params = [params for (op, params) in log if "insert into" in op][0]
    # Source is the 4th positional param in the existing schema's order
    source = insert_params[3]
    assert source == "llm"


def test_bulk_record_uses_correct_sources_for_each_field():
    get_conn, log = _factory()
    header = {
        "supplier_name": "ACME Co Ltd",  # template override
        "buyer_name": "WidgetCo",        # filename rescue
        "invoice_id": "INV-100",         # default (llm)
        "invoice_amount": 1234.5,        # default (llm)
        "_internal": "skip",             # leading underscore — skipped
        "created_date": "2026-05-04",    # audit col — skipped
    }
    sanitizer_rejection = SimpleNamespace(
        field="payment_terms", raw_value="weird input",
    )
    rows = record_extraction_provenance(
        get_conn,
        record_id="INV3", doc_type="Invoice",
        header={**header, "payment_terms": None},
        rescued_fields={"buyer_name"},
        template_overrides={"supplier_name"},
        sanitizer_rejections=[sanitizer_rejection],
    )
    # supplier_name + buyer_name + invoice_id + invoice_amount + payment_terms = 5
    assert rows == 5

    # Inspect the (field_name, source) pairs from the INSERT params
    inserted = []
    for op, params in log:
        if "insert into" in op:
            field_name, source = params[2], params[3]
            inserted.append((field_name, source))
    assert ("supplier_name", "template") in inserted
    assert ("buyer_name", "filename") in inserted
    assert ("invoice_id", "llm") in inserted
    assert ("invoice_amount", "llm") in inserted
    assert ("payment_terms", "sanitizer") in inserted
    # Skipped fields must NOT appear
    field_names = [f for (f, _) in inserted]
    assert "_internal" not in field_names
    assert "created_date" not in field_names


def test_bulk_record_with_structural_default_source():
    get_conn, log = _factory()
    rows = record_extraction_provenance(
        get_conn,
        record_id="INV4", doc_type="Invoice",
        header={"invoice_id": "I-1", "invoice_amount": 99.0},
        default_source="structural",
    )
    assert rows == 2
    inserts = [params for (op, params) in log if "insert into" in op]
    assert all(p[3] == "structural" for p in inserts)
