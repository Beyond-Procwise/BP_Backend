"""Guard: every db_column declared in extraction_schemas/*.yaml exists in the DB.

This is the reason migrations must land before YAML. loader._verify_db_consistency
raises SchemaDriftError on a missing column, and it runs on EVERY schema load —
so a YAML-first commit takes extraction down at the next process restart, not at
the next test run. This test moves that failure into CI.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.services.extraction_v3.yaml_schema.loader import (  # noqa: E402
    SchemaDriftError,
    load_all_schemas,
    load_doc_schema_path,
)

EXPECTED_DOC_TYPES = {"contract", "invoice", "purchase_order", "quote"}


def test_every_schema_loads_against_the_live_database():
    schemas = load_all_schemas()
    assert EXPECTED_DOC_TYPES.issubset(set(schemas)), (
        f"missing schemas: {EXPECTED_DOC_TYPES - set(schemas)}"
    )
    for name, schema in schemas.items():
        assert schema.fields, f"{name}.yaml declared no fields"


def test_guard_rejects_a_column_that_does_not_exist(tmp_path):
    """Prove the guard actually fails. A guard that has never been seen red is
    not a guard."""
    bogus = tmp_path / "quote.yaml"
    bogus.write_text(textwrap.dedent("""
        doc_type: quote
        db_table: proc.bp_quote_stg
        db_lines_table: null
        fields:
          - name: definitely_not_a_column
            type: string
            required: false
            db_column: definitely_not_a_column
            canonical_labels: ["Nope"]
    """).strip())

    with pytest.raises(SchemaDriftError) as exc:
        load_doc_schema_path(bogus)
    assert "definitely_not_a_column" in str(exc.value)


def test_guard_rejects_a_missing_lines_table_column(tmp_path):
    bogus = tmp_path / "quote.yaml"
    bogus.write_text(textwrap.dedent("""
        doc_type: quote
        db_table: proc.bp_quote_stg
        db_lines_table: proc.bp_quote_line_items_stg
        fields:
          - name: quote_id
            type: string
            required: true
            db_column: quote_id
            canonical_labels: ["Quote No"]
        line_items:
          primary_extractor: qwen_vlm
          fields:
            - name: not_a_line_column
              type: string
              required: false
              db_column: not_a_line_column
              canonical_labels: ["Nope"]
    """).strip())

    with pytest.raises(SchemaDriftError) as exc:
        load_doc_schema_path(bogus)
    assert "not_a_line_column" in str(exc.value)
