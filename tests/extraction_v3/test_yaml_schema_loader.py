import pytest
from pathlib import Path
from src.services.extraction_v3.yaml_schema import (
    load_doc_schema, load_doc_schema_path, load_all_schemas, SchemaDriftError
)


def test_load_invoice_schema():
    s = load_doc_schema("invoice")
    assert s.doc_type == "invoice"
    assert s.db_table == "proc.bp_invoice"
    inv_id = next(f for f in s.fields if f.name == "invoice_id")
    assert inv_id.required is True
    assert inv_id.canonical_labels  # non-empty
    sup = next(f for f in s.fields if f.name == "supplier_name")
    assert sup.db_column is None
    assert sup.resolves_to_db_column == "supplier_id"


def test_load_all_schemas_includes_four_doctypes():
    schemas = load_all_schemas()
    assert set(schemas.keys()) == {"invoice", "purchase_order", "quote", "contract"}


def test_drift_fails_loud(tmp_path):
    bad = tmp_path / "broken.yaml"
    bad.write_text("""\
doc_type: broken
db_table: proc.does_not_exist
fields:
  - name: x
    type: string
    required: true
    db_column: nope
    canonical_labels: ["x"]
    extractors: [layoutlmv3]
""")
    with pytest.raises(SchemaDriftError) as exc:
        load_doc_schema_path(bad)
    assert "does_not_exist" in str(exc.value)


def test_drift_fails_for_missing_column(tmp_path):
    bad = tmp_path / "bad_col.yaml"
    bad.write_text("""\
doc_type: bad_col
db_table: proc.bp_invoice
fields:
  - name: x
    type: string
    required: true
    db_column: column_that_does_not_exist
    canonical_labels: ["x"]
    extractors: [layoutlmv3]
""")
    with pytest.raises(SchemaDriftError) as exc:
        load_doc_schema_path(bad)
    assert "column_that_does_not_exist" in str(exc.value)
