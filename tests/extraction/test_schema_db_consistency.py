"""Guard: every db_column declared in extraction_schemas/*.yaml exists in the DB.

This is the reason migrations must land before YAML. loader._verify_db_consistency
raises SchemaDriftError on a missing column, and it runs on EVERY schema load —
so a YAML-first commit takes extraction down at the next process restart, not at
the next test run. This test moves that failure into CI.

Two things the loader's own check does NOT cover, and this file does:

1. The loader validates `db_table` only — which is `proc.bp_*_stg`. But
   persistence.write_raw (persistence.py:170, 188-196) INSERTs into
   `proc.bp_*_raw` with NO column filter, so a field whose column landed on
   `_stg` alone would load cleanly, pass the loader's guard, and then kill every
   extraction of that doc type with UndefinedColumn at INSERT time. `_trgt` is
   the final destination and is checked for the same reason.
2. The loader checks whatever database `.env` happens to name. Every migration
   in this codebase is supposed to be applied to BOTH `bp_sqldb` and
   `bp_testdb`; parametrising over both is what actually enforces that.
   DB_NAME is overridden per-test via the environment — no committed config is
   touched.
"""
from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path

import psycopg2
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.services.extraction.promotion import (  # noqa: E402
    _LINE_RAW_TO_STG,
    _RAW_TO_STG,
)
from src.services.extraction_v3.yaml_schema.loader import (  # noqa: E402
    SCHEMAS_DIR,
    DocSchema,
    SchemaDriftError,
    _get_db_columns,
    _parse_table_ref,
    load_all_schemas,
    load_doc_schema_path,
)

EXPECTED_DOC_TYPES = {"contract", "invoice", "purchase_order", "quote"}

# Every migration in this repo is applied to both. Naming them explicitly is the
# point — reading the name out of .env is exactly the hole this closes.
DATABASES = ["bp_sqldb", "bp_testdb"]


@pytest.fixture
def db_name(request, monkeypatch):
    """Point Settings at one specific database for the duration of a test.

    The environment variable wins over the `.env` file in pydantic-settings, so
    this redirects every Settings() built downstream without editing any
    committed config.
    """
    monkeypatch.setenv("DB_NAME", request.param)
    return request.param


def _connect(dbname: str):
    from config.settings import Settings

    s = Settings()
    return psycopg2.connect(
        host=s.db_host, dbname=dbname, user=s.db_user,
        password=s.db_password, port=s.db_port,
    )


@pytest.mark.parametrize("db_name", DATABASES, indirect=True)
def test_every_schema_loads_against_the_live_database(db_name):
    schemas = load_all_schemas()
    assert EXPECTED_DOC_TYPES.issubset(set(schemas)), (
        f"{db_name}: missing schemas: {EXPECTED_DOC_TYPES - set(schemas)}"
    )
    for name, schema in schemas.items():
        assert schema.fields, f"{name}.yaml declared no fields"


# --- the layers the loader does not check ------------------------------------


def _sibling_tables(doc_type: str, db_table: str, raw_table: str | None) -> dict[str, str]:
    """The other physical layers the same columns have to exist on.

    `contract` is the odd one out: its db_table is proc.bp_contracts (not a
    _stg table) and it has no _trgt layer at all — see promotion.py:29.
    """
    layers: dict[str, str] = {}
    if raw_table:
        layers["_raw"] = raw_table
    if db_table.endswith("_stg"):
        layers["_trgt"] = db_table[: -len("_stg")] + "_trgt"
    return layers


def _missing_columns(conn, table_ref: str, wanted: set[str]) -> set[str] | None:
    """Columns in `wanted` absent from table_ref. None when the table does not
    exist (a layer a doc type genuinely does not have)."""
    tbl_schema, tbl_name = _parse_table_ref(table_ref)
    with conn.cursor() as cur:
        present = _get_db_columns(cur, tbl_schema, tbl_name)
    if not present:
        return None
    return wanted - present


def _declared_columns(fields) -> set[str]:
    return {f.db_column for f in fields if f.db_column is not None}


@pytest.mark.parametrize("db_name", DATABASES, indirect=True)
def test_every_declared_column_exists_on_raw_and_trgt(db_name):
    """write_raw does not filter columns. A column that exists on _stg alone
    passes the loader and then raises UndefinedColumn on the very next INSERT."""
    checked: list[str] = []
    with _connect(db_name) as conn:
        for path in sorted(SCHEMAS_DIR.glob("*.yaml")):
            import yaml

            schema = DocSchema(**yaml.safe_load(path.read_text()))
            if not schema.fields:
                continue
            doc_type = schema.doc_type
            raw_table = (_RAW_TO_STG.get(doc_type) or (None, None))[0]

            header_cols = _declared_columns(schema.fields)
            for layer, table in _sibling_tables(doc_type, schema.db_table, raw_table).items():
                missing = _missing_columns(conn, table, header_cols)
                if missing is None:
                    continue  # this doc type has no such layer
                assert not missing, (
                    f"{db_name}: {doc_type}{layer} table {table} is missing "
                    f"{sorted(missing)} — declared in {path.name} and written "
                    f"unfiltered by persistence.write_raw"
                )
                checked.append(f"{doc_type}{layer}")

            if schema.db_lines_table and schema.line_items:
                line_cols = _declared_columns(schema.line_items.fields)
                line_raw = (_LINE_RAW_TO_STG.get(doc_type) or (None, None))[0]
                for layer, table in _sibling_tables(
                    doc_type, schema.db_lines_table, line_raw
                ).items():
                    missing = _missing_columns(conn, table, line_cols)
                    if missing is None:
                        continue
                    assert not missing, (
                        f"{db_name}: {doc_type} line items{layer} table {table} "
                        f"is missing {sorted(missing)} — declared in {path.name}"
                    )
                    checked.append(f"{doc_type}.lines{layer}")

    # A silently-empty sweep is the failure mode this whole file exists to stop.
    assert len(checked) >= 12, f"{db_name}: only checked {checked}"


@pytest.mark.parametrize("db_name", DATABASES, indirect=True)
def test_the_raw_trgt_sweep_actually_fails_on_a_missing_column(db_name):
    """Prove the guard goes red. A sweep that has never been seen fail is not a
    guard — it is decoration."""
    with _connect(db_name) as conn:
        missing = _missing_columns(
            conn, "proc.bp_invoice_raw", {"invoice_id", "definitely_not_a_column"}
        )
    assert missing == {"definitely_not_a_column"}


@pytest.mark.parametrize("db_name", DATABASES, indirect=True)
def test_a_table_that_does_not_exist_is_reported_as_absent_not_as_complete(db_name):
    """The skip path must be distinguishable from a clean pass, or a typo'd
    table name would look like success."""
    with _connect(db_name) as conn:
        assert _missing_columns(conn, "proc.bp_no_such_table", {"anything"}) is None


# --- the loader's own guard, proven red --------------------------------------


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


def test_env_override_is_what_selects_the_database(monkeypatch):
    """The parametrisation above is only meaningful if DB_NAME actually
    redirects Settings — .env names one database, and reading it was the hole."""
    from config.settings import Settings

    for db in DATABASES:
        monkeypatch.setenv("DB_NAME", db)
        assert Settings().db_name == db
    assert os.environ.get("DB_NAME") in DATABASES
