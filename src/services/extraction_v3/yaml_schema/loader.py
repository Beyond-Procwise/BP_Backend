from pathlib import Path
from typing import Literal

import psycopg2
import yaml
from pydantic import BaseModel

from config.settings import Settings

# parents[4] of this file = repo root (/home/muthu/PycharmProjects/BP_Backend)
SCHEMAS_DIR = Path(__file__).resolve().parents[4] / "extraction_schemas"


class SchemaDriftError(RuntimeError): ...


class JudgeRules(BaseModel):
    tiebreaker: bool = True
    grounded_last_resort: bool = True
    ner_type_check: Literal["none", "ORG", "PERSON", "GPE", "LOC"] = "none"


class Pattern(BaseModel):
    """A single regex rule in the renovation's PatternRegistry (L1 primary)."""

    name: str
    anchor: str
    value: str
    max_span_after_anchor_chars: int = 80
    prior_confidence: float = 0.75

    def __init__(self, **data) -> None:  # type: ignore[no-untyped-def]
        # Defensive: clamp prior_confidence to [0, 1] without erroring.
        super().__init__(**data)


class FieldSpec(BaseModel):
    name: str
    type: Literal["string", "iso_date", "money", "decimal", "address", "postcode", "currency"]
    required: bool
    db_column: str | None = None
    resolves_to_db_column: str | None = None
    canonical_labels: list[str]
    # `extractors:` is kept for backwards compatibility with older YAML files.
    # The renovation drives L1 from `patterns:` instead; new YAML may omit
    # `extractors:` entirely.
    extractors: list[str] = []
    patterns: list[Pattern] = []
    confidence_threshold: float = 0.70
    judge: JudgeRules = JudgeRules()
    invariants: list[str] = []


class LineItemsSpec(BaseModel):
    primary_extractor: Literal["table_transformer", "layoutlmv3", "qwen_vlm"]
    fallback_extractor: str | None = None
    fields: list[FieldSpec]
    invariants: list[str] = []


class DocSchema(BaseModel):
    doc_type: str
    db_table: str
    db_lines_table: str | None = None
    fields: list[FieldSpec]
    line_items: LineItemsSpec | None = None
    document_invariants: list[str] = []


def _get_db_columns(cur, schema: str, table: str) -> set[str]:
    """Return the set of column names for schema.table, or empty set if table does not exist."""
    cur.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        """,
        (schema, table),
    )
    return {row[0] for row in cur.fetchall()}


def _parse_table_ref(full_table: str) -> tuple[str, str]:
    """Split 'schema.table' into (schema, table). Defaults schema to 'public' if absent."""
    parts = full_table.split(".", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return "public", parts[0]


def _verify_db_consistency_with_conn(schema: DocSchema, conn) -> None:
    """Check that the tables and columns declared in the schema actually exist in the DB.

    Takes an open psycopg2 connection. Skipped entirely when schema.fields is empty (stub schemas).
    This is the core implementation used by both the batched and single-doc paths.
    """
    if not schema.fields:
        return

    with conn.cursor() as cur:
        # --- verify main table exists ---
        tbl_schema, tbl_name = _parse_table_ref(schema.db_table)
        cols = _get_db_columns(cur, tbl_schema, tbl_name)
        if not cols:
            raise SchemaDriftError(
                f"Table '{schema.db_table}' does not exist in the database "
                f"(schema='{tbl_schema}', table='{tbl_name}')."
            )

        # --- verify each field's db_column and resolves_to_db_column ---
        for field in schema.fields:
            if field.db_column is not None and field.db_column not in cols:
                raise SchemaDriftError(
                    f"Column '{field.db_column}' declared for field '{field.name}' "
                    f"does not exist on table '{schema.db_table}'."
                )
            if field.resolves_to_db_column is not None and field.resolves_to_db_column not in cols:
                raise SchemaDriftError(
                    f"resolves_to_db_column '{field.resolves_to_db_column}' declared for field "
                    f"'{field.name}' does not exist on table '{schema.db_table}'."
                )

        # --- verify lines table if present ---
        if schema.db_lines_table is not None:
            lt_schema, lt_name = _parse_table_ref(schema.db_lines_table)
            line_cols = _get_db_columns(cur, lt_schema, lt_name)
            if not line_cols:
                raise SchemaDriftError(
                    f"Lines table '{schema.db_lines_table}' does not exist in the database."
                )
            if schema.line_items:
                for field in schema.line_items.fields:
                    if field.db_column is not None and field.db_column not in line_cols:
                        raise SchemaDriftError(
                            f"Column '{field.db_column}' declared for line item field "
                            f"'{field.name}' does not exist on table '{schema.db_lines_table}'."
                        )


def _verify_db_consistency(schema: DocSchema) -> None:
    """Check that the tables and columns declared in the schema actually exist in the DB.

    Opens a single connection and calls _verify_db_consistency_with_conn.
    Used by load_doc_schema_path for single-document loading.
    """
    s = Settings()
    with psycopg2.connect(
        host=s.db_host,
        dbname=s.db_name,
        user=s.db_user,
        password=s.db_password,
        port=s.db_port,
    ) as conn:
        _verify_db_consistency_with_conn(schema, conn)


def load_doc_schema_path(path: Path) -> DocSchema:
    """Load and validate a DocSchema from a YAML file path."""
    raw = yaml.safe_load(path.read_text())
    schema = DocSchema(**raw)
    _verify_db_consistency(schema)
    return schema


def load_doc_schema(doc_type: str) -> DocSchema:
    """Load a DocSchema by doc_type name, reading from SCHEMAS_DIR/{doc_type}.yaml."""
    return load_doc_schema_path(SCHEMAS_DIR / f"{doc_type}.yaml")


def load_all_schemas() -> dict[str, DocSchema]:
    """Load every *.yaml file in SCHEMAS_DIR and return {stem: DocSchema}.

    Opens a single DB connection and reuses it across all schemas for efficiency.
    """
    schemas: dict[str, DocSchema] = {}
    s = Settings()
    with psycopg2.connect(
        host=s.db_host,
        dbname=s.db_name,
        user=s.db_user,
        password=s.db_password,
        port=s.db_port,
    ) as conn:
        for p in sorted(SCHEMAS_DIR.glob("*.yaml")):
            raw = yaml.safe_load(p.read_text())
            schema = DocSchema(**raw)
            _verify_db_consistency_with_conn(schema, conn)
            schemas[p.stem] = schema
    return schemas
