"""Single-transaction persistence: writes proc.bp_<doctype> + line items +
proc.bp_extraction_provenance_v3 atomically. Any failure rolls back all of it.

C9 contract: every committed value carries an evidence span; if we cannot
record provenance we do not commit the extraction data either.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from src.services.extraction_v3.schemas.result import ExtractionResult, CommittedField
from src.services.extraction_v3.yaml_schema.loader import load_doc_schema, DocSchema
from src.services.db import get_conn

log = logging.getLogger(__name__)


def _doc_pk_field(doc_type: str) -> str:
    return {
        "invoice": "invoice_id",
        "purchase_order": "po_id",
        "quote": "quote_id",
        "contract": "contract_id",
    }.get(doc_type, "id")


def _split_line_items(
    committed: list[CommittedField],
) -> tuple[list[CommittedField], dict[int, dict[str, CommittedField]]]:
    """Separate header CommittedFields from line-item CommittedFields.

    Line item field_paths look like 'line_items[3].amount'.
    Returns (header_fields, line_items_by_idx).
    """
    header: list[CommittedField] = []
    lines: dict[int, dict[str, CommittedField]] = {}
    for cf in committed:
        if cf.field_path.startswith("line_items["):
            try:
                idx = int(cf.field_path.split("[", 1)[1].split("]", 1)[0])
                key = cf.field_path.split("].", 1)[1]
            except (IndexError, ValueError):
                header.append(cf)
                continue
            lines.setdefault(idx, {})[key] = cf
        else:
            header.append(cf)
    return header, lines


def _build_header_insert(
    cur: Any,
    doc_type: str,
    doc_pk: str,
    schema: DocSchema,
    header_cfs: list[CommittedField],
) -> None:
    """INSERT/UPSERT the header row into proc.bp_<doctype>."""
    field_to_db_col: dict[str, str] = {}
    for f in schema.fields:
        if f.db_column:
            field_to_db_col[f.name] = f.db_column

    pk_col = _doc_pk_field(doc_type)
    cols = [pk_col]
    vals: list[Any] = [doc_pk]

    for cf in header_cfs:
        db_col = field_to_db_col.get(cf.field_path)
        if not db_col:
            # Field doesn't map directly to a column (e.g. supplier_name →
            # resolves to supplier_id via a lookup; not handled in Plan 1).
            continue
        if db_col == pk_col:
            continue  # already in the list as the PK
        cols.append(db_col)
        vals.append(cf.value)

    placeholders = ",".join(["%s"] * len(vals))
    col_list = ",".join(cols)
    update_cols = [c for c in cols if c != pk_col]
    if update_cols:
        update_set = ",".join(f"{c}=EXCLUDED.{c}" for c in update_cols)
        conflict_clause = f"ON CONFLICT ({pk_col}) DO UPDATE SET {update_set}"
    else:
        # Only the PK is being inserted; nothing to update on conflict.
        conflict_clause = f"ON CONFLICT ({pk_col}) DO NOTHING"
    sql = (
        f"INSERT INTO {schema.db_table} ({col_list}) VALUES ({placeholders}) "
        f"{conflict_clause}"
    )
    cur.execute(sql, vals)


def _build_line_items_inserts(
    cur: Any,
    doc_type: str,
    doc_pk: str,
    schema: DocSchema,
    lines_by_idx: dict[int, dict[str, CommittedField]],
) -> None:
    if not schema.db_lines_table or not schema.line_items:
        return

    line_field_db_cols: dict[str, str] = {}
    for f in schema.line_items.fields:
        if f.db_column:
            line_field_db_cols[f.name] = f.db_column

    parent_fk = _doc_pk_field(doc_type)
    line_pk_col = {
        "invoice": "invoice_line_id",
        "purchase_order": "po_line_id",
        "quote": "quote_line_id",
    }.get(doc_type, "line_id")

    # Delete any prior lines for this doc (idempotency for re-extracts).
    cur.execute(
        f"DELETE FROM {schema.db_lines_table} WHERE {parent_fk} = %s", (doc_pk,)
    )

    for idx, fields_dict in sorted(lines_by_idx.items()):
        cols = [line_pk_col, parent_fk]
        vals: list[Any] = [f"{doc_pk}-L{idx}", doc_pk]

        # Add line_no column when the schema doesn't map it explicitly.
        if "line_no" not in line_field_db_cols:
            cols.append("line_no")
            vals.append(idx)

        for fname, cf in fields_dict.items():
            db_col = line_field_db_cols.get(fname)
            if not db_col:
                continue
            cols.append(db_col)
            vals.append(cf.value)

        placeholders = ",".join(["%s"] * len(vals))
        col_list = ",".join(cols)
        sql = (
            f"INSERT INTO {schema.db_lines_table} ({col_list}) VALUES ({placeholders}) "
            f"ON CONFLICT ({line_pk_col}) DO NOTHING"
        )
        cur.execute(sql, vals)


def _build_provenance_inserts(
    cur: Any,
    doc_type: str,
    doc_pk: str,
    committed: list[CommittedField],
    pipeline_version: str,
) -> None:
    rows = []
    for cf in committed:
        rows.append((
            doc_type,
            doc_pk,
            cf.field_path,
            cf.value,
            cf.page,
            cf.bbox[0],
            cf.bbox[1],
            cf.bbox[2],
            cf.bbox[3],
            cf.evidence_text,
            cf.model,
            cf.model_confidence,
            json.dumps(cf.judge_actions) if cf.judge_actions else None,
            cf.final_confidence,
            pipeline_version,
        ))
    if not rows:
        return
    cur.executemany(
        """
        INSERT INTO proc.bp_extraction_provenance_v3
        (doc_type, doc_pk, field_path, value, page,
         bbox_x0, bbox_y0, bbox_x1, bbox_y1,
         evidence_text, model, model_confidence,
         judge_actions, final_confidence, pipeline_version)
        VALUES (%s, %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s)
        """,
        rows,
    )


def persist(result: ExtractionResult) -> None:
    """Persist an ExtractionResult atomically. Any failure rolls back everything.

    Writes:
      1. Header row to proc.bp_<doctype>  (UPSERT on conflict — idempotent).
      2. Line items to proc.bp_<doctype>_line_items  (DELETE + INSERT — clean rewrite).
      3. One provenance row per CommittedField to proc.bp_extraction_provenance_v3.

    If ANY of the three writes fails the entire transaction is rolled back, so a
    failed provenance write never leaves orphan extraction data behind.

    Raises ValueError if doc_pk is None (cannot write a row without a primary key).
    """
    if result.doc_pk is None:
        raise ValueError(
            f"cannot persist ExtractionResult without doc_pk for {result.doc_type}"
        )

    schema = load_doc_schema(result.doc_type)
    header_cfs, lines_by_idx = _split_line_items(result.committed)

    with get_conn() as conn:
        # get_conn() returns connections with autocommit=True by default.
        # Override to get explicit transaction semantics for this call.
        prior_autocommit = conn.autocommit
        conn.autocommit = False
        try:
            with conn.cursor() as cur:
                _build_header_insert(
                    cur, result.doc_type, result.doc_pk, schema, header_cfs
                )
                _build_line_items_inserts(
                    cur, result.doc_type, result.doc_pk, schema, lines_by_idx
                )
                _build_provenance_inserts(
                    cur,
                    result.doc_type,
                    result.doc_pk,
                    result.committed,
                    result.pipeline_version,
                )
            conn.commit()
            log.info(
                "persist OK doc_type=%s doc_pk=%s fields=%d",
                result.doc_type,
                result.doc_pk,
                len(result.committed),
            )
        except Exception:
            conn.rollback()
            log.exception(
                "persist ROLLBACK doc_type=%s doc_pk=%s",
                result.doc_type,
                result.doc_pk,
            )
            raise
        finally:
            conn.autocommit = prior_autocommit
