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
from src.services.extraction_v3.supplier_resolver import resolve_or_create_supplier

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


def _get_col_max_lengths(cur: Any, table: str) -> dict[str, int]:
    """Return a map of column_name → character_maximum_length for varchar columns.

    Columns without a max length (text, integer, etc.) are not included.
    Used to prevent StringDataRightTruncation errors when persisting extracted values.
    """
    parts = table.split(".", 1)
    tbl_schema = parts[0] if len(parts) == 2 else "public"
    tbl_name = parts[1] if len(parts) == 2 else parts[0]
    cur.execute(
        """
        SELECT column_name, character_maximum_length
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
          AND character_maximum_length IS NOT NULL
        """,
        (tbl_schema, tbl_name),
    )
    return {row[0]: row[1] for row in cur.fetchall()}


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

    # Fetch column length constraints to avoid varchar truncation errors
    col_max_lengths = _get_col_max_lengths(cur, schema.db_table)

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
        # Truncate string values that exceed the column's varchar limit
        val = cf.value
        max_len = col_max_lengths.get(db_col)
        if max_len is not None and isinstance(val, str) and len(val) > max_len:
            log.warning(
                "Truncating field %s (col=%s) from %d to %d chars for %s",
                cf.field_path, db_col, len(val), max_len, doc_pk,
            )
            val = val[:max_len]
        vals.append(val)

    # Audit columns — always set on every V3 write.
    _AUDIT_AGENT = "ExtractionV3"
    for audit_col in ("created_date", "last_modified_date"):
        if audit_col not in cols:
            cols.append(audit_col)
            vals.append("NOW()")  # placeholder — swapped below
    for audit_col in ("created_by", "last_modified_by"):
        if audit_col not in cols:
            cols.append(audit_col)
            vals.append(_AUDIT_AGENT)
    # Replace the "NOW()" placeholder strings with a SQL NOW() expression by
    # switching to a parameterised query that has a literal NOW() in the SQL.
    # We do this by building the values list without NOW() and injecting it
    # directly into the SQL string for those two cols.
    _now_cols = {"created_date", "last_modified_date"}
    final_cols = []
    final_vals = []
    now_col_positions = []
    for i, (c, v) in enumerate(zip(cols, vals)):
        if c in _now_cols:
            now_col_positions.append(len(final_cols))
            final_cols.append(c)
            # value slot will be filled by NOW() in SQL, not a parameter
        else:
            final_cols.append(c)
            final_vals.append(v)

    # Build placeholders: %s for params, NOW() for timestamp cols
    ph_parts = []
    param_idx = 0
    for c in final_cols:
        if c in _now_cols:
            ph_parts.append("NOW()")
        else:
            ph_parts.append("%s")
    cols = final_cols
    vals = final_vals

    placeholders = ",".join(ph_parts)  # mix of %s and NOW()
    col_list = ",".join(cols)
    update_cols = [c for c in cols if c != pk_col]
    if update_cols:
        # For ON CONFLICT SET, use NOW() for timestamp audit columns
        set_parts = []
        for c in update_cols:
            if c in _now_cols:
                set_parts.append(f"{c}=NOW()")
            else:
                set_parts.append(f"{c}=EXCLUDED.{c}")
        update_set = ",".join(set_parts)
        conflict_clause = f"ON CONFLICT ({pk_col}) DO UPDATE SET {update_set}"
    else:
        # Only the PK is being inserted; nothing to update on conflict.
        conflict_clause = f"ON CONFLICT ({pk_col}) DO NOTHING"
    sql = (
        f"INSERT INTO {schema.db_table} ({col_list}) VALUES ({placeholders}) "
        f"{conflict_clause}"
    )
    cur.execute(sql, vals)


def _coerce_to_db_value(raw: str, line_items_spec, field_name: str) -> Any:
    """Coerce a raw string value to a DB-compatible type for line item fields.

    Money/decimal fields (unit_price, line_total, quantity) are stored as
    numeric(18,2) in the DB. The extracted value may be "$120.00" — we strip
    currency symbols and commas before inserting.

    Returns the original string for non-numeric fields.
    """
    if line_items_spec is None:
        return raw
    for f in line_items_spec.fields:
        if f.name != field_name:
            continue
        if f.type in ("money", "decimal"):
            from src.services.extraction_v2.parsers.amounts import parse_amount
            parsed = parse_amount(raw)
            if parsed is not None:
                return str(parsed)
        break
    return raw


_LINE_ITEMS_COL_MAX_LENGTHS: dict[str, dict[str, int]] = {}  # table → {col: max_len}
_LINE_ITEMS_COL_TYPES: dict[str, dict[str, str]] = {}  # table → {col: data_type}


def _ensure_col_metadata(cur: Any, table: str) -> None:
    """Cache column data_type and max_length for a lines table."""
    if table in _LINE_ITEMS_COL_TYPES:
        return
    parts = table.split(".", 1)
    tbl_schema = parts[0] if len(parts) == 2 else "public"
    tbl_name = parts[1] if len(parts) == 2 else parts[0]
    cur.execute(
        """
        SELECT column_name, data_type, character_maximum_length
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        """,
        (tbl_schema, tbl_name),
    )
    max_lens: dict[str, int] = {}
    col_types: dict[str, str] = {}
    for col, dtype, max_len in cur.fetchall():
        col_types[col] = dtype
        if max_len is not None:
            max_lens[col] = max_len
    _LINE_ITEMS_COL_MAX_LENGTHS[table] = max_lens
    _LINE_ITEMS_COL_TYPES[table] = col_types


def _coerce_val_for_col(val: Any, col_name: str, lines_table: str) -> Any:
    """Coerce a value to match the DB column type/length constraints.

    - varchar(N): truncate string to N chars
    - integer / smallint: convert "16.00" → 16 (int)
    - numeric: strip currency symbols, convert to string
    """
    if not isinstance(val, str):
        return val
    col_types = _LINE_ITEMS_COL_TYPES.get(lines_table, {})
    dtype = col_types.get(col_name, "")
    if dtype in ("integer", "smallint", "bigint"):
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return val
    max_lens = _LINE_ITEMS_COL_MAX_LENGTHS.get(lines_table, {})
    max_len = max_lens.get(col_name)
    if max_len is not None and len(val) > max_len:
        return val[:max_len]
    return val


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

    # Fetch column metadata (types, max lengths) for this lines table once
    _ensure_col_metadata(cur, schema.db_lines_table)

    # Detect which line-sequence column this table uses (line_no or line_number).
    # Only add it if it exists in the DB and isn't already covered by the schema.
    _line_seq_candidates = ["line_no", "line_number"]
    _line_seq_col = None
    if not any(c in line_field_db_cols.values() for c in _line_seq_candidates):
        # Fetch the actual column names from the DB to find the right one
        cur.execute(
            """
            SELECT column_name FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
              AND column_name IN ('line_no', 'line_number')
            LIMIT 1
            """,
            tuple(schema.db_lines_table.split(".", 1)) if "." in schema.db_lines_table
            else ("public", schema.db_lines_table),
        )
        row = cur.fetchone()
        if row:
            _line_seq_col = row[0]

    for idx, fields_dict in sorted(lines_by_idx.items()):
        cols = [line_pk_col, parent_fk]
        vals: list[Any] = [f"{doc_pk}-L{idx}", doc_pk]

        # Add the line sequence column (line_no or line_number) when it exists
        # in the table and isn't already mapped by the schema.
        if _line_seq_col:
            cols.append(_line_seq_col)
            vals.append(idx)

        for fname, cf in fields_dict.items():
            db_col = line_field_db_cols.get(fname)
            if not db_col:
                continue
            cols.append(db_col)
            # Coerce money/decimal string values ("$120.00") to plain numeric
            # before inserting into numeric DB columns (which reject currency symbols).
            val = cf.value
            if isinstance(val, str):
                val = _coerce_to_db_value(val, schema.line_items, fname)
            # Apply column-level type/length constraints (integer truncation,
            # varchar max-length, etc.)
            val = _coerce_val_for_col(val, db_col, schema.db_lines_table)
            vals.append(val)

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


def _resolve_supplier_fields(
    schema: DocSchema,
    header_cfs: list[CommittedField],
    conn: Any,
) -> list[CommittedField]:
    """For any field with resolves_to_db_column, call supplier resolver and
    inject a synthetic CommittedField for the resolved column.

    Specifically: when ``supplier_name`` is committed AND ``supplier_id`` is
    NOT already committed, call resolve_or_create_supplier to get a canonical
    supplier_id and inject it into header_cfs.

    supplier_name itself is NOT written to bp_invoice (it has db_column=null).
    This function does NOT remove it from header_cfs; _build_header_insert
    already skips fields with no db_column mapping.
    """
    # Build lookup of field_path → resolves_to_db_column for schema fields
    resolves_map: dict[str, str] = {}
    for f in schema.fields:
        if f.resolves_to_db_column:
            resolves_map[f.name] = f.resolves_to_db_column

    if not resolves_map:
        return header_cfs

    # Check what's already in header_cfs (avoid overwriting an existing supplier_id)
    existing_field_paths = {cf.field_path for cf in header_cfs}

    extra: list[CommittedField] = []
    for cf in header_cfs:
        target_col = resolves_map.get(cf.field_path)
        if not target_col:
            continue
        # target_col is e.g. "supplier_id". Skip if already committed.
        if target_col in existing_field_paths:
            log.debug(
                "persist: %s already committed — skipping resolver", target_col
            )
            continue

        # Call the resolver (supplier_name → supplier_id)
        resolved_id = resolve_or_create_supplier(cf.value, conn)
        if resolved_id:
            log.info(
                "persist: resolved %s='%s' → %s='%s'",
                cf.field_path, cf.value, target_col, resolved_id,
            )
            synthetic = CommittedField(
                field_path=target_col,
                value=resolved_id,
                page=cf.page,
                bbox=cf.bbox,
                evidence_text=cf.evidence_text,
                model=cf.model,
                model_confidence=cf.model_confidence,
                judge_actions=cf.judge_actions,
                final_confidence=cf.final_confidence,
            )
            extra.append(synthetic)
        else:
            log.warning(
                "persist: supplier resolver returned None for '%s' — supplier_id will be NULL",
                cf.value,
            )

    return header_cfs + extra


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
            # --- Supplier resolution (runs inside transaction so any INSERT
            #     into bp_supplier is rolled back on failure) ---
            header_cfs = _resolve_supplier_fields(schema, header_cfs, conn)

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
