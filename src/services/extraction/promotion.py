"""Promotion: copy a _raw row's field columns into _stg, run supplier
resolution, delete the _raw row.

Triggered automatically on a clean extraction (no blocking discrepancy)
and on HITL fix completion via NOTIFY on
'extraction_raw_ready_for_promotion'.
"""
from __future__ import annotations

import json
import logging
import select
from typing import Any

import psycopg2

from config.settings import Settings
from src.services.db import get_conn

log = logging.getLogger(__name__)

_RAW_TO_STG = {
    "invoice": ("proc.bp_invoice_raw", "proc.bp_invoice_stg"),
    "purchase_order": ("proc.bp_purchase_order_raw", "proc.bp_purchase_order_stg"),
    "quote": ("proc.bp_quote_raw", "proc.bp_quote_stg"),
    "contract": ("proc.bp_contract_raw", "proc.bp_contracts"),
}

# Line-item table promotion pairs (raw → stg). Contract has no line items.
_LINE_RAW_TO_STG = {
    "invoice": ("proc.bp_invoice_line_items_raw", "proc.bp_invoice_line_items_stg"),
    "purchase_order": ("proc.bp_po_line_items_raw", "proc.bp_po_line_items_stg"),
    "quote": ("proc.bp_quote_line_items_raw", "proc.bp_quote_line_items_stg"),
}

# Line-item PK column name in _stg per doc_type (TEXT NOT NULL — promote
# computes "<doc_pk>-L<line_no>" before INSERT).
_LINE_STG_PK = {
    "invoice": "invoice_line_id",
    "purchase_order": "po_line_id",
    "quote": "quote_line_id",
}
# Line-no column name on _stg (matches the index column on _raw).
_LINE_STG_INDEX = {
    "invoice": "line_no",
    "purchase_order": "line_number",
    "quote": "line_number",
}

# The column that uniquely identifies a document in _stg (used by ON CONFLICT).
# Re-extraction of an existing invoice/PO/quote/contract upserts the _stg row
# rather than failing the promotion.
_STG_PK = {
    "invoice": "invoice_id",
    "purchase_order": "po_id",
    "quote": "quote_id",
    "contract": "contract_id",
}

# Control columns on _raw that must NOT be copied to _stg
_CONTROL_COLS = {
    "raw_id", "doc_pk_candidate", "source_file", "process_monitor_id",
    "pipeline_version", "extracted_at", "parser_snapshot", "promotion_status",
    "promoted_at", "trace_id", "raw_payload",
}


def _stg_columns(cur, stg_table: str) -> list[str]:
    schema, table = stg_table.split(".")
    cur.execute(
        """SELECT column_name FROM information_schema.columns
            WHERE table_schema=%s AND table_name=%s""",
        (schema, table),
    )
    return [r[0] for r in cur.fetchall()]


def promote(raw_id: int, doc_type: str) -> dict[str, Any]:
    """Copy _raw flat columns into _stg, delete _raw, update audit cols.

    Resolves supplier_name → supplier_id via the existing
    supplier_resolver when supplier_id is unset and supplier_name is present.

    Returns {ok: bool, doc_pk: str | None, reason: str | None}.
    """
    from src.services.extraction_v3.supplier_resolver import resolve_or_create_supplier

    raw_t, stg_t = _RAW_TO_STG[doc_type]
    with get_conn() as conn:
        conn.autocommit = False
        cur = conn.cursor()
        try:
            # 1. Read _raw row
            cur.execute(f"SELECT * FROM {raw_t} WHERE raw_id = %s", (raw_id,))
            row = cur.fetchone()
            if row is None:
                conn.rollback()
                return {"ok": False, "reason": "raw_row_missing"}
            col_names = [d.name for d in cur.description]
            raw_data = dict(zip(col_names, row))

            # Resolve supplier_name → supplier_id when needed
            if (
                raw_data.get("supplier_id") is None
                and raw_data.get("supplier_name")
            ):
                try:
                    sid = resolve_or_create_supplier(raw_data["supplier_name"], conn)
                    if sid:
                        raw_data["supplier_id"] = sid
                except Exception as exc:
                    log.warning("supplier resolve failed for %r: %s",
                                raw_data.get("supplier_name"), exc)

            # 2. Intersect with _stg columns
            stg_cols = _stg_columns(cur, stg_t)
            target_cols = [c for c in stg_cols if c in raw_data and c not in _CONTROL_COLS]
            target_vals = [raw_data[c] for c in target_cols]
            # Skip if nothing to promote
            if not target_cols:
                conn.rollback()
                return {"ok": False, "reason": "no_overlapping_columns"}

            placeholders = ", ".join(["%s"] * len(target_cols))
            col_clause = ", ".join(target_cols)
            # Upsert: re-extraction updates the existing _stg row rather than
            # failing on PK collision. ON CONFLICT requires a unique column;
            # we use _STG_PK (doc-type's pk column).
            pk_col = _STG_PK[doc_type]
            updates = ", ".join(
                f"{c} = EXCLUDED.{c}" for c in target_cols if c != pk_col
            )
            if updates:
                sql = (
                    f"INSERT INTO {stg_t} ({col_clause}) VALUES ({placeholders}) "
                    f"ON CONFLICT ({pk_col}) DO UPDATE SET {updates}"
                )
            else:
                sql = (
                    f"INSERT INTO {stg_t} ({col_clause}) VALUES ({placeholders}) "
                    f"ON CONFLICT ({pk_col}) DO NOTHING"
                )
            cur.execute(sql, target_vals)

            # 3. Promote line items if any exist for this raw_id
            if doc_type in _LINE_RAW_TO_STG:
                line_raw_t, line_stg_t = _LINE_RAW_TO_STG[doc_type]
                line_stg_cols = _stg_columns(cur, line_stg_t)
                # Read all line_items_raw rows for this raw_id
                cur.execute(
                    f"SELECT * FROM {line_raw_t} WHERE raw_id = %s "
                    f"ORDER BY line_raw_id", (raw_id,),
                )
                line_rows = cur.fetchall()
                if line_rows:
                    line_cols = [d.name for d in cur.description]
                    # Idempotent: remove any prior line items for this doc_pk
                    doc_pk = raw_data.get("doc_pk_candidate")
                    if doc_pk:
                        pk_col = _STG_PK[doc_type]
                        cur.execute(
                            f"DELETE FROM {line_stg_t} WHERE {pk_col} = %s",
                            (doc_pk,),
                        )
                    line_pk_col = _LINE_STG_PK.get(doc_type)
                    line_idx_col = _LINE_STG_INDEX.get(doc_type)
                    for lrow in line_rows:
                        lrow_data = dict(zip(line_cols, lrow))
                        # carry doc_pk onto the line for FK
                        if doc_pk:
                            lrow_data[_STG_PK[doc_type]] = doc_pk
                        # Generate line PK ("<doc_pk>-L<line_no>") if the _stg
                        # has a NOT NULL line PK column.
                        if doc_pk and line_pk_col and line_pk_col in line_stg_cols:
                            line_no_val = lrow_data.get(line_idx_col) if line_idx_col else None
                            if line_no_val is None:
                                line_no_val = lrow_data.get("line_raw_id")
                            lrow_data[line_pk_col] = f"{doc_pk}-L{line_no_val}"
                        target_line_cols = [
                            c for c in line_stg_cols
                            if c in lrow_data and c not in (
                                "line_raw_id", "raw_id", "created_date",
                                "created_by", "last_modified_by", "last_modified_date",
                            )
                        ]
                        if not target_line_cols:
                            continue
                        line_vals = [lrow_data[c] for c in target_line_cols]
                        line_ph = ", ".join(["%s"] * len(target_line_cols))
                        line_cc = ", ".join(target_line_cols)
                        cur.execute(
                            f"INSERT INTO {line_stg_t} ({line_cc}) VALUES ({line_ph})",
                            line_vals,
                        )

            # 4. Update _raw to promoted (audit) then delete
            cur.execute(
                f"UPDATE {raw_t} SET promotion_status='promoted', promoted_at=NOW() "
                f"WHERE raw_id=%s", (raw_id,),
            )
            # Keep the _raw row as an audit trail OR delete? Spec says delete on
            # clean promotion; keep on discrepancy. Delete here — the data is
            # now in _stg, and provenance_v3 retains the bbox/evidence.
            cur.execute(f"DELETE FROM {raw_t} WHERE raw_id=%s", (raw_id,))

            conn.commit()
            return {"ok": True, "doc_pk": raw_data.get("doc_pk_candidate")}
        except Exception as exc:
            conn.rollback()
            log.exception("promotion failed for raw_id=%s doc_type=%s: %s",
                          raw_id, doc_type, exc)
            return {"ok": False, "reason": str(exc)}


def apply_hitl_fixes_and_promote(raw_id: int, doc_type: str) -> dict[str, Any]:
    """Read all resolved discrepancies for this raw_id, apply their
    resolved_value updates to the _raw row, then promote.

    Called by the NOTIFY listener when the DB trigger fires
    'extraction_raw_ready_for_promotion'.
    """
    raw_t = _RAW_TO_STG[doc_type][0]
    with get_conn() as conn:
        conn.autocommit = False
        cur = conn.cursor()
        try:
            cur.execute("""
                SELECT field_name, resolved_value, resolution_action
                  FROM proc.bp_extraction_discrepancy
                 WHERE raw_id=%s AND status='resolved'
                   AND blocks_promotion=TRUE
            """, (raw_id,))
            fixes = cur.fetchall()
            for field_name, resolved_value, action in fixes:
                if action == "apply_value":
                    cur.execute(
                        f"UPDATE {raw_t} SET {field_name} = %s WHERE raw_id=%s",
                        (resolved_value, raw_id),
                    )
                elif action == "keep_null":
                    cur.execute(
                        f"UPDATE {raw_t} SET {field_name} = NULL WHERE raw_id=%s",
                        (raw_id,),
                    )
                # 'dismiss' does nothing to _raw
            conn.commit()
        except Exception as exc:
            conn.rollback()
            log.exception("apply_hitl_fixes failed: %s", exc)
            return {"ok": False, "reason": str(exc)}

    # Now promote
    return promote(raw_id, doc_type)


def run_listener(stop_event=None) -> None:
    """Listen on 'extraction_raw_ready_for_promotion' channel and process
    NOTIFY events sequentially. Run in a dedicated worker."""
    s = Settings()
    conn = psycopg2.connect(
        host=s.db_host, dbname=s.db_name, user=s.db_user,
        password=s.db_password, port=s.db_port,
    )
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("LISTEN extraction_raw_ready_for_promotion;")
    log.info("promotion listener armed on channel extraction_raw_ready_for_promotion")
    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            if not select.select([conn], [], [], 1.0)[0]:
                continue
            conn.poll()
            while conn.notifies:
                notify = conn.notifies.pop(0)
                try:
                    payload = json.loads(notify.payload)
                except Exception:
                    log.warning("malformed notify payload: %r", notify.payload)
                    continue
                doc_type = payload.get("doc_type")
                raw_id = payload.get("raw_id")
                if not (doc_type and raw_id):
                    continue
                log.info("notify received: raw_id=%s doc_type=%s", raw_id, doc_type)
                result = apply_hitl_fixes_and_promote(int(raw_id), doc_type)
                log.info("promotion result: %s", result)
    finally:
        try:
            conn.close()
        except Exception:
            pass
