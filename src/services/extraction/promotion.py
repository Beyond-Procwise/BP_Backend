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
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Optional

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


# ---------------------------------------------------------------------------
# Confidence + discrepancy helpers
# ---------------------------------------------------------------------------

# Fields used to compute the per-row confidence score. Required fields
# count 2x (they're the critical ones); secondary fields count 1x. The
# audit columns and FK-resolved values are excluded so the score reflects
# only what the extraction layers produced.
_EXPECTED_FIELDS_FOR_CONFIDENCE: dict[str, list[str]] = {
    "invoice": [
        "invoice_id", "invoice_date", "invoice_amount", "currency",       # required
        "supplier_id", "buyer_id", "tax_amount", "tax_percent",
        "invoice_total_incl_tax", "country", "region",
        "exchange_rate_to_usd", "converted_amount_usd", "payment_terms",
    ],
    "purchase_order": [
        "po_id", "supplier_name", "order_date", "total_amount", "currency",  # required
        "supplier_id", "buyer_id", "tax_amount", "tax_percent",
        "total_amount_incl_tax", "ship_to_country", "delivery_region",
        "exchange_rate_to_usd", "converted_amount_usd", "payment_terms",
        "expected_delivery_date",
    ],
    "quote": [
        "quote_id", "quote_date", "currency", "total_amount_incl_tax",     # required
        "supplier_id", "buyer_id", "total_amount", "tax_amount", "tax_percent",
        "country", "region", "supplier_address", "buyer_address",
    ],
}


def _compute_confidence_score(
    doc_type: str, row: dict[str, Any], required: set[str],
) -> Decimal | None:
    """Return a 0–100 score reflecting how complete this row is.

    Required fields contribute 2 points each (filled) or 0 (NULL).
    Secondary fields contribute 1 point each. Score = (achieved / max) × 100.

    A row that has every required field filled but no secondaries lands at
    ~50%. A row with everything filled lands at 100%. NULL on a required
    field caps the score below 50%.
    """
    expected = _EXPECTED_FIELDS_FOR_CONFIDENCE.get(doc_type)
    if not expected:
        return None
    achieved = 0
    maximum = 0
    for f in expected:
        weight = 2 if f in required else 1
        maximum += weight
        val = row.get(f)
        if val is not None and val != "":
            achieved += weight
    if maximum == 0:
        return None
    pct = (achieved / maximum) * 100.0
    return Decimal(f"{pct:.2f}")


# Field triples per doc_type used for tax/total reconciliation:
# (subtotal_field, tax_field, total_incl_tax_field). The relation we
# enforce is: subtotal + tax_amount ≈ total_incl_tax (rounding ≤ 0.50).
_TAX_TOTAL_TRIPLE = {
    "invoice": ("invoice_amount", "tax_amount", "invoice_total_incl_tax"),
    "purchase_order": ("total_amount", "tax_amount", "total_amount_incl_tax"),
    "quote": ("total_amount", "tax_amount", "total_amount_incl_tax"),
}
_TAX_PERCENT_TRIPLE = {
    "invoice": ("invoice_amount", "tax_percent", "tax_amount"),
    "purchase_order": ("total_amount", "tax_percent", "tax_amount"),
    "quote": ("total_amount", "tax_percent", "tax_amount"),
}
_DEFAULT_DISCREPANCY_TOLERANCE = Decimal("0.50")


def _to_decimal(v: Any) -> Decimal | None:
    if v is None:
        return None
    try:
        return Decimal(str(v))
    except (InvalidOperation, ValueError, TypeError):
        return None


def _check_tax_total_consistency(
    cur, doc_type: str, raw_id: int, row: dict[str, Any],
) -> int:
    """Record discrepancies when extracted money fields don't reconcile.

    Two relations enforced (both must hold when all three values are present):
      1. subtotal + tax_amount ≈ total_incl_tax  (tolerance 0.50)
      2. subtotal × tax_percent / 100 ≈ tax_amount  (tolerance 0.50)

    On mismatch: INSERT into proc.bp_extraction_discrepancy with
    severity='warning', blocks_promotion=false (per user direction —
    surface but don't auto-fix). Returns the number of discrepancies
    logged for this row.
    """
    n_logged = 0
    pk_col = _STG_PK.get(doc_type)
    doc_pk = row.get(pk_col) if pk_col else None
    source_file = row.get("source_file")

    def _log(field_name: str, issue: str, expected, computed, notes):
        cur.execute(
            """
            INSERT INTO proc.bp_extraction_discrepancy
                (doc_type, raw_id, source_file, doc_pk_candidate,
                 field_name, raw_value, expected_value, computed_value,
                 issue_type, severity, status, notes, blocks_promotion)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                doc_type, raw_id, source_file, str(doc_pk) if doc_pk else None,
                field_name,
                str(row.get(field_name)) if row.get(field_name) is not None else None,
                str(expected) if expected is not None else None,
                str(computed) if computed is not None else None,
                issue, "warning", "open", notes, False,
            ),
        )

    # 1. subtotal + tax = total
    sub_f, tax_f, tot_f = _TAX_TOTAL_TRIPLE.get(doc_type, (None, None, None))
    if sub_f and tax_f and tot_f:
        s = _to_decimal(row.get(sub_f))
        t = _to_decimal(row.get(tax_f))
        g = _to_decimal(row.get(tot_f))
        if s is not None and t is not None and g is not None:
            expected_total = (s + t).quantize(Decimal("0.01"))
            if abs(expected_total - g) > _DEFAULT_DISCREPANCY_TOLERANCE:
                _log(
                    field_name=tot_f,
                    issue="sum_mismatch",
                    expected=expected_total,
                    computed=g,
                    notes=(
                        f"{sub_f}({s}) + {tax_f}({t}) = {expected_total} "
                        f"but {tot_f}={g}. Diff "
                        f"{(g - expected_total).copy_abs()}."
                    ),
                )
                n_logged += 1

    # 2. subtotal × tax_percent / 100 = tax_amount
    sub_f2, pct_f, tax_f2 = _TAX_PERCENT_TRIPLE.get(doc_type, (None, None, None))
    if sub_f2 and pct_f and tax_f2:
        s = _to_decimal(row.get(sub_f2))
        p = _to_decimal(row.get(pct_f))
        t = _to_decimal(row.get(tax_f2))
        if s is not None and p is not None and t is not None and p > 0:
            expected_tax = (s * p / Decimal("100")).quantize(Decimal("0.01"))
            if abs(expected_tax - t) > _DEFAULT_DISCREPANCY_TOLERANCE:
                _log(
                    field_name=tax_f2,
                    issue="tax_percent_mismatch",
                    expected=expected_tax,
                    computed=t,
                    notes=(
                        f"{sub_f2}({s}) × {pct_f}({p}%) = {expected_tax} "
                        f"but {tax_f2}={t}."
                    ),
                )
                n_logged += 1

    return n_logged


def promote(raw_id: int, doc_type: str) -> dict[str, Any]:
    """Copy _raw flat columns into _stg, delete _raw, update audit cols.

    Resolves supplier_name → supplier_id via the existing
    supplier_resolver when supplier_id is unset and supplier_name is present.

    Returns {ok: bool, doc_pk: str | None, reason: str | None}.

    NOTE: As of the AgentNick promotion (2026-05-21), `extraction.dispatch`
    runs the context_layer synthesis BEFORE writing _raw — so by the time
    we get here, raw_data already carries AgentNick's authoritative values.
    We no longer re-call synthesize() here; that would burn a second LLM
    pass and risk drift. HITL-applied fixes (apply_hitl_fixes_and_promote)
    are honoured because they patch _raw columns BEFORE this function runs.
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

            # Required-field set — used for the AgentNick audit log below
            # and for honouring HITL apply_value fixes that clear noisy
            # values into NULL.
            try:
                from src.services.extraction.pattern_registry import get_registry
                _reg = get_registry(doc_type)
                _required = {f.name for f in _reg.schema.fields if f.required}
            except Exception:  # noqa: BLE001
                _required = set()

            # Load full_text from parser_snapshot for the filename-hint
            # supplier-resolution fallback (used a few lines down when
            # context_layer couldn't determine the supplier). Without this
            # the fallback's grounding check always fails and rows like
            # the RUBILOGY/VALUED-MERCHANT quotes land in _stg with NULL
            # supplier_id even when the doc clearly names the supplier.
            ps = raw_data.get("parser_snapshot")
            full_text = ""
            if isinstance(ps, dict):
                full_text = (ps.get("full_text") or "")
            elif isinstance(ps, str):
                try:
                    full_text = (json.loads(ps).get("full_text") or "")
                except Exception:  # noqa: BLE001
                    full_text = ""

            # 2. Supplier resolution. Two cases:
            #    (a) invoice/po: schema has both supplier_name and supplier_id;
            #        synthesizer fills supplier_name → resolver derives ID.
            #    (b) quote: schema only has supplier_id, which the synthesizer
            #        treats as a name string. Resolve to a real SUP-* ID if
            #        the current value doesn't already look like one.
            supplier_name_for_resolve = None
            if raw_data.get("supplier_name"):
                if raw_data.get("supplier_id") is None:
                    supplier_name_for_resolve = raw_data["supplier_name"]
            elif doc_type == "quote":
                sid_value = raw_data.get("supplier_id")
                if (
                    isinstance(sid_value, str)
                    and sid_value
                    and not sid_value.startswith("SUP-")
                ):
                    # Treat the synthesized value as a name; resolve it.
                    supplier_name_for_resolve = sid_value
                    # Temporarily clear so the resolver writes the real ID.
                    raw_data["supplier_id"] = None

            # 2b. Filename-hint fallback. When the context layer left
            # supplier_id NULL but the filename clearly named a supplier
            # AND that supplier (as a stem) appears in the full document
            # text, use the filename name to resolve. The grounding
            # safeguard (presence in full_text) preserves the
            # no-fabrication contract.
            if supplier_name_for_resolve is None and raw_data.get("supplier_id") is None:
                src_file = raw_data.get("source_file")
                if src_file:
                    try:
                        from src.services.extraction.context_layer import (
                            parse_filename_hints, _supplier_name_grounded,
                        )
                        fh = parse_filename_hints(src_file)
                        fh_supplier = (fh.get("supplier") or "").strip()
                        if (
                            fh_supplier
                            and full_text
                            and _supplier_name_grounded(fh_supplier, full_text)
                        ):
                            log.info(
                                "AgentNick: filename-hint fallback "
                                "supplier=%r for %s (Qwen left supplier_id NULL)",
                                fh_supplier, src_file,
                            )
                            supplier_name_for_resolve = fh_supplier
                    except Exception as exc:  # noqa: BLE001
                        log.debug("filename-hint fallback failed: %s", exc)

            if supplier_name_for_resolve:
                try:
                    sid = resolve_or_create_supplier(supplier_name_for_resolve, conn)
                    if sid:
                        raw_data["supplier_id"] = sid
                except Exception as exc:  # noqa: BLE001
                    log.warning("supplier resolve failed for %r: %s",
                                supplier_name_for_resolve, exc)

            # 1c. Tax/total consistency check — flag (don't auto-fix).
            # If invoice_amount (or total_amount) + tax_amount ≠ *_total_incl_tax
            # by more than 0.50 (rounding tolerance), record a discrepancy.
            # Per "no auto-fix" rule — we surface the mismatch for human
            # review, but the extracted values are kept as-is.
            _discrepancies_logged = _check_tax_total_consistency(
                cur, doc_type, raw_id, raw_data,
            )

            # 1d. Audit columns — every stg row is stamped with the system
            # principal (AgentNick) and current timestamps. created_* are
            # set only when the row is brand-new in stg; last_modified_*
            # update on every UPSERT (including re-extractions).
            #
            # We compute "is_new" by probing stg for the doc_pk *before*
            # the INSERT/UPDATE; if no row exists, this is a new record.
            now = datetime.now(timezone.utc)
            agent_principal = "AgentNick"
            pk_col_for_check = _STG_PK[doc_type]
            doc_pk_val = raw_data.get(pk_col_for_check) or raw_data.get(
                "doc_pk_candidate"
            )
            is_new_row = True
            if doc_pk_val:
                cur.execute(
                    f"SELECT 1 FROM {stg_t} WHERE {pk_col_for_check} = %s LIMIT 1",
                    (doc_pk_val,),
                )
                is_new_row = cur.fetchone() is None
            if is_new_row:
                raw_data["created_date"] = now
                raw_data["created_by"] = agent_principal
            raw_data["last_modified_date"] = now
            raw_data["last_modified_by"] = agent_principal

            # 1e. Confidence score — percentage of NON-NULL "expected"
            # fields per `_EXPECTED_FIELDS_FOR_CONFIDENCE`. Required fields
            # are weighted 2x because they're the critical ones.
            raw_data["confidence_score"] = _compute_confidence_score(
                doc_type, raw_data, _required,
            )

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
            # AgentNick audit trail — one structured INFO line per row, so
            # the operator can grep journalctl for the agent's activity.
            log.info(
                "AgentNick: persisted doc_type=%s doc_pk=%s confidence=%s%% "
                "discrepancies=%d new_row=%s",
                doc_type,
                raw_data.get("doc_pk_candidate"),
                raw_data.get("confidence_score"),
                _discrepancies_logged,
                is_new_row,
            )
            return {
                "ok": True,
                "doc_pk": raw_data.get("doc_pk_candidate"),
                "confidence_score": float(raw_data.get("confidence_score"))
                    if raw_data.get("confidence_score") is not None else None,
                "discrepancies_logged": _discrepancies_logged,
                "is_new_row": is_new_row,
            }
        except Exception as exc:
            conn.rollback()
            log.exception("promotion failed for raw_id=%s doc_type=%s: %s",
                          raw_id, doc_type, exc)
            return {"ok": False, "reason": str(exc)}


def _detect_doc_type(raw_id: int) -> Optional[str]:
    """Return the doc_type whose _raw table contains raw_id, or None.

    Used to recover from legacy discrepancy rows that may have a stale
    doc_type relative to their raw_id.
    """
    with get_conn() as conn:
        cur = conn.cursor()
        for dt, (raw_t, _) in _RAW_TO_STG.items():
            cur.execute(f"SELECT 1 FROM {raw_t} WHERE raw_id = %s", (raw_id,))
            if cur.fetchone() is not None:
                return dt
    return None


def apply_hitl_fixes_and_promote(raw_id: int, doc_type: str) -> dict[str, Any]:
    """Read all resolved discrepancies for this raw_id, apply their
    resolved_value updates to the _raw row, then promote.

    Called by the NOTIFY listener when the DB trigger fires
    'extraction_raw_ready_for_promotion'.

    If the supplied doc_type doesn't have a _raw row matching raw_id (e.g.
    a legacy discrepancy row whose doc_type is stale relative to its
    raw_id), detect the correct doc_type by scanning the four _raw tables
    and re-route.
    """
    if doc_type not in _RAW_TO_STG:
        log.warning("unknown doc_type %r for raw_id=%s; attempting detect", doc_type, raw_id)
        doc_type = _detect_doc_type(raw_id) or doc_type
    else:
        # Verify the row exists in the claimed _raw table; otherwise re-detect.
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                f"SELECT 1 FROM {_RAW_TO_STG[doc_type][0]} WHERE raw_id = %s",
                (raw_id,),
            )
            if cur.fetchone() is None:
                detected = _detect_doc_type(raw_id)
                if detected and detected != doc_type:
                    log.info(
                        "doc_type override for raw_id=%s: %s → %s "
                        "(legacy discrepancy.doc_type was stale)",
                        raw_id, doc_type, detected,
                    )
                    doc_type = detected
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
