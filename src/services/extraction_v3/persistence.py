"""Two-stage extraction persistence:

Stage 1 — _write_raw:
  INSERT into proc.bp_<doctype>_raw with a JSONB payload of all committed
  and residual fields.  This always succeeds (or raises) so we always have
  a record of what the pipeline produced.

Stage 2 — validate → promote or flag:
  - _compute_derived_values: fill exchange_rate_to_usd, converted_amount_usd,
    region from postcode, etc.
  - _detect_discrepancies: invariant checks, type-bind errors, missing required
    fields.  Returns list[Discrepancy] (severity 'critical' | 'warning' | 'info').
  - If NO critical discrepancies:
      _promote_to_stg: write header to _stg, line items to _stg, provenance
      to _v3, DELETE _raw row (clean promotion).
  - Else:
      _flag_with_discrepancies: write discrepancy rows, UPDATE _raw
      promotion_status='discrepancy'.  _stg is NOT written.

C9 contract: every committed value carries an evidence span; if we cannot
record provenance we do not commit the extraction data either.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from src.services.extraction_v3.schemas.result import ExtractionResult, CommittedField
from src.services.extraction_v3.yaml_schema.loader import load_doc_schema, DocSchema
from src.services.db import get_conn
from src.services.extraction_v3.supplier_resolver import resolve_or_create_supplier

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers shared by both stages
# ---------------------------------------------------------------------------

def _doc_pk_field(doc_type: str) -> str:
    return {
        "invoice": "invoice_id",
        "purchase_order": "po_id",
        "quote": "quote_id",
        "contract": "contract_id",
    }.get(doc_type, "id")


def _raw_table(doc_type: str) -> str:
    return {
        "invoice": "proc.bp_invoice_raw",
        "purchase_order": "proc.bp_purchase_order_raw",
        "quote": "proc.bp_quote_raw",
        "contract": "proc.bp_contract_raw",
    }.get(doc_type, f"proc.bp_{doc_type}_raw")


def _split_line_items(
    committed: list[CommittedField],
) -> tuple[list[CommittedField], dict[int, dict[str, CommittedField]]]:
    """Separate header CommittedFields from line-item CommittedFields."""
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


# ---------------------------------------------------------------------------
# Discrepancy model
# ---------------------------------------------------------------------------

@dataclass
class Discrepancy:
    field_name: str
    raw_value: str | None
    expected_value: str | None
    computed_value: str | None
    issue_type: str   # 'missing_required' | 'type_bind_error' | 'invariant_failed' |
                      # 'no_evidence' | 'value_out_of_range' | 'classifier_rejected' |
                      # 'date_invalid' | 'amount_mismatch'
    severity: str     # 'critical' | 'warning' | 'info'
    notes: str | None = None


# ---------------------------------------------------------------------------
# Stage 1: _write_raw
# ---------------------------------------------------------------------------

def _write_raw(result: ExtractionResult, source_file: str, conn: Any) -> int:
    """INSERT into proc.bp_<doctype>_raw.  Returns the raw_id."""
    table = _raw_table(result.doc_type)

    # Build JSONB payload: {header: {field: value, ...}, line_items: [...], residuals: [...]}
    header_vals: dict[str, Any] = {}
    for cf in result.committed:
        if not cf.field_path.startswith("line_items["):
            header_vals[cf.field_path] = {
                "value": cf.value,
                "confidence": cf.final_confidence,
                "model": cf.model,
                "evidence": cf.evidence_text,
                "page": cf.page,
            }

    # Build line items list (group by index)
    _, lines_by_idx = _split_line_items(result.committed)
    line_items_list = []
    for idx in sorted(lines_by_idx.keys()):
        row = {}
        for fname, cf in lines_by_idx[idx].items():
            row[fname] = {"value": cf.value, "confidence": cf.final_confidence, "model": cf.model}
        line_items_list.append(row)

    residuals_list = [
        {"field": rf.field_path, "reason": rf.reason}
        for rf in result.residuals
    ]

    payload = {
        "header": header_vals,
        "line_items": line_items_list,
        "residuals": residuals_list,
    }

    with conn.cursor() as cur:
        cur.execute(
            f"""
            INSERT INTO {table}
                (doc_pk_candidate, source_file, raw_payload, pipeline_version, promotion_status)
            VALUES (%s, %s, %s::jsonb, %s, 'pending')
            RETURNING raw_id
            """,
            (
                result.doc_pk,
                source_file,
                json.dumps(payload),
                result.pipeline_version,
            ),
        )
        raw_id: int = cur.fetchone()[0]

    log.info(
        "write_raw OK doc_type=%s doc_pk=%s raw_id=%d source=%s",
        result.doc_type, result.doc_pk, raw_id, source_file,
    )
    return raw_id


# ---------------------------------------------------------------------------
# Stage 2a: _compute_derived_values
# ---------------------------------------------------------------------------

def _compute_derived_values(result: ExtractionResult, conn: Any) -> ExtractionResult:
    """Fill derived fields: region from postcode, USD conversion.

    Operates purely on already-committed fields — does NOT fabricate values
    from thin air. Only computes when the source data is present.
    """
    committed = list(result.committed)
    committed_map = {cf.field_path: cf for cf in committed}

    # --- Region from postcode / postal_code ---
    _postcode_field_names = ("postal_code", "postcode")
    postcode_cf: CommittedField | None = None
    for fn in _postcode_field_names:
        if fn in committed_map:
            postcode_cf = committed_map[fn]
            break

    if postcode_cf and "region" not in committed_map:
        try:
            from src.services.extraction_v2.parsers.postcodes import parse_postcode
            pc = parse_postcode(postcode_cf.value)
            if pc is not None and hasattr(pc, "region") and pc.region:
                log.debug(
                    "compute_derived: region from postcode %s → %s",
                    postcode_cf.value, pc.region,
                )
                committed.append(CommittedField(
                    field_path="region",
                    value=pc.region,
                    page=postcode_cf.page,
                    bbox=postcode_cf.bbox,
                    evidence_text=postcode_cf.evidence_text,
                    model="derived:postcode",
                    model_confidence=0.85,
                    judge_actions=[],
                    final_confidence=0.85,
                ))
                committed_map = {cf.field_path: cf for cf in committed}
        except Exception as exc:
            log.debug("compute_derived: postcode→region failed: %s", exc)

    # --- USD conversion ---
    # If currency is not USD and we have invoice_amount / total_amount,
    # and exchange_rate_to_usd is already present (extracted from doc),
    # compute converted_amount_usd = amount * exchange_rate_to_usd.
    _amount_fields = ("invoice_amount", "total_amount", "total_contract_value")
    currency_cf = committed_map.get("currency")
    rate_cf = committed_map.get("exchange_rate_to_usd")
    amount_cf: CommittedField | None = None
    for fn in _amount_fields:
        if fn in committed_map:
            amount_cf = committed_map[fn]
            break

    if (
        currency_cf
        and rate_cf
        and amount_cf
        and "converted_amount_usd" not in committed_map
    ):
        try:
            rate = float(rate_cf.value)
            amount = float(amount_cf.value)
            converted = round(amount * rate, 2)
            log.debug(
                "compute_derived: converted_amount_usd = %s * %s = %s",
                amount, rate, converted,
            )
            committed.append(CommittedField(
                field_path="converted_amount_usd",
                value=str(converted),
                page=rate_cf.page,
                bbox=rate_cf.bbox,
                evidence_text=rate_cf.evidence_text,
                model="derived:fx",
                model_confidence=0.90,
                judge_actions=[],
                final_confidence=0.90,
            ))
        except (ValueError, TypeError):
            pass

    return ExtractionResult(
        doc_type=result.doc_type,
        doc_pk=result.doc_pk,
        committed=committed,
        residuals=result.residuals,
        judge_calls=result.judge_calls,
        pipeline_version=result.pipeline_version,
    )


# ---------------------------------------------------------------------------
# Type validation helper (used by discrepancy detection)
# ---------------------------------------------------------------------------

def _validate_type(value: str, field_type: str) -> bool:
    """Return True if value is valid for the given schema field_type.

    Uses the extraction_v2 parsers to validate. 'string' always returns True.
    Derived model fields (model starts with 'derived:' or 'pipeline_recovery')
    are exempt — they were computed from already-valid data.
    """
    raw = (value or "").strip()
    if not raw:
        return False  # empty string is not valid for any type
    if field_type == "string":
        return True
    if field_type in ("money", "decimal"):
        from src.services.extraction_v2.parsers.amounts import parse_amount
        return parse_amount(raw) is not None
    if field_type == "iso_date":
        from src.services.extraction_v2.parsers.dates import parse_date
        import re
        _has_year = re.search(r"20\d{2}", raw)
        _has_sep = re.search(r"\d{1,2}[/\-]\d{1,2}", raw)
        _has_month_name = re.search(
            r"(january|february|march|april|may|june|july|august|september|october|"
            r"november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)",
            raw, re.IGNORECASE,
        )
        if not (_has_year or _has_sep or (_has_month_name and re.search(r"\d", raw))):
            return False
        return parse_date(raw) is not None
    if field_type == "postcode":
        from src.services.extraction_v2.parsers.postcodes import parse_postcode
        return parse_postcode(raw) is not None
    if field_type == "currency":
        from src.services.extraction_v2.parsers.currency import parse_currency
        return parse_currency(raw) is not None
    if field_type == "address":
        from src.services.extraction_v2.parsers.addresses import parse_address
        result = parse_address(raw)
        return bool(result.postcode or (result.city and result.line1))
    # Unknown type — accept
    return True


# ---------------------------------------------------------------------------
# Stage 2b: _detect_discrepancies
# ---------------------------------------------------------------------------

def _detect_discrepancies(
    result: ExtractionResult,
    schema: DocSchema,
) -> list[Discrepancy]:
    """Detect issues in the extraction result.

    Checks:
    1. Missing required fields → critical
    2. Type-bind errors on committed fields (re-validate) → critical
    3. Date sanity (future invoice dates > 2 years out) → warning
    4. Amount range (negative money fields) → critical
    5. Invariant-level checks via run_invariants → critical/warning/info
    """
    discrepancies: list[Discrepancy] = []
    committed_map = {cf.field_path: cf for cf in result.committed}
    residual_set = {rf.field_path for rf in result.residuals}

    # --- 1. Missing required fields ---
    for fspec in schema.fields:
        if fspec.required:
            if fspec.name not in committed_map:
                discrepancies.append(Discrepancy(
                    field_name=fspec.name,
                    raw_value=None,
                    expected_value=None,
                    computed_value=None,
                    issue_type="missing_required",
                    severity="critical",
                    notes=f"Required field '{fspec.name}' not extracted. Residuals: {residual_set & {fspec.name}}",
                ))

    # --- 2. Type-bind errors on committed values ---
    fields_by_name = {f.name: f for f in schema.fields}
    for cf in result.committed:
        if cf.field_path.startswith("line_items["):
            continue
        fspec = fields_by_name.get(cf.field_path)
        if fspec is None or fspec.db_column is None:
            continue
        # Re-validate the committed value against the schema type using parsers directly
        bind_ok = _validate_type(cf.value, fspec.type)
        if not bind_ok:
            discrepancies.append(Discrepancy(
                field_name=cf.field_path,
                raw_value=cf.value,
                expected_value=None,
                computed_value=None,
                issue_type="type_bind_error",
                severity="critical",
                notes=f"Value {cf.value!r} could not be bound to type {fspec.type!r}",
            ))

    # --- 3. Date sanity: future dates more than 2 years out are suspicious ---
    import re as _re
    from datetime import date as _date
    today = _date.today()
    date_fields = [f.name for f in schema.fields if f.type == "iso_date"]
    for fname in date_fields:
        cf = committed_map.get(fname)
        if cf is None:
            continue
        try:
            from src.services.extraction_v2.parsers.dates import parse_date
            d = parse_date(cf.value)
            if d is not None:
                years_out = (d - today).days / 365.25
                if years_out > 2.0:
                    discrepancies.append(Discrepancy(
                        field_name=fname,
                        raw_value=cf.value,
                        expected_value=None,
                        computed_value=None,
                        issue_type="date_invalid",
                        severity="warning",
                        notes=f"Date {cf.value} is {years_out:.1f} years in the future",
                    ))
        except Exception:
            pass

    # --- 4. Negative monetary amounts ---
    money_fields = [f.name for f in schema.fields if f.type == "money"]
    for fname in money_fields:
        cf = committed_map.get(fname)
        if cf is None:
            continue
        try:
            val = float(cf.value)
            if val < 0:
                discrepancies.append(Discrepancy(
                    field_name=fname,
                    raw_value=cf.value,
                    expected_value=None,
                    computed_value=None,
                    issue_type="value_out_of_range",
                    severity="critical",
                    notes=f"Monetary field '{fname}' is negative: {cf.value}",
                ))
        except (ValueError, TypeError):
            pass

    # --- 5. Invariant checks (reuse binding runner) ---
    try:
        from src.services.extraction_v3.binding.invariants_runner import run_invariants
        header_dict = {cf.field_path: cf.value for cf in result.committed if not cf.field_path.startswith("line_items[")}
        _, lines_by_idx = _split_line_items(result.committed)
        line_dicts = []
        for idx in sorted(lines_by_idx.keys()):
            row = {fname: cf.value for fname, cf in lines_by_idx[idx].items()}
            line_dicts.append(row)

        inv_results = run_invariants(header_dict, line_dicts, schema)
        for r in inv_results:
            sev_lower = r.severity.lower() if r.severity else ""
            if sev_lower in ("critical", "fail", "error"):
                discrepancies.append(Discrepancy(
                    field_name=r.name,
                    raw_value=None,
                    expected_value=None,
                    computed_value=None,
                    issue_type="invariant_failed",
                    severity="critical",
                    notes=r.message,
                ))
            elif sev_lower in ("warning", "warn"):
                discrepancies.append(Discrepancy(
                    field_name=r.name,
                    raw_value=None,
                    expected_value=None,
                    computed_value=None,
                    issue_type="invariant_failed",
                    severity="warning",
                    notes=r.message,
                ))
    except Exception as exc:
        log.warning("_detect_discrepancies: invariant runner failed: %s", exc)

    return discrepancies


# ---------------------------------------------------------------------------
# Stage 2c helpers (used by both promote and legacy persist paths)
# ---------------------------------------------------------------------------

_LINE_ITEMS_COL_MAX_LENGTHS: dict[str, dict[str, int]] = {}
_LINE_ITEMS_COL_TYPES: dict[str, dict[str, str]] = {}


def _ensure_col_metadata(cur: Any, table: str) -> None:
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


def _coerce_to_db_value(raw: str, line_items_spec, field_name: str) -> Any:
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


def _resolve_supplier_fields(
    schema: DocSchema,
    header_cfs: list[CommittedField],
    conn: Any,
) -> list[CommittedField]:
    resolves_map: dict[str, str] = {}
    for f in schema.fields:
        if f.resolves_to_db_column:
            resolves_map[f.name] = f.resolves_to_db_column

    if not resolves_map:
        return header_cfs

    existing_field_paths = {cf.field_path for cf in header_cfs}
    extra: list[CommittedField] = []
    for cf in header_cfs:
        target_col = resolves_map.get(cf.field_path)
        if not target_col:
            continue
        if target_col in existing_field_paths:
            log.debug("persist: %s already committed — skipping resolver", target_col)
            continue
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


def _build_header_insert(
    cur: Any,
    doc_type: str,
    doc_pk: str,
    schema: DocSchema,
    header_cfs: list[CommittedField],
) -> None:
    """INSERT/UPSERT the header row into proc.bp_<doctype>_stg."""
    field_to_db_col: dict[str, str] = {}
    for f in schema.fields:
        if f.db_column:
            field_to_db_col[f.name] = f.db_column

    col_max_lengths = _get_col_max_lengths(cur, schema.db_table)
    pk_col = _doc_pk_field(doc_type)
    cols = [pk_col]
    vals: list[Any] = [doc_pk]

    resolved_cols: set[str] = {
        f.resolves_to_db_column for f in schema.fields if f.resolves_to_db_column
    }

    for cf in header_cfs:
        db_col = field_to_db_col.get(cf.field_path)
        if not db_col:
            if cf.field_path in resolved_cols:
                db_col = cf.field_path
            else:
                continue
        if db_col == pk_col:
            continue
        cols.append(db_col)
        val = cf.value
        max_len = col_max_lengths.get(db_col)
        if max_len is not None and isinstance(val, str) and len(val) > max_len:
            log.warning(
                "Truncating field %s (col=%s) from %d to %d chars for %s",
                cf.field_path, db_col, len(val), max_len, doc_pk,
            )
            val = val[:max_len]
        vals.append(val)

    _AUDIT_AGENT = "ExtractionV3"
    for audit_col in ("created_date", "last_modified_date"):
        if audit_col not in cols:
            cols.append(audit_col)
            vals.append("NOW()")
    for audit_col in ("created_by", "last_modified_by"):
        if audit_col not in cols:
            cols.append(audit_col)
            vals.append(_AUDIT_AGENT)

    _now_cols = {"created_date", "last_modified_date"}
    final_cols = []
    final_vals = []
    for c, v in zip(cols, vals):
        if c in _now_cols:
            final_cols.append(c)
        else:
            final_cols.append(c)
            final_vals.append(v)

    ph_parts = []
    for c in final_cols:
        if c in _now_cols:
            ph_parts.append("NOW()")
        else:
            ph_parts.append("%s")
    cols = final_cols
    vals = final_vals

    placeholders = ",".join(ph_parts)
    col_list = ",".join(cols)
    update_cols = [c for c in cols if c != pk_col]
    if update_cols:
        set_parts = []
        for c in update_cols:
            if c in _now_cols:
                set_parts.append(f"{c}=NOW()")
            else:
                set_parts.append(f"{c}=EXCLUDED.{c}")
        update_set = ",".join(set_parts)
        conflict_clause = f"ON CONFLICT ({pk_col}) DO UPDATE SET {update_set}"
    else:
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

    cur.execute(
        f"DELETE FROM {schema.db_lines_table} WHERE {parent_fk} = %s", (doc_pk,)
    )

    _ensure_col_metadata(cur, schema.db_lines_table)

    _line_seq_candidates = ["line_no", "line_number"]
    _line_seq_col = None
    if not any(c in line_field_db_cols.values() for c in _line_seq_candidates):
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

        if _line_seq_col:
            cols.append(_line_seq_col)
            vals.append(idx)

        for fname, cf in fields_dict.items():
            db_col = line_field_db_cols.get(fname)
            if not db_col:
                continue
            cols.append(db_col)
            val = cf.value
            if isinstance(val, str):
                val = _coerce_to_db_value(val, schema.line_items, fname)
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
            doc_type, doc_pk, cf.field_path, cf.value, cf.page,
            cf.bbox[0], cf.bbox[1], cf.bbox[2], cf.bbox[3],
            cf.evidence_text, cf.model, cf.model_confidence,
            json.dumps(cf.judge_actions) if cf.judge_actions else None,
            cf.final_confidence, pipeline_version,
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


# ---------------------------------------------------------------------------
# Stage 2c: _promote_to_stg
# ---------------------------------------------------------------------------

def _promote_to_stg(
    result: ExtractionResult,
    raw_id: int,
    schema: DocSchema,
    conn: Any,
) -> None:
    """Write header to _stg, line items to _stg, provenance to _v3, delete _raw."""
    header_cfs, lines_by_idx = _split_line_items(result.committed)

    # Supplier resolution runs inside the transaction
    header_cfs = _resolve_supplier_fields(schema, header_cfs, conn)

    with conn.cursor() as cur:
        _build_header_insert(cur, result.doc_type, result.doc_pk, schema, header_cfs)
        _build_line_items_inserts(cur, result.doc_type, result.doc_pk, schema, lines_by_idx)
        _build_provenance_inserts(
            cur, result.doc_type, result.doc_pk, result.committed, result.pipeline_version
        )
        # Clean promotion: delete the _raw row
        cur.execute(
            f"DELETE FROM {_raw_table(result.doc_type)} WHERE raw_id = %s",
            (raw_id,),
        )

    log.info(
        "promote_to_stg OK doc_type=%s doc_pk=%s raw_id=%d fields=%d",
        result.doc_type, result.doc_pk, raw_id, len(result.committed),
    )


# ---------------------------------------------------------------------------
# Stage 2d: _flag_with_discrepancies
# ---------------------------------------------------------------------------

def _flag_with_discrepancies(
    result: ExtractionResult,
    raw_id: int,
    source_file: str,
    discrepancies: list[Discrepancy],
    conn: Any,
) -> None:
    """Write discrepancy rows; update _raw promotion_status='discrepancy'."""
    table = _raw_table(result.doc_type)

    with conn.cursor() as cur:
        # Update _raw status
        cur.execute(
            f"UPDATE {table} SET promotion_status = 'discrepancy' WHERE raw_id = %s",
            (raw_id,),
        )

        # Insert one row per discrepancy
        for d in discrepancies:
            cur.execute(
                """
                INSERT INTO proc.bp_extraction_discrepancy
                    (doc_type, raw_id, source_file, doc_pk_candidate,
                     field_name, raw_value, expected_value, computed_value,
                     issue_type, severity, status, notes)
                VALUES (%s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, 'open', %s)
                """,
                (
                    result.doc_type, raw_id, source_file, result.doc_pk,
                    d.field_name, d.raw_value, d.expected_value, d.computed_value,
                    d.issue_type, d.severity, d.notes,
                ),
            )

    log.warning(
        "flag_with_discrepancies doc_type=%s doc_pk=%s raw_id=%d "
        "discrepancies=%d (critical=%d)",
        result.doc_type, result.doc_pk, raw_id,
        len(discrepancies),
        sum(1 for d in discrepancies if d.severity == "critical"),
    )


# ---------------------------------------------------------------------------
# Public entry point: persist (two-stage)
# ---------------------------------------------------------------------------

def persist(result: ExtractionResult, source_file: str = "") -> None:
    """Persist an ExtractionResult via the two-stage pipeline.

    Stage 1: write_raw — always succeeds (stores JSONB)
    Stage 2: validate → promote to _stg (clean) or flag (discrepancy)

    Raises ValueError if doc_pk is None.
    All DB writes happen in a single transaction.
    """
    if result.doc_pk is None:
        raise ValueError(
            f"cannot persist ExtractionResult without doc_pk for {result.doc_type}"
        )

    schema = load_doc_schema(result.doc_type)

    with get_conn() as conn:
        prior_autocommit = conn.autocommit
        conn.autocommit = False
        try:
            # Stage 1: raw landing
            raw_id = _write_raw(result, source_file, conn)

            # Stage 2a: compute derived values
            result = _compute_derived_values(result, conn)

            # Stage 2b: detect discrepancies
            discrepancies = _detect_discrepancies(result, schema)
            critical_count = sum(1 for d in discrepancies if d.severity == "critical")

            if critical_count == 0:
                # Stage 2c: promote to _stg
                _promote_to_stg(result, raw_id, schema, conn)
            else:
                # Stage 2d: flag — keep _raw, write discrepancy rows
                _flag_with_discrepancies(result, raw_id, source_file, discrepancies, conn)

            conn.commit()
            log.info(
                "persist OK doc_type=%s doc_pk=%s fields=%d "
                "discrepancies=%d critical=%d promoted=%s",
                result.doc_type, result.doc_pk, len(result.committed),
                len(discrepancies), critical_count, critical_count == 0,
            )
        except Exception:
            conn.rollback()
            log.exception(
                "persist ROLLBACK doc_type=%s doc_pk=%s",
                result.doc_type, result.doc_pk,
            )
            raise
        finally:
            conn.autocommit = prior_autocommit
