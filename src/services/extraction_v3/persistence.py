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
import re
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
    """Fill derived fields: region from postcode, USD conversion, zero-tax.

    Operates purely on already-committed fields — does NOT fabricate values
    from thin air. Only computes when the source data is present.
    """
    committed = list(result.committed)
    committed_map = {cf.field_path: cf for cf in committed}

    # --- Sanitise extracted region values (reject garbage before any derivation) ---
    # The 'region' canonical_labels are broad (Region, State, Province, County,
    # Territory) and frequently match non-geographic text. Null-out values that
    # are clearly not geographic region names.
    _region_cf = committed_map.get("region")
    if _region_cf is not None:
        _rv = (_region_cf.value or "").strip()
        _is_garbage_region = (
            len(_rv) > 40              # full address blocks, company+address, etc.
            or "@" in _rv              # email addresses
            or _rv.count(" ") > 5     # multi-word = likely full address line
            or any(ch.isdigit() for ch in _rv)  # digits → serial, address, postcode
        )
        if _is_garbage_region:
            log.info(
                "compute_derived: dropping garbage region value %r for %s",
                _rv, result.doc_pk,
            )
            committed = [cf for cf in committed if cf.field_path != "region"]
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

    # --- Region from country (for invoices without a postal_code field) ---
    # Invoice schema has no postal_code extraction target. When country is
    # present and region is still absent, derive region from the country name
    # for single-region countries, or leave NULL for ambiguous multi-region ones.
    # For US/AU/CA/GB where region=state/province matters, extract from the
    # supplier address block if country is known.
    if "region" not in committed_map and result.doc_type == "invoice":
        _country_cf = committed_map.get("country")
        if _country_cf is not None:
            _country_val = (_country_cf.value or "").strip()
            # Single-region/territory countries → country IS the region
            _SINGLE_REGION_COUNTRIES = {
                "Singapore": "Singapore",
                "Hong Kong": "Hong Kong",
                "New Zealand": "New Zealand",
                "Ireland": "Ireland",
                "Luxembourg": "Luxembourg",
                "Malta": "Malta",
                "Iceland": "Iceland",
                "Cyprus": "Cyprus",
            }
            _derived_region = _SINGLE_REGION_COUNTRIES.get(_country_val)
            if _derived_region:
                log.info(
                    "compute_derived: single-region country %r → region=%r for %s",
                    _country_val, _derived_region, result.doc_pk,
                )
                committed.append(CommittedField(
                    field_path="region",
                    value=_derived_region,
                    page=_country_cf.page,
                    bbox=_country_cf.bbox,
                    evidence_text=_country_cf.evidence_text,
                    model="derived:country",
                    model_confidence=0.75,
                    judge_actions=[],
                    final_confidence=0.75,
                ))
                committed_map = {cf.field_path: cf for cf in committed}

    # --- Tax-exempt detection: derive tax_amount=0 and tax_percent=0 ---
    # When invoice_amount == invoice_total_incl_tax (within 1% tolerance) and
    # neither tax_amount nor tax_percent was extracted, the document is
    # tax-exempt. Setting them to "0.0" prevents NULL on key columns.
    if result.doc_type == "invoice":
        invoice_amount_cf = committed_map.get("invoice_amount")
        total_cf = committed_map.get("invoice_total_incl_tax")
        tax_cf = committed_map.get("tax_amount")
        tax_pct_cf = committed_map.get("tax_percent")
        if (
            invoice_amount_cf is not None
            and total_cf is not None
            and tax_cf is None
            and tax_pct_cf is None
        ):
            try:
                amt = float(invoice_amount_cf.value)
                tot = float(total_cf.value)
                # Within 1% tolerance: tax-exempt invoice
                if amt > 0 and abs(amt - tot) / max(amt, 1.0) < 0.01:
                    log.info(
                        "compute_derived: tax-exempt invoice detected for %s "
                        "(invoice_amount=%.2f == total=%.2f); setting tax_amount=0 tax_percent=0",
                        result.doc_pk, amt, tot,
                    )
                    # Use invoice_amount_cf as anchor (its bbox/page/evidence are reliable)
                    for fp, label_val in (("tax_amount", "0.0"), ("tax_percent", "0.0")):
                        committed.append(CommittedField(
                            field_path=fp,
                            value=label_val,
                            page=invoice_amount_cf.page,
                            bbox=invoice_amount_cf.bbox,
                            evidence_text="derived:tax-exempt (invoice_amount==total)",
                            model="derived:tax_exempt",
                            model_confidence=0.80,
                            judge_actions=[],
                            final_confidence=0.80,
                        ))
                    committed_map = {cf.field_path: cf for cf in committed}
            except (ValueError, TypeError):
                pass

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

    return result.model_copy(update={"committed": committed})


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

    # --- 6. Total-amount reconciliation (warning, never blocks promotion) ---
    # User policy: capture mismatch in discrepancy table, do NOT auto-correct.
    # Pull subtotal/tax/shipping_fee/total_amount_incl_tax from committed.
    def _f(name: str) -> float | None:
        cf = committed_map.get(name)
        if cf is None or not cf.value:
            return None
        try:
            return float(cf.value)
        except (ValueError, TypeError):
            return None

    # Field names per doc type
    if result.doc_type in ("invoice",):
        total_field = "invoice_total_incl_tax"
        sub_field = "invoice_amount"  # pre-tax invoice amount
    elif result.doc_type in ("purchase_order", "po"):
        total_field = "total_amount_incl_tax"
        sub_field = "total_amount"  # in PO mapper, pre-tax sits in total_amount
    elif result.doc_type == "quote":
        total_field = "total_amount_incl_tax"
        sub_field = "total_amount"
    else:
        total_field = sub_field = None

    if total_field and sub_field:
        sub = _f(sub_field)
        tax = _f("tax_amount")
        shipping = _f("shipping_fee")
        total = _f(total_field)
        if sub is not None and total is not None:
            expected = sub + (tax or 0.0) + (shipping or 0.0)
            if abs(expected - total) > 1.0:
                bits = [f"subtotal={sub:.2f}"]
                if tax is not None: bits.append(f"tax={tax:.2f}")
                if shipping is not None: bits.append(f"shipping={shipping:.2f}")
                bits.append(f"sum={expected:.2f}")
                bits.append(f"printed_total={total:.2f}")
                discrepancies.append(Discrepancy(
                    field_name=total_field,
                    raw_value=str(total),
                    expected_value=f"{expected:.2f}",
                    computed_value=f"{expected - total:+.2f}",
                    issue_type="amount_mismatch",
                    severity="warning",
                    notes="Reconciliation: " + ", ".join(bits) +
                          " — discrepancy captured, values NOT auto-corrected.",
                ))

    return discrepancies


def _write_discrepancies_only(
    result: ExtractionResult,
    raw_id: int,
    source_file: str,
    discrepancies: list[Discrepancy],
    conn: Any,
) -> None:
    """Write discrepancy rows WITHOUT changing promotion_status.

    Used when promotion succeeded (no critical) but warning/info-level
    discrepancies exist that should still be visible in the discrepancy table.
    """
    if not discrepancies:
        return
    with conn.cursor() as cur:
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
            # Garbage extracted into an integer column (e.g. quantity='sets',
            # quantity='2 Total: 11,555.50'). Returning the raw string would
            # crash the INSERT and roll back the whole transaction. Drop to
            # NULL instead so the row still lands and the bad cell is visible
            # for downstream review.
            return None
    if dtype in ("numeric", "real", "double precision", "decimal"):
        # Strip currency symbols / commas before letting Postgres parse.
        cleaned = re.sub(r"[£$€¥₹£,\s]", "", val).strip()
        if not cleaned:
            return None
        try:
            float(cleaned)
            return cleaned
        except ValueError:
            return None
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

    bad_rows = 0
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
        # Per-row resilience: a single bad line shouldn't kill the others.
        # Use a savepoint so an INSERT failure rolls back only this row, not
        # the whole _stg promotion. The header + good lines still land.
        cur.execute(f"SAVEPOINT line_{idx}")
        try:
            cur.execute(sql, vals)
            cur.execute(f"RELEASE SAVEPOINT line_{idx}")
        except Exception as exc:
            cur.execute(f"ROLLBACK TO SAVEPOINT line_{idx}")
            cur.execute(f"RELEASE SAVEPOINT line_{idx}")
            bad_rows += 1
            log.warning(
                "_build_line_items_inserts: skipping bad line %d for %s/%s: %s -- vals=%r",
                idx, doc_type, doc_pk, exc, vals,
            )
    if bad_rows:
        log.warning(
            "_build_line_items_inserts: %d/%d line rows skipped for %s/%s",
            bad_rows, len(lines_by_idx), doc_type, doc_pk,
        )


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
        # Mark _raw as promoted (KEEP the row for audit). Previous behaviour
        # deleted the _raw row on clean promotion; user requirement is that
        # _raw permanently holds the engine's raw output so _stg's computed
        # values can always be recomputed from source.
        cur.execute(
            f"UPDATE {_raw_table(result.doc_type)} "
            f"SET promotion_status = 'promoted' WHERE raw_id = %s",
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

def _mark_raw_status(doc_type: str, raw_id: int, status: str, note: str = "") -> None:
    """Best-effort: open a fresh tx, update _raw promotion_status. Never raises."""
    try:
        with get_conn() as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(
                    f"UPDATE {_raw_table(doc_type)} "
                    f"SET promotion_status = %s WHERE raw_id = %s",
                    (status, raw_id),
                )
        if note:
            log.info("_raw status set: doc_type=%s raw_id=%d -> %s (%s)",
                     doc_type, raw_id, status, note)
    except Exception as exc:
        log.warning("_mark_raw_status failed for raw_id=%d: %s", raw_id, exc)


def persist(result: ExtractionResult, source_file: str = "") -> int | None:
    """Persist an ExtractionResult via two independent transactions.

    Tx1 (always commits, even if doc_pk is None):
        write _raw row (JSONB payload of engine output). doc_pk_candidate
        is nullable; rows without a PK land with promotion_status='no_pk'
        so they're queryable for manual review.

    Tx2 (best-effort; only runs when doc_pk is present; failure does NOT
        lose _raw):
        compute derived values → detect discrepancies →
        promote to _stg (no critical) OR flag with discrepancies.
        On Tx2 exception, _raw is marked promotion_status='failed' via a
        third small Tx so failures are queryable.

    Returns the raw_id of the written _raw row, or None if Tx1 itself fails
    (DB unreachable, constraint violation, etc.). Even an extractor that
    produced no PK results in a _raw row whose raw_id is returned.
    """
    schema = load_doc_schema(result.doc_type)

    # ===== Tx1: write _raw and commit =====================================
    with get_conn() as conn:
        prior_autocommit = conn.autocommit
        conn.autocommit = False
        try:
            raw_id = _write_raw(result, source_file, conn)
            conn.commit()
        except Exception:
            conn.rollback()
            log.exception(
                "persist Tx1 (write_raw) ROLLBACK doc_type=%s doc_pk=%s",
                result.doc_type, result.doc_pk,
            )
            raise
        finally:
            conn.autocommit = prior_autocommit

    # No PK → can't promote. Mark _raw and stop here.
    if result.doc_pk is None:
        log.warning(
            "persist: no doc_pk extracted for %s -- _raw raw_id=%d preserved, "
            "marked promotion_status='no_pk' for manual review (source=%s)",
            result.doc_type, raw_id, source_file,
        )
        _mark_raw_status(result.doc_type, raw_id, "no_pk", source_file)
        return raw_id

    # ===== Tx2: compute, detect, promote or flag ==========================
    # If Tx2 fails, _raw is still in place (Tx1 already committed). Mark
    # _raw status='failed' via a third small Tx so the failure is queryable.
    with get_conn() as conn:
        prior_autocommit = conn.autocommit
        conn.autocommit = False
        try:
            result = _compute_derived_values(result, conn)
            discrepancies = _detect_discrepancies(result, schema)
            critical_count = sum(1 for d in discrepancies if d.severity == "critical")

            if critical_count == 0:
                _promote_to_stg(result, raw_id, schema, conn)
                # Still surface warning/info discrepancies (e.g. amount_mismatch)
                # so they're queryable even on a successful promotion.
                non_critical = [d for d in discrepancies if d.severity != "critical"]
                if non_critical:
                    _write_discrepancies_only(result, raw_id, source_file, non_critical, conn)
            else:
                _flag_with_discrepancies(result, raw_id, source_file, discrepancies, conn)

            conn.commit()
            log.info(
                "persist OK doc_type=%s doc_pk=%s raw_id=%d fields=%d "
                "discrepancies=%d critical=%d promoted=%s",
                result.doc_type, result.doc_pk, raw_id, len(result.committed),
                len(discrepancies), critical_count, critical_count == 0,
            )
        except Exception as exc:
            conn.rollback()
            log.exception(
                "persist Tx2 (compute/detect/promote) FAILED -- _raw is preserved "
                "doc_type=%s doc_pk=%s raw_id=%d",
                result.doc_type, result.doc_pk, raw_id,
            )
            # Mark _raw as failed in a fresh tx so it's queryable
            _mark_raw_status(result.doc_type, raw_id, "failed", str(exc)[:200])
            # Don't re-raise: _raw is preserved; caller treats this as "ok"
            # for the purpose of process_monitor status (extraction landed).
        finally:
            conn.autocommit = prior_autocommit

    return raw_id
