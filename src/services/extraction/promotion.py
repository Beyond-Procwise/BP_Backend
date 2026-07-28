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


def _table_columns(cur, qualified_table: str) -> list[str]:
    """Return actual column names for any schema-qualified table.

    Used as an identifier allowlist before interpolating column names
    into SQL (SQL-injection defence for HITL-supplied field_name values).
    """
    schema, table = qualified_table.split(".")
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


# Money fields that represent what the DOCUMENT says. If one of these was absent
# from the document and we computed it, that is an inference and must be recorded.
# FX columns (exchange_rate_to_usd / converted_amount_usd) are deliberately NOT
# listed: those are legitimately derived by design, not read off the page.
_INFERRABLE_MONEY: dict[str, tuple[str, ...]] = {
    "invoice": ("invoice_amount", "tax_amount", "invoice_total_incl_tax"),
    "purchase_order": ("total_amount", "tax_amount", "total_amount_incl_tax"),
    "quote": ("total_amount", "tax_amount", "total_amount_incl_tax"),
}


def _log_derived_money(
    cur, doc_type: str, raw_id: int, captured: dict[str, Any], derived: dict[str, Any],
) -> int:
    """Record every money field the pipeline INFERRED rather than read.

    _compute_derived fills a missing tax_amount as subtotal x pct and a missing
    total as subtotal + tax. That is inference, and until now it happened silently:
    the resulting row satisfies subtotal + tax == total by construction, so the
    reconciliation check passes and nothing downstream can tell a figure the
    document stated from one we invented. Invoice_INV618706 was booked with a tax
    of 116.96 and a total of 701.75 that appear nowhere on the page.

    The value is still written (the safety net is useful) — but it is now declared.
    Non-blocking: this is provenance, not an error.
    """
    fields = _INFERRABLE_MONEY.get(doc_type)
    if not fields:
        return 0
    pk_col = _STG_PK.get(doc_type)
    doc_pk = derived.get(pk_col) if pk_col else None
    source_file = derived.get("source_file")
    n = 0
    for f in fields:
        was_absent = captured.get(f) in (None, "")
        now_present = derived.get(f) not in (None, "")
        if was_absent and now_present:
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
                    f, None, None, str(derived.get(f)),
                    "value_derived", "warning", "open",
                    (
                        f"{f} was not captured from the document — it was COMPUTED "
                        f"as {derived.get(f)} by the derived-value safety net. The "
                        f"figure does not appear on the page; treat it as inferred, "
                        f"not as source data."
                    ),
                    False,
                ),
            )
            n += 1
    return n


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

    def _log(field_name: str, issue: str, expected, computed, notes,
             severity: str = "warning"):
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
                issue, severity, "open", notes, False,
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

    # 1b. Net exceeds gross — not a reconciliation gap but an IMPOSSIBILITY.
    #
    # Check 1 above fires whenever subtotal + tax != total, which on this corpus is
    # 84 findings across 41 documents: usually a discount, shipping or rounding line
    # the header does not carry, and a warning is the right response. This check is
    # deliberately narrower. A net-of-tax figure can never exceed the tax-inclusive
    # figure for the same document; tax is not negative. When it does, one of the two
    # values is simply wrong, and no amount of human triage will reconcile them.
    #
    # Recorded as CRITICAL so it is distinguishable from the routine reconciliation
    # warnings it would otherwise be buried among. Kept non-blocking, in line with the
    # standing "surface, don't auto-fix" rule for money fields — the reporting layer
    # refuses to publish the value instead (see the gateway's Quotes query).
    #
    # Live example: quote ORB-Q-6612 carries total_amount 3,335,591.00 against
    # total_amount_incl_tax 1,315,200.00 with tax 219,200.00. Its own tax line implies
    # a net of 1,096,000.00, so the recorded net is out by a factor of three. It is not
    # corrected here: the value must come from the document, not from our arithmetic.
    if sub_f and tot_f:
        s = _to_decimal(row.get(sub_f))
        g = _to_decimal(row.get(tot_f))
        if s is not None and g is not None and (s - g) > _DEFAULT_DISCREPANCY_TOLERANCE:
            t = _to_decimal(row.get(tax_f)) if tax_f else None
            implied = f" Its tax line implies a net of {(g - t).quantize(Decimal('0.01'))}." if t is not None else ""
            _log(
                field_name=sub_f,
                issue="net_exceeds_gross",
                expected=g,
                computed=s,
                notes=(
                    f"{sub_f}({s}) exceeds {tot_f}({g}), which is impossible: a "
                    f"net-of-tax amount cannot be larger than the tax-inclusive "
                    f"amount for the same document.{implied} One of the two values "
                    f"was mis-read and the document needs re-extracting."
                ),
                severity="critical",
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

            # Safety net: guarantee COMPUTABLE columns (exchange_rate_to_usd,
            # converted_amount_usd, tax_amount, *_total_incl_tax) are populated
            # at promotion time, even when the upstream context layer did not
            # run/compute them (e.g. an LLM hiccup under load). Pure arithmetic
            # over columns already present — no fabrication; fills missing
            # tax/total values and sets the deterministic FX conversion.
            # Genuine document miscalculations are NOT touched here — those are
            # flagged in the discrepancy table by dispatch.
            # Snapshot what the DOCUMENT actually printed, before anything is
            # derived from it. _check_tax_total_consistency below must judge the
            # captured values, not values we computed ourselves: _compute_derived
            # fills a missing tax_amount as subtotal x pct and a missing total as
            # subtotal + tax, so a derived row satisfies subtotal + tax == total
            # BY CONSTRUCTION and the check can never fire. That is how
            # Invoice_INV618706 passed reconciliation on 584.79 + 116.96 = 701.75 —
            # three figures that were all wrong, and all derived from each other.
            captured_data = dict(raw_data)

            try:
                from src.services.extraction.context_layer import _compute_derived
                raw_data = _compute_derived(raw_data)
            except Exception:  # noqa: BLE001
                log.debug("derived-compute safety net skipped", exc_info=True)

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
                    sid = resolve_or_create_supplier(
                        supplier_name_for_resolve, conn, doc_type=doc_type)
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
            # Judged against captured_data (pre-derivation), so the check sees what
            # the document printed. A figure the document did not state is absent
            # here and the check skips it — there is nothing to reconcile. A figure
            # the document DID state and got wrong is now caught instead of being
            # silently smoothed over by our own arithmetic.
            _discrepancies_logged = _check_tax_total_consistency(
                cur, doc_type, raw_id, captured_data,
            )
            # Declare anything the safety net inferred. Without this a computed
            # tax/total is indistinguishable from one the document actually printed.
            _discrepancies_logged += _log_derived_money(
                cur, doc_type, raw_id, captured_data, raw_data,
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

            # 1f. Carry the look-forward deal (deal_id/deal_name) from
            # process_monitor onto the staged row, so deal-tagged docs are
            # self-describing in _stg. The authoritative grouping still lives on
            # process_monitor; this just mirrors it. Best-effort, never blocks.
            pm_id = raw_data.get("process_monitor_id")
            if pm_id is not None and not raw_data.get("deal_id"):
                try:
                    cur.execute(
                        "SELECT deal_id, deal_name FROM proc.process_monitor WHERE id = %s",
                        (pm_id,))
                    pm = cur.fetchone()
                    if pm and pm[0]:
                        raw_data["deal_id"] = pm[0]
                        raw_data["deal_name"] = pm[1]
                except Exception:  # noqa: BLE001
                    log.debug("process_monitor deal carry skipped", exc_info=True)

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

            # 4. Mark _raw as promoted and RETAIN the row.
            # _raw is a permanent retention tier: it holds the engine's original
            # extraction output so the computed _stg values can always be traced
            # back to and compared against the raw source. Do NOT delete on clean
            # promotion (matches extraction_v3/persistence.py, which also keeps
            # _raw). The row is marked promotion_status='promoted' for audit.
            cur.execute(
                f"UPDATE {raw_t} SET promotion_status='promoted', promoted_at=NOW() "
                f"WHERE raw_id=%s", (raw_id,),
            )

            # Advance the document's process_monitor status to 'Staged' (in _stg)
            # so the status board reflects pipeline progress. Guarded so it never
            # regresses a doc that already reached a deal/target state.
            _pm_id = raw_data.get("process_monitor_id")
            if _pm_id is not None:
                cur.execute(
                    "UPDATE proc.process_monitor SET status='Staged', lastmodified_date=NOW() "
                    "WHERE id=%s AND status IN ('Extracted','Extraction_InReview','Running')",
                    (_pm_id,),
                )

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
                "process_monitor_id": raw_data.get("process_monitor_id"),
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
            # Build an allowlist of real columns for raw_t ONCE, then validate
            # every field_name before interpolating it into an UPDATE identifier.
            # This prevents SQL-injection via a crafted field_name in the
            # discrepancy table (e.g. "x = NULL, promotion_status").
            allowed_cols = set(_table_columns(cur, raw_t))
            for field_name, resolved_value, action in fixes:
                if field_name not in allowed_cols:
                    log.error(
                        "HITL fix: rejected unknown/invalid column name %r for %s "
                        "(raw_id=%s) — skipping to prevent SQL injection",
                        field_name, raw_t, raw_id,
                    )
                    continue
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


def promote_pending(doc_types=("invoice", "quote", "purchase_order", "contract"),
                    limit: Optional[int] = None) -> dict[str, Any]:
    """Catch-up promotion for _raw rows stranded at promotion_status='pending'.

    A 'pending' row means its ``extraction_raw_ready_for_promotion`` NOTIFY was
    never processed (e.g. the listener was down, or the notify was lost). This
    promotes them through the same path as the event-driven listener so no
    document is silently stuck before _stg. Idempotent.
    """
    out: dict[str, Any] = {"promoted": 0, "failed": 0, "by_type": {}}
    pending: list[tuple[int, str]] = []
    with get_conn() as conn:
        conn.autocommit = True
        cur = conn.cursor()
        for dt in doc_types:
            mapping = _RAW_TO_STG.get(dt)
            if not mapping:
                continue
            raw_t = mapping[0]
            try:
                cur.execute(
                    f"SELECT raw_id FROM {raw_t} WHERE promotion_status='pending' "
                    f"ORDER BY raw_id" + (f" LIMIT {int(limit)}" if limit else ""))
                for (rid,) in cur.fetchall():
                    pending.append((int(rid), dt))
            except Exception:  # table may not exist in some envs
                log.debug("promote_pending scan skipped for %s", raw_t, exc_info=True)
    for rid, dt in pending:
        try:
            res = apply_hitl_fixes_and_promote(rid, dt)
            ok = bool(res and res.get("ok"))
            out["promoted" if ok else "failed"] += 1
            out["by_type"][dt] = out["by_type"].get(dt, 0) + (1 if ok else 0)
        except Exception:
            out["failed"] += 1
            log.exception("promote_pending failed raw_id=%s doc_type=%s", rid, dt)
    if pending:
        log.info("promote_pending: %s", out)
    return out


def run_listener(stop_event=None, on_promoted=None) -> None:
    """Listen on 'extraction_raw_ready_for_promotion' channel and process
    NOTIFY events sequentially. Run in a dedicated worker.

    ``on_promoted(result, payload)`` — optional callback invoked after each
    successful _raw -> _stg promotion, so downstream consumers can drive the
    rest of the chain (stg -> trgt -> deal-linking -> mining) on the event
    rather than on a timer. Best-effort: callback errors never break the loop."""
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
                if on_promoted is not None and isinstance(result, dict) and result.get("ok"):
                    try:
                        on_promoted(result, payload)
                    except Exception:
                        log.exception("on_promoted callback failed (non-fatal)")
    finally:
        try:
            conn.close()
        except Exception:
            pass
