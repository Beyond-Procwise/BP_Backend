"""Adapter: engine dict -> ExtractionResult.

Filename PK fallback (last-resort, see _derive_pk_from_filename):
    Some test/sample PDFs put the PO/invoice number ONLY in the filename
    (e.g. 'MASTER_PO10_Furniture_BoldColour.pdf') and have no extractable
    label inside the body text. Rather than dropping these as no_pk, we
    derive the PK from the filename and surface it as a low-confidence
    committed field. The discrepancy detector still flags missing labels.


The v4 engine produces a flat dict keyed by reference field names. The existing
persist() pipeline expects an ExtractionResult with a list of CommittedField
entries keyed by YAML schema field names. This adapter performs the rename and
the wrap so persist() can run unchanged (and we keep its supplier resolver,
discrepancy detection, region/USD derivation, and audit handling).

Reference key -> YAML schema field name:
- INVOICE: invoice_id, po_id, supplier_name, buyer_id, requested_by,
  requested_date, invoice_date, due_date, invoice_paid_date, payment_terms,
  currency, invoice_amount, tax_percent, tax_amount, invoice_total_incl_tax,
  exchange_rate_to_usd, converted_amount_usd, country, region
  (drops: requisition_id, invoice_status, ai_flag_required, trigger_*,
   created_*, last_modified_*)
- PO: po_id, supplier_name, buyer_id, requisition_id, requested_by,
  requested_date, order_date, expected_delivery_date, payment_terms,
  total_amount, ship_to_country, delivery_region
  (renames: currency_code -> currency)
  (drops: incoterm_*, delivery_address_*, delivery_city, postal_code,
   base_currency, po_status, exchange_rate_to_usd, total_amount_usd,
   ai_flag_required, trigger_*, created_*, contract_id)
- QUOTE: quote_id, supplier_id (name resolves to id via persist), buyer_id,
  supplier_address, buyer_address, quote_date, validity_date, po_id, currency,
  total_amount, tax_percent, tax_amount, total_amount_incl_tax, country, region
  (drops: deal_id, ai_flag_required, trigger_*, created_*, last_modified_*)

Line items (all doc types) keep:
- item_description, quantity, unit_price, line_amount, tax_percent,
  tax_amount, total_amount_incl_tax (invoice/quote) or just first 4 (PO)
"""
from __future__ import annotations

import logging
from typing import Any

from src.services.extraction_v3.schemas.result import (
    CommittedField,
    ExtractionResult,
)

log = logging.getLogger(__name__)

PIPELINE_VERSION = "v4.0.0-hybrid"
MODEL_NAME = "hybrid_v4"
DEFAULT_CONFIDENCE = 0.90

# YAML schema fields with type: iso_date. The reference engine emits
# DD/MM/YYYY strings; Postgres datestyle=MDY chokes on them. Normalize to
# ISO YYYY-MM-DD before wrapping as CommittedField so persist's INSERT works.
_DATE_FIELDS = frozenset({
    "invoice_date", "due_date", "requested_date", "invoice_paid_date",
    "order_date", "expected_delivery_date",
    "quote_date", "validity_date",
})


def _to_iso_date(value: Any) -> str:
    """Coerce a date-like string to ISO YYYY-MM-DD; return original on failure."""
    if value is None:
        return ""
    s = str(value).strip()
    if not s:
        return ""
    # Already ISO?
    import re as _re
    if _re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s
    try:
        from src.services.extraction_v2.parsers.dates import parse_date
        parsed = parse_date(s)
        if parsed is not None:
            # parse_date returns datetime.date (or str). Normalise to ISO.
            if hasattr(parsed, "strftime"):
                return parsed.strftime("%Y-%m-%d")
            return str(parsed)
    except Exception as exc:
        log.debug("adapter: date parse failed for %r (%s)", s, exc)
    return s

# ----- header field renames per doc type -----------------------------------

_INVOICE_HEADER_RENAME = {
    "invoice_id": "invoice_id",
    "po_id": "po_id",
    "supplier_name": "supplier_name",
    "buyer_id": "buyer_id",
    "requested_by": "requested_by",
    "requested_date": "requested_date",
    "invoice_date": "invoice_date",
    "due_date": "due_date",
    "invoice_paid_date": "invoice_paid_date",
    "payment_terms": "payment_terms",
    "currency": "currency",
    "invoice_amount": "invoice_amount",
    "tax_percent": "tax_percent",
    "tax_amount": "tax_amount",
    "invoice_total_incl_tax": "invoice_total_incl_tax",
    "exchange_rate_to_usd": "exchange_rate_to_usd",
    "converted_amount_usd": "converted_amount_usd",
    "country": "country",
    "region": "region",
}

_PO_HEADER_RENAME = {
    "po_id": "po_id",
    "supplier_name": "supplier_name",
    # supplier_id is resolved from supplier_name by dispatch._resolve_supplier
    # (writes the SUP- id back as a separate committed field).
    "buyer_id": "buyer_id",
    "requisition_id": "requisition_id",
    "requested_by": "requested_by",
    "requested_date": "requested_date",
    "order_date": "order_date",
    "expected_delivery_date": "expected_delivery_date",
    "payment_terms": "payment_terms",
    "currency_code": "currency",  # rename
    "total_amount": "total_amount",
    "tax_percent": "tax_percent",
    "tax_amount": "tax_amount",
    "total_amount_incl_tax": "total_amount_incl_tax",
    "ship_to_country": "ship_to_country",
    "delivery_region": "delivery_region",
    "delivery_address_line1": "delivery_address_line1",
    "delivery_address_line2": "delivery_address_line2",
    "delivery_city": "delivery_city",
    "postal_code": "postal_code",
    "exchange_rate_to_usd": "exchange_rate_to_usd",
    "converted_amount_usd": "converted_amount_usd",
}

_QUOTE_HEADER_RENAME = {
    "quote_id": "quote_id",
    "supplier_id": "supplier_id",  # YAML field is supplier_id, value is name (resolver runs)
    "buyer_id": "buyer_id",
    "supplier_address": "supplier_address",
    "buyer_address": "buyer_address",
    "quote_date": "quote_date",
    "validity_date": "validity_date",
    "po_id": "po_id",
    "currency": "currency",
    "total_amount": "total_amount",
    "tax_percent": "tax_percent",
    "tax_amount": "tax_amount",
    "total_amount_incl_tax": "total_amount_incl_tax",
    "country": "country",
    "region": "region",
}

# ----- line-item field renames per doc type --------------------------------

_INVOICE_LINE_RENAME = {
    "item_id": "item_description",   # ref's "item_id" holds the description
    "quantity": "quantity",
    "unit_price": "unit_price",
    "line_total": "line_amount",     # rename
    "tax_percent": "tax_percent",
    "tax_amount": "tax_amount",
    "total_with_tax": "total_amount_incl_tax",  # rename
}

_PO_LINE_RENAME = {
    "item_description": "item_description",
    "quantity": "quantity",
    "unit_price": "unit_price",
    "line_total_amount": "line_amount",  # rename (db_column is line_total)
}

_QUOTE_LINE_RENAME = {
    "item_description": "item_description",
    "quantity": "quantity",
    "unit_of_measure": "unit_of_measure",
    "unit_price": "unit_price",
    "line_total": "line_amount",         # rename (db_column is line_total)
    "tax_percent": "tax_percent",
    "tax_amount": "tax_amount",
    "total_amount": "total_amount",
}


# ----- helpers --------------------------------------------------------------


def _is_filled(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    return True


def _to_committed(field_path: str, value: Any) -> CommittedField:
    """Wrap a field/value pair as a CommittedField with placeholder grounding.

    The hybrid engine doesn't supply per-field bbox/page/evidence_text. We use
    sentinel values (page=1, bbox=(0,0,0,0), evidence_text=value) so the
    persistence layer can route the row through type-binding and discrepancy
    detection without crashing on missing grounding.
    """
    str_val = str(value) if not isinstance(value, str) else value
    str_val = str_val.strip()
    return CommittedField(
        field_path=field_path,
        value=str_val,
        page=1,
        bbox=(0.0, 0.0, 0.0, 0.0),
        evidence_text=str_val,
        model=MODEL_NAME,
        model_confidence=DEFAULT_CONFIDENCE,
        judge_actions=[],
        final_confidence=DEFAULT_CONFIDENCE,
    )


def _wrap_header(rename_map: dict[str, str], header: dict) -> list[CommittedField]:
    out: list[CommittedField] = []
    for src, dst in rename_map.items():
        val = header.get(src)
        if not _is_filled(val):
            continue
        if dst in _DATE_FIELDS:
            iso = _to_iso_date(val)
            if not iso:
                continue
            val = iso
        out.append(_to_committed(dst, val))
    return out


def _wrap_line_items(
    rename_map: dict[str, str],
    items: list[dict],
) -> list[CommittedField]:
    out: list[CommittedField] = []
    for idx, item in enumerate(items):
        for src, dst in rename_map.items():
            val = item.get(src)
            if not _is_filled(val):
                continue
            out.append(_to_committed(f"line_items[{idx}].{dst}", val))
    return out


def _doc_pk(doc_type: str, header: dict) -> str | None:
    pk_field = {
        "invoice": "invoice_id",
        "purchase_order": "po_id",
        "po": "po_id",
        "quote": "quote_id",
    }.get(doc_type)
    if not pk_field:
        return None
    val = header.get(pk_field)
    if isinstance(val, str):
        val = val.strip()
    return val if _is_filled(val) else None


# ----- public API -----------------------------------------------------------


def _derive_pk_from_filename(doc_type: str, source_file: str) -> str | None:
    """Last-resort PK from filename for docs whose body lacks a clear label.

    Patterns recognized:
      - 'PO10', 'MASTER_PO10_*', 'PO_10' -> PO10  (purchase_order)
      - 'INV-1234', 'Invoice for INV-XYZ' -> INV-1234 (invoice)
      - 'QUOTE_WSG100025', 'WSG100025' -> WSG100025 (quote)
    Returns None if no usable pattern found.
    """
    import os
    import re as _re
    if not source_file:
        return None
    base = os.path.basename(source_file)
    # Strip extension
    name = os.path.splitext(base)[0]
    # Use lookbehind for non-alphanumeric (allows _, -, space) since \b
    # treats _ as a word char and won't match between _ and P.
    if doc_type in ("purchase_order", "po"):
        m = _re.search(r"(?:^|[^A-Z0-9])(PO[\s_\-]*\d{1,6})(?:[^0-9]|$)", name, _re.IGNORECASE)
        if m:
            return _re.sub(r"[\s_\-]", "", m.group(1)).upper()
    elif doc_type == "invoice":
        m = _re.search(r"(?:^|[^A-Z0-9])(INV[\-/]?\d[\w\-/]{2,20})", name, _re.IGNORECASE)
        if m:
            return m.group(1).upper()
    elif doc_type == "quote":
        m = _re.search(
            r"(?:^|[^A-Z0-9])(QUT[\-/]?\d[\w\-/]+|WSG\d{3,}|QTE[\-/]?\d[\w\-/]+)",
            name, _re.IGNORECASE,
        )
        if m:
            return m.group(1).upper()
    return None


def to_extraction_result(
    engine_output: dict,
    doc_type: str,
    source_file: str = "",
) -> ExtractionResult:
    """Convert engine output dict -> ExtractionResult.

    Args:
        engine_output: dict from run_data_extraction(). Must contain
            "<doc>_data" and "line_items" keys.
        doc_type: "invoice" | "purchase_order" | "po" | "quote".
        source_file: original file path (used for filename-PK fallback).

    Returns:
        ExtractionResult ready for persist_v3 / persist().
    """
    dt = doc_type.lower().strip()
    if dt == "po":
        dt = "purchase_order"

    if dt == "invoice":
        header = engine_output.get("invoice_data") or {}
        line_items = engine_output.get("line_items") or []
        rename = _INVOICE_HEADER_RENAME
        line_rename = _INVOICE_LINE_RENAME
        pk_field = "invoice_id"
    elif dt == "purchase_order":
        header = engine_output.get("po_data") or {}
        line_items = engine_output.get("line_items") or []
        rename = _PO_HEADER_RENAME
        line_rename = _PO_LINE_RENAME
        pk_field = "po_id"
    elif dt == "quote":
        header = engine_output.get("quote_data") or {}
        line_items = engine_output.get("line_items") or []
        rename = _QUOTE_HEADER_RENAME
        line_rename = _QUOTE_LINE_RENAME
        pk_field = "quote_id"
    else:
        raise ValueError(f"Unsupported doc_type for adapter: {doc_type!r}")

    # Filename PK fallback BEFORE wrapping — so the wrapped CommittedField
    # carries the derived value too, not just the ExtractionResult.doc_pk.
    if not _is_filled(header.get(pk_field)) and source_file:
        derived = _derive_pk_from_filename(dt, source_file)
        if derived:
            log.info(
                "adapter: derived %s=%r from filename %s",
                pk_field, derived, source_file,
            )
            header = dict(header)
            header[pk_field] = derived

    committed = _wrap_header(rename, header)
    committed.extend(_wrap_line_items(line_rename, line_items))

    pk = _doc_pk(dt, header)
    if pk is None:
        log.warning(
            "adapter: no doc_pk for %s -- header keys=%s",
            dt, list(header.keys()),
        )

    return ExtractionResult(
        doc_type=dt,
        doc_pk=pk,
        committed=committed,
        residuals=[],
        judge_calls=0,
        pipeline_version=PIPELINE_VERSION,
    )
