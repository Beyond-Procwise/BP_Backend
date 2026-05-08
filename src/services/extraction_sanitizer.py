"""Field-type validation gate for extraction output.

Runs between extraction and persistence. Catches garbage values that the
LLM and structural extractor sometimes emit when they can't anchor the
real value — filename fragments captured as supplier_name, document-type
labels stored as parties, PO numbers stored as buyer_id, bare digits or
fragments stored as payment_terms, tax = subtotal mistakes, far-future
dates.

The sanitizer never raises on bad input. It NULLs fields that fail
validation and returns a list of Rejection records describing what was
removed and why, so the caller can persist them as discrepancies for
later review.

This module has no DB dependency by design — it operates on dicts and is
trivially unit-testable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Predicate building blocks
# ---------------------------------------------------------------------------

# Tokens that the LLM commonly captures as supplier_name when no real
# supplier is found in the body.
_DOC_TYPE_LABEL = re.compile(
    r"^\s*(invoice|quote|purchase\s*order|po|bill|receipt|"
    r"property\s+invoice|resource\s+rate\s+card|estimated)\s*$",
    re.I,
)

# Filename markers — anything containing these is almost certainly a
# filename fragment that leaked into the supplier field. Includes a
# catch-all for trailing/leading underscore which is a strong filename
# signal (real company names never end in "_").
_FILENAME_MARKER = re.compile(
    r"_watermark|_split[_\s]+tables|_duplicate|_scenario|_no[_\s]+vat|"
    r"\b(quote|invoice|po)[_\-\s]*scenario\b|"
    r"_\s*$|"                  # trailing underscore (Office Clean_)
    r"^\s*_|"                  # leading underscore
    r"_[A-Za-z0-9]+_",          # _x_ embedded (Quote_Scenario_)
    re.I,
)

# Label-with-value mash: "INVOICENO: 132666", "QUOTE NO 1283"
_LABEL_VALUE_MASH = re.compile(
    r"^\s*(invoice|quote|po|purchase\s*order)\s*no\.?\s*[:#]?\s*\d+",
    re.I,
)

# Person-name + professional title (Dana Parker DDS, Dr. John Smith)
_PERSON_TITLE = re.compile(
    r"^\s*(dr\.?|prof\.?|mr\.?|mrs\.?|ms\.?|miss|sir|dame)\s+\w+",
    re.I,
)
_TRAILING_TITLE = re.compile(
    r"\b(dds|md|phd|esq|jr|sr|mba|cpa|frcs)\b\.?\s*$",
    re.I,
)

# UK postcode (full or outcode-incode) and US ZIP — strict: matches when
# the WHOLE string is a postcode.
_POSTCODE = re.compile(
    r"^\s*(?:[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}|"          # UK full
    r"\d{5}(?:-\d{4})?)\s*$",                              # US ZIP
    re.I,
)
# Loose postcode pattern that matches anywhere in a string — used in
# combination with a "street address starts with number" signal to
# detect full address blocks ("10 Redkiln Way Horsham RH13 5QH").
_POSTCODE_ANYWHERE = re.compile(
    r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b|"             # UK
    r"\b\d{5}(?:-\d{4})?\b",                               # US ZIP
    re.I,
)
_STREET_ADDRESS_LEAD = re.compile(
    r"^\s*\d+(?:st|nd|rd|th|[a-z])?\s+[A-Z][a-z]+",         # "10 Redkiln", "5a King", "3rd Floor"
)
# Street-type vocabulary — combined with a leading number this is a
# strong address signal even without a postcode in the string.
_STREET_TYPE = re.compile(
    r"\b(?:road|rd|street|st|avenue|ave|drive|dr|lane|ln|way|place|pl|"
    r"close|crescent|court|ct|terrace|boulevard|blvd|highway|hwy|"
    r"square|sq|park|gardens?|mews|grove|hill|view|walk|"
    r"floor|suite|building|house|chambers|estate)\b",
    re.I,
)

# Document IDs (PO/INV/QUT prefix + digits, with optional dashes)
_DOC_ID = re.compile(
    r"^\s*(?:PO|INV|QUT|QTE|BILL|PUR)[\-\s]?\d[\d\-]*\s*$",
    re.I,
)

# Bare numeric "PO-style" IDs (5+ digits)
_BARE_NUMERIC_ID = re.compile(r"^\s*\d{5,}\s*$")

# URL / email / domain patterns — never a company name
_URL_OR_EMAIL = re.compile(
    r"^\s*(?:https?://|www\.|[\w\.\-]+@[\w\.\-]+\.|[\w\-]+\.(?:com|co\.uk|net|"
    r"org|io|ai|biz|info|ltd|us|uk|eu|de|fr|nl)\s*[/]?)",
    re.I,
)


def looks_like_postcode(s: str) -> bool:
    if not s or not isinstance(s, str):
        return False
    return bool(_POSTCODE.match(s))


def looks_like_address(s: str) -> bool:
    """True if `s` is a street-address block rather than a party name.

    Triggers on either:
    - postcode present anywhere in the string (UK/US format), OR
    - leading number + street name + street-type vocabulary
      ("10 Redkiln Way", "45 Market Street")
    """
    if not s or not isinstance(s, str):
        return False
    if _POSTCODE_ANYWHERE.search(s):
        return bool(_STREET_ADDRESS_LEAD.match(s)) or len(s) > 30
    # No postcode, but street-type vocabulary + leading number = address
    return bool(_STREET_ADDRESS_LEAD.match(s)) and bool(_STREET_TYPE.search(s))


def looks_like_url_or_email(s: str) -> bool:
    """True if `s` is a URL, email, or bare domain (never a company name)."""
    if not s or not isinstance(s, str):
        return False
    return bool(_URL_OR_EMAIL.search(s))


def looks_like_doc_id(s: str) -> bool:
    if not s or not isinstance(s, str):
        return False
    return bool(_DOC_ID.match(s)) or bool(_BARE_NUMERIC_ID.match(s))


def looks_like_person_with_title(s: str) -> bool:
    if not s or not isinstance(s, str):
        return False
    return bool(_PERSON_TITLE.match(s)) or bool(_TRAILING_TITLE.search(s))


def is_garbage_party_name(s: Optional[str]) -> bool:
    """Return True if `s` is clearly NOT a real party (supplier/buyer) name.

    A party name must be a company-like string. We reject:
    - empty / very short strings (<3 chars after trim)
    - document-type labels ("Invoice", "Quote")
    - filename fragments (containing _watermark, _scenario, …)
    - label+value mashes ("INVOICENO: 123")
    - postcodes
    - bare numeric IDs
    - mostly-digits strings
    """
    if not s or not isinstance(s, str):
        return True
    t = s.strip()
    if len(t) < 3:
        return True
    if _DOC_TYPE_LABEL.match(t):
        return True
    if _FILENAME_MARKER.search(t):
        return True
    if _LABEL_VALUE_MASH.match(t):
        return True
    if looks_like_postcode(t):
        return True
    if looks_like_address(t):
        return True
    if looks_like_url_or_email(t):
        return True
    if looks_like_doc_id(t):
        return True
    digits = sum(c.isdigit() for c in t)
    if digits and digits / len(t) > 0.5:
        return True
    return False


# ---------------------------------------------------------------------------
# Payment terms vocabulary
# ---------------------------------------------------------------------------

_PAYMENT_TERM_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bnet\s*(\d{1,3})\b", re.I), "Net {0}"),
    (re.compile(r"within\s+(\d{1,3})\s+days?", re.I), "Net {0}"),
    (re.compile(r"(\d{1,3})\s+days?\s+net", re.I), "Net {0}"),
    (re.compile(r"(?:due\s+(?:on\s+)?receipt|upon\s+receipt|on\s+receipt)", re.I), "Due on Receipt"),
    (re.compile(r"(?:cod|cash\s+on\s+delivery)", re.I), "COD"),
    (re.compile(r"upon\s+delivery", re.I), "Upon Delivery"),
]

# Bare tokens that are NOT payment terms (extractor noise)
_PAYMENT_TERM_GARBAGE_EXACT = {
    "", "&", "payments", "full", "net", "& conditions", "and conditions",
    "0", "1", "2", "3", "n/a", "na", "tbd",
}


def canonical_payment_terms(raw: Optional[str]) -> Optional[str]:
    """Normalize a free-text payment terms string to a canonical form
    (e.g. 'Net 30', 'Due on Receipt', 'COD') or return None when the
    input is garbage/unrecognized.
    """
    if not raw or not isinstance(raw, str):
        return None
    s = raw.strip()
    low = s.lower()
    if low in _PAYMENT_TERM_GARBAGE_EXACT:
        return None
    # Bare digits like "90" are not payment terms in any vocabulary
    if s.isdigit():
        return None
    # Multi-line junk (Bank details, paypal mashes) — refuse
    if "\n" in s and len(s) > 60:
        return None
    if "paypal" in low or "bank code" in low or "www." in low:
        return None

    for pat, fmt in _PAYMENT_TERM_RULES:
        m = pat.search(s)
        if m:
            return fmt.format(*m.groups()) if m.groups() else fmt
    return None


# ---------------------------------------------------------------------------
# Sanitizer
# ---------------------------------------------------------------------------

@dataclass
class Rejection:
    field: str
    extracted_value: Any
    reason: str
    severity: str = "warning"


# Field roles per doc_type — what each field should be type-checked as.
_PARTY_FIELDS = {
    "Invoice":        ("supplier_id", "supplier_name", "buyer_id"),
    "Purchase_Order": ("supplier_id", "supplier_name", "buyer_id"),
    "Quote":          ("supplier_id", "supplier_name", "buyer_id"),
}

_DATE_FIELDS = {
    "Invoice":        ("invoice_date", "due_date", "invoice_paid_date"),
    "Purchase_Order": ("order_date", "expected_delivery_date", "requested_date"),
    "Quote":          ("quote_date", "validity_date"),
}

_AMOUNT_FIELDS = {
    "Invoice":        ("invoice_amount", "tax_amount", "invoice_total_incl_tax"),
    "Purchase_Order": ("total_amount", "tax_amount", "total_amount_incl_tax"),
    "Quote":          ("total_amount", "tax_amount", "total_amount_incl_tax"),
}

_PK_FIELD = {
    "Invoice": "invoice_id",
    "Purchase_Order": "po_id",
    "Quote": "quote_id",
}

# Date sanity bounds: nothing before 2000, nothing more than 5 years ahead.
_DATE_MIN = date(2000, 1, 1)


def _date_too_far_future(today: date | None = None) -> date:
    today = today or date.today()
    return today + timedelta(days=365 * 5)


def _parse_iso_date(val: Any) -> Optional[date]:
    if isinstance(val, date):
        return val
    if not val or not isinstance(val, str):
        return None
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", val.strip())
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


class ExtractionSanitizer:
    """Validate/normalise extracted fields before persistence.

    Returns (sanitized_header, sanitized_line_items, rejections). Never
    raises — bad values become NULL and are reported in rejections.
    """

    def sanitize(
        self,
        header: Dict[str, Any],
        line_items: List[Dict[str, Any]],
        doc_type: str,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Rejection]]:
        clean = dict(header)
        rejections: List[Rejection] = []

        self._sanitize_parties(clean, doc_type, rejections)
        self._sanitize_po_reference(clean, doc_type, rejections)
        self._sanitize_payment_terms(clean, rejections)
        self._sanitize_amounts(clean, doc_type, rejections)
        self._sanitize_dates(clean, doc_type, rejections)
        clean_lines = self._sanitize_line_items(line_items, doc_type, rejections)

        return clean, clean_lines, rejections

    def _sanitize_line_items(
        self,
        line_items: List[Dict[str, Any]],
        doc_type: str,
        rejections: List[Rejection],
    ) -> List[Dict[str, Any]]:
        """Detect qty × unit_price ≠ line_total inconsistencies and NULL the
        broken pair so we don't persist false data. Keep line_total — it's
        the most-trustworthy field (matches what the document literally
        shows). Never overwrite with a computed value because the doc
        layout sometimes uses qty as 'months' or 'units per pack' that
        don't match the simple math model.
        """
        out: List[Dict[str, Any]] = []
        for it in line_items or []:
            new = dict(it)
            try:
                qty = new.get("quantity")
                unit = new.get("unit_price")
                # Per schema, line totals live in different columns by doc type
                lt = (new.get("line_amount") or new.get("line_total")
                      or new.get("total_amount"))
                if (qty is not None and unit is not None and lt is not None):
                    qty_f = float(qty)
                    unit_f = float(unit)
                    lt_f = float(lt)
                    if qty_f > 0 and unit_f > 0 and lt_f > 0:
                        expected = qty_f * unit_f
                        # 1% tolerance, or 10p — whichever is larger
                        tol = max(lt_f * 0.01, 0.10)
                        if abs(expected - lt_f) > tol:
                            rejections.append(Rejection(
                                field="line_item.qty_unit_inconsistent",
                                extracted_value=(
                                    f"qty={qty} × unit={unit} = {expected:.2f} "
                                    f"but line_total={lt}"
                                ),
                                reason="qty_unit_inconsistent",
                            ))
                            # NULL the inconsistent pair, keep line_total
                            new["quantity"] = None
                            new["unit_price"] = None
            except (ValueError, TypeError):
                pass  # leave as-is on parse failure
            out.append(new)
        return out

    # ------------------------------------------------------------------
    def _sanitize_parties(self, header: Dict, doc_type: str, rejections: List[Rejection]) -> None:
        for field in _PARTY_FIELDS.get(doc_type, ()):
            val = header.get(field)
            if val is None:
                continue
            # Canonical SUP- IDs are pre-validated upstream — accept them.
            if isinstance(val, str) and val.startswith("SUP-") and len(val) >= 6:
                continue
            if is_garbage_party_name(val):
                rejections.append(Rejection(
                    field=field,
                    extracted_value=val,
                    reason=f"garbage_party_name: {self._classify_garbage(val)}",
                ))
                header[field] = None
                continue
            # Person-with-title is suspicious but not always wrong.
            # For buyer_id, it's almost always wrong; for supplier_name,
            # it might be a sole-proprietor — accept with warning.
            if field == "buyer_id" and looks_like_person_with_title(str(val)):
                rejections.append(Rejection(
                    field=field, extracted_value=val,
                    reason="garbage_party_name: person_with_title",
                ))
                header[field] = None

    @staticmethod
    def _classify_garbage(val: str) -> str:
        if _DOC_TYPE_LABEL.match(val):
            return "doc_type_label"
        if _FILENAME_MARKER.search(val):
            return "filename_fragment"
        if _LABEL_VALUE_MASH.match(val):
            return "label_value_mash"
        if looks_like_postcode(val):
            return "postcode"
        if looks_like_address(val):
            return "address"
        if looks_like_url_or_email(val):
            return "url_or_email"
        if looks_like_doc_id(val):
            return "doc_id"
        digits = sum(c.isdigit() for c in val)
        if digits and digits / len(val) > 0.5:
            return "mostly_digits"
        return "too_short_or_empty"

    # ------------------------------------------------------------------
    def _sanitize_po_reference(self, header: Dict, doc_type: str, rejections: List[Rejection]) -> None:
        if doc_type not in ("Invoice", "Quote"):
            return
        po = header.get("po_id")
        if po is None:
            return
        s = str(po).strip()
        if not s:
            header["po_id"] = None
            return
        # Self-referencing po_id is always wrong (po_id == invoice_id/quote_id)
        pk_field = _PK_FIELD[doc_type]
        if header.get(pk_field) and s == str(header[pk_field]):
            rejections.append(Rejection(
                field="po_id", extracted_value=po,
                reason="self_reference: po_id == " + pk_field,
            ))
            header["po_id"] = None
            return
        # Strip prefix-less numeric → add PO prefix
        if re.match(r"^\d{4,}$", s):
            header["po_id"] = "PO" + s
            return
        # Reject obviously non-PO (single digit, very short)
        if not re.match(r"^(PO|PUR)[\-\s]?\d{3,}$", s, re.I) and not re.match(r"^\d{4,}$", s):
            rejections.append(Rejection(
                field="po_id", extracted_value=po,
                reason="invalid_po_format",
            ))
            header["po_id"] = None
            return
        # Already prefixed → uppercase the PO part for consistency
        header["po_id"] = re.sub(r"^po", "PO", s, flags=re.I)

    # ------------------------------------------------------------------
    def _sanitize_payment_terms(self, header: Dict, rejections: List[Rejection]) -> None:
        raw = header.get("payment_terms")
        if raw is None:
            return
        canonical = canonical_payment_terms(str(raw))
        if canonical is None:
            if str(raw).strip():  # only log if there was actual content
                rejections.append(Rejection(
                    field="payment_terms", extracted_value=raw,
                    reason="unrecognized_terms_vocabulary",
                ))
            header["payment_terms"] = None
        else:
            header["payment_terms"] = canonical

    # ------------------------------------------------------------------
    def _sanitize_amounts(self, header: Dict, doc_type: str, rejections: List[Rejection]) -> None:
        amt_fields = _AMOUNT_FIELDS.get(doc_type, ())
        if len(amt_fields) < 2:
            return
        subtotal_field, tax_field = amt_fields[0], amt_fields[1]
        subtotal = header.get(subtotal_field)
        tax = header.get(tax_field)
        try:
            sub_f = float(subtotal) if subtotal is not None else None
            tax_f = float(tax) if tax is not None else None
        except (ValueError, TypeError):
            return

        # Tax == subtotal is the classic "extractor confused tax with amount"
        # bug. Real-world tax rates are 0-30%, not 100%.
        if (sub_f is not None and tax_f is not None
                and sub_f > 100 and abs(sub_f - tax_f) < 0.01):
            rejections.append(Rejection(
                field=tax_field, extracted_value=tax,
                reason=f"tax_equals_subtotal: extractor confused {tax_field} with {subtotal_field}",
                severity="error",
            ))
            header[tax_field] = None
            if header.get("tax_percent") is not None:
                header["tax_percent"] = None

        # tax_percent should be 0-30 in practice. Anything ≥ 50 is almost
        # always an extraction error (often the value is the tax_amount
        # captured into the percent field).
        tp = header.get("tax_percent")
        try:
            tp_f = float(tp) if tp is not None else None
        except (ValueError, TypeError):
            tp_f = None
        if tp_f is not None and tp_f >= 50:
            rejections.append(Rejection(
                field="tax_percent", extracted_value=tp,
                reason=f"implausible_tax_rate: {tp_f}% — likely tax_amount captured as percent",
            ))
            header["tax_percent"] = None

    # ------------------------------------------------------------------
    def _sanitize_dates(self, header: Dict, doc_type: str, rejections: List[Rejection]) -> None:
        max_date = _date_too_far_future()
        for field in _DATE_FIELDS.get(doc_type, ()):
            val = header.get(field)
            if val is None:
                continue
            d = _parse_iso_date(val)
            if d is None:
                # Non-ISO date — might be valid but the schema expects ISO.
                # The orchestrator already has its own date sanity check,
                # so we leave it alone here.
                continue
            if d < _DATE_MIN or d > max_date:
                rejections.append(Rejection(
                    field=field, extracted_value=val,
                    reason=f"date_out_of_range: {d.isoformat()}",
                ))
                header[field] = None
