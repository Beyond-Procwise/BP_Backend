"""Procurement-document invariants.

A *Validator* is a deterministic check applied to a (header, line_items)
pair. Each returns a :class:`ValidatorResult` describing whether the
invariant held, the residual error if not, the fields it touched, and a
severity that drives the orchestrator's response.

Composing them into a :class:`ValidatorChain` produces a
:class:`ValidationReport` that the calibrated-confidence layer
consumes — any ``critical`` failure forces the record into the review
queue regardless of all other signals.

The invariants implemented here are the 10 procurement-standard checks
from the SOTA survey, minus PO linkage (which lives in
``po_linkage.py`` because it needs a DB read).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)


__all__ = [
    "Severity", "ValidatorResult", "ValidationReport", "Validator",
    "ValidatorChain", "DEFAULT_VALIDATORS",
    "LineArithmetic", "SubtotalClosure", "TaxClosure", "GrandTotalClosure",
    "CurrencyConsistency", "DateSanity", "VendorIdentity",
    "QuantitySign", "RoundOffBucket", "TaxTotalConfusion",
]


class Severity:
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class ValidatorResult:
    name: str
    passed: bool
    severity: str = Severity.WARNING
    residual: float = 0.0
    fields: tuple[str, ...] = ()
    message: str = ""

    @classmethod
    def ok(cls, name: str, fields: Iterable[str] = ()) -> "ValidatorResult":
        return cls(name=name, passed=True, fields=tuple(fields))

    @classmethod
    def fail(cls, name: str, message: str, *,
             severity: str = Severity.WARNING,
             residual: float = 0.0,
             fields: Iterable[str] = ()) -> "ValidatorResult":
        return cls(
            name=name, passed=False, severity=severity,
            residual=residual, fields=tuple(fields), message=message,
        )

    @classmethod
    def na(cls, name: str) -> "ValidatorResult":
        """Not applicable to this document — same as passing, just makes
        intent explicit in audit logs."""
        return cls(name=name, passed=True, severity=Severity.INFO,
                   message="not_applicable")


@dataclass
class ValidationReport:
    results: list[ValidatorResult] = field(default_factory=list)

    def __bool__(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def critical_failures(self) -> list[ValidatorResult]:
        return [r for r in self.results
                if not r.passed and r.severity == Severity.CRITICAL]

    @property
    def warnings(self) -> list[ValidatorResult]:
        return [r for r in self.results
                if not r.passed and r.severity == Severity.WARNING]

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def total_count(self) -> int:
        return len(self.results)

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 1.0
        applicable = [r for r in self.results
                      if r.message != "not_applicable"]
        if not applicable:
            return 1.0
        return sum(1 for r in applicable if r.passed) / len(applicable)


class Validator:
    """Base class — subclasses override :meth:`check`."""
    name: str = "validator"

    def applicable(self, doc_type: str) -> bool:
        return True

    def check(self, header: dict, line_items: list,
              doc_type: str) -> ValidatorResult:  # pragma: no cover
        raise NotImplementedError


class ValidatorChain:
    def __init__(self, validators: list[Validator]):
        self.validators = list(validators)

    def run(self, header: dict, line_items: list, doc_type: str) -> ValidationReport:
        results: list[ValidatorResult] = []
        for v in self.validators:
            if not v.applicable(doc_type):
                results.append(ValidatorResult.na(v.name))
                continue
            try:
                results.append(v.check(header, line_items, doc_type))
            except Exception as exc:
                logger.debug("validator %s raised: %s", v.name, exc, exc_info=True)
                results.append(ValidatorResult.fail(
                    v.name, f"validator_exception: {exc}",
                    severity=Severity.WARNING,
                ))
        return ValidationReport(results=results)


# -- helpers ---------------------------------------------------------------

def _f(v) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _approx(a: float, b: float, tol_abs: float = 0.05, tol_rel: float = 0.005) -> bool:
    """Return True when ``|a-b| <= max(tol_abs, tol_rel * max(|a|,|b|))``."""
    diff = abs(a - b)
    if diff <= tol_abs:
        return True
    scale = max(abs(a), abs(b), 1.0)
    return diff <= tol_rel * scale


def _grand_total(header: dict) -> Optional[float]:
    for k in ("invoice_total_incl_tax", "total_amount_incl_tax",
              "total_amount", "invoice_amount"):
        v = _f(header.get(k))
        if v is not None:
            return v
    return None


def _subtotal(header: dict) -> Optional[float]:
    for k in ("subtotal", "invoice_amount", "total_amount"):
        v = _f(header.get(k))
        if v is not None:
            return v
    return None


def _tax_amount(header: dict) -> Optional[float]:
    return _f(header.get("tax_amount"))


def _tax_rate(header: dict) -> Optional[float]:
    rate = _f(header.get("tax_percent")) or _f(header.get("tax_rate"))
    if rate is None:
        return None
    if rate > 1.0:  # given as percent
        rate = rate / 100.0
    return rate


def _line_amount(item: dict) -> Optional[float]:
    return _f(item.get("line_amount") or item.get("line_total"))


def _line_qty(item: dict) -> Optional[float]:
    return _f(item.get("quantity") or item.get("qty"))


def _line_unit_price(item: dict) -> Optional[float]:
    return _f(item.get("unit_price") or item.get("price"))


# -- the 10 invariants -----------------------------------------------------

class LineArithmetic(Validator):
    """qty * unit_price ≈ line_amount, per row."""
    name = "line_arithmetic"

    def check(self, header, line_items, doc_type):
        if not line_items:
            return ValidatorResult.na(self.name)
        residual_total = 0.0
        bad_rows: list[int] = []
        for i, item in enumerate(line_items):
            qty = _line_qty(item)
            price = _line_unit_price(item)
            amount = _line_amount(item)
            if qty is None or price is None or amount is None:
                continue
            expected = qty * price
            if not _approx(expected, amount):
                bad_rows.append(i)
                residual_total += abs(expected - amount)
        if not bad_rows:
            return ValidatorResult.ok(self.name)
        return ValidatorResult.fail(
            self.name,
            f"{len(bad_rows)} of {len(line_items)} lines fail qty×price≈amount "
            f"(rows {bad_rows[:5]})",
            severity=Severity.WARNING,
            residual=residual_total,
            fields=("line_items.line_amount", "line_items.quantity",
                    "line_items.unit_price"),
        )


class SubtotalClosure(Validator):
    """Σ line.amount ≈ subtotal (when both are present)."""
    name = "subtotal_closure"

    def check(self, header, line_items, doc_type):
        subtotal = _subtotal(header)
        if subtotal is None or not line_items:
            return ValidatorResult.na(self.name)
        line_sum = 0.0
        any_amount = False
        for item in line_items:
            la = _line_amount(item)
            if la is not None:
                line_sum += la
                any_amount = True
        if not any_amount:
            return ValidatorResult.na(self.name)
        if _approx(line_sum, subtotal):
            return ValidatorResult.ok(self.name, fields=("subtotal",))
        return ValidatorResult.fail(
            self.name,
            f"line sum ({line_sum:.2f}) != subtotal ({subtotal:.2f})",
            severity=Severity.WARNING,
            residual=abs(line_sum - subtotal),
            fields=("subtotal", "line_items.line_amount"),
        )


class TaxClosure(Validator):
    """tax_rate × subtotal ≈ tax_amount (when all three are present)."""
    name = "tax_closure"

    def check(self, header, line_items, doc_type):
        rate = _tax_rate(header)
        subtotal = _subtotal(header)
        tax = _tax_amount(header)
        if rate is None or subtotal is None or tax is None:
            return ValidatorResult.na(self.name)
        expected = subtotal * rate
        if _approx(expected, tax):
            return ValidatorResult.ok(self.name, fields=("tax_amount",))
        return ValidatorResult.fail(
            self.name,
            f"subtotal×rate ({expected:.2f}) != tax_amount ({tax:.2f})",
            severity=Severity.WARNING,
            residual=abs(expected - tax),
            fields=("tax_amount", "tax_percent", "subtotal"),
        )


class GrandTotalClosure(Validator):
    """subtotal + tax (+ freight + charges − discount) ≈ grand_total."""
    name = "grand_total_closure"

    def check(self, header, line_items, doc_type):
        subtotal = _subtotal(header)
        tax = _tax_amount(header) or 0.0
        freight = _f(header.get("freight_amount")) or 0.0
        charges = _f(header.get("charges_amount")) or 0.0
        discount = _f(header.get("discount_amount")) or 0.0
        total = _grand_total(header)
        if subtotal is None or total is None:
            return ValidatorResult.na(self.name)
        expected = subtotal + tax + freight + charges - discount
        if _approx(expected, total):
            return ValidatorResult.ok(self.name, fields=("invoice_total_incl_tax",))
        return ValidatorResult.fail(
            self.name,
            f"subtotal+tax+adjustments ({expected:.2f}) != total ({total:.2f})",
            severity=Severity.WARNING,
            residual=abs(expected - total),
            fields=("subtotal", "tax_amount", "invoice_total_incl_tax"),
        )


class CurrencyConsistency(Validator):
    """All monetary fields share one currency (when currency is recorded)."""
    name = "currency_consistency"

    def check(self, header, line_items, doc_type):
        header_ccy = (header.get("currency") or "").strip().upper() or None
        line_ccys = set()
        for item in line_items or []:
            v = (item.get("currency") or "").strip().upper()
            if v:
                line_ccys.add(v)
        ccys = set()
        if header_ccy:
            ccys.add(header_ccy)
        ccys.update(line_ccys)
        if len(ccys) <= 1:
            return ValidatorResult.ok(self.name, fields=("currency",))
        return ValidatorResult.fail(
            self.name,
            f"mixed currencies present: {sorted(ccys)}",
            severity=Severity.CRITICAL,
            fields=("currency",),
        )


class DateSanity(Validator):
    """invoice_date <= due_date; ISO-8601 form; not far-future."""
    name = "date_sanity"
    _ISO = re.compile(r"^\d{4}-\d{2}-\d{2}")

    def check(self, header, line_items, doc_type):
        invoice_date = header.get("invoice_date") or header.get("order_date") or header.get("quote_date")
        due_date = header.get("due_date")
        if invoice_date and not self._ISO.match(str(invoice_date)):
            return ValidatorResult.fail(
                self.name,
                f"invoice/order/quote_date not ISO-8601: {invoice_date!r}",
                severity=Severity.WARNING,
                fields=("invoice_date",),
            )
        if invoice_date and due_date and self._ISO.match(str(due_date)):
            try:
                from datetime import date
                d1 = date.fromisoformat(str(invoice_date)[:10])
                d2 = date.fromisoformat(str(due_date)[:10])
                if d2 < d1:
                    return ValidatorResult.fail(
                        self.name,
                        f"due_date ({d2}) precedes invoice_date ({d1})",
                        severity=Severity.WARNING,
                        fields=("invoice_date", "due_date"),
                    )
                if (d1.year > 2100) or (d2.year > 2100):
                    return ValidatorResult.fail(
                        self.name,
                        f"far-future date(s) detected: invoice={d1}, due={d2}",
                        severity=Severity.WARNING,
                        fields=("invoice_date", "due_date"),
                    )
            except Exception:
                pass
        return ValidatorResult.ok(self.name)


class VendorIdentity(Validator):
    """supplier_id/name is non-empty and not garbage-shaped (URL/doc-id/postcode/etc.).

    Garbage tokens are matched on word boundaries so legitimate names
    containing them as substrings (e.g. "City Of Newport" → "po" in
    "Newport") are NOT flagged. URL/email markers (http, @, .com, .co.)
    don't need word boundaries because they're already self-anchoring.
    """
    name = "vendor_identity"
    # Word-bounded tokens that look like document-id labels rather than
    # company names.
    _GARBAGE_WORDS = re.compile(
        r"\b(invoice|quote|order|nbr|bank|sort\s*code|iban|swift|"
        r"payment|payable|remittance)\b",
        re.IGNORECASE,
    )
    # Self-anchoring markers (URLs, emails) — don't need \b.
    _GARBAGE_MARKERS = re.compile(
        r"(https?://|www\.|@\w|\.com\b|\.co\.\w|\.co\.uk\b)",
        re.IGNORECASE,
    )
    # Standalone "PO" or "NUMBER" in supplier names — usually doc-id leakage.
    # We only flag when the WHOLE field is dominated by such a label, not
    # when it's part of a real name (e.g. "PO Box" addresses are kept).
    _DOMINANT_LABEL = re.compile(
        r"^\s*(invoice|po|number|nbr)\s*[#:.\-]?\s*\d",
        re.IGNORECASE,
    )

    def check(self, header, line_items, doc_type):
        sid = (header.get("supplier_id") or "").strip()
        sname = (header.get("supplier_name") or "").strip()
        identifier = sid or sname
        if not identifier:
            return ValidatorResult.fail(
                self.name, "supplier_id and supplier_name both empty",
                severity=Severity.CRITICAL,
                fields=("supplier_id", "supplier_name"),
            )
        # The three garbage detectors are checked in order. Each catches a
        # distinct failure mode while leaving legitimate names untouched.
        if self._DOMINANT_LABEL.search(identifier):
            return ValidatorResult.fail(
                self.name,
                f"supplier identifier looks like a document label: {identifier!r}",
                severity=Severity.CRITICAL,
                fields=("supplier_id", "supplier_name"),
            )
        if self._GARBAGE_MARKERS.search(identifier):
            return ValidatorResult.fail(
                self.name,
                f"supplier identifier contains URL/email markers: {identifier!r}",
                severity=Severity.CRITICAL,
                fields=("supplier_id", "supplier_name"),
            )
        if self._GARBAGE_WORDS.search(identifier):
            return ValidatorResult.fail(
                self.name,
                f"supplier identifier contains label tokens: {identifier!r}",
                severity=Severity.CRITICAL,
                fields=("supplier_id", "supplier_name"),
            )
        if len(identifier) < 2:
            return ValidatorResult.fail(
                self.name,
                f"supplier identifier too short: {identifier!r}",
                severity=Severity.WARNING,
                fields=("supplier_id", "supplier_name"),
            )
        return ValidatorResult.ok(self.name, fields=("supplier_id",))


class QuantitySign(Validator):
    """All line.qty share a sign (credit-note vs invoice consistency).

    A document with mixed +/- quantities is almost always an OCR error
    (a `7` mistaken for `-7` or similar). We flag it as critical so the
    record is reviewed before persistence proceeds with bad numbers.
    """
    name = "quantity_sign"

    def check(self, header, line_items, doc_type):
        if not line_items:
            return ValidatorResult.na(self.name)
        signs = set()
        for item in line_items:
            q = _line_qty(item)
            if q is None or q == 0:
                continue
            signs.add("neg" if q < 0 else "pos")
        if len(signs) <= 1:
            return ValidatorResult.ok(self.name)
        return ValidatorResult.fail(
            self.name,
            "mixed-sign quantities — likely OCR/extraction error",
            severity=Severity.CRITICAL,
            fields=("line_items.quantity",),
        )


class RoundOffBucket(Validator):
    """If subtotal+tax differs from grand_total by < 0.05 × grand_total,
    classify as round-off (info, not failure). Otherwise propagate the
    grand_total_closure failure unchanged. This validator only EMITS info
    notes; it never fails."""
    name = "round_off_bucket"

    def check(self, header, line_items, doc_type):
        subtotal = _subtotal(header)
        tax = _tax_amount(header) or 0.0
        total = _grand_total(header)
        if subtotal is None or total is None:
            return ValidatorResult.na(self.name)
        expected = subtotal + tax
        diff = abs(expected - total)
        if diff == 0:
            return ValidatorResult.ok(self.name)
        threshold = 0.05 * max(abs(total), 1.0)
        if diff <= threshold:
            return ValidatorResult(
                name=self.name, passed=True, severity=Severity.INFO,
                residual=diff, message=f"round_off_diff={diff:.2f}",
            )
        return ValidatorResult.ok(self.name)


class TaxTotalConfusion(Validator):
    """Detect when tax_amount == invoice_total_incl_tax — structurally impossible.

    If tax_amount equals the grand total and both are > 0, the extractor has
    assigned the grand-total token to the tax_amount field. This is a critical
    extraction error: tax can never equal the total (it would imply 0 subtotal).

    Fix: null-out tax_amount and flag for re-extraction.
    """
    name = "tax_total_confusion"

    def applicable(self, doc_type: str) -> bool:
        return doc_type in ("invoice", "quote")

    def check(self, header, line_items, doc_type):
        tax = _tax_amount(header)
        total = _f(header.get("invoice_total_incl_tax") or header.get("total_amount_incl_tax"))
        subtotal = _subtotal(header)
        if tax is None or total is None or tax <= 0 or total <= 0:
            return ValidatorResult.na(self.name)
        # Identical values: tax assigned the grand-total token
        if abs(tax - total) < 0.01:
            return ValidatorResult.fail(
                self.name,
                f"tax_amount ({tax:.2f}) == invoice_total_incl_tax ({total:.2f}) — "
                "extractor confused grand-total token with tax field; tax cannot equal total",
                severity=Severity.CRITICAL,
                fields=("tax_amount", "invoice_total_incl_tax"),
            )
        # Sanity: tax > subtotal (would mean negative net, impossible for normal invoices)
        if subtotal is not None and subtotal > 0 and tax >= subtotal:
            return ValidatorResult.fail(
                self.name,
                f"tax_amount ({tax:.2f}) >= subtotal ({subtotal:.2f}) — "
                "tax cannot exceed or equal the pre-tax subtotal",
                severity=Severity.CRITICAL,
                fields=("tax_amount", "invoice_amount"),
            )
        return ValidatorResult.ok(self.name, fields=("tax_amount",))


# -- default chain (PO linkage lives in po_linkage.py because it does DB I/O) --

DEFAULT_VALIDATORS: list[Validator] = [
    LineArithmetic(),
    SubtotalClosure(),
    TaxClosure(),
    GrandTotalClosure(),
    CurrencyConsistency(),
    DateSanity(),
    VendorIdentity(),
    QuantitySign(),
    RoundOffBucket(),
]
