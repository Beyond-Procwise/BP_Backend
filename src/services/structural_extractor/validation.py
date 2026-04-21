"""Validation gates for extracted values: anchor verification, math reconciliation,
cross-field consistency."""
from __future__ import annotations

from dataclasses import dataclass, field

from src.services.structural_extractor.parsing.model import ParsedDocument
from src.services.structural_extractor.types import ExtractedValue


@dataclass
class ValidationReport:
    passed: bool
    failures: list[str] = field(default_factory=list)


def verify_anchors(header: dict, doc: ParsedDocument) -> ValidationReport:
    """Ensure every extracted (non-derived/inferred/lookup) value has an
    anchor that can be traced back to the parsed source."""
    failures: list[str] = []
    norm_full = doc.full_text.lower().replace(" ", "") if doc.full_text else ""
    for field_name, ev in header.items():
        if not isinstance(ev, ExtractedValue):
            continue
        if ev.provenance != "extracted":
            # derived / inferred / lookup: no anchor to verify
            continue
        if ev.anchor_ref is None:
            failures.append(f"{field_name}: no anchor_ref for extracted value")
            continue
        if ev.anchor_text:
            if ev.anchor_text.lower().replace(" ", "") not in norm_full:
                failures.append(
                    f"{field_name}: anchor_text {ev.anchor_text!r} not in doc.full_text"
                )
    return ValidationReport(passed=not failures, failures=failures)


def _ev_zero() -> ExtractedValue:
    return ExtractedValue(value=0, provenance="extracted", source="structural", attempt=1)


def verify_math(header: dict, line_items: list[dict], tol: float = 0.01) -> ValidationReport:
    """Reconcile header amounts and line-item arithmetic."""
    failures: list[str] = []
    sub_key = "invoice_amount" if "invoice_amount" in header else "total_amount"
    total_key = (
        "invoice_total_incl_tax"
        if "invoice_total_incl_tax" in header
        else "total_amount_incl_tax"
    )
    sub = header.get(sub_key)
    tax = header.get("tax_amount")
    total = header.get(total_key)
    if sub and tax and total:
        try:
            if abs(float(sub.value) + float(tax.value) - float(total.value)) > tol:
                failures.append(
                    f"header math: {sub_key}({sub.value}) + tax_amount({tax.value}) "
                    f"!= {total_key}({total.value})"
                )
        except (TypeError, ValueError) as e:
            failures.append(f"header math type error: {e}")

    # Line item: qty * unit_price == line_total
    for idx, item in enumerate(line_items):
        qty = item.get("quantity")
        price = item.get("unit_price")
        lt = item.get("line_total") or item.get("line_amount")
        if qty and price and lt:
            try:
                if abs(float(qty.value) * float(price.value) - float(lt.value)) > tol:
                    failures.append(
                        f"line {idx + 1}: qty({qty.value}) x unit_price({price.value}) "
                        f"!= line_total({lt.value})"
                    )
            except Exception:
                continue

    # Sum(line_total) == subtotal
    if line_items and sub:
        try:
            total_of_lines = sum(
                float((it.get("line_total") or it.get("line_amount") or _ev_zero()).value)
                for it in line_items
                if it.get("line_total") or it.get("line_amount")
            )
            if abs(total_of_lines - float(sub.value)) > tol:
                failures.append(
                    f"sum line_totals ({total_of_lines}) != {sub_key}({sub.value})"
                )
        except Exception:
            pass

    return ValidationReport(passed=not failures, failures=failures)
