"""ScaleMismatch invariant: catches 10x decimal-point misreads (I-39).

For invoices, when |line_sum - invoice_amount| / min(line_sum, invoice_amount)
suggests an order-of-magnitude misread (ratio > 9), flag as CRITICAL.
"""
from __future__ import annotations

from src.services.extraction_v2.invariants import Validator, ValidatorResult, Severity, _f


class ScaleMismatch(Validator):
    name = "scale_mismatch"

    def applicable(self, doc_type: str) -> bool:
        return doc_type in ("invoice", "purchase_order", "quote")

    def check(self, header: dict, line_items: list, doc_type: str) -> ValidatorResult:
        # Pick the canonical "header total" field per doc_type
        if doc_type == "invoice":
            header_total = _f(header.get("invoice_amount")) or _f(header.get("invoice_total_incl_tax"))
        elif doc_type == "purchase_order":
            header_total = _f(header.get("total_amount")) or _f(header.get("total_amount_incl_tax"))
        elif doc_type == "quote":
            header_total = _f(header.get("total_amount"))
        else:
            return ValidatorResult.ok(self.name)

        line_sum = sum(
            _f(li.get("line_amount") or li.get("amount") or li.get("line_total")) or 0.0
            for li in (line_items or [])
        )

        if not header_total or not line_sum:
            return ValidatorResult.ok(self.name)

        large = max(line_sum, header_total)
        small = max(min(line_sum, header_total), 0.01)
        ratio = large / small
        if ratio > 9:
            return ValidatorResult.fail(
                self.name,
                f"scale mismatch: line_sum={line_sum:.2f} vs header_total={header_total:.2f} "
                f"(ratio {ratio:.1f}× — likely decimal-point misread)",
                severity=Severity.CRITICAL,
            )
        return ValidatorResult.ok(self.name)
