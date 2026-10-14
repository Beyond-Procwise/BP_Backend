"""The deal's verdict (spec §9.2). Pure.

Precedence: Blocked -> Needs review -> Incomplete -> Matched with notes -> Matched.
"""
from __future__ import annotations

from decimal import Decimal

from .model import NOTE, DocumentSet, Finding, Links, Result, Severity, Verdict


def verdict(ds: DocumentSet, links: Links, findings: list[Finding],
            results: list[Result]) -> Verdict:
    s1 = sum(1 for f in findings if f.severity == Severity.S1)
    s2 = sum(1 for f in findings if f.severity == Severity.S2)
    notes = (sum(1 for r in results if r.outcome in NOTE)
             + sum(1 for f in findings if f.severity == Severity.S3))
    invoiced = {p.doc_id for p in links.invoice_po.values() if p is not None}
    incomplete = (not ds.pos or not ds.invoices or bool(links.no_ref) or bool(links.bad_refs)
                  or any(p.doc_id not in invoiced for p in ds.pos))
    exposure = sum(((f.exposure_gbp or Decimal("0")) for f in findings
                    if f.severity >= Severity.S2), Decimal("0"))
    label = ("Blocked" if s1 else "Needs review" if s2 else "Incomplete" if incomplete
             else "Matched with notes" if notes else "Matched")
    return Verdict(ds.deal_id, label, s1, s2, notes, exposure, incomplete)
