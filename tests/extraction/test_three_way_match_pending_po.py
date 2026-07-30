"""po_not_found must soften to po_pending_review when the cited PO was
uploaded in the same corpus but is still held in extraction review.

Session ses-20260730-UJF3: invoices citing PO-2025-0257 / PO-2025-0270 were
flagged critical "does not exist in the system" while both POs sat in
Discrepancy_Review two rows away. The flag was true of the promoted tiers
and wrong in spirit.
"""
from src.services.extraction import three_way_match as twm


def _run(monkeypatch, *, po_promoted, po_in_raw):
    monkeypatch.setattr(twm, "_load_po", lambda po_id: (po_promoted, []))
    monkeypatch.setattr(twm, "_po_uploaded_but_unpromoted", lambda po_id: po_in_raw)
    return twm.check_against_po(
        "invoice", {"po_id": "PO-2025-0257", "invoice_amount": "100"}, [])


def test_cited_po_stuck_in_review_yields_warning_not_critical(monkeypatch):
    findings = _run(monkeypatch, po_promoted=None, po_in_raw=True)
    assert len(findings) == 1
    d = findings[0]
    assert d.issue_type == "po_pending_review"
    assert d.severity == "warning"
    assert d.raw_value == "PO-2025-0257"
    assert "held in extraction review" in d.notes


def test_genuinely_absent_po_still_critical(monkeypatch):
    findings = _run(monkeypatch, po_promoted=None, po_in_raw=False)
    assert len(findings) == 1
    assert findings[0].issue_type == "po_not_found"
    assert findings[0].severity == "critical"
