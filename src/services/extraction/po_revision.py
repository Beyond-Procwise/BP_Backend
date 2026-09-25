"""A re-issued purchase order keeps its own identity, and says whether it is approved.

Mirrors context_layer.canonical_quote_revision. A PO is re-issued as "Revision 3" or
"Change Order 2"; each issue is a distinct document, and the invoice is matched against the
latest APPROVED one. If revisions shared a primary key, last-write-wins would keep whichever
finished extracting last — the same defect that once collapsed nine quotes into five rows.

The revision is read deterministically from the text adjacent to THIS PO's number, so a
covering letter citing another PO cannot retag this one. Revision 1 (or 0, an "original
issue") stays unsuffixed: adding a suffix would give the same PO a second identity. Output
matches the gateway's parser (spendiq.match.ts poBase / poRevisionOf): "<number> (Rev n)".
"""
from __future__ import annotations

import re

# "Rev 3", "Rev. 3", "Revision 3", "Change Order 2", "CO 2", "Amendment 2", "Version 2",
# within a short window after the PO number (a line break or a label between is allowed).
_PO_REV_RE = re.compile(
    r"^[\s\-–—:,/()#]{0,12}(?:[A-Za-z ]{0,20}?\s)?"
    r"(?:rev(?:ision)?\.?|change\s+order|c\.?o\.?|amendment|amd\.?|version|v)\s*(?:no\.?|number|#)?\s*[:\-]?\s*(\d{1,3})\b",
    re.IGNORECASE)
_SUFFIX_RE = re.compile(r"\s*\(\s*rev\b.*$", re.IGNORECASE)

_APPROVAL_WORDS = [
    ("cancelled", re.compile(r"\b(?:cancell?ed|void(?:ed)?|withdrawn)\b", re.IGNORECASE)),
    ("rejected", re.compile(r"\b(?:rejected|declined|not\s+approved)\b", re.IGNORECASE)),
    ("pending", re.compile(r"\b(?:pending(?:\s+approval)?|awaiting\s+approval|draft|for\s+approval|submitted)\b", re.IGNORECASE)),
    ("approved", re.compile(r"\b(?:approved|authori[sz]ed|released|issued)\b", re.IGNORECASE)),
]


def po_base(po_id: str | None) -> str:
    return _SUFFIX_RE.sub("", str(po_id or "")).strip()


_REV_NUMBER_RE = re.compile(r"\(\s*rev\s*(\d+)", re.IGNORECASE)


def revision_of(po_id: str | None) -> int | None:
    """The revision a stored key carries ("4500018832 (Rev 3)" -> 3), None when bare."""
    m = _REV_NUMBER_RE.search(str(po_id or ""))
    return int(m.group(1)) if m else None


def latest_approved(cited: str | None, pos: dict):
    """The PO an invoice citing ``cited`` is measured against, from ``pos`` (id -> doc).

    An invoice prints the bare number, so it resolves to the highest revision of that
    number whose approval is stated as approved or not stated at all. A citation that
    names a revision keeps it, and with no approved revision the cited PO stands. Each
    doc may carry ``revision`` and ``approval``; the id's own suffix is the fallback.
    """
    if not cited:
        return None
    if revision_of(cited) is not None:
        return pos.get(cited)
    base = po_base(cited)
    live = [p for pid, p in pos.items()
            if po_base(pid) == base
            and getattr(p, "approval", None) in (None, "approved")]
    if not live:
        return pos.get(cited)
    return max(live, key=lambda p: getattr(p, "revision", None) or revision_of(p.doc_id) or 1)


def _ident(po_id) -> str:
    """One row's identity, as linking_engine._norm_po_id: '506789 (Rev 3)' -> '506789rev3'."""
    return re.sub(r"^po", "", re.sub(r"[^a-z0-9]", "", str(po_id or "").lower()))


def pick_key(row: dict, cited) -> tuple:
    """Rank of a PO row for a citation of ``cited``, higher is better; the Python twin of
    linking_engine._PO_PICK_ORDER_SQL. The revision the citation names; else the latest
    revision approved or unstated; else the row the citation matches."""
    same = _ident(row.get("po_id")) == _ident(cited)
    approved = row.get("approval_status") in (None, "approved")
    rev = row.get("po_revision")
    rev = int(rev) if rev not in (None, "") else (revision_of(row.get("po_id")) or 1)
    return (revision_of(cited) is not None and same, approved, rev if approved else 0, same)


def canonical_po_revision(po_id: str | None, full_text: str | None) -> tuple[str | None, int | None]:
    """(po_id carrying the revision the document states, the revision number or None)."""
    if not po_id or not full_text:
        return po_id, None
    base = po_base(po_id)
    if not base:
        return po_id, None
    text = str(full_text)
    idx = text.find(base)
    while idx >= 0:
        m = _PO_REV_RE.match(text[idx + len(base): idx + len(base) + 60])
        if m:
            rev = int(m.group(1))
            return (base if rev <= 1 else f"{base} (Rev {rev})"), rev
        idx = text.find(base, idx + len(base))
    return po_id, None


def normalise_approval(raw: str | None) -> str | None:
    """'approved' | 'pending' | 'rejected' | 'cancelled' | None (the document does not say).

    Ordered most-negative first, so "Not approved" is rejected, not approved.
    """
    if not raw:
        return None
    for value, rx in _APPROVAL_WORDS:
        if rx.search(str(raw)):
            return value
    return None
