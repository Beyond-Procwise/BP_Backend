import re
from src.services.structural_extractor.parsing.model import (
    ParsedDocument, BBox, CellRef, ColumnRef, NodeRef
)
from src.services.structural_extractor.types import ExtractedValue

ANCHOR_TOKENS = {"payment terms", "payment due", "terms", "payment terms:", "terms:"}
NET_RE = re.compile(r"^Net$", re.IGNORECASE)
WITHIN_RE = re.compile(r"^within$", re.IGNORECASE)

# Section-break markers: once we hit one of these tokens, we've left the
# payment terms block and entered a different section (bank details, etc).
SECTION_BREAKS = {
    "payment details:", "payment details",
    "bank:", "bank",
    "bank name:", "sort code:", "account number:", "account:",
    "this invoice references:", "invoice references:", "references:", "reference:",
    "iban:", "swift/bic:", "swift:",
    "notes:",
}


def _is_section_break(token_text: str) -> bool:
    low = token_text.lower().strip()
    if low in SECTION_BREAKS:
        return True
    # Also catch any ALL-CAPS token ending in ":" that has uppercase content
    # (heuristic for an inline section label).
    if token_text.endswith(":") and token_text[:-1].isupper() and len(token_text) > 3:
        return True
    # Catch Title-Case colon-terminated tokens — these are common inline
    # section labels in authored documents ('Currency:', 'Notes:', 'Ref:',
    # 'Bank:'). Requires a leading uppercase AND an alpha-only body (no
    # digits, no punctuation beyond the terminal colon) so we don't
    # accidentally match a candidate value like 'Net:' or '30:'.
    if len(token_text) > 2 and token_text.endswith(":"):
        body = token_text[:-1]
        if body and body[0].isupper() and body.isalpha():
            return True
    return False


def _clean_value(text: str) -> str:
    # Trim whitespace + trailing punctuation like ",", ";" that often hangs
    # off the end of a natural-language phrase.
    return text.strip().rstrip(",;")


def extract_payment_terms(doc: ParsedDocument, doc_type: str) -> dict[str, ExtractedValue]:
    out: dict[str, ExtractedValue] = {}
    tokens = doc.tokens
    # Strategy 1: anchor on "Payment Terms" / "Terms" / "Payment Due"
    # label, then collect up to 6 tokens, stopping at:
    #   (a) any SECTION_BREAK token (PAYMENT DETAILS:, BANK:, IBAN:, etc),
    #   (b) a Y-line change (the value fits on one line in all 4 docs),
    #   (c) any ALL-CAPS colon-terminated token (inline section label).
    MAX_TOKENS = 6

    def _is_anchor_phrase(text: str) -> bool:
        """Return True if the text equals one of ANCHOR_TOKENS ignoring case and
        trailing colons. Accepts both 'Payment Terms' (single structured cell)
        and 'Payment Terms:' variants."""
        norm = text.lower().strip().rstrip(":")
        return norm in {a.rstrip(":") for a in ANCHOR_TOKENS}

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        # Single-token anchor (XLSX/CSV cells often carry the full label).
        single_match = _is_anchor_phrase(tok.text)
        # Two-word anchor (PDF/DOCX split tokens like 'Payment' 'Terms').
        two_match = False
        if i + 1 < len(tokens):
            combo = f"{tok.text} {tokens[i + 1].text}"
            two_match = _is_anchor_phrase(combo)
        if not (single_match or two_match):
            i += 1
            continue

        start = i + (2 if two_match else 1)
        anchor_tok = tokens[start - 1]
        anchor_y = anchor_tok.anchor.y0 if isinstance(anchor_tok.anchor, BBox) else None
        # For XLSX the label and value live in adjacent cells on the same row
        # ('Payment Terms' | 'Net 14'). Take the next same-row cell verbatim —
        # that cell's entire text IS the payment-terms value.
        if isinstance(anchor_tok.anchor, CellRef) and start < len(tokens):
            nxt = tokens[start]
            if (
                isinstance(nxt.anchor, CellRef)
                and nxt.anchor.row == anchor_tok.anchor.row
                and nxt.anchor.sheet == anchor_tok.anchor.sheet
            ):
                val_text = _clean_value(nxt.text)
                if val_text:
                    out["payment_terms"] = ExtractedValue(
                        value=val_text, provenance="extracted",
                        anchor_text=val_text, anchor_ref=nxt.anchor,
                        source="structural", confidence=1.0, attempt=1,
                    )
                    return out

        value_tokens = []
        for j in range(start, min(start + MAX_TOKENS, len(tokens))):
            if _is_section_break(tokens[j].text):
                break
            # Y-line change: if this token's y differs from the anchor
            # by more than 4pt (one line height) AND we already have at
            # least one value token, stop here.
            if (
                anchor_y is not None
                and value_tokens
                and isinstance(tokens[j].anchor, BBox)
                and abs(tokens[j].anchor.y0 - anchor_y) > 4
            ):
                break
            value_tokens.append(tokens[j])
        text = " ".join(t.text for t in value_tokens)[:200]
        text = _clean_value(text)
        if text:
            out["payment_terms"] = ExtractedValue(
                value=text, provenance="extracted",
                anchor_text=text,
                anchor_ref=value_tokens[0].anchor if value_tokens else tok.anchor,
                source="structural", confidence=1.0, attempt=1,
            )
            return out
        i += 1

    # Strategy 2: "Net N" or "within N days" standalone
    for i, t in enumerate(tokens):
        if NET_RE.match(t.text) and i + 1 < len(tokens) and tokens[i+1].text.isdigit():
            out["payment_terms"] = ExtractedValue(
                value=f"Net {tokens[i+1].text}", provenance="extracted",
                anchor_text=f"{t.text} {tokens[i+1].text}",
                anchor_ref=t.anchor, source="structural", confidence=1.0, attempt=1,
            )
            return out
        if WITHIN_RE.match(t.text) and i + 2 < len(tokens) and tokens[i+1].text.isdigit():
            val = f"within {tokens[i+1].text} {tokens[i+2].text}"
            val = _clean_value(val)
            out["payment_terms"] = ExtractedValue(
                value=val, provenance="extracted",
                anchor_text=val, anchor_ref=t.anchor,
                source="structural", confidence=1.0, attempt=1,
            )
            return out
    # Strategy 3: search for phrase "within N days" in full_text (fallback)
    m = re.search(r"within\s+(\d+)\s+days", doc.full_text or "", re.IGNORECASE)
    if m:
        out["payment_terms"] = ExtractedValue(
            value=m.group(0), provenance="extracted",
            anchor_text=m.group(0),
            anchor_ref=tokens[0].anchor if tokens else None,
            source="structural", confidence=0.9, attempt=1,
        )
    return out
