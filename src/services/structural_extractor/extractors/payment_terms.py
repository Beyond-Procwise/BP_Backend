import re
from src.services.structural_extractor.parsing.model import ParsedDocument, BBox
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
    for i in range(len(tokens) - 1):
        two_words = f"{tokens[i].text} {tokens[i+1].text}".lower().rstrip(":")
        if two_words in ANCHOR_TOKENS:
            # Collect value tokens until a section break or token cap.
            anchor_y = None
            if i + 1 < len(tokens) and isinstance(tokens[i + 1].anchor, BBox):
                anchor_y = tokens[i + 1].anchor.y0
            value_tokens = []
            for j in range(i + 2, min(i + 2 + MAX_TOKENS, len(tokens))):
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
                    anchor_ref=value_tokens[0].anchor if value_tokens else tokens[i].anchor,
                    source="structural", confidence=1.0, attempt=1,
                )
                return out
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
