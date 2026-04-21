import re
from src.services.structural_extractor.parsing.model import ParsedDocument
from src.services.structural_extractor.types import ExtractedValue

ANCHOR_TOKENS = {"payment terms", "payment due", "terms", "payment terms:", "terms:"}
NET_RE = re.compile(r"^Net$", re.IGNORECASE)
WITHIN_RE = re.compile(r"^within$", re.IGNORECASE)


def extract_payment_terms(doc: ParsedDocument, doc_type: str) -> dict[str, ExtractedValue]:
    out: dict[str, ExtractedValue] = {}
    tokens = doc.tokens
    # Strategy 1: anchor on "Payment Terms" / "Terms" label, take next 5-10 tokens
    for i in range(len(tokens) - 1):
        two_words = f"{tokens[i].text} {tokens[i+1].text}".lower().rstrip(":")
        if two_words in ANCHOR_TOKENS:
            value_tokens = tokens[i+2:i+10]
            text = " ".join(t.text for t in value_tokens)[:200]
            if text.strip():
                out["payment_terms"] = ExtractedValue(
                    value=text.strip(), provenance="extracted",
                    anchor_text=text.strip(),
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
