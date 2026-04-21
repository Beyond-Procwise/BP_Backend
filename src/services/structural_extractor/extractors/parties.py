from src.services.structural_extractor.parsing.model import ParsedDocument, Token, BBox
from src.services.structural_extractor.types import ExtractedValue
from src.services.structural_extractor.discovery.type_entities import find_candidates
from src.services.structural_extractor.discovery.schema import FieldType
from src.services.structural_extractor.discovery.proximity import inferred_label

BUYER_ANCHORS = {"bill to", "billed to", "invoice to", "ship to", "sold to", "customer", "invoice for"}
SUPPLIER_ANCHORS = {"from", "remit to", "payable to", "vendor", "supplier"}


def _find_anchor_position(tokens: list, phrases: set) -> tuple | None:
    """Scan for any of the given (space-separated) anchor phrases and return (page, y, x) of the last token of the match."""
    n = len(tokens)
    # Try up to 3-token phrases
    for i in range(n):
        for span in (3, 2):
            if i + span > n:
                continue
            joined = " ".join(tokens[j].text for j in range(i, i + span)).lower().rstrip(":;,.")
            if joined in phrases:
                last = tokens[i + span - 1]
                if isinstance(last.anchor, BBox):
                    return (last.anchor.page, last.anchor.y1, last.anchor.x0)
        # 1-token
        single = tokens[i].text.lower().rstrip(":;,.")
        if single in phrases:
            last = tokens[i]
            if isinstance(last.anchor, BBox):
                return (last.anchor.page, last.anchor.y1, last.anchor.x0)
    return None


def _org_below_or_right_of(anchor_pos, org_cands) -> object | None:
    """Find the ORG candidate that is spatially just below (or same-line to right of) the label anchor."""
    if not anchor_pos:
        return None
    page, y_max, x_anchor = anchor_pos
    best = None
    best_dy = 1e9
    for c in org_cands:
        tok = c.tokens[0]
        if not isinstance(tok.anchor, BBox):
            continue
        if tok.anchor.page != page:
            continue
        # Must be below the anchor OR on same line to the right
        dy = tok.anchor.y0 - y_max
        if dy < -3:
            continue
        # within 80pt vertically
        if dy > 80:
            continue
        if dy < best_dy:
            best_dy = dy
            best = c
    return best


def extract_parties(doc: ParsedDocument, doc_type: str) -> dict[str, ExtractedValue]:
    out: dict[str, ExtractedValue] = {}
    org_cands = find_candidates(doc, FieldType.ORG)

    # --- Buyer: look for buyer anchor (Bill To, Invoice To, etc.) ---
    buyer_pos = _find_anchor_position(doc.tokens, BUYER_ANCHORS)
    buyer_org = _org_below_or_right_of(buyer_pos, org_cands) if buyer_pos else None
    if buyer_org:
        out["buyer_id"] = ExtractedValue(
            value=buyer_org.text, provenance="extracted", anchor_text=buyer_org.text,
            anchor_ref=buyer_org.tokens[0].anchor, source="structural",
            confidence=1.0, attempt=1,
        )
    else:
        # Also try: same-line label from ORG's inferred_label
        for c in org_cands:
            lbl = inferred_label(c, doc.tokens).lower().strip(":;,.")
            if any(a in lbl for a in BUYER_ANCHORS):
                out["buyer_id"] = ExtractedValue(
                    value=c.text, provenance="extracted", anchor_text=c.text,
                    anchor_ref=c.tokens[0].anchor, source="structural",
                    confidence=1.0, attempt=1,
                )
                break

    # --- Supplier: look for supplier anchor ---
    supplier_pos = _find_anchor_position(doc.tokens, SUPPLIER_ANCHORS)
    supplier_org = _org_below_or_right_of(supplier_pos, org_cands) if supplier_pos else None
    if supplier_org:
        out["supplier_id"] = ExtractedValue(
            value=supplier_org.text, provenance="extracted", anchor_text=supplier_org.text,
            anchor_ref=supplier_org.tokens[0].anchor, source="structural",
            confidence=1.0, attempt=1,
        )

    # Supplier letterhead fallback: top-of-page text above the buyer anchor.
    if "supplier_id" not in out:
        buyer_val = out.get("buyer_id").value if "buyer_id" in out else None
        # Boundary: top of buyer block (or y=180 if no buyer found).
        y_boundary = buyer_pos[1] if buyer_pos else 180.0
        # Gather tokens with y0 < y_boundary (above the buyer block) and collect lines.
        top_tokens = [
            t for t in doc.tokens
            if isinstance(t.anchor, BBox) and t.anchor.page == 1 and t.anchor.y0 < y_boundary
        ]
        # Bucket by y-line
        lines: dict[int, list] = {}
        for t in top_tokens:
            yb = int(t.anchor.y0 / 4)
            lines.setdefault(yb, []).append(t)
        # Build candidate letterhead strings: each line's full text
        candidates_text = []
        for yb in sorted(lines.keys()):
            line_toks = sorted(lines[yb], key=lambda t: t.anchor.x0)
            line_text = " ".join(t.text for t in line_toks).strip(" .:")
            # Skip obvious header tokens like 'Invoice.' or 'Invoice'
            if not line_text or len(line_text) < 3:
                continue
            if line_text.lower() in {"invoice", "invoice.", "purchase order", "quote", "quotation"}:
                continue
            if line_text == buyer_val:
                continue
            candidates_text.append((line_toks, line_text))
        # Prefer a known-ORG line; else first non-trivial letterhead line.
        picked = None
        if org_cands:
            for line_toks, line_text in candidates_text:
                for c in org_cands:
                    if c.text in line_text or line_text in c.text:
                        picked = (line_toks, c.text if c.text != buyer_val else line_text)
                        break
                if picked:
                    break
        if picked is None and candidates_text:
            line_toks, line_text = candidates_text[0]
            picked = (line_toks, line_text)
        if picked:
            line_toks, text = picked
            out["supplier_id"] = ExtractedValue(
                value=text, provenance="extracted", anchor_text=text,
                anchor_ref=line_toks[0].anchor, source="structural",
                confidence=0.8, attempt=1,
            )
    return out
