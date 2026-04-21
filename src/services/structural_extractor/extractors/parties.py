import re
from src.services.structural_extractor.parsing.model import ParsedDocument, Token, BBox
from src.services.structural_extractor.types import ExtractedValue
from src.services.structural_extractor.discovery.type_entities import find_candidates
from src.services.structural_extractor.discovery.schema import FieldType, fields_for, type_of
from src.services.structural_extractor.discovery.proximity import inferred_label

BUYER_ANCHORS = {
    "bill to", "billed to", "billed to:", "invoice to", "ship to",
    "sold to", "customer:", "invoice for",
}
SUPPLIER_ANCHORS = {
    "from", "remit to", "payable to", "vendor", "supplier",
    # PO convention: "Recipient:" is the party receiving the PO (the supplier).
    "recipient", "recipient:",
}

# Corporate suffixes used for letterhead-merge heuristics (mixed case + upper).
CORP_SUFFIXES_ANY = {
    "ltd", "ltd.", "llc", "inc", "inc.", "limited", "plc",
    "corp", "company", "gmbh", "ag", "sa",
}

# Stop-words indicating a line is NOT a supplier name (document metadata).
LETTERHEAD_STOPWORDS = {
    "invoice", "invoice.", "purchase order", "quote", "quotation",
    "date:", "date", "no.", "no", "number:",
}


def _find_anchor_position(tokens: list, phrases: set) -> tuple | None:
    """Scan for any of the given (space-separated) anchor phrases and return (page, y, x) of the last token of the match.

    Multi-token phrases take priority over single-token ones at the same
    index, and single-token phrases must carry a trailing colon (or be
    part of a known compound) — a bare 'customer' as section title is
    too ambiguous otherwise.
    """
    n = len(tokens)
    for i in range(n):
        # Prefer 3- and 2-token phrases first.
        matched = False
        for span in (3, 2):
            if i + span > n:
                continue
            joined = " ".join(tokens[j].text for j in range(i, i + span)).lower().rstrip(":;,.")
            # Also try with trailing colon preserved
            joined_colon = " ".join(tokens[j].text for j in range(i, i + span)).lower()
            if joined in phrases or joined_colon in phrases:
                last = tokens[i + span - 1]
                if isinstance(last.anchor, BBox):
                    return (last.anchor.page, last.anchor.y1, last.anchor.x0)
                matched = True
                break
        if matched:
            continue
        # Single-token phrase: require the token to end with a colon OR
        # be the exact phrase including the colon. This rejects standalone
        # section-title words like 'CUSTOMER' that happen to match.
        raw = tokens[i].text
        low = raw.lower()
        if low in phrases and raw.endswith(":"):
            last = tokens[i]
            if isinstance(last.anchor, BBox):
                return (last.anchor.page, last.anchor.y1, last.anchor.x0)
        # Also: if the raw token (with colon) IS in phrases
        if low in phrases and ":" in raw:
            last = tokens[i]
            if isinstance(last.anchor, BBox):
                return (last.anchor.page, last.anchor.y1, last.anchor.x0)
    return None


def _org_anchor(c) -> BBox | None:
    """Return the BBox anchor of an ORG candidate (Candidate or _OrgView)."""
    if hasattr(c, "anchor") and isinstance(c.anchor, BBox):
        return c.anchor
    if hasattr(c, "tokens") and c.tokens:
        a = c.tokens[0].anchor
        if isinstance(a, BBox):
            return a
    return None


def _org_below_or_right_of(anchor_pos, org_cands, x_tolerance: float = 80.0) -> object | None:
    """Find the ORG candidate that is spatially just below (or same-line to
    right of) the label anchor, within x_tolerance horizontally.

    The x filter prevents picking up an ORG in a sibling column when the
    document has side-by-side BILL TO / PAYABLE TO blocks (common in
    invoices laid out as two-column metadata).
    """
    if not anchor_pos:
        return None
    page, y_max, x_anchor = anchor_pos
    best = None
    best_dy = 1e9
    for c in org_cands:
        anchor = _org_anchor(c)
        if anchor is None:
            continue
        if anchor.page != page:
            continue
        dy = anchor.y0 - y_max
        if dy < -3:
            continue
        if dy > 80:
            continue
        # Horizontal alignment: the ORG must start within x_tolerance of
        # the anchor's x. Labels like "BILL TO:" are left-aligned with
        # their value; a value in the next column is a different party.
        if abs(anchor.x0 - x_anchor) > x_tolerance:
            continue
        if dy < best_dy:
            best_dy = dy
            best = c
    return best


def _find_all_caps_orgs(doc: ParsedDocument) -> list[tuple[str, BBox]]:
    """Find multi-word ALL-CAPS runs that look like organization names.

    Fallback for ORG detection when no corporate suffix is present. A
    multi-word run of uppercase word tokens (e.g., 'DESIGN HOUSE AGENCY',
    'ASSURITY LTD') is returned with the bbox of the first token.

    X-gap filtering: break runs when horizontal gap between tokens
    exceeds 40pt — this prevents merging two side-by-side columns
    ('ASSURITY LTD  ...  DESIGN HOUSE AGENCY') into one run.
    """
    lines: dict[tuple, list[Token]] = {}
    for t in doc.tokens:
        if not isinstance(t.anchor, BBox):
            continue
        yb = int(t.anchor.y0 / 4)
        lines.setdefault((t.anchor.page, yb), []).append(t)

    runs: list[tuple[str, BBox]] = []
    for key in sorted(lines.keys()):
        toks = sorted(lines[key], key=lambda t: t.anchor.x0)
        # Find contiguous runs of >=2 uppercase word-like tokens on same line.
        i = 0
        while i < len(toks):
            if _is_upper_word(toks[i]):
                j = i + 1
                while j < len(toks) and _is_upper_word(toks[j]):
                    # Break if x-gap between consecutive tokens is too wide.
                    gap = toks[j].anchor.x0 - toks[j - 1].anchor.x1
                    if gap > 40:
                        break
                    j += 1
                span = toks[i:j]
                if len(span) >= 2:
                    text = " ".join(t.text for t in span)
                    # Reject pure-postcode / pure-alphanumeric-code runs
                    # like 'RH13 5QH' or 'B7 4AX' — these are addresses,
                    # not orgs.
                    if not _is_postcode_pair(span):
                        runs.append((text, span[0].anchor))
                i = j
            else:
                i += 1
    return runs


def _is_postcode_pair(span: list[Token]) -> bool:
    """Heuristic: two-token run that looks like a UK postcode (outcode + incode)."""
    if len(span) != 2:
        return False
    a, b = span[0].text.strip(".,"), span[1].text.strip(".,")
    out_re = re.compile(r"^[A-Z]{1,2}\d[A-Z0-9]?$")
    in_re = re.compile(r"^\d[A-Z]{2}$")
    return bool(out_re.match(a) and in_re.match(b))


def _find_mixed_case_orgs_y_aware(doc: ParsedDocument) -> list[tuple[str, BBox]]:
    """Y-aware corp-suffix detector: finds 'Word Word Ltd' etc. on a single line.

    Complements the global _find_org (whose walk-back crosses line
    boundaries in non-y-sorted token lists — producing junk spans like
    'Item Description Assurity Ltd' when Assurity Ltd appears on one
    line but earlier-indexed tokens come from a different y).
    """
    lines: dict[tuple, list[Token]] = {}
    for t in doc.tokens:
        if not isinstance(t.anchor, BBox):
            continue
        yb = int(t.anchor.y0 / 4)
        lines.setdefault((t.anchor.page, yb), []).append(t)

    runs: list[tuple[str, BBox]] = []
    for key in sorted(lines.keys()):
        toks = sorted(lines[key], key=lambda t: t.anchor.x0)
        for i, t in enumerate(toks):
            clean = t.text.strip(".,;:").lower()
            if clean in CORP_SUFFIXES_ANY:
                # Walk back while tokens look like name words (capitalized
                # or all-uppercase) and on the same line.
                start = i
                while start > 0:
                    prev = toks[start - 1]
                    ptxt = prev.text.strip(".,;:")
                    if not ptxt:
                        break
                    if ptxt[0].isupper() or ptxt.isupper():
                        start -= 1
                    else:
                        break
                if start == i:
                    continue  # just a bare 'Ltd' with nothing before
                span = toks[start:i + 1]
                text = " ".join(tok.text for tok in span)
                runs.append((text, span[0].anchor))
    return runs


def _is_upper_word(tok: Token) -> bool:
    t = tok.text.strip().rstrip(".,:;")
    if not t or len(t) < 2:
        return False
    # Token must contain at least one alpha and all alpha chars uppercase
    if not any(ch.isalpha() for ch in t):
        return False
    for ch in t:
        if ch.isalpha() and not ch.isupper():
            return False
    return True


def _supplier_letterhead(doc: ParsedDocument, y_boundary: float, buyer_val: str | None) -> tuple[str, BBox] | None:
    """Find supplier text in the top-of-page letterhead.

    Returns (text, anchor) or None. Merges up to 3 adjacent letterhead
    lines when the lines sit close together (within 10 y-buckets) and
    neither contains a stopword like 'Invoice' or 'DATE:'. This is how
    we recover 'ELEANOR PRICE' + 'CREATIVE STUDIO' as one name, or
    'DESIGN HOUSE AGENCY' (which has no corp suffix).
    """
    top_tokens = [
        t for t in doc.tokens
        if isinstance(t.anchor, BBox) and t.anchor.page == 1 and t.anchor.y0 < y_boundary
    ]
    if not top_tokens:
        return None

    lines: dict[int, list[Token]] = {}
    for t in top_tokens:
        yb = int(t.anchor.y0 / 4)
        lines.setdefault(yb, []).append(t)

    # Build line descriptors with bucket key, tokens, and text
    descriptors: list[tuple[int, list[Token], str]] = []
    for yb in sorted(lines.keys()):
        line_toks = sorted(lines[yb], key=lambda t: t.anchor.x0)
        line_text = " ".join(t.text for t in line_toks).strip(" .:")
        if not line_text or len(line_text) < 3:
            continue
        low = line_text.lower().strip(":. ")
        if low in LETTERHEAD_STOPWORDS:
            continue
        # Also skip lines containing "DATE:" / "NO." tokens (invoice-metadata)
        toks_up = [t.text.upper().rstrip(":.") for t in line_toks]
        if any(tu in {"DATE", "NO"} for tu in toks_up):
            continue
        if buyer_val and line_text == buyer_val:
            continue
        descriptors.append((yb, line_toks, line_text))

    if not descriptors:
        return None

    # Prefer lines whose words are mostly uppercase (looks like a letterhead
    # rather than prose). If none, fall back to the first line.
    def _mostly_upper(d):
        _, toks, txt = d
        upper_count = sum(1 for t in toks if _is_upper_word(t))
        return upper_count >= max(2, len(toks) // 2)

    preferred = [d for d in descriptors if _mostly_upper(d)]
    pool = preferred if preferred else descriptors

    # Merge adjacent buckets (within 10 y-buckets) starting at the first
    # line in pool.
    first = pool[0]
    merged_text = first[2]
    merged_tokens = list(first[1])
    last_bucket = first[0]
    for yb, toks, txt in pool[1:]:
        if yb - last_bucket <= 10:
            merged_text = f"{merged_text} {txt}"
            merged_tokens.extend(toks)
            last_bucket = yb
        else:
            break

    return merged_text, merged_tokens[0].anchor


def _anchor_to_text(anchor_pos, tokens: list) -> tuple[str, BBox] | None:
    """Find the text directly below an anchor position when ORG detection misses.

    Collects tokens on the same page within 60pt below the anchor y,
    horizontally near the anchor x, and returns the first non-empty
    contiguous text line.
    """
    if not anchor_pos:
        return None
    page, y_max, x_anchor = anchor_pos
    # Collect tokens below anchor, within 60pt, within horizontal band.
    near: list[Token] = []
    for t in tokens:
        if not isinstance(t.anchor, BBox) or t.anchor.page != page:
            continue
        dy = t.anchor.y0 - y_max
        if dy < -3 or dy > 60:
            continue
        if t.anchor.x1 < x_anchor - 20 or t.anchor.x0 > x_anchor + 200:
            continue
        near.append(t)
    if not near:
        return None
    # Group by y-bucket, take the first bucket
    lines: dict[int, list[Token]] = {}
    for t in near:
        yb = int(t.anchor.y0 / 4)
        lines.setdefault(yb, []).append(t)
    first_key = min(lines.keys())
    first_line = sorted(lines[first_key], key=lambda t: t.anchor.x0)
    # Strip email addresses, phone patterns; keep the first chunk of words
    keep: list[Token] = []
    for t in first_line:
        if "@" in t.text:
            break
        if t.text.startswith("+"):
            break
        keep.append(t)
    if not keep:
        return None
    text = " ".join(t.text for t in keep).strip(" .:")
    if not text:
        return None
    return text, keep[0].anchor


def extract_parties(doc: ParsedDocument, doc_type: str) -> dict[str, ExtractedValue]:
    out: dict[str, ExtractedValue] = {}

    # Build a unified ORG candidate list from multiple detectors. The
    # global find_candidates can return spans that cross line boundaries
    # (walk-back doesn't respect Y), so we add two more sources and
    # let the positional logic downstream pick the right one.
    class _OrgView:
        __slots__ = ("text", "anchor")

        def __init__(self, text, anchor):
            self.text = text
            self.anchor = anchor

    org_views: list[_OrgView] = []

    # (1) Global corp-suffix detector, but accept only single-line spans.
    for c in find_candidates(doc, FieldType.ORG):
        toks = c.tokens
        if not toks:
            continue
        anchors = [t.anchor for t in toks if isinstance(t.anchor, BBox)]
        if not anchors:
            continue
        # Reject if span crosses more than 5pt vertically — walk-back
        # produced junk. The real on-line ORG will be found by the
        # y-aware detector instead.
        if max(a.y0 for a in anchors) - min(a.y0 for a in anchors) > 5:
            continue
        org_views.append(_OrgView(c.text, anchors[0]))

    # (2) Y-aware corp-suffix detector (recovers DHA / ELEANOR cases).
    for text, anchor in _find_mixed_case_orgs_y_aware(doc):
        # Dedup against existing views (same text + near y)
        dup = False
        for ov in org_views:
            if ov.text == text and abs(ov.anchor.y0 - anchor.y0) < 5:
                dup = True
                break
        if not dup:
            org_views.append(_OrgView(text, anchor))

    # (3) All-caps multi-word runs (fallback for suffix-less orgs like
    # 'DESIGN HOUSE AGENCY').
    caps_orgs = _find_all_caps_orgs(doc)
    for text, anchor in caps_orgs:
        dup = False
        for ov in org_views:
            if abs(ov.anchor.y0 - anchor.y0) < 5 and (ov.text in text or text in ov.text):
                dup = True
                break
        if not dup:
            org_views.append(_OrgView(text, anchor))

    org_cands = org_views  # reuse same downstream logic — needs .text + .anchor

    # --- Buyer: look for buyer anchor (Bill To, Invoice To, etc.) ---
    buyer_pos = _find_anchor_position(doc.tokens, BUYER_ANCHORS)
    buyer_org = _org_below_or_right_of(buyer_pos, org_cands) if buyer_pos else None

    buyer_val: str | None = None
    buyer_anchor: BBox | None = None
    if buyer_org:
        buyer_val = buyer_org.text
        buyer_anchor = _org_anchor(buyer_org)
    else:
        # Try all-caps fallback: find nearest caps-org below buyer anchor
        if buyer_pos:
            page, y_max, x_anchor = buyer_pos
            best: tuple[str, BBox] | None = None
            best_dy = 1e9
            for text, anchor in caps_orgs:
                if anchor.page != page:
                    continue
                dy = anchor.y0 - y_max
                if dy < -3 or dy > 80:
                    continue
                if abs(anchor.x0 - x_anchor) > 80:
                    continue
                if dy < best_dy:
                    best_dy = dy
                    best = (text, anchor)
            if best:
                buyer_val, buyer_anchor = best
        # Also try: same-line label from ORG's inferred_label.
        # Note: inferred_label requires a Candidate; we only call it for
        # views that have .tokens. _OrgView instances don't, so we skip.
        if buyer_val is None:
            for c in org_cands:
                if not hasattr(c, "tokens"):
                    continue
                lbl = inferred_label(c, doc.tokens).lower().strip(":;,.")
                if any(a in lbl for a in BUYER_ANCHORS):
                    buyer_val = c.text
                    buyer_anchor = _org_anchor(c)
                    break
        # Fallback: raw text below the buyer anchor
        if buyer_val is None and buyer_pos:
            res = _anchor_to_text(buyer_pos, doc.tokens)
            if res:
                buyer_val, buyer_anchor = res

    if buyer_val:
        out["buyer_id"] = ExtractedValue(
            value=buyer_val, provenance="extracted", anchor_text=buyer_val,
            anchor_ref=buyer_anchor, source="structural",
            confidence=1.0, attempt=1,
        )

    # --- Supplier: look for supplier anchor ---
    supplier_pos = _find_anchor_position(doc.tokens, SUPPLIER_ANCHORS)
    supplier_org = _org_below_or_right_of(supplier_pos, org_cands) if supplier_pos else None

    supplier_val: str | None = None
    supplier_anchor: BBox | None = None
    if supplier_org:
        supplier_val = supplier_org.text
        supplier_anchor = _org_anchor(supplier_org)
    elif supplier_pos:
        # Try all-caps fallback near supplier anchor
        page, y_max, x_anchor = supplier_pos
        best: tuple[str, BBox] | None = None
        best_dy = 1e9
        for text, anchor in caps_orgs:
            if anchor.page != page:
                continue
            dy = anchor.y0 - y_max
            if dy < -3 or dy > 80:
                continue
            # Exclude the buyer's text if we already picked it
            if buyer_val and text == buyer_val:
                continue
            if dy < best_dy:
                best_dy = dy
                best = (text, anchor)
        if best:
            supplier_val, supplier_anchor = best
        else:
            res = _anchor_to_text(supplier_pos, doc.tokens)
            if res:
                supplier_val, supplier_anchor = res

    # Supplier letterhead fallback: top-of-page text above the buyer anchor.
    if supplier_val is None:
        # Boundary: top of buyer block (or y=180 if no buyer found).
        y_boundary = buyer_pos[1] if buyer_pos else 180.0
        res = _supplier_letterhead(doc, y_boundary, buyer_val)
        if res:
            supplier_val, supplier_anchor = res

    # Very last fallback: if an ORG candidate exists that's NOT the buyer,
    # pick the top-most one.
    if supplier_val is None and org_cands:
        def _y_key(c):
            a = _org_anchor(c)
            return a.y0 if a else 1e9
        for c in sorted(org_cands, key=_y_key):
            if c.text == buyer_val:
                continue
            supplier_val = c.text
            supplier_anchor = _org_anchor(c)
            break

    # Schema: POs use 'supplier_name' as the primary ORG field, invoices
    # use 'supplier_id'. Emit whichever the schema declares for this doc
    # (some schemas declare both — emit both for symmetry).
    supplier_fields = [
        f for f in fields_for(doc_type)
        if type_of(doc_type, f) == FieldType.ORG and f.startswith("supplier")
    ]
    if supplier_val and supplier_fields:
        for sf in supplier_fields:
            out[sf] = ExtractedValue(
                value=supplier_val, provenance="extracted", anchor_text=supplier_val,
                anchor_ref=supplier_anchor, source="structural",
                confidence=0.9, attempt=1,
            )
    elif supplier_val:
        out["supplier_id"] = ExtractedValue(
            value=supplier_val, provenance="extracted", anchor_text=supplier_val,
            anchor_ref=supplier_anchor, source="structural",
            confidence=0.9, attempt=1,
        )
    return out
