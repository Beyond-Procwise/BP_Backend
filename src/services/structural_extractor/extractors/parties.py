import re
from src.services.structural_extractor.parsing.model import (
    ParsedDocument, Token, BBox, CellRef, ColumnRef, NodeRef
)
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

# Section-header / anchor phrases that must never be returned as a party name.
# These are structural labels, not organization names. Matched against the
# normalized (upper, stripped) line text.
ANCHOR_PHRASE_REJECTS = {
    "BILLED TO", "BILL TO", "BILLED TO:", "BILL TO:",
    "INVOICE TO", "INVOICE TO:",
    "SHIP TO", "SHIP TO:", "SOLD TO", "SOLD TO:",
    "INVOICE", "PAYMENT", "PAYMENT INFORMATION", "PAYMENT DETAILS",
    "CUSTOMER", "CUSTOMER:", "VENDOR", "VENDOR:", "SUPPLIER", "SUPPLIER:",
    "RECIPIENT", "RECIPIENT:",
}


def _is_anchor_phrase(text: str) -> bool:
    """Return True if the given text looks like a structural section-header
    phrase (all-caps, <=3 words, ends in 'TO'/':' or matches a known label).
    These must never be returned as a party name.
    """
    if not text:
        return False
    norm = text.strip().upper().rstrip(":;,.")
    if norm in ANCHOR_PHRASE_REJECTS:
        return True
    # Strip trailing colon and re-check
    if text.strip().upper() in ANCHOR_PHRASE_REJECTS:
        return True
    # Heuristic: short all-caps phrase ending in 'TO' is almost always an
    # anchor label ('PAYABLE TO', 'REMIT TO', etc.).
    words = norm.split()
    if 1 <= len(words) <= 3 and norm.isupper() and words[-1] == "TO":
        return True
    return False


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
                    if _is_postcode_pair(span):
                        i = j
                        continue
                    # Reject anchor-phrase labels ('BILLED TO:', 'PAYMENT
                    # INFORMATION', etc.) — they look like ORG runs
                    # structurally but are section headers, not names.
                    if _is_anchor_phrase(text):
                        i = j
                        continue
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

    # Split same-y-bucket tokens into horizontal column-groups whenever
    # the x-gap between consecutive tokens exceeds 40pt. Otherwise a
    # letterhead on the LEFT ('AUARIUS MARKETING') and invoice metadata
    # on the RIGHT ('Invoice No. INV-25-050') appear as one logical line,
    # and a 'NO'/'DATE' keyword on the right poisons the whole line.
    line_groups: list[tuple[int, list[Token]]] = []
    for yb in sorted(lines.keys()):
        toks_sorted = sorted(lines[yb], key=lambda t: t.anchor.x0)
        group: list[Token] = []
        for t in toks_sorted:
            if not group:
                group.append(t)
                continue
            prev = group[-1]
            gap = t.anchor.x0 - prev.anchor.x1
            if gap > 40:
                line_groups.append((yb, group))
                group = [t]
            else:
                group.append(t)
        if group:
            line_groups.append((yb, group))

    # Build line descriptors with bucket key, tokens, and text
    descriptors: list[tuple[int, list[Token], str]] = []
    for yb, line_toks in line_groups:
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
        # Reject anchor-phrase labels — they can appear in the top band
        # (e.g. 'BILLED TO' when AQUARIUS puts the buyer block at y=168).
        if _is_anchor_phrase(line_text):
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


def _structured_buyer_anchor_tokens(doc: ParsedDocument) -> list[Token]:
    """Return tokens marking the END of a buyer-anchor phrase.

    For a 2-word label ('Bill' 'To:'), the END token is 'To:' — the value
    lookup starts AFTER this token. For a single-token label ('Bill To' in
    an XLSX cell or 'Customer:' alone) the END token IS the label token.

    Matches 'Bill To', 'Bill To:', 'Billed To', 'Invoice To', 'Ship To',
    'Sold To', 'Customer:', 'Invoice For'.
    """
    matches: list[Token] = []
    buyer_phrases = {p.rstrip(":") for p in BUYER_ANCHORS}
    tokens = doc.tokens
    n = len(tokens)
    for i, t in enumerate(tokens):
        single_norm = t.text.lower().strip().rstrip(":;,.")
        if single_norm in buyer_phrases:
            matches.append(t)
            continue
        # Two-word match (DOCX splits "Bill To:" into 'Bill' 'To:').
        # Return the SECOND token — value lookup advances past it.
        if i + 1 < n:
            combo = f"{t.text} {tokens[i + 1].text}".lower().rstrip(":;,. ")
            if combo in buyer_phrases:
                matches.append(tokens[i + 1])
    return matches


def _structured_supplier_anchor_tokens(doc: ParsedDocument) -> list[Token]:
    """Return tokens marking the END of a supplier-anchor phrase.

    Same semantics as the buyer helper — multi-token labels ('Remit'
    'To:') return the LAST token so value lookup starts after it.
    """
    matches: list[Token] = []
    supplier_phrases = {p.rstrip(":") for p in SUPPLIER_ANCHORS}
    tokens = doc.tokens
    n = len(tokens)
    for i, t in enumerate(tokens):
        single_norm = t.text.lower().strip().rstrip(":;,.")
        if single_norm in supplier_phrases:
            matches.append(t)
            continue
        if i + 1 < n:
            combo = f"{t.text} {tokens[i + 1].text}".lower().rstrip(":;,. ")
            if combo in supplier_phrases:
                matches.append(tokens[i + 1])
    return matches


def _value_token_after_label(label_tok: Token, tokens: list[Token]) -> Token | None:
    """Return the candidate value token for a given label anchor.

    XLSX: the value is the cell to the right of the label cell on the
    same row ('Bill To' in A5, 'Assurity Ltd' in B5).

    DOCX: the value sits in the next paragraph (Assurity Ltd in P10
    immediately after 'Bill To:' in P9). If the label is followed in
    the same paragraph by the value (e.g. 'Supplier: WidgetCo Ltd'),
    the in-paragraph remainder wins.
    """
    anchor = label_tok.anchor
    # XLSX: find next token on same row with col > label col.
    if isinstance(anchor, CellRef):
        same_sheet = [
            t for t in tokens
            if isinstance(t.anchor, CellRef)
            and t.anchor.sheet == anchor.sheet
            and t.anchor.row == anchor.row
            and t.anchor.col > anchor.col
            and t.text.strip()
        ]
        same_sheet.sort(key=lambda t: t.anchor.col)
        return same_sheet[0] if same_sheet else None
    # DOCX paragraph anchor.
    if isinstance(anchor, NodeRef) and anchor.kind == "paragraph":
        # Find tokens AFTER label in the same paragraph — sometimes
        # 'Supplier:' 'WidgetCo' 'Ltd' live in one paragraph.
        in_para_after = [
            t for t in tokens
            if isinstance(t.anchor, NodeRef)
            and t.anchor.kind == "paragraph"
            and t.anchor.paragraph_index == anchor.paragraph_index
            and t.order > label_tok.order
            and t.text.strip()
        ]
        if in_para_after:
            # Combine all tokens following the label in the same paragraph.
            # Use the first one as the anchor; the downstream caller joins
            # texts from a span.
            return in_para_after[0]
        # Next paragraph: find smallest paragraph_index > current whose
        # first token is non-empty.
        next_para_toks = [
            t for t in tokens
            if isinstance(t.anchor, NodeRef)
            and t.anchor.kind == "paragraph"
            and t.anchor.paragraph_index is not None
            and anchor.paragraph_index is not None
            and t.anchor.paragraph_index > anchor.paragraph_index
            and t.text.strip()
        ]
        if next_para_toks:
            min_pidx = min(t.anchor.paragraph_index for t in next_para_toks)
            return next(
                (t for t in next_para_toks if t.anchor.paragraph_index == min_pidx),
                None,
            )
    return None


def _paragraph_text(doc: ParsedDocument, paragraph_index: int) -> str:
    toks = [
        t for t in doc.tokens
        if isinstance(t.anchor, NodeRef)
        and t.anchor.kind == "paragraph"
        and t.anchor.paragraph_index == paragraph_index
    ]
    return " ".join(t.text for t in toks)


def _extract_parties_structured(doc: ParsedDocument, doc_type: str) -> dict[str, ExtractedValue]:
    """Extract buyer + supplier for DOCX / XLSX documents.

    Strategy:
    1. Buyer — find a "Bill To" / "Invoice To" / "Customer" label; return
       the value in the next cell (XLSX) or next paragraph (DOCX).
    2. Supplier — find a "Supplier" / "From" / "Remit To" label. If absent,
       fall back to the letterhead: top cell of the first sheet (XLSX) or
       first paragraph (DOCX) that looks like an org name (all-caps multi-
       word OR ends in a corporate suffix).
    """
    out: dict[str, ExtractedValue] = {}
    tokens = doc.tokens

    org_cands = find_candidates(doc, FieldType.ORG)
    # Build a map: token.order -> org candidate text (prefer the longest
    # candidate starting at that token).
    org_by_start: dict[int, tuple[str, Token]] = {}
    for c in org_cands:
        if not c.tokens:
            continue
        key = c.tokens[0].order
        prev = org_by_start.get(key)
        if prev is None or len(c.text) > len(prev[0]):
            org_by_start[key] = (c.text, c.tokens[0])

    # --- Buyer ---
    buyer_val: str | None = None
    buyer_anchor = None
    for label_tok in _structured_buyer_anchor_tokens(doc):
        val_tok = _value_token_after_label(label_tok, tokens)
        if val_tok is None:
            continue
        # Prefer: if the value token IS an ORG candidate start, use the
        # full candidate text.
        if val_tok.order in org_by_start:
            buyer_val, _ = org_by_start[val_tok.order]
            buyer_anchor = val_tok.anchor
            break
        # XLSX: cell text may be a multi-word org like 'Assurity Ltd' —
        # already one token. Use it verbatim.
        if isinstance(val_tok.anchor, CellRef):
            text = val_tok.text.strip()
            if text:
                buyer_val = text
                buyer_anchor = val_tok.anchor
                break
        # DOCX: if val_tok is paragraph-anchored and NOT an org candidate,
        # take the whole paragraph as the buyer string (up to 5 words, to
        # avoid pulling in the subsequent address line).
        if isinstance(val_tok.anchor, NodeRef) and val_tok.anchor.kind == "paragraph":
            # If label and value are in the same paragraph (e.g. 'Supplier:
            # WidgetCo Ltd'), limit to tokens after the label.
            pidx = val_tok.anchor.paragraph_index
            if label_tok.anchor.kind == "paragraph" and label_tok.anchor.paragraph_index == pidx:
                para_toks = [
                    t for t in tokens
                    if isinstance(t.anchor, NodeRef)
                    and t.anchor.paragraph_index == pidx
                    and t.order > label_tok.order
                    and t.text.strip()
                ]
            else:
                para_toks = [
                    t for t in tokens
                    if isinstance(t.anchor, NodeRef)
                    and t.anchor.paragraph_index == pidx
                    and t.text.strip()
                ]
            if para_toks:
                text = " ".join(t.text for t in para_toks[:6])
                buyer_val = text
                buyer_anchor = para_toks[0].anchor
                break

    if buyer_val:
        out["buyer_id"] = ExtractedValue(
            value=buyer_val, provenance="extracted", anchor_text=buyer_val,
            anchor_ref=buyer_anchor, source="structural", confidence=1.0,
            attempt=1,
        )

    # --- Supplier ---
    supplier_val: str | None = None
    supplier_anchor = None
    for label_tok in _structured_supplier_anchor_tokens(doc):
        val_tok = _value_token_after_label(label_tok, tokens)
        if val_tok is None:
            continue
        if val_tok.order in org_by_start:
            supplier_val, _ = org_by_start[val_tok.order]
            supplier_anchor = val_tok.anchor
            break
        if isinstance(val_tok.anchor, CellRef):
            text = val_tok.text.strip()
            if text and text != buyer_val:
                supplier_val = text
                supplier_anchor = val_tok.anchor
                break
        if isinstance(val_tok.anchor, NodeRef) and val_tok.anchor.kind == "paragraph":
            pidx = val_tok.anchor.paragraph_index
            if label_tok.anchor.kind == "paragraph" and label_tok.anchor.paragraph_index == pidx:
                para_toks = [
                    t for t in tokens
                    if isinstance(t.anchor, NodeRef)
                    and t.anchor.paragraph_index == pidx
                    and t.order > label_tok.order
                    and t.text.strip()
                ]
            else:
                para_toks = [
                    t for t in tokens
                    if isinstance(t.anchor, NodeRef)
                    and t.anchor.paragraph_index == pidx
                    and t.text.strip()
                ]
            if para_toks:
                text = " ".join(t.text for t in para_toks[:6])
                if text and text != buyer_val:
                    supplier_val = text
                    supplier_anchor = para_toks[0].anchor
                    break

    # Supplier letterhead fallback: top-of-document ORG candidate.
    if supplier_val is None:
        # Sort candidates by document order (first occurrence first), so
        # the letterhead ORG wins over buyer-block ORGs.
        ordered = sorted(
            org_cands,
            key=lambda c: c.tokens[0].order if c.tokens else 10**9,
        )
        for c in ordered:
            if c.text == buyer_val:
                continue
            # Reject anchor phrases ('PURCHASE ORDER') — these are
            # document titles, not org names.
            if _is_anchor_phrase(c.text):
                continue
            # Reject short generic-label candidates (already rejected by
            # anchor phrase, but double-check).
            supplier_val = c.text
            supplier_anchor = c.tokens[0].anchor if c.tokens else None
            break

    # Emit supplier into schema-declared fields (Purchase_Order uses
    # supplier_name + supplier_id; Invoice uses supplier_id).
    supplier_fields = [
        f for f in fields_for(doc_type)
        if type_of(doc_type, f) == FieldType.ORG and f.startswith("supplier")
    ]
    if supplier_val and supplier_fields:
        for sf in supplier_fields:
            out[sf] = ExtractedValue(
                value=supplier_val, provenance="extracted",
                anchor_text=supplier_val, anchor_ref=supplier_anchor,
                source="structural", confidence=0.9, attempt=1,
            )
    elif supplier_val:
        out["supplier_id"] = ExtractedValue(
            value=supplier_val, provenance="extracted",
            anchor_text=supplier_val, anchor_ref=supplier_anchor,
            source="structural", confidence=0.9, attempt=1,
        )
    return out


def extract_parties(doc: ParsedDocument, doc_type: str) -> dict[str, ExtractedValue]:
    out: dict[str, ExtractedValue] = {}

    # Structured-table formats (DOCX paragraphs, XLSX cells) expose party
    # names through explicit label+value adjacency rather than spatial
    # bbox positioning. Handle them via a dedicated branch — the PDF/BBox
    # logic below is kept untouched for the 14-doc PDF golden set.
    if doc.source_format in ("docx", "xlsx"):
        return _extract_parties_structured(doc, doc_type)

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
            if _is_anchor_phrase(c.text):
                continue
            supplier_val = c.text
            supplier_anchor = _org_anchor(c)
            break

    # Final guard: never return an anchor phrase as the supplier name.
    if supplier_val and _is_anchor_phrase(supplier_val):
        supplier_val = None
        supplier_anchor = None

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
