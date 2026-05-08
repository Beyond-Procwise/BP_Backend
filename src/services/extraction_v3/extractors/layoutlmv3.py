"""LayoutLMv3-backed header-field extractor.

PLAN 1 IMPLEMENTATION: This uses canonical-label proximity over the
ParsedDocument's tokens (which carry layout-aware bboxes from Docling /
PaddleOCR). The pre-trained microsoft/layoutlmv3-base has no procurement
labels, so token classification is deferred to Plan 2 (fine-tune).

The "LayoutLMv3" name is preserved for module identity — this slot will
be replaced by the fine-tuned token classifier in Plan 2 without
disturbing downstream consumers (since the contract is just "produce
Candidates").

Algorithm (canonical-label proximity):
  1. For each schema field with "layoutlmv3" in its extractors list,
     iterate over its canonical_labels.
  2. For each label, scan all tokens in every page for a fuzzy text match
     (rapidfuzz.fuzz.ratio >= LABEL_MATCH_MIN_SCORE, or a contains check
     for partial multi-word overlap).
  3. The value token is the closest non-label token that is either:
     - immediately to the right of the label token on the same line, or
     - directly below it within a small vertical band.
  4. For typed fields (money, decimal, iso_date, currency), the proximity
     search is type-aware: only tokens that parse successfully for the
     target type are accepted, scanning up to MAX_TYPE_SCAN_RADIUS tokens
     in reading order from the label position. This prevents label text
     from landing in numeric/date/currency fields.
  5. Emit a Candidate with confidence = (label_match_score / 100) * 0.9,
     capped to [0.10, 0.95] to reflect the Plan 1 baseline uncertainty.

Currency field special case: if no label-proximate currency token is
found, a global scan of ALL tokens is performed for ISO-4217 codes
(USD/GBP/EUR/...) and currency symbols ($/£/€). The most-frequent
match is emitted as a fallback candidate at confidence 0.55.

Substring guarantee: every Candidate.evidence_text is taken DIRECTLY from
a token's text and is verified to appear literally in parsed.full_text before
being emitted.  Note that Docling's markdown exporter HTML-escapes certain
characters (e.g. & → &amp;) so a token's literal text may not always appear
in full_text verbatim; those tokens are silently skipped to preserve the
guarantee rather than emitting an ungrounded candidate.

DO NOT import transformers at module level — Plan 1 does not run inference
through any model heads. Fine-tuned token classification lands in Plan 2.
"""
from __future__ import annotations

import re
from collections import Counter

from rapidfuzz import fuzz

from src.services.extraction_v3.extractors.base import Extractor
from src.services.extraction_v3.schemas.candidate import Candidate
from src.services.extraction_v3.schemas.parsed_document import ParsedDocument, Token
from src.services.extraction_v3.yaml_schema.loader import DocSchema, FieldSpec
from src.services.extraction_v3.yaml_schema.registry import register_extractor


# ---------------------------------------------------------------------------
# Proximity constants — tuned for typical A4/Letter invoice/PO/quote layouts.
# Revisit if adding doc types with unusual typography (e.g. multi-column forms).
# ---------------------------------------------------------------------------
LABEL_MATCH_MIN_SCORE: int = 78      # rapidfuzz ratio threshold [0–100]
RIGHT_OF_LABEL_MAX_DX: float = 500.0  # px; max horizontal gap to the right of a label
BELOW_LABEL_MAX_DY: float = 50.0      # px; max vertical gap below a label
SAME_LINE_TOLERANCE: float = 1.0      # multiplier of label height for same-line check
BELOW_X_TOLERANCE: float = 100.0     # px; max horizontal offset for "below" candidates

# How many nearby candidate tokens to inspect when type-checking
MAX_TYPE_SCAN_RADIUS: int = 6

# ---------------------------------------------------------------------------
# Currency detection constants
# ---------------------------------------------------------------------------
_ISO_CURRENCY_CODES = frozenset({
    "GBP", "USD", "EUR", "JPY", "INR", "CAD", "AUD", "CHF", "CNY", "NZD",
    "ZAR", "SGD", "HKD", "AED", "SEK", "NOK", "DKK", "PLN", "CZK",
})
_SYMBOL_TO_ISO = {"$": "USD", "£": "GBP", "€": "EUR", "¥": "JPY", "₹": "INR"}
# Regex: either a currency symbol or a 3-letter ISO code (word-boundary aware)
_CURRENCY_TOKEN_RE = re.compile(
    r"(?:^|(?<=\s))(USD|GBP|EUR|JPY|INR|CAD|AUD|CHF|CNY|NZD|ZAR|SGD|HKD|AED|"
    r"SEK|NOK|DKK|PLN|CZK|[$£€¥₹])(?:$|(?=\s|[^A-Z]))",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Label-shape detection (shared with sbert_anchor logic)
# ---------------------------------------------------------------------------
_LABEL_RE = re.compile(
    r"^(?:[A-Z][A-Za-z ]{1,40}:?\s*|[A-Z]{2,}(?:\s+[A-Z]{2,})*:?\s*)$"
)
_COLON_LABEL_RE = re.compile(r".+:$")
_ALL_CAPS_WORDS_RE = re.compile(r"^([A-Z]{2,}\s*)+$")


def _is_label_shaped(text: str) -> bool:
    """Return True if *text* looks like a field label rather than a value.

    Recognises:
      - Tokens ending with ":"  (e.g. "Invoice Number:", "BILL TO:")
      - All-caps multi-word tokens with no digits (e.g. "DELIVER TO", "SHIP VIA")
      - Title-case short phrases with no digits (e.g. "Invoice Date")
      - Mixed-case colon-terminated phrases (e.g. "Requested By:")
    """
    s = text.strip()
    if len(s) < 2 or len(s) > 80:
        return False
    # Ends with colon — classic label marker
    if s.endswith(":"):
        return True
    # All-caps, possibly multi-word, no digits
    cleaned = s.replace(":", "").strip()
    words = cleaned.split()
    if words and all(w.isupper() and w.isalpha() for w in words) and len(words) >= 1 and len(s) >= 3:
        return True
    return False


# ---------------------------------------------------------------------------
# Typed-value validators for proximity filtering
# ---------------------------------------------------------------------------

def _try_parse_money(text: str) -> bool:
    """Return True iff text can be coerced to a Money value."""
    from src.services.extraction_v2.parsers.amounts import parse_amount
    return parse_amount(text) is not None


_MONTH_ONLY_RE = re.compile(
    r"^(january|february|march|april|may|june|july|august|september|october|"
    r"november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\.?$",
    re.IGNORECASE,
)


def _try_parse_date(text: str) -> bool:
    """Return True iff text can be coerced to a date.

    Rejects bare month names (e.g. "October") that dateparser would expand to
    "October 1 <current_year>" — these are almost certainly label proximity
    accidents rather than actual dates.
    """
    from src.services.extraction_v2.parsers.dates import parse_date
    s = text.strip()
    # A bare month name has no year info — reject to avoid current-year hallucinations
    if _MONTH_ONLY_RE.match(s):
        return False
    return parse_date(s) is not None


def _try_parse_currency(text: str) -> bool:
    """Return True iff text looks like an ISO-4217 code or currency symbol."""
    s = text.strip()
    if s in _SYMBOL_TO_ISO:
        return True
    if s.upper() in _ISO_CURRENCY_CODES:
        return True
    # Could be embedded: "$1,234" — extract the symbol
    if _CURRENCY_TOKEN_RE.search(s):
        return True
    return False


def _extract_currency_from_token(text: str) -> str | None:
    """Extract the ISO-4217 code from a token that may embed a currency symbol/code."""
    s = text.strip()
    if s in _SYMBOL_TO_ISO:
        return _SYMBOL_TO_ISO[s]
    if s.upper() in _ISO_CURRENCY_CODES:
        return s.upper()
    # Search embedded
    m = _CURRENCY_TOKEN_RE.search(s)
    if m:
        tok = m.group(1)
        if tok in _SYMBOL_TO_ISO:
            return _SYMBOL_TO_ISO[tok]
        upper = tok.upper()
        if upper in _ISO_CURRENCY_CODES:
            return upper
    return None


def _is_document_header(text: str) -> bool:
    """Return True if this token looks like a document-type header (not a field label).

    Document headers are short phrases like "INVOICE", "PURCHASE ORDER",
    "Interior Design Purchase Order" that appear at the top of a page and name
    the document type rather than labelling a specific field.

    Detects both:
    - ALL-CAPS versions: "PURCHASE ORDER", "INVOICE"
    - Title-Case versions: "Interior Design Purchase Order"
    - Mixed: "Purchase Order Form"
    """
    s = text.strip()
    # Must not contain a colon (colons indicate field labels)
    if ":" in s:
        return False
    words = s.split()
    if not words or len(words) > 8:
        return False
    # Must match known document-type vocabulary (case-insensitive)
    doc_type_words = {
        "invoice", "purchase", "order", "quotation", "quote", "contract",
        "statement", "receipt", "estimate", "proposal", "bill", "credit",
        "memo", "note",
    }
    words_lower = [w.lower() for w in words if w.isalpha()]
    return bool(set(words_lower) & doc_type_words)


_TABLE_HEADER_MONEY_RE = re.compile(r"[\$£€¥₹]\s*\d|^\d[\d,\.]+$")
_TABLE_HEADER_NUMERIC_RE = re.compile(r"^\d{5,}$")  # long digit-only string = likely an ID value


def _is_in_table_header_row(tok: Token, all_tokens: list[Token]) -> bool:
    """Return True if *tok* appears to be in a table column-header row.

    Heuristic: a token is in a column-header row if there are at least 2
    other tokens on the same horizontal line (within ±0.8 * token_height)
    AND all co-row tokens are short (≤25 chars) and contain no money/decimal
    values (i.e. look like column labels, not data values like "$450.00").

    Exclusions (returns False immediately):
    - The candidate token itself is a long digit string (≥5 digits) → it's
      an ID value, not in a column-header row even if surrounded by short words.
    - The candidate token itself contains a currency symbol followed by a digit.

    This guards against "Supplier" column header → "Qty" column header
    proximity matches that corrupt supplier_name extraction, while allowing
    "lnvoice" (OCR'd "Invoice") → "1234567890" (numeric PO ID) to pass.
    """
    tok_text = tok.text.strip()

    # If the candidate itself is a numeric ID (≥5 digits) or money → not a column header
    if _TABLE_HEADER_NUMERIC_RE.match(tok_text):
        return False
    if _TABLE_HEADER_MONEY_RE.search(tok_text):
        return False

    ty0, ty1 = tok.bbox[1], tok.bbox[3]
    tok_y_centre = (ty0 + ty1) / 2.0
    tok_height = max(ty1 - ty0, 1.0)

    row_peers: list[Token] = []
    for t in all_tokens:
        if t is tok:
            continue
        t_y_centre = (t.bbox[1] + t.bbox[3]) / 2.0
        if abs(t_y_centre - tok_y_centre) < tok_height * 0.8:
            row_peers.append(t)

    if len(row_peers) < 2:
        # Fewer than 2 peers on same line — not a multi-column header row
        return False

    # All peers must be "column-header-like": short (≤25 chars), no money/decimal content
    for peer in row_peers:
        peer_text = peer.text.strip()
        if len(peer_text) > 25:
            return False  # long text = not a column header
        if _TABLE_HEADER_MONEY_RE.search(peer_text):
            return False  # money value = not a column header row
        if _TABLE_HEADER_NUMERIC_RE.match(peer_text):
            return False  # numeric ID value = not a column header row

    return True  # all peers are short non-value tokens = table header row


def _token_satisfies_type(token: Token, field_type: str) -> bool:
    """Return True iff the token's text satisfies the type constraint."""
    text = token.text.strip()
    if not text:
        return False
    if field_type in ("money", "decimal"):
        return _try_parse_money(text)
    if field_type == "iso_date":
        return _try_parse_date(text)
    if field_type == "currency":
        return _try_parse_currency(text)
    # string / address / postcode: reject if purely label-shaped
    if _is_label_shaped(text):
        return False
    # Accept alphanumeric content (IDs can be purely numeric like "1000587")
    return bool(re.search(r"[a-zA-Z0-9]", text))


# ---------------------------------------------------------------------------
# Intra-token extraction
# ---------------------------------------------------------------------------
# Regex to find "Label: value" pattern WITHIN a single compound token.
# This handles documents where Docling merges "Invoice Number: INV-001 Date: ..."
# into a single token block.
_INTRA_LABEL_VALUE_RE = re.compile(
    r"(?:^|(?<=\s))([A-Za-z][A-Za-z\s#\(\)\.\/]{0,40}?)\s*:\s*(.+?)(?=\s+[A-Za-z][A-Za-z\s#\(\)\.\/]{1,40}?\s*:|$)",
    re.DOTALL,
)


def _extract_intra_token(token_text: str, label: str, field_type: str) -> str | None:
    """Try to extract a typed value from within a compound token text.

    A compound token looks like:
      "Invoice Number: INV-001 Invoice Date: 22nd Sep 2022 ..."
      "Subtotal $3,955.00 Discount (10%) -$395.50 ..."
      "Invoice Number: CM-2022-01"  (single field per token)

    Algorithm:
      1. Try "Label: value" pattern match for the canonical label.
      2. For money fields, also try finding the first/last amount in the
         token text following the label mention.
      3. Return the extracted text if it satisfies the type, or None.
    """
    from src.services.extraction_v2.parsers.amounts import parse_amount
    from src.services.extraction_v2.parsers.dates import parse_date

    text = token_text.strip()
    label_lower = label.lower().strip()

    # Strategy 0: "Label # value" pattern (common in US-style quote/PO headers)
    # Handles "Quotation  # WSG100025", "Invoice # INV-001", "PO # 12345"
    # where "#" acts as separator between label and value.
    label_escaped_0 = re.escape(label.strip().rstrip("#").strip())
    hash_pattern = re.compile(
        rf"(?:^|(?<=\s)){label_escaped_0}\s*#\s*(.+?)(?:\s|$)",
        re.IGNORECASE,
    )
    mh = hash_pattern.search(text)
    if mh:
        raw_val = mh.group(1).strip()
        if _validate_extracted_value(raw_val, field_type):
            return raw_val

    # Strategy 1: explicit "Label: value" pattern within the compound token
    # Build a pattern for this specific label
    label_escaped = re.escape(label.strip())
    # Allow partial match and handle multi-word labels
    label_pattern = re.compile(
        rf"(?:^|(?<=\s)){label_escaped}\s*:\s*(.+?)(?=\s+[A-Za-z]{{2,}}[A-Za-z\s#\(\)\.]*\s*:|$)",
        re.IGNORECASE | re.DOTALL,
    )
    m = label_pattern.search(text)
    if m:
        raw_val = m.group(1).strip()
        if _validate_extracted_value(raw_val, field_type):
            return raw_val

    # Strategy 2: for money fields, find label mention then parse nearest amount
    if field_type in ("money", "decimal"):
        # Find the label position (case-insensitive)
        pos = text.lower().find(label_lower)
        if pos >= 0:
            after = text[pos + len(label):].lstrip(": \t")
            # Scan for first parseable money token in the remainder
            # Split on whitespace and look for amounts
            parts = re.split(r"\s+", after)
            for part in parts[:8]:  # look at next 8 words
                part_clean = re.sub(r"[,\s]$", "", part.strip())
                if part_clean and parse_amount(part_clean) is not None:
                    return part_clean
            # Also try the first "amount-shaped" substring using regex
            amt_m = re.search(r"[\$£€]?\d[\d,\.]+(?:\.\d{2})?", after)
            if amt_m:
                raw_val = amt_m.group(0).strip()
                if parse_amount(raw_val) is not None:
                    return raw_val

    # Strategy 3: for date fields, find label then parse nearest date
    if field_type == "iso_date":
        pos = text.lower().find(label_lower)
        if pos >= 0:
            after = text[pos + len(label):].lstrip(": \t")
            # Take the first 40 chars and try to parse as date
            candidate_snippet = after[:60].strip()
            # Try progressively shorter snippets
            for end in [60, 40, 25, 15]:
                snippet = candidate_snippet[:end].strip()
                if snippet and parse_date(snippet) is not None:
                    return snippet

    # Strategy 4: for currency, find label then look for ISO code or symbol
    if field_type == "currency":
        pos = text.lower().find(label_lower)
        if pos >= 0:
            after = text[pos + len(label):].lstrip(": \t")
            parts = re.split(r"\s+", after)
            for part in parts[:3]:
                iso = _extract_currency_from_token(part.strip())
                if iso:
                    return iso
            # Also check if there's a currency symbol in amounts
            iso = _extract_currency_from_token(after[:20])
            if iso:
                return iso

    return None


def _validate_extracted_value(raw_val: str, field_type: str) -> bool:
    """Return True if raw_val is plausible for field_type."""
    from src.services.extraction_v2.parsers.amounts import parse_amount
    from src.services.extraction_v2.parsers.dates import parse_date

    s = raw_val.strip()
    if not s:
        return False
    if field_type in ("money", "decimal"):
        return parse_amount(s) is not None
    if field_type == "iso_date":
        return parse_date(s) is not None
    if field_type == "currency":
        return _try_parse_currency(s)
    # string: non-empty, not purely a label phrase.
    # Accept alphanumeric IDs (e.g. "1000587"), mixed strings, etc.
    # Only reject tokens that look EXACTLY like a label (ends with ":" or all-caps words).
    if _is_label_shaped(s):
        return False
    # Require non-empty content (alphanumeric or punctuation, but not only whitespace)
    return bool(re.search(r"[a-zA-Z0-9]", s))


@register_extractor("layoutlmv3")
class LayoutLMv3Extractor(Extractor):
    """Plan 1 baseline: canonical-label proximity extractor with type-aware value selection.

    Registered as "layoutlmv3" so the YAML schema's extractor lists resolve
    to this class.  Plan 2 will replace the body of this class with a
    fine-tuned LayoutLMv3ForTokenClassification inference pass while keeping
    the class name and registration key unchanged.
    """

    def produce_candidates(
        self,
        parsed: ParsedDocument,
        schema: DocSchema,
    ) -> list[Candidate]:
        candidates: list[Candidate] = []
        for field in schema.fields:
            if "layoutlmv3" not in field.extractors:
                continue
            field_cands = self._candidates_for_field(parsed, field)
            candidates.extend(field_cands)

            # Currency field: if no label-proximate candidate was produced,
            # fall back to a global token scan for currency symbols/codes.
            if field.name == "currency" and not field_cands:
                fallback = self._currency_global_scan(parsed, field)
                if fallback:
                    candidates.append(fallback)

            # Money/decimal fields: if no token-proximity candidates were found,
            # try scanning the markdown representation of the full_text for
            # "| Label   | £amount |" patterns (common when Docling renders tables
            # as markdown with no discrete bboxed tokens for table cells).
            if field.type in ("money", "decimal") and not field_cands:
                md_cands = self._markdown_table_scan(parsed, field)
                candidates.extend(md_cands)

        return candidates

    # ---------------------------------------------------------------------- #
    # Internal helpers                                                         #
    # ---------------------------------------------------------------------- #

    def _candidates_for_field(
        self,
        parsed: ParsedDocument,
        field: FieldSpec,
    ) -> list[Candidate]:
        out: list[Candidate] = []
        for label in field.canonical_labels:
            for page in parsed.pages:
                for label_token, label_score in self._find_label_tokens(page.tokens, label):
                    confidence = min(0.95, max(0.10, (label_score / 100.0) * 0.9))

                    # --- Path A: compound token (label + value in same token text) ---
                    # This handles "Invoice Number: INV-001 Date: ..." single-token blocks.
                    intra_value = _extract_intra_token(
                        label_token.text, label, field.type
                    )
                    if intra_value:
                        # Substring guarantee: intra_value must appear in full_text
                        if intra_value in parsed.full_text:
                            # For currency, normalize to ISO code
                            if field.type == "currency":
                                iso = _extract_currency_from_token(intra_value)
                                display_val = iso if iso else intra_value
                            else:
                                display_val = intra_value
                            out.append(Candidate(
                                field=field.name,
                                value=display_val,
                                page=label_token.page,
                                bbox=label_token.bbox,
                                evidence_text=intra_value,
                                model="layoutlmv3",
                                confidence=confidence,
                            ))
                            continue  # found within this token; skip proximity search

                    # --- Path B: separate token (label token → nearby value token) ---
                    value_token, norm_value = self._next_value_token(
                        page.tokens, label_token, field.type
                    )
                    if value_token is None:
                        continue
                    # Substring guard — the raw token text must appear in full_text.
                    raw_tok_text = value_token.text.strip()
                    if raw_tok_text not in parsed.full_text:
                        continue
                    # Sanity guard: if the label token IS the document header
                    # (e.g. "PURCHASE ORDER", "INVOICE"), skip proximity-based
                    # value candidates entirely.  The document-type header is not
                    # a field label; it names the document.  Any value embedded in
                    # a header token is handled by Path A (intra-token) above.
                    if _is_document_header(label_token.text):
                        continue
                    evidence_text = raw_tok_text
                    out.append(
                        Candidate(
                            field=field.name,
                            value=norm_value if norm_value else raw_tok_text,
                            page=value_token.page,
                            bbox=value_token.bbox,
                            evidence_text=evidence_text,
                            model="layoutlmv3",
                            confidence=confidence,
                        )
                    )
        return out

    def _find_label_tokens(
        self,
        tokens: list[Token],
        label: str,
    ) -> list[tuple[Token, float]]:
        """Return tokens whose text is a fuzzy match for *label*, with scores.

        Extended to detect compound tokens that contain the label embedded
        (e.g. "Invoice Number: INV-001 Invoice Date: ..." contains both labels).
        """
        matches: list[tuple[Token, float]] = []
        label_lower = label.lower().strip()
        # Normalize label for whitespace-insensitive matching
        label_lower_norm = re.sub(r"\s+", " ", label_lower)
        for tok in tokens:
            tok_text_raw = tok.text.strip()
            tok_text = tok_text_raw.lower().rstrip(":").rstrip(".")
            # Normalize token text for whitespace-insensitive containment checks
            tok_text_norm = re.sub(r"\s+", " ", tok_text)
            score = fuzz.ratio(tok_text, label_lower)
            if score >= LABEL_MATCH_MIN_SCORE:
                matches.append((tok, float(score)))
            elif label_lower_norm in tok_text_norm:
                # Embedded label match — label appears somewhere in a larger token.
                # The token is a compound "Label: value" or "Label # value" block.
                # We grant a fixed score of 80.0 (= LABEL_MATCH_MIN_SCORE + 2) so
                # Path A (intra-token extraction) can handle it.  Previously the
                # formula 80*ratio+20 under-scored long compound tokens (e.g.
                # "Quotation  # WSG100024" with ratio=0.41 → 52.7 < threshold).
                # Whitespace is normalised to handle double-space OCR/formatting artefacts.
                partial_score = max(float(score), 80.0)
                matches.append((tok, min(90.0, partial_score)))
            elif tok_text_norm in label_lower_norm:
                matches.append((tok, max(float(score), 80.0)))
        return matches

    def _next_value_token(
        self,
        tokens: list[Token],
        label_tok: Token,
        field_type: str = "string",
    ) -> tuple[Token | None, str | None]:
        """Return (value_token, normalized_value) geometrically nearest to *label_tok*.

        Search order (lower sort key = preferred):
          1. Same line, immediately to the right  (sort key = dx)
          2. Below the label, similar x-range     (sort key = dy + 1000)

        For typed fields (money, decimal, iso_date, currency), only tokens
        that parse successfully are accepted. We scan up to MAX_TYPE_SCAN_RADIUS
        candidates in rank order before giving up, so a single wrong token
        doesn't permanently block the right one.

        Returns (None, None) if no suitable token is found.
        """
        lx0, ly0, lx1, ly1 = label_tok.bbox
        label_height = max(ly1 - ly0, 1.0)  # guard against zero-height bbox

        ranked: list[tuple[float, Token]] = []
        for tok in tokens:
            if tok is label_tok:
                continue
            tx0, ty0, tx1, ty1 = tok.bbox

            # --- same-line heuristic ---
            same_line = abs(ty0 - ly0) < label_height * SAME_LINE_TOLERANCE
            dx = tx0 - lx1
            if same_line and 0.0 < dx < RIGHT_OF_LABEL_MAX_DX:
                ranked.append((dx, tok))
                continue

            # --- below-the-label heuristic ---
            dy = ty0 - ly1
            if 0.0 < dy < BELOW_LABEL_MAX_DY and abs(tx0 - lx0) < BELOW_X_TOLERANCE:
                ranked.append((dy + 1000.0, tok))

        if not ranked:
            return None, None
        ranked.sort(key=lambda pair: pair[0])

        # For untyped string fields: skip tokens that are themselves label-shaped.
        # For typed fields: pick the first token that satisfies the type constraint.
        is_typed = field_type in ("money", "decimal", "iso_date", "currency")
        scan_limit = MAX_TYPE_SCAN_RADIUS if is_typed else 4

        for _, tok in ranked[:scan_limit]:
            if is_typed:
                if _token_satisfies_type(tok, field_type):
                    norm = None
                    if field_type == "currency":
                        norm = _extract_currency_from_token(tok.text.strip())
                    return tok, norm
            else:
                # Free-text: skip if the token is itself label-shaped
                if _is_label_shaped(tok.text):
                    continue
                tok_stripped = tok.text.strip()
                # Must contain at least one alphanumeric character
                if not re.search(r"[a-zA-Z0-9]", tok_stripped):
                    continue
                # Skip compound tokens that embed a "Label: value" sub-pattern.
                # Two conditions (either is sufficient):
                # 1. Token is long (>80 chars) — address blocks, multi-field blobs
                # 2. Token contains ":<text>" where before the colon is a short
                #    alpha-only phrase (≤40 chars) — classic "Field: value" structure
                if len(tok_stripped) > 80:
                    continue
                if ":" in tok_stripped:
                    colon_pos = tok_stripped.index(":")
                    before_colon = tok_stripped[:colon_pos].strip()
                    # Short label-like before-colon = label-value compound.
                    # Allows "#", ".", digits in label like "Invoice #", "P.O."
                    if before_colon and len(before_colon) <= 40 and re.match(r"^[A-Za-z][A-Za-z0-9 #\.\-/&]*$", before_colon):
                        continue
                # Skip tokens that are in a table column-header row context.
                # e.g. "Supplier" → "Qty" (next column header) must be suppressed.
                if _is_in_table_header_row(tok, tokens):
                    continue
                return tok, None

        return None, None

    # Country → default currency mapping for country-based inference
    _COUNTRY_TO_CURRENCY: dict[str, str] = {
        "united kingdom": "GBP", "uk": "GBP", "england": "GBP",
        "united states": "USD", "usa": "USD", "us": "USD",
        "australia": "AUD",
        "canada": "CAD",
        "india": "INR",
        "japan": "JPY",
        "china": "CNY",
        "germany": "EUR", "france": "EUR", "spain": "EUR", "italy": "EUR",
        "netherlands": "EUR", "belgium": "EUR", "austria": "EUR",
        "new zealand": "NZD",
        "singapore": "SGD",
        "hong kong": "HKD",
        "south africa": "ZAR",
        "united arab emirates": "AED", "uae": "AED",
        "sweden": "SEK",
        "norway": "NOK",
        "denmark": "DKK",
        "poland": "PLN",
    }

    # Regex: ISO currency code in parentheses, e.g. "(USD)" or "[ USD ]"
    # Also catches standalone ISO codes on word boundaries in text.
    _CURRENCY_IN_TEXT_RE = re.compile(
        r"[\(\[]\s*(USD|GBP|EUR|JPY|INR|CAD|AUD|CHF|CNY|NZD|ZAR|SGD|HKD|AED|"
        r"SEK|NOK|DKK|PLN|CZK)\s*[\)\]]"
        r"|(?<![A-Z])(USD|GBP|EUR|JPY|INR|CAD|AUD|CHF|CNY|NZD|ZAR|SGD|HKD|AED|"
        r"SEK|NOK|DKK|PLN|CZK)(?![A-Z])",
        re.IGNORECASE,
    )

    def _currency_global_scan(
        self,
        parsed: ParsedDocument,
        field: FieldSpec,
    ) -> Candidate | None:
        """Global scan for currency symbols/codes when no label match was found.

        Pass 1: Counts all tokens whose text contains an ISO-4217 code or symbol.
        Pass 1b: If no token match, scan full_text for ISO codes (e.g. "(USD)" in
                 column headers like "Unit Price (USD)" — common in Canva/sparse PDFs).
        Pass 2: If nothing found, infers from country mentions in the full text.
        Emits at confidence 0.55 (token match), 0.50 (text scan), 0.40 (country inference).
        """
        counter: Counter[str] = Counter()
        token_for: dict[str, Token] = {}
        page_for: dict[str, int] = {}

        for page in parsed.pages:
            for tok in page.tokens:
                iso = _extract_currency_from_token(tok.text.strip())
                if iso:
                    counter[iso] += 1
                    if iso not in token_for:
                        token_for[iso] = tok
                        page_for[iso] = page.index

        if counter:
            best_iso, _ = counter.most_common(1)[0]
            tok = token_for[best_iso]
            raw_text = tok.text.strip()
            # Substring guarantee: raw token must be in full_text
            if raw_text in parsed.full_text:
                return Candidate(
                    field=field.name,
                    value=best_iso,
                    page=page_for[best_iso],
                    bbox=tok.bbox,
                    evidence_text=raw_text,
                    model="layoutlmv3",
                    confidence=0.55,
                )

        # Pass 1b: scan full_text for ISO codes embedded in column headers / text
        # This handles sparse-token PDFs (Canva style) where "Unit Price (USD)" is
        # only in the markdown export and has no discrete bboxed token.
        text_iso_counter: Counter[str] = Counter()
        text_evidence: dict[str, str] = {}
        for m in self._CURRENCY_IN_TEXT_RE.finditer(parsed.full_text):
            code = (m.group(1) or m.group(2) or "").upper().strip()
            if not code:
                continue
            iso = _SYMBOL_TO_ISO.get(code, code if code in _ISO_CURRENCY_CODES else None)
            if iso:
                text_iso_counter[iso] += 1
                if iso not in text_evidence:
                    # Use the matched substring as evidence (guaranteed in full_text)
                    text_evidence[iso] = m.group(0).strip()

        if text_iso_counter:
            best_iso, _ = text_iso_counter.most_common(1)[0]
            ev_text = text_evidence[best_iso]
            if ev_text in parsed.full_text:
                return Candidate(
                    field=field.name,
                    value=best_iso,
                    page=0,
                    bbox=(0.0, 0.0, 0.0, 0.0),
                    evidence_text=ev_text,
                    model="layoutlmv3",
                    confidence=0.50,
                )

        # Pass 2: infer from country mentions in the document text
        full_lower = parsed.full_text.lower()
        for country_text, iso_code in self._COUNTRY_TO_CURRENCY.items():
            if country_text in full_lower:
                # Find a token that contains the country text for evidence
                evidence_tok = None
                for page in parsed.pages:
                    for tok in page.tokens:
                        if country_text in tok.text.lower():
                            evidence_tok = tok
                            ev_page = page.index
                            break
                    if evidence_tok:
                        break
                if evidence_tok is None:
                    continue
                raw_text = evidence_tok.text.strip()
                if raw_text not in parsed.full_text:
                    continue
                return Candidate(
                    field=field.name,
                    value=iso_code,
                    page=ev_page,
                    bbox=evidence_tok.bbox,
                    evidence_text=raw_text,
                    model="layoutlmv3",
                    confidence=0.40,  # lower confidence: inferred from country, not explicit
                )

        return None

    # Markdown table row: a line starting with "|" and containing at least two "|"
    _MD_TABLE_LINE_RE = re.compile(r"^\|(.+)\|$", re.MULTILINE)

    def _markdown_table_scan(
        self,
        parsed: ParsedDocument,
        field: FieldSpec,
    ) -> list[Candidate]:
        """Scan full_text for markdown table rows that contain a canonical label.

        Handles multi-column tables (e.g. 6-column Canva POs) by splitting each
        row into all cells and checking whether ANY cell matches a canonical label.
        The value is taken from the LAST cell that satisfies the type constraint.

        Emits candidates at confidence up to 0.85 (label-match weighted).
        Only runs when no token-proximity candidates were found for the field.
        Substring guarantee: candidate value must appear verbatim in full_text.
        """
        from src.services.extraction_v2.parsers.amounts import parse_amount
        from src.services.extraction_v2.parsers.dates import parse_date
        from rapidfuzz import fuzz

        out: list[Candidate] = []
        label_lower_set = {lbl.lower().rstrip(":").strip() for lbl in field.canonical_labels}

        for m in self._MD_TABLE_LINE_RE.finditer(parsed.full_text):
            row_text = m.group(1)  # everything between leading | and trailing |
            cells = [c.strip() for c in row_text.split("|")]

            # Skip separator rows (only dashes/colons in cells)
            if all(re.match(r"^[-:\s]*$", c) for c in cells if c):
                continue

            # Check if any cell matches a canonical label
            best_score = 0.0
            label_cell_idx = -1
            for i, cell in enumerate(cells):
                cell_lower = cell.lower().rstrip(":").strip()
                if not cell_lower:
                    continue
                for lbl_lower in label_lower_set:
                    score = fuzz.ratio(cell_lower, lbl_lower)
                    if score > best_score:
                        best_score = score
                        label_cell_idx = i

            if best_score < LABEL_MATCH_MIN_SCORE or label_cell_idx < 0:
                continue

            # Find the value: scan cells AFTER the label cell (right-to-left preference),
            # pick the last cell that satisfies the type constraint.
            value_str = None
            for cell in reversed(cells[label_cell_idx + 1:]):
                candidate_val = cell.strip()
                if not candidate_val:
                    continue
                if field.type in ("money", "decimal"):
                    if parse_amount(candidate_val) is not None:
                        value_str = candidate_val
                        break
                elif field.type == "iso_date":
                    if parse_date(candidate_val) is not None:
                        value_str = candidate_val
                        break
                else:
                    # string field: accept non-empty, non-label cell
                    if not re.match(r"^[-:\s]*$", candidate_val) and not _is_label_shaped(candidate_val):
                        value_str = candidate_val
                        break

            if not value_str:
                continue

            # Substring guarantee: the value must appear in full_text
            if value_str not in parsed.full_text:
                continue

            out.append(Candidate(
                field=field.name,
                value=value_str,
                page=0,
                bbox=(0.0, 0.0, 0.0, 0.0),
                evidence_text=value_str,
                model="layoutlmv3",
                confidence=min(0.85, (best_score / 100.0) * 0.9),
            ))

        return out
