"""Supplier name → supplier_id resolution with auto-create.

Looks up proc.bp_supplier for an existing match using:
  1. Exact case-insensitive match on supplier_name or trading_name.
  2. rapidfuzz WRatio fuzzy match (threshold >= 85) against all supplier names.
  3. If no match: auto-create a new proc.bp_supplier row with a slug-derived
     supplier_id (e.g. ``SUP-MGMSouvenirShop``) and return it.

A LogisticRegression char-ngram classifier (``models/supplier_name_classifier.joblib``)
is also applied before resolution.  Any candidate whose P(valid) < _CLF_THRESHOLD
is dropped so contaminated strings never reach the database.

Never raises — returns None on any unrecoverable error so the caller can
persist supplier_id = NULL and put the row into the review queue.
"""
from __future__ import annotations

import logging
from src.services.governed_limits import limit as _governed_limit
import os
import re
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Classifier lazy-loading
# ---------------------------------------------------------------------------
# Probability threshold below which a candidate is treated as invalid.
# 0.44 is the empirically-derived cutoff: all known-bad strings score <= 0.435,
# all known-good strings (including edge-cases like "Lane Bryant") score >= 0.446.
_CLF_THRESHOLD: float = 0.44

# Module-level sentinel: None = not yet attempted; False = load failed.
_clf_model: dict[str, Any] | None | bool = None  # None → not loaded yet


def _load_classifier() -> dict[str, Any] | None:
    """Load the supplier-name classifier from disk exactly once.

    Returns the model dict (keys: vectorizer, clf, threshold) or None if the
    model file is missing / joblib is unavailable.  Thread safety is not
    critical here because worst-case two threads load it simultaneously — the
    outcome is identical and the duplicate object is GC'd immediately.
    """
    global _clf_model
    if _clf_model is not None and _clf_model is not False:
        return _clf_model  # type: ignore[return-value]
    if _clf_model is False:
        return None  # previous load attempt failed — don't retry

    # Locate the model file relative to this source tree.
    _HERE = Path(__file__).parent
    # Walk up to project root (BP_Backend/) then into models/
    _project_root = _HERE
    for _ in range(6):
        candidate = _project_root / "models" / "supplier_name_classifier.joblib"
        if candidate.exists():
            break
        _project_root = _project_root.parent
    else:
        log.warning(
            "supplier_resolver: classifier model not found; "
            "falling back to rule-based filter only"
        )
        _clf_model = False
        return None

    try:
        import joblib  # type: ignore[import]

        data = joblib.load(str(candidate))
        if not isinstance(data, dict) or "vectorizer" not in data or "clf" not in data:
            raise ValueError("unexpected model dict structure")
        _clf_model = data
        log.info(
            "supplier_resolver: loaded classifier from %s (threshold=%.2f)",
            candidate,
            data.get("threshold", _CLF_THRESHOLD),
        )
        return data
    except Exception as exc:
        log.warning("supplier_resolver: classifier load failed (%s); using rules only", exc)
        _clf_model = False
        return None


def _classifier_accepts(name: str) -> bool:
    """Return True if the name passes the ML classifier (or if classifier unavailable).

    Uses P(valid) >= _CLF_THRESHOLD as the acceptance criterion.
    """
    model = _load_classifier()
    if model is None:
        return True  # no model → accept (rule-based filter still applies)

    threshold = float(model.get("threshold", _CLF_THRESHOLD))
    try:
        vec = model["vectorizer"]
        clf = model["clf"]
        X = vec.transform([name])
        prob = clf.predict_proba(X)[0, 1]  # P(class=1 → valid)
        accept = prob >= threshold
        if not accept:
            log.debug(
                "supplier_resolver: classifier rejected %r (P(valid)=%.3f < %.2f)",
                name, prob, threshold,
            )
        return accept
    except Exception as exc:
        log.debug("supplier_resolver: classifier predict failed (%s); accepting by default", exc)
        return True


# ---------------------------------------------------------------------------
# Rule-based garbage filter
# ---------------------------------------------------------------------------
# Minimum name length to attempt resolution / creation.
_MIN_NAME_LEN = 3
# rapidfuzz threshold (0-100). Raised from 85 → 92: at 85, two suppliers that
# share only a business suffix ("Perry Ltd" vs "UrbEdge Facilities Management
# Ltd" → 85.5 via WRatio) got falsely merged because the suffix inflated the
# partial-token score. 92 makes the match require substantial overlap on the
# distinctive part of the name. Below this, we auto-create a new supplier.
_FUZZY_THRESHOLD = 92

# Supplier-match review band. A close-call decision (near, but not clearly the
# same or clearly distinct) is FLAGGED for human confirmation rather than acted
# on silently. Straddles _FUZZY_THRESHOLD:
#   [_REVIEW_LOW, 92)  → auto-CREATED a supplier, but it's a possible duplicate.
#   [92, _REVIEW_HIGH) → auto-LINKED to a supplier, but it's a possible false merge.
# Outside the band the auto-decision is confident and no review is raised.
# SupplierIdentityPolicy (P9): these bands decide when two names are one
# company, and therefore whose bank details an invoice is paid against.
def _REVIEW_LOW() -> float:
    return _governed_limit("supplier_identity", "review_low",
                           env="SUPPLIER_REVIEW_LOW")


def _REVIEW_HIGH() -> float:
    return _governed_limit("supplier_identity", "review_high",
                           env="SUPPLIER_REVIEW_HIGH")
_REVIEW_ENABLED = os.getenv("SUPPLIER_REVIEW_ENABLED", "1") not in ("0", "false", "False")


def _lookup_alias(name: str, cur) -> str | None:
    """Return the canonical supplier_id for a human-confirmed alias, or None."""
    try:
        cur.execute(
            "SELECT supplier_id FROM proc.bp_supplier_alias "
            "WHERE LOWER(alias_name) = LOWER(%s) LIMIT 1",
            (name,),
        )
        row = cur.fetchone()
        return row[0] if row else None
    except Exception:  # noqa: BLE001 - table may not exist yet; fail open
        return None


def _emit_review(cur, *, extracted_name, decision, chosen_id, candidate_id,
                 candidate_name, score, doc_type, doc_pk, trace_id) -> int | None:
    """Flag a close-call supplier decision for human review. Best-effort.

    Returns the new review_id, or None if skipped (already flagged / disabled)
    or on error.
    """
    if not _REVIEW_ENABLED:
        return None
    try:
        # Skip if an equivalent pair is already flagged (either direction) or
        # already resolved (confirmed/rejected) — never re-flag a decided pair.
        cur.execute(
            "SELECT 1 FROM proc.bp_supplier_review WHERE "
            "((LOWER(extracted_name) = LOWER(%s) AND candidate_supplier_id = %s) OR "
            " (chosen_supplier_id = %s AND candidate_supplier_id = %s) OR "
            " (chosen_supplier_id = %s AND candidate_supplier_id = %s)) "
            "AND status IN ('pending','confirmed','rejected') LIMIT 1",
            (extracted_name, candidate_id, chosen_id, candidate_id, candidate_id, chosen_id),
        )
        if cur.fetchone():
            return None
        cur.execute(
            "INSERT INTO proc.bp_supplier_review "
            "(extracted_name, decision, chosen_supplier_id, candidate_supplier_id, "
            " candidate_supplier_name, score, doc_type, doc_pk, trace_id) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING review_id",
            (extracted_name, decision, chosen_id, candidate_id, candidate_name,
             score, doc_type, doc_pk, trace_id),
        )
        rid = cur.fetchone()[0]
        log.info("supplier_resolver: flagged review %r (%s, candidate=%s, score=%.1f)",
                 extracted_name, decision, candidate_id, score)
        return rid
    except Exception:  # noqa: BLE001
        log.debug("supplier_resolver: review emit failed", exc_info=True)
        return None


def _create_supplier(cur, display_name: str) -> str:
    """Insert a new bp_supplier row (SUP-<slug>) and return its id."""
    new_id = f"SUP-{_slug(display_name)}"
    cur.execute(
        "SELECT supplier_id FROM proc.bp_supplier WHERE supplier_id = %s LIMIT 1",
        (new_id,),
    )
    if cur.fetchone():
        log.info("supplier_resolver: collision resolved — reusing %s", new_id)
        return new_id
    cur.execute(
        "INSERT INTO proc.bp_supplier "
        "(supplier_id, supplier_name, trading_name, created_date, created_by) "
        "VALUES (%s, %s, %s, NOW(), %s)",
        (new_id, display_name, display_name, "ExtractionV3-AutoDiscovery"),
    )
    log.info("supplier_resolver: auto-created supplier '%s' → %s", display_name, new_id)
    return new_id

# Business-entity suffixes stripped BEFORE the WRatio comparison so the
# distinctive part of the name dominates the score. Without this, every
# "X Ltd" / "Y INC" pair scores ~85 against each other regardless of stem.
# `Ld` / `Lt` are OCR of `Ltd` with a character dropped, and they are not cosmetic: the
# suffix is stripped BEFORE the fuzzy comparison, so an unrecognised one strips
# asymmetrically. "PeopleFirst HR Solutions Ltd" reduced to "PeopleFirst HR" while the
# scanned "PeopleFirst HR Solutions Ld" reduced to nothing at all — 90.0 against a
# threshold of 92, so the same company was minted a second time under a second id, and
# every downstream join (ranking, deal linking) then failed on the split.
_BIZ_SUFFIX_RE = re.compile(
    r"\s*[,\.]?\s*\b(?:LLC|Ltd|Ld|Lt|Limited|Inc|Incorporated|Pvt|Pvt\.?\s*Ltd|"
    r"Private\s+Limited|GmbH|Corp|Corporation|Co\.?|Company|Studios|Agency|"
    r"Group|Solutions|Services|Holdings|Enterprises?|Partnership|LLP|"
    r"PLC|AG|S\.?A\.?|N\.?V\.?|S\.?L\.?|S\.?r\.?l\.?|B\.?V\.?)\b\.?\s*$",
    re.IGNORECASE,
)


def _strip_biz_suffix(name: str) -> str:
    """Strip business-entity suffix for a fairer WRatio comparison.

    "Perry Ltd"                          → "Perry"
    "UrbEdge Facilities Management Ltd"  → "UrbEdge Facilities Management"
    "Wade INC"                           → "Wade"
    "FASHION ITEMS INC"                  → "FASHION ITEMS"
    """
    if not name:
        return name
    prev = None
    out = name
    # Strip up to 3 trailing suffixes (e.g. "X Co Ltd") — bounded loop, no while-True.
    for _ in range(3):
        prev = out
        out = _BIZ_SUFFIX_RE.sub("", out).strip(" ,.")
        if out == prev:
            break
    return out or name

# Noise tokens that indicate the extracted value is not a real supplier name.
_NOISE_LOWER = (
    "bank ", "bank,", "banking", " bank", "trust ", " trust",
    "credit union", "savings", "branch", "sort code", "iban",
    "swift", "bsb", "routing", "invoice", "purchase order",
    "bill to", "remit", "payable", "payment",
)

# Label/header phrases that are never supplier names (extracted from table headers / form labels).
_LABEL_PHRASES_LOWER = frozenset({
    "client information", "order details", "billing information",
    "contact information", "account information", "supplier information",
    "vendor information", "company information", "customer information",
    "ship to", "delivered to", "sold to", "attention", "attn:",
    "order summary", "invoice details", "billing details",
    "item description", "product description", "service description",
})

# Document-reference patterns — strings that look like doc IDs (INV-…, PO-…, etc.)
_DOC_REF_RE = re.compile(
    r'^\s*(INV|PO|REC|ORD|REF|DOC|SER|QUOT?|BILL|RFQ)\s*[-#]?\s*[\d\-A-Z/]{3,}',
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Street keyword detection — used by _is_address_contaminated
# ---------------------------------------------------------------------------
_STREET_KEYWORDS = frozenset({
    "street", "st.", " st ", "road", "rd.", " rd ", "avenue", "ave.", " ave ",
    "drive", "dr.", " dr ", "boulevard", "blvd.", " blvd ", "lane", "ln.",
    " ln ", "way", "court", "ct.", " ct ", "plaza", "place", "pl.",
    "parkway", "pkwy", "highway", "hwy", "terrace", "trail", "close",
    "crescent", "grove", "mews", "row", "square", "walk",
})

# Footer / closing phrases that appear on invoices but are NOT supplier names.
_FOOTER_PHRASES_LOWER = (
    "thank you for your business",
    "thank you for your order",
    "thank you for choosing",
    "thanks for your business",
    "please remit",
    "please make payment",
    "make payable to",
    "make checks payable",
    "payment due",
    "please pay",
    "for inquiries",
    "for questions",
)

# City-state patterns (word, comma, 2-letter state code)
_CITY_STATE_RE = re.compile(
    r"^[A-Za-z\s]{3,30},\s*[A-Z]{2}$"
)

# Date-like strings
_DATE_RE = re.compile(
    r"^\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}$"
    r"|^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2},?\s+\d{4}$",
    re.IGNORECASE,
)

# Money-like string
_MONEY_RE = re.compile(r"^[\$£€¥₹]?\s*[\d,]+(\.\d{1,2})?$")

# Email address
_EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")

# Single attention/generic words that are never supplier names
_SINGLE_GARBAGE_WORDS = frozenset({
    "attn", "attention", "from", "vendor", "supplier", "company",
    "name", "sender", "bill", "invoice", "to", "re", "ref",
})


# ---------------------------------------------------------------------------
# Table-header / form-label lexicon
# ---------------------------------------------------------------------------
# Words that belong to a document's STRUCTURE (column headers, form labels,
# term/date labels) rather than to any company's identity. A supplier name is
# rejected when it is made up ENTIRELY of these — that is what makes the rule
# generalise instead of being a blocklist of observed strings: any permutation
# or subset of a table header row ("Qty Unit Price", "Item No Rate Amount",
# "Delivery Deadline", "Terms and Conditions") collapses to the same signal,
# while a real company name always contributes at least one distinctive word.
#
# Deliberately EXCLUDES corporate-entity words (Ltd, Inc, Group, Solutions,
# Services, Holdings, Company, ...) — those legitimately make up most of a real
# name and are handled by _BIZ_SUFFIX_RE / _CORP_SUFFIX_WORDS.
_LABEL_TOKENS = frozenset({
    # column headers
    "description", "descriptions", "desc", "item", "items", "line", "lines",
    "qty", "quantity", "quantities", "unit", "units", "price", "prices",
    "rate", "rates", "amount", "amounts", "amt", "cost", "costs", "value",
    "total", "totals", "subtotal", "sub", "grand", "net", "gross", "sum",
    "count", "no", "num", "number", "ref", "reference", "sku", "code",
    "currency", "detail", "details", "summary", "particulars", "product",
    "products", "service", "services_desc", "category", "type", "status",
    # tax / money labels
    "tax", "taxes", "vat", "gst", "discount", "discounts", "charge",
    "charges", "fee", "fees", "payment", "payments", "pay", "paid",
    "balance", "due", "outstanding", "advance", "deposit",
    # date / term labels
    "date", "dates", "day", "days", "week", "weeks", "month", "months",
    "year", "years", "period", "start", "end", "deadline", "term", "terms",
    "condition", "conditions", "valid", "validity", "expiry", "issued",
    # logistics labels
    "delivery", "deliver", "delivered", "dispatch", "shipping", "shipment",
    "freight", "carriage", "handling", "packing",
    # misc form furniture
    "notes", "note", "remarks", "remark", "signature", "sign", "signed",
    "page", "pages", "per", "each", "and", "the", "for", "of", "to", "from",
    "incl", "excl", "including", "excluding", "hrs", "hours", "hour",
})

# Entity words after which a bare number is never part of the name — a real
# company does not put a figure AFTER its legal suffix ("Lester Group 807"),
# so the number is a neighbouring table cell that got concatenated on.
_ENTITY_TAIL_WORDS = frozenset({
    "group", "ltd", "limited", "llc", "inc", "corp", "corporation", "co",
    "company", "plc", "llp", "gmbh", "solutions", "services", "holdings",
    "enterprises", "enterprise", "studios", "agency", "partners", "trading",
})

_TOKEN_SPLIT_RE = re.compile(r"[\s/|]+")


def _label_tokens(name: str) -> list[str]:
    """Normalised word tokens, punctuation stripped, empties dropped."""
    out = []
    for raw in _TOKEN_SPLIT_RE.split(name.lower()):
        tok = raw.strip(".,:;()[]{}\"'-&")
        if tok:
            out.append(tok)
    return out


def _is_header_fragment(name: str) -> bool:
    """True when the string is a table-header / form-label fragment.

    Two conditions, both anchored on the label lexicon:
      A. EVERY word is a structural label word  → the whole string is furniture
         ("Description Qty Unit Price", "days Tax", "Delivery Deadline",
          "CONDITIONS", "Terms and Conditions").
      B. THREE OR MORE CONSECUTIVE words are label words → a real name that a
         header row got concatenated onto ("Acme Description Qty Unit Price").
         Three-in-a-row is required so an ordinary name that happens to reuse
         one or two of these words ("Total Fitness", "Express Delivery Ltd",
         "City of Newport") is untouched.
    """
    toks = _label_tokens(name)
    if not toks:
        return False

    # A — entirely structural.
    if all(t in _LABEL_TOKENS for t in toks):
        return True

    # B — a run of >= 3 label words embedded in a longer string.
    run = 0
    for t in toks:
        run = run + 1 if t in _LABEL_TOKENS else 0
        if run >= 3:
            return True
    return False


def _is_table_cell_artifact(name: str) -> bool:
    """True when the string carries the fingerprint of merged table cells.

    C. A bare number trailing an entity suffix ("Lester Group 807") — the
       figure is a neighbouring cell, never part of the legal name.
    D. Two or more very short mixed letter+digit tokens ("... C1 6s") — cell
       codes / OCR row markers. Two are required so genuine names built on one
       such token ("3M", "O2", "B&Q") are unaffected.
    """
    toks = _label_tokens(name)
    if len(toks) >= 2 and toks[-1].isdigit() and toks[-2] in _ENTITY_TAIL_WORDS:
        return True

    short_mixed = sum(
        1 for t in toks
        if len(t) <= 3 and any(c.isdigit() for c in t) and any(c.isalpha() for c in t)
    )
    return short_mixed >= 2


def _is_address_contaminated(name: str) -> bool:
    """Return True if the candidate contains street/address keywords with digits
    or other address-indicating patterns.

    This catches cases like:
    - "Mill St. Main" (street keyword)
    - "Construction Masters 270 Construction Road Drive Dayton, OH 25143" (street + digit)
    - "Hott Street Oklahoma City" (street keyword + known pattern)
    - "Co. Unit 7" (unit number indicator)
    """
    lo = name.lower()
    has_street_kw = any(kw in lo for kw in _STREET_KEYWORDS)
    has_digit = bool(re.search(r'\d', name))

    # Street keyword with digit → definitely an address
    if has_street_kw and has_digit:
        return True

    # Long string with a street keyword is very likely an address line
    if has_street_kw and len(name) > 30:
        return True

    # Street keyword present in a short string but followed by non-company text.
    # e.g. "Mill St. Main" — "St." is followed by a bare word (no company suffix).
    # Accept "Main St." as a potential company name (starts with word, ends with St.)
    # but reject "Mill St. Main" (street keyword in the middle).
    # Corporate suffix words — if a street keyword is followed by one of these,
    # the string is likely a COMPANY NAME not an address (e.g. "Main Street Solutions LLC").
    _CORP_SUFFIX_WORDS = frozenset({
        "inc", "inc.", "ltd", "ltd.", "llc", "corp", "corp.", "gmbh", "sa",
        "limited", "industries", "group", "holdings", "studios", "services",
        "systems", "solutions", "enterprises", "consulting", "technology",
        "technologies", "partners", "associates", "agency", "company", "co.",
        "international", "global", "digital", "media", "design", "creative",
    })

    if has_street_kw:
        # Use word-boundary aware search for each street keyword to avoid
        # matching "st" inside "street" or "solution". Build a regex for
        # the full keyword (as a word) then check what follows it.
        for kw in _STREET_KEYWORDS:
            kw_core = kw.strip()
            if not kw_core:
                continue
            # Build a word-boundary-aware pattern. For keywords that end with
            # a non-word char (like "st." "rd."), only anchor the START with \b
            # since \b before a trailing dot/punctuation doesn't work.
            if kw_core[-1].isalnum():
                kw_pattern = re.compile(
                    r'\b' + re.escape(kw_core) + r'\b', re.IGNORECASE
                )
            else:
                kw_pattern = re.compile(
                    r'\b' + re.escape(kw_core), re.IGNORECASE
                )
            m = kw_pattern.search(lo)
            if not m:
                continue
            after = lo[m.end():].strip()
            if after and not after.startswith(('.', ',')):
                # Check if the text after the street keyword starts with a corporate
                # suffix word (e.g. "Main Street Solutions LLC" → keep as valid company)
                words_after = after.split()
                first_word_after = words_after[0].rstrip('.,') if words_after else ''
                if first_word_after in _CORP_SUFFIX_WORDS:
                    continue  # legitimate company name — not an address

                # Short string where the street keyword appears in the MIDDLE (not last).
                # Heuristic: abbreviated street types (St., Rd., Ave., Dr., Blvd., Ln., Ct., Pl.)
                # in the middle of a 3-word string usually indicate an address fragment
                # (e.g. "Mill St. Main", "123 Rd. Fork"). BUT non-abbreviated forms
                # (Lane, Drive, Road, Avenue) commonly appear in company names
                # (e.g. "Lane Bryant", "Park Avenue Group"). Accept ambiguous non-abbreviated
                # short strings without digits as company names.
                _ABBREV_STREET_KWS = frozenset({
                    'st.', 'rd.', 'ave.', 'dr.', 'blvd.', 'ln.', 'ct.', 'pl.',
                    'pkwy', 'hwy',
                })
                total_words = len(lo.split())
                if total_words <= 3 and not has_digit:
                    if kw_core.lower() not in _ABBREV_STREET_KWS:
                        continue  # non-abbreviated street word in short string — keep as company name
                    # Abbreviated street type in middle of 3-word string → address fragment

                # More text after the street keyword with no corporate suffix → address
                return True

    # Unit number indicator: "Unit N", "Suite N", "Apt N", "Bldg N" with digit
    _UNIT_RE = re.compile(r'\b(unit|suite|apt|bldg|floor|fl\.?|building|rm\.?|room)\s*[\d#]+', re.IGNORECASE)
    if _UNIT_RE.search(name):
        return True

    return False


def _is_garbage_name(name: str) -> bool:
    """Return True if `name` is obviously not a real supplier name."""
    return _garbage_reason(name) is not None


def _garbage_reason(name: str) -> str | None:
    """Return a short reason code if `name` is not a real supplier name, else None.

    Rejects:
    - Too short (< 3 chars)
    - All digits / punctuation
    - Known noise tokens (bank, routing, etc.)
    - Email addresses
    - Address-contaminated strings (street keyword + digit)
    - Footer / closing phrases
    - City+state patterns (e.g. "Oklahoma City, OK")
    - Pure date strings
    - Pure money strings
    - Single generic/attention words
    """
    stripped = name.strip()
    lo = stripped.lower()

    # Length guard
    if len(lo) < _MIN_NAME_LEN:
        return "too_short"

    # All digits / punctuation
    if re.match(r'^[\d\W_]+$', lo):
        return "no_letters"

    # Email address
    if _EMAIL_RE.search(stripped):
        return "email"

    # Noise markers (legacy list)
    if any(m in lo for m in _NOISE_LOWER):
        return "noise_token"

    # Address contamination (street keyword with number OR long address)
    if _is_address_contaminated(stripped):
        return "address"

    # Footer / closing phrases
    if any(lo.startswith(fp) or fp in lo for fp in _FOOTER_PHRASES_LOWER):
        return "footer_phrase"

    # City+state pattern ("Oklahoma City, OK")
    if _CITY_STATE_RE.match(stripped):
        return "city_state"

    # Date-like string
    if _DATE_RE.match(stripped):
        return "date"

    # Money-like string
    if _MONEY_RE.match(stripped):
        return "money"

    # Label / header phrases (e.g. "Client Information", "Order Details")
    if lo in _LABEL_PHRASES_LOWER:
        return "label_phrase"

    # Document-reference strings (e.g. "INV-B-23476 PO", "PO-12345")
    if _DOC_REF_RE.match(stripped):
        return "doc_reference"

    # Single word that is a generic attention/label word
    words = stripped.split()
    if len(words) == 1 and lo in _SINGLE_GARBAGE_WORDS:
        return "generic_word"

    # Very short single word (≤3 chars) with no corporate suffix
    _CORP_SUFFIXES = ("inc", "ltd", "llc", "corp", "gmbh", "sa", "co.", "co,",
                      "limited", "industries", "group", "holdings", "studios",
                      "services", "systems", "solutions", "enterprises")
    if len(lo) <= 3 and not any(lo.endswith(s) for s in _CORP_SUFFIXES):
        return "too_short_no_suffix"

    # Table-header / form-label fragment ("Description Qty Unit Price")
    if _is_header_fragment(stripped):
        return "header_fragment"

    # Merged-table-cell artefact ("Lester Group 807", "... C1 6s")
    if _is_table_cell_artifact(stripped):
        return "table_cell_artifact"

    return None
def _record_rejection(conn, name: str, reason: str, *, doc_type=None, doc_pk=None,
                      trace_id=None) -> None:
    """Log a rejected supplier-name candidate. Best-effort, never raises.

    Nothing is silently dropped: the literal extracted value is preserved so a
    false positive is visible and can be overturned by a human. Rolls back to a
    savepoint on failure so a missing table cannot poison the caller's txn.
    """
    if conn is None:
        return
    try:
        with conn.cursor() as cur:
            cur.execute("SAVEPOINT sp_supplier_reject")
            try:
                cur.execute(
                    "INSERT INTO proc.bp_supplier_name_reject "
                    "(extracted_name, reason, doc_type, doc_pk, trace_id) "
                    "VALUES (%s,%s,%s,%s,%s) "
                    "ON CONFLICT (LOWER(extracted_name)) DO UPDATE SET "
                    "  seen_count = proc.bp_supplier_name_reject.seen_count + 1, "
                    "  last_seen = now()",
                    (name, reason, doc_type, doc_pk, trace_id),
                )
                cur.execute("RELEASE SAVEPOINT sp_supplier_reject")
            except Exception:
                cur.execute("ROLLBACK TO SAVEPOINT sp_supplier_reject")
                raise
    except Exception:  # noqa: BLE001 — audit log must never break extraction
        log.debug("supplier_resolver: reject-log write failed for %r", name, exc_info=True)


def _slug(name: str) -> str:
    """Derive a slug from a supplier name for use as SUP-<slug>."""
    # Title-case, remove non-alphanumeric, truncate to 40 chars.
    title = re.sub(r'\s+', '', name.title())
    slug = re.sub(r'[^A-Za-z0-9]', '', title)[:40] or "Unknown"
    return slug


def resolve_or_create_supplier(name: str, conn, *, doc_type: str | None = None,
                               doc_pk: str | None = None, trace_id: str | None = None) -> str | None:
    """Resolve supplier `name` to a canonical supplier_id.

    Steps:
    1. Validate name is not garbage.
    2. If name already starts with "SUP-", check if it exists in bp_supplier
       and return it as-is (prevents re-slugging canonical IDs).
    3. Exact match (case-insensitive) on supplier_name / trading_name.
    4. Fuzzy match (rapidfuzz WRatio >= 85) across all suppliers.
    5. Auto-create new row: INSERT supplier_id = "SUP-<slug>".

    Args:
        name: Raw supplier name string (e.g. "MGM Souvenir Shop").
        conn: Live psycopg2 connection (caller owns it; we do NOT close it).

    Returns:
        supplier_id string (e.g. "SUP-MGMSouvenirShop") or None on failure.
    """
    if not name or not isinstance(name, str):
        return None

    name = name.strip()
    if not name:
        return None
    reason = _garbage_reason(name)
    if reason:
        log.debug("supplier_resolver: rejected %r (%s)", name, reason)
        _record_rejection(conn, name, reason, doc_type=doc_type, doc_pk=doc_pk,
                          trace_id=trace_id)
        return None

    # ML classifier gate — drops contaminated strings that pass rule-based filter
    if not _classifier_accepts(name):
        log.debug("supplier_resolver: classifier rejected name: %r", name)
        _record_rejection(conn, name, "classifier", doc_type=doc_type, doc_pk=doc_pk,
                          trace_id=trace_id)
        return None

    # --- Already canonical? ---
    if name.startswith("SUP-") and len(name) >= 6:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM proc.bp_supplier WHERE supplier_id = %s LIMIT 1",
                    (name,),
                )
                if cur.fetchone():
                    log.debug("supplier_resolver: canonical SUP- exists: %s", name)
                    return name
        except Exception:
            log.debug("supplier_resolver: canonical check failed", exc_info=True)
        # Already a SUP- form but NOT in table — fall through to create it.
        # Treat the part after "SUP-" as the display name.
        display_name = name[4:]
    else:
        display_name = name

    try:
        with conn.cursor() as cur:
            # --- 0. Human-confirmed alias (deterministic; never re-flagged) ---
            alias_id = _lookup_alias(display_name, cur)
            if alias_id:
                log.debug("supplier_resolver: alias '%s' → %s", display_name, alias_id)
                return alias_id

            # --- 1. Exact match ---
            cur.execute(
                """
                SELECT supplier_id FROM proc.bp_supplier
                WHERE LOWER(supplier_name) = LOWER(%s)
                   OR LOWER(trading_name) = LOWER(%s)
                LIMIT 1
                """,
                (display_name, display_name),
            )
            row = cur.fetchone()
            if row:
                log.info("supplier_resolver: exact match '%s' → %s", display_name, row[0])
                return row[0]

            # --- 2. Fuzzy: find the single best candidate ---
            best_id: str | None = None
            best_name: str | None = None
            best_score = 0.0
            try:
                from rapidfuzz import fuzz

                cur.execute("SELECT supplier_id, supplier_name FROM proc.bp_supplier")
                q_stem = _strip_biz_suffix(display_name) or display_name
                for sid, sname in cur.fetchall():
                    if not sname:
                        continue
                    c_stem = _strip_biz_suffix(sname) or sname
                    score = fuzz.WRatio(q_stem, c_stem)
                    if score > best_score:
                        best_score, best_id, best_name = score, sid, sname
            except ImportError:
                log.warning("supplier_resolver: rapidfuzz not available; skipping fuzzy match")

            # --- 3. Decide (unchanged): link at/above threshold, else create ---
            if best_score >= _FUZZY_THRESHOLD and best_id:
                chosen, decision = best_id, "linked"
                log.info("supplier_resolver: fuzzy match '%s' → %s (score=%.1f)",
                         display_name, best_id, best_score)
            else:
                chosen, decision = _create_supplier(cur, display_name), "created"

            # --- 4. Flag close calls for human review (best-effort) ---
            if chosen and best_id and _REVIEW_LOW() <= best_score < _REVIEW_HIGH():
                _emit_review(
                    cur, extracted_name=display_name, decision=decision,
                    chosen_id=chosen, candidate_id=best_id, candidate_name=best_name,
                    score=float(best_score), doc_type=doc_type, doc_pk=doc_pk,
                    trace_id=trace_id,
                )
            return chosen

    except Exception:
        log.exception("supplier_resolver: unexpected error resolving '%s'", display_name)
        return None


# ---------------------------------------------------------------------------
# Human review resolution (confirm / reject) — drives bp_supplier_alias so the
# decision is durable and future documents of the variant resolve consistently.
# The document's literal extracted value is never altered.
# ---------------------------------------------------------------------------
def _add_alias(cur, alias_name: str, supplier_id: str, actor: str) -> None:
    cur.execute(
        "SELECT alias_id FROM proc.bp_supplier_alias WHERE LOWER(alias_name) = LOWER(%s)",
        (alias_name,),
    )
    row = cur.fetchone()
    if row:
        cur.execute(
            "UPDATE proc.bp_supplier_alias SET supplier_id = %s, created_by = %s WHERE alias_id = %s",
            (supplier_id, actor, row[0]),
        )
    else:
        cur.execute(
            "INSERT INTO proc.bp_supplier_alias (alias_name, supplier_id, created_by) VALUES (%s, %s, %s)",
            (alias_name, supplier_id, actor),
        )


def confirm_review(review_id: int, reviewer: str, conn) -> dict:
    """Human says the extracted name IS the candidate supplier: alias it to the
    canonical candidate so every future document of that variant resolves there."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT extracted_name, candidate_supplier_id, status "
            "FROM proc.bp_supplier_review WHERE review_id = %s FOR UPDATE",
            (review_id,),
        )
        row = cur.fetchone()
        if not row:
            raise ValueError(f"review {review_id} not found")
        name, candidate, status = row
        if status != "pending":
            raise ValueError(f"review {review_id} is '{status}', not pending")
        _add_alias(cur, name, candidate, reviewer)
        cur.execute(
            "UPDATE proc.bp_supplier_review SET status = 'confirmed', reviewed_by = %s, "
            "reviewed_date = now() WHERE review_id = %s",
            (reviewer, review_id),
        )
    conn.commit()
    return {"review_id": review_id, "status": "confirmed", "alias": name, "supplier_id": candidate}


def reject_review(review_id: int, reviewer: str, conn) -> dict:
    """Human says the extracted name is a DISTINCT supplier from the candidate:
    ensure it has its own supplier_id and alias it there (so it isn't re-flagged
    or re-merged). Already-persisted rows are not repointed (v1)."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT extracted_name, decision, chosen_supplier_id, candidate_supplier_id, status "
            "FROM proc.bp_supplier_review WHERE review_id = %s FOR UPDATE",
            (review_id,),
        )
        row = cur.fetchone()
        if not row:
            raise ValueError(f"review {review_id} not found")
        name, decision, chosen, candidate, status = row
        if status != "pending":
            raise ValueError(f"review {review_id} is '{status}', not pending")
        # If it was auto-linked into the candidate, it needs its own supplier now.
        distinct_id = _create_supplier(cur, name) if decision == "linked" else chosen
        _add_alias(cur, name, distinct_id, reviewer)
        cur.execute(
            "UPDATE proc.bp_supplier_review SET status = 'rejected', reviewed_by = %s, "
            "reviewed_date = now() WHERE review_id = %s",
            (reviewer, review_id),
        )
    conn.commit()
    return {"review_id": review_id, "status": "rejected", "alias": name, "supplier_id": distinct_id}


def sweep_supplier_duplicates(conn, min_score: float | None = None) -> dict:
    """Pairwise fuzzy-scan of all bp_supplier rows; flag likely-duplicate pairs
    into the review queue for human confirm/reject.

    Idempotent: skips pairs already aliased or already flagged/decided (either
    direction). Canonical = the older row (by created_date, then supplier_id);
    the other becomes the review's extracted_name (the one that would be merged
    on confirm). O(n²) — fine for the current supplier count.
    """
    import datetime as _dt
    if min_score is None:
        min_score = _governed_limit("supplier_identity", "sweep_min_score",
                                    env="SUPPLIER_SWEEP_MIN_SCORE")
    try:
        from rapidfuzz import fuzz
    except ImportError:
        return {"flagged": 0, "error": "rapidfuzz unavailable"}

    flagged = 0
    compared = 0
    with conn.cursor() as cur:
        cur.execute(
            "SELECT supplier_id, supplier_name, created_date FROM proc.bp_supplier "
            "WHERE supplier_name IS NOT NULL AND length(trim(supplier_name)) >= %s",
            (_MIN_NAME_LEN,),
        )
        items = [(sid, sname, _strip_biz_suffix(sname) or sname, cdate)
                 for sid, sname, cdate in cur.fetchall()]
        n = len(items)
        for i in range(n):
            ai, aname, astem, adate = items[i]
            akey = (adate or _dt.datetime.min, ai)
            for j in range(i + 1, n):
                bi, bname, bstem, bdate = items[j]
                compared += 1
                score = fuzz.WRatio(astem, bstem)
                if score < min_score:
                    continue
                bkey = (bdate or _dt.datetime.min, bi)
                # Older = canonical (candidate to keep); newer = extracted (to merge).
                if akey <= bkey:
                    canon_id, canon_name, other_id, other_name = ai, aname, bi, bname
                else:
                    canon_id, canon_name, other_id, other_name = bi, bname, ai, aname
                if _lookup_alias(other_name, cur):
                    continue  # already resolved to something
                rid = _emit_review(
                    cur, extracted_name=other_name, decision="existing_dup",
                    chosen_id=other_id, candidate_id=canon_id, candidate_name=canon_name,
                    score=float(score), doc_type=None, doc_pk=None, trace_id=None,
                )
                if rid:
                    flagged += 1
    conn.commit()
    log.info("supplier_resolver: duplicate sweep flagged %d pairs (min_score=%.0f, compared=%d, suppliers=%d)",
             flagged, min_score, compared, n)
    return {"flagged": flagged, "compared": compared, "min_score": min_score, "suppliers": n}
