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

# Business-entity suffixes stripped BEFORE the WRatio comparison so the
# distinctive part of the name dominates the score. Without this, every
# "X Ltd" / "Y INC" pair scores ~85 against each other regardless of stem.
_BIZ_SUFFIX_RE = re.compile(
    r"\s*[,\.]?\s*\b(?:LLC|Ltd|Limited|Inc|Incorporated|Pvt|Pvt\.?\s*Ltd|"
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
    """Return True if `name` is obviously not a real supplier name.

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
        return True

    # All digits / punctuation
    if re.match(r'^[\d\W_]+$', lo):
        return True

    # Email address
    if _EMAIL_RE.search(stripped):
        return True

    # Noise markers (legacy list)
    if any(m in lo for m in _NOISE_LOWER):
        return True

    # Address contamination (street keyword with number OR long address)
    if _is_address_contaminated(stripped):
        return True

    # Footer / closing phrases
    if any(lo.startswith(fp) or fp in lo for fp in _FOOTER_PHRASES_LOWER):
        return True

    # City+state pattern ("Oklahoma City, OK")
    if _CITY_STATE_RE.match(stripped):
        return True

    # Date-like string
    if _DATE_RE.match(stripped):
        return True

    # Money-like string
    if _MONEY_RE.match(stripped):
        return True

    # Label / header phrases (e.g. "Client Information", "Order Details")
    if lo in _LABEL_PHRASES_LOWER or any(lo == phrase for phrase in _LABEL_PHRASES_LOWER):
        return True

    # Document-reference strings (e.g. "INV-B-23476 PO", "PO-12345")
    if _DOC_REF_RE.match(stripped):
        return True

    # Single word that is a generic attention/label word
    words = stripped.split()
    if len(words) == 1 and lo in _SINGLE_GARBAGE_WORDS:
        return True

    # Very short single word (≤3 chars) with no corporate suffix
    _CORP_SUFFIXES = ("inc", "ltd", "llc", "corp", "gmbh", "sa", "co.", "co,",
                      "limited", "industries", "group", "holdings", "studios",
                      "services", "systems", "solutions", "enterprises")
    if len(lo) <= 3 and not any(lo.endswith(s) for s in _CORP_SUFFIXES):
        return True

    return False


def _slug(name: str) -> str:
    """Derive a slug from a supplier name for use as SUP-<slug>."""
    # Title-case, remove non-alphanumeric, truncate to 40 chars.
    title = re.sub(r'\s+', '', name.title())
    slug = re.sub(r'[^A-Za-z0-9]', '', title)[:40] or "Unknown"
    return slug


def resolve_or_create_supplier(name: str, conn) -> str | None:
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
    if not name or _is_garbage_name(name):
        log.debug("supplier_resolver: garbage name rejected: %r", name)
        return None

    # ML classifier gate — drops contaminated strings that pass rule-based filter
    if not _classifier_accepts(name):
        log.debug("supplier_resolver: classifier rejected name: %r", name)
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
                log.info(
                    "supplier_resolver: exact match '%s' → %s", display_name, row[0]
                )
                return row[0]

            # --- 2. Fuzzy match ---
            try:
                from rapidfuzz import fuzz, process as rf_process

                cur.execute(
                    "SELECT supplier_id, supplier_name FROM proc.bp_supplier"
                )
                all_suppliers = cur.fetchall()  # list of (id, name)
                if all_suppliers:
                    choices = {row[0]: row[1] for row in all_suppliers if row[1]}
                    # Strip business suffix from query AND each candidate so
                    # we score on the distinctive stem. Falls back to the raw
                    # name if stripping leaves nothing meaningful.
                    q_stem = _strip_biz_suffix(display_name) or display_name
                    best_id: str | None = None
                    best_score = 0.0
                    for sid, sname in choices.items():
                        c_stem = _strip_biz_suffix(sname) or sname
                        score = fuzz.WRatio(q_stem, c_stem)
                        if score > best_score:
                            best_score = score
                            best_id = sid
                    if best_score >= _FUZZY_THRESHOLD and best_id:
                        log.info(
                            "supplier_resolver: fuzzy match '%s' → %s (score=%.1f)",
                            display_name, best_id, best_score,
                        )
                        return best_id
            except ImportError:
                log.warning("supplier_resolver: rapidfuzz not available; skipping fuzzy match")

            # --- 3. Auto-create ---
            slug = _slug(display_name)
            new_id = f"SUP-{slug}"

            # Check collision (rare but possible with very similar names)
            # Note: bp_supplier has no PK constraint so we guard manually.
            cur.execute(
                "SELECT supplier_id FROM proc.bp_supplier WHERE supplier_id = %s LIMIT 1",
                (new_id,),
            )
            if cur.fetchone():
                # ID already exists — return it (another thread beat us to it)
                log.info(
                    "supplier_resolver: collision resolved — reusing %s", new_id
                )
                return new_id

            cur.execute(
                """
                INSERT INTO proc.bp_supplier
                    (supplier_id, supplier_name, trading_name,
                     created_date, created_by)
                VALUES (%s, %s, %s, NOW(), %s)
                """,
                (new_id, display_name, display_name, "ExtractionV3-AutoDiscovery"),
            )
            log.info(
                "supplier_resolver: auto-created supplier '%s' → %s",
                display_name, new_id,
            )
            return new_id

    except Exception:
        log.exception(
            "supplier_resolver: unexpected error resolving '%s'", display_name
        )
        return None
