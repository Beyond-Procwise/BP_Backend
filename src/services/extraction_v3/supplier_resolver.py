"""Supplier name → supplier_id resolution with auto-create.

Looks up proc.bp_supplier for an existing match using:
  1. Exact case-insensitive match on supplier_name or trading_name.
  2. rapidfuzz WRatio fuzzy match (threshold >= 85) against all supplier names.
  3. If no match: auto-create a new proc.bp_supplier row with a slug-derived
     supplier_id (e.g. ``SUP-MGMSouvenirShop``) and return it.

Never raises — returns None on any unrecoverable error so the caller can
persist supplier_id = NULL and put the row into the review queue.
"""
from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

# Minimum name length to attempt resolution / creation.
_MIN_NAME_LEN = 3
# rapidfuzz threshold (0-100). 85 is tight enough to avoid false merges.
_FUZZY_THRESHOLD = 85

# Noise tokens that indicate the extracted value is not a real supplier name.
_NOISE_LOWER = (
    "bank ", "bank,", "banking", " bank", "trust ", " trust",
    "credit union", "savings", "branch", "sort code", "iban",
    "swift", "bsb", "routing", "invoice", "purchase order",
    "bill to", "remit", "payable", "payment",
)


def _is_garbage_name(name: str) -> bool:
    """Return True if `name` is obviously not a real supplier name."""
    lo = name.lower().strip()
    if len(lo) < _MIN_NAME_LEN:
        return True
    # All digits / punctuation
    if re.match(r'^[\d\W_]+$', lo):
        return True
    # Noise markers
    if any(m in lo for m in _NOISE_LOWER):
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
                    # Build list of (supplier_id, score)
                    best_id: str | None = None
                    best_score = 0.0
                    for sid, sname in choices.items():
                        score = fuzz.WRatio(display_name, sname)
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
