"""Known shipping carrier blocklist for supplier_name candidate filtering.

When extracting supplier_name, shipping carriers that appear in "SHIP VIA:",
"DELIVERED BY:", "CARRIER:", etc. sections must be excluded. This module
provides the blocklist and the matching function.

The blocklist is intentionally a post-extraction filter (not a primary
extractor). It acts on spaCy NER output after entity detection.

Usage:
    from src.services.extraction_v3.extractors._carrier_blocklist import is_carrier

    if is_carrier(candidate_value):
        # drop the candidate
"""
from __future__ import annotations

# Canonical carrier names (mixed-case for readability).
# Matching is case-insensitive substring matching on the normalized candidate.
KNOWN_CARRIERS: frozenset[str] = frozenset({
    "UPS",
    "USPS",
    "FedEx",
    "DHL",
    "GLS",
    "TNT",
    "Aramex",
    "Canada Post",
    "Royal Mail",
    "DPD",
    "Hermes",
    "Yodel",
    "Hermes Logistics",
    "UPS Express",
    "UPS Ground",
    "UPS Next Day Air",
    "FedEx Ground",
    "FedEx Express",
    "FedEx Freight",
    "USPS Priority",
    "USPS First Class",
    "Blue Dart",
    "Purolator",
    "LaPoste",
    "Poste Italiane",
    "Deutsche Post",
    "Australia Post",
    "New Zealand Post",
    "Japan Post",
    "China Post",
    "SF Express",
    "OnTrac",
    "LSO",
    "Lasership",
    "Spee-Dee",
    "Lone Star",
    "Pilot Freight",
    "ABF Freight",
    "Old Dominion",
    "XPO Logistics",
    "XPO",
    "Estes Express",
    "R+L Carriers",
    "Saia",
    "Forward Air",
})

# Lowercase versions for fast O(1) prefix matching
_CARRIER_LOWER: frozenset[str] = frozenset(c.lower() for c in KNOWN_CARRIERS)


def is_carrier(value: str) -> bool:
    """Return True if the value contains a known carrier name (case-insensitive).

    Uses substring matching so "UPS Express Tracking" → True.
    The value is stripped and lowercased before comparison.

    This is a post-extraction filter — never used as a primary extractor.
    """
    if not value:
        return False
    val_lower = value.strip().lower()
    # Exact match first (fast path)
    if val_lower in _CARRIER_LOWER:
        return True
    # Substring check: does the candidate value CONTAIN a carrier name?
    for carrier_lower in _CARRIER_LOWER:
        if carrier_lower in val_lower:
            return True
    return False
