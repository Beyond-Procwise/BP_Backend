"""Deterministic address parser.

Splits a multi-line address block into structured fields. The strategy
anchors on the postcode (the most reliable signal) and back-walks to
identify city, then segments the remaining text into address lines.

This is the highest-leverage parser in the V2 pipeline — it covers the
single most-common gap class observed in production. Logic is purely
deterministic; no ML or LLM involvement.

Contract: ``parse_address(raw)`` always returns a :class:`ParsedAddress`,
possibly with None fields when the input is unparseable. Never raises.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from src.services.extraction_v2.types import InvalidValue, Postcode


__all__ = ["ParsedAddress", "parse_address"]


@dataclass(frozen=True)
class ParsedAddress:
    """Structured address block."""
    line1: Optional[str]
    line2: Optional[str]
    city: Optional[str]
    postcode: Optional[str]   # normalized form ("RH13 5QH"); None if unparseable
    country: Optional[str]


# Postcode patterns — searched anywhere in the input
_UK_POSTCODE_RE = re.compile(
    r"\b([A-Z]{1,2}\d[A-Z\d]?)\s*(\d[A-Z]{2})\b",
    re.IGNORECASE,
)
_US_ZIP_RE = re.compile(r"\b(\d{5})(?:-(\d{4}))?\b")

# Country-name aliases — used to detect country mention even before postcode
_COUNTRY_ALIASES = {
    "united kingdom": "United Kingdom",
    "uk": "United Kingdom",
    "great britain": "United Kingdom",
    "england": "United Kingdom",
    "scotland": "United Kingdom",
    "wales": "United Kingdom",
    "united states": "United States",
    "usa": "United States",
    "u.s.a.": "United States",
    "u.s.": "United States",
}

# US two-letter state codes — appear between city and ZIP in US addresses
# ("123 Main St, Springfield, IL 62701"). Used to strip the state code so
# the city is correctly identified as "Springfield" not "IL".
_US_STATES = frozenset({
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN",
    "IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV",
    "NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN",
    "TX","UT","VT","VA","WA","WV","WI","WY","DC",
})


# UK-county / region words that often appear before the postcode and aren't
# part of the city name. Used to strip them off so we don't grab "West Sussex"
# as the city when the real city is "Horsham".
_UK_REGIONS = {
    "west sussex", "east sussex", "west midlands", "east midlands",
    "greater london", "greater manchester", "merseyside",
    "south yorkshire", "west yorkshire", "north yorkshire", "east yorkshire",
    "surrey", "hampshire", "kent", "essex", "yorkshire", "lancashire",
    "cheshire", "norfolk", "suffolk", "devon", "cornwall", "somerset",
    "dorset", "wiltshire", "gloucestershire", "warwickshire",
    "leicestershire", "nottinghamshire", "derbyshire", "staffordshire",
    "berkshire", "oxfordshire", "hertfordshire", "buckinghamshire",
    "bedfordshire", "cambridgeshire", "northamptonshire",
    "lincolnshire", "rutland", "shropshire", "herefordshire",
    "worcestershire", "cumbria", "northumberland", "durham",
    "tyne and wear", "middlesex",
}


def parse_address(raw: Optional[str]) -> ParsedAddress:
    """Parse an address block. Never raises."""
    blank = ParsedAddress(None, None, None, None, None)
    if not raw or not isinstance(raw, str):
        return blank

    text = raw.strip()
    if not text:
        return blank

    # Normalize whitespace within the block but preserve newlines as soft
    # boundaries (they help separate lines).
    text = re.sub(r"[ \t]+", " ", text)
    # Remove trailing punctuation that's just decoration
    text = text.rstrip(" .,;:")

    postcode, country, postcode_match_span = _find_postcode(text)
    city = _find_city(text, postcode_match_span) if postcode else None

    # Build the street portion. The street is everything that ISN'T the
    # postcode, the city, the country alias, or trailing region words.
    if postcode:
        # Handle BOTH layouts:
        #   "Street, City, Postcode, Country"   (UK conventional)
        #   "Street, Postcode, City, Country"   (also seen in production)
        # Strategy: take the text BEFORE the postcode as the street base,
        # then strip the city, region, and country tokens out wherever
        # they appear in the leading or trailing portion.
        leading = text[: postcode_match_span[0]]
        trailing = text[postcode_match_span[1]:]
        street_block = (leading + " " + trailing).strip(" \t\n,.;-")
    else:
        street_block = text

    # Strip city / regions / country tokens out of the street block
    if city:
        # Remove the city token wherever it sits (with surrounding commas)
        street_block = re.sub(
            r"[,\s]*\b" + re.escape(city) + r"\b[,\s]*", " ",
            street_block, flags=re.IGNORECASE,
        )
    for region in _UK_REGIONS:
        street_block = re.sub(
            r"[,\s]*\b" + re.escape(region) + r"\b[,\s]*", " ",
            street_block, flags=re.IGNORECASE,
        )
    # Strip US state codes (only as standalone 2-letter tokens, not parts of words)
    street_block = re.sub(
        r"(?:[,\s])\b(?:" + "|".join(_US_STATES) + r")\b(?=[,\s]|$)", " ",
        street_block,
    )
    for alias in sorted(_COUNTRY_ALIASES, key=len, reverse=True):
        street_block = re.sub(
            r"[,\s]*\b" + re.escape(alias) + r"\b[,\s]*", " ",
            street_block, flags=re.IGNORECASE,
        )
    # Collapse multi-spaces and stray separators
    street_block = re.sub(r"[ \t]+", " ", street_block).strip(" ,;.-\n")

    line1, line2 = _split_lines(street_block)

    if not country:
        country = _detect_country_alias(text)

    return ParsedAddress(
        line1=line1 or None,
        line2=line2 or None,
        city=city,
        postcode=postcode,
        country=country,
    )


# ---------------------------------------------------------------------------

def _find_postcode(text: str) -> tuple[Optional[str], Optional[str], tuple[int, int]]:
    """Return (normalized_postcode, country, (start, end)) of FIRST postcode."""
    m = _UK_POSTCODE_RE.search(text)
    if m:
        try:
            normalized = str(Postcode(m.group(0)))
            return normalized, "United Kingdom", m.span()
        except InvalidValue:
            pass

    m = _US_ZIP_RE.search(text)
    if m:
        try:
            normalized = str(Postcode(m.group(0).replace("-", "")))
            return normalized, "United States", m.span()
        except InvalidValue:
            pass

    return None, None, (0, 0)


def _find_city(text: str, postcode_span: tuple[int, int]) -> Optional[str]:
    """Find the city name relative to the postcode.

    Strategy: try the chunk IMMEDIATELY AFTER the postcode first
    (handles "Street, Postcode, City" layout); fall back to the chunk
    immediately BEFORE (the conventional "Street, City, Postcode").

    Strips known UK regions and street-modifier words so that
    "Horsham West Sussex RH13 5QH" yields "Horsham", not "West Sussex".
    """
    candidate = _city_after_postcode(text, postcode_span) or _city_before_postcode(text, postcode_span)
    if not candidate:
        return None
    return _validate_city(candidate)


def _city_after_postcode(text: str, postcode_span: tuple[int, int]) -> Optional[str]:
    after = text[postcode_span[1]:].lstrip(" \t\n,;:.-")
    if not after:
        return None
    # First chunk before the next comma/newline. Skip if it's a country alias.
    chunk = re.split(r"[,\n]", after, maxsplit=1)[0].strip(" \t\n.-")
    if not chunk:
        return None
    if chunk.lower() in _COUNTRY_ALIASES:
        return None
    return chunk


def _city_before_postcode(text: str, postcode_span: tuple[int, int]) -> Optional[str]:
    before = text[: postcode_span[0]].rstrip(" \t\n,;:.-")
    if not before:
        return None

    # Strip a known UK region appearing immediately before the postcode
    low = before.lower()
    for region in sorted(_UK_REGIONS, key=len, reverse=True):
        if low.endswith(region):
            before = before[: -len(region)].rstrip(" \t\n,;:.-")
            break

    # Strip a US state code (2 letters) immediately before the postcode
    m = re.search(r"[,\s](" + "|".join(_US_STATES) + r")\s*$", before)
    if m:
        before = before[: m.start()].rstrip(" \t\n,;:.-")

    if not before:
        return None

    chunks = [c.strip() for c in re.split(r"[,\n]", before) if c.strip()]
    if not chunks:
        return None
    candidate = chunks[-1]

    # Trim multi-word phrases down to 1-2 trailing capitalized words
    words = candidate.split()
    if not words:
        return None

    if len(words) > 3:
        tail = []
        for w in reversed(words):
            if not w[:1].isupper():
                break
            tail.insert(0, w)
            if len(tail) >= 2:
                break
        candidate = " ".join(tail) if tail else words[-1]

    return candidate


def _validate_city(candidate: str) -> Optional[str]:
    if candidate.lower() in _UK_REGIONS:
        return None
    if candidate.lower() in {"floor", "suite", "unit", "block"}:
        return None
    if re.fullmatch(r"\d+", candidate):
        return None
    if candidate.lower() in _COUNTRY_ALIASES:
        return None
    return candidate


def _split_lines(street_block: str) -> tuple[Optional[str], Optional[str]]:
    """Split the street portion into line1 / line2.

    The caller has already removed city / region / country tokens, so
    what's left should be street/building lines. Prefer newlines as
    boundaries (most reliable); fall back to comma boundaries.
    """
    if not street_block:
        return None, None

    # Split on newlines first
    parts = [p.strip(" ,;.-") for p in street_block.split("\n") if p.strip(" ,;.-")]

    if not parts:
        return None, None

    if len(parts) == 1:
        # Single line — split on comma if it's a multi-clause string
        # (e.g., "Unit 12, Meridian Business Park")
        text = parts[0]
        # Heuristic: if the FIRST comma-separated chunk looks like a
        # complete address-line on its own (≥10 chars), keep it as
        # line1 and put the rest in line2.
        if "," in text:
            chunks = [c.strip() for c in text.split(",")]
            chunks = [c for c in chunks if c]
            if len(chunks) >= 2 and len(chunks[0]) >= 10:
                # First chunk is line1 (likely "Unit 12, Building Name"
                # is one logical line — keep them together if the unit
                # number is short)
                if re.match(r"^(unit|suite|building|floor|block|apt)\s", chunks[0], re.I):
                    # Combine first two chunks ("Unit 12, Meridian Business Park")
                    line1 = chunks[0] + ", " + chunks[1]
                    line2 = ", ".join(chunks[2:]) if len(chunks) > 2 else None
                    return line1, line2
                return chunks[0], ", ".join(chunks[1:]) if len(chunks) > 1 else None
        return text, None

    # Two or more parts → first is line1, rest concatenated as line2
    return parts[0], ", ".join(parts[1:])


def _detect_country_alias(text: str) -> Optional[str]:
    """Detect country mentions even when postcode wasn't found."""
    low = text.lower()
    # Sort longest-first so "United Kingdom" wins over "uk"
    for alias in sorted(_COUNTRY_ALIASES, key=len, reverse=True):
        if re.search(r"\b" + re.escape(alias) + r"\b", low):
            return _COUNTRY_ALIASES[alias]
    return None
