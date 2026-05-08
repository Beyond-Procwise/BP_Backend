"""spaCy NER candidate generator.

For each field whose YAML judge.ner_type_check is non-'none', this extractor
finds spaCy entities of the required type in parsed.full_text and emits each
as a candidate. Acts as both:
  1. a positive evidence source (fields like supplier_name benefit from
     "named-entity-shaped strings in this doc that look like organizations")
  2. an implicit negative signal — if the legacy paths picked up a value
     that's not an ORG, the L3 orchestrator can compare to spaCy's choices.

Improvements over baseline:
  - country field: GPE entities are post-processed through a city/state →
    country lookup table so "Las Vegas" → "United States", "London" → "United
    Kingdom", etc.  Country-shaped GPEs (e.g. "United States") pass through
    unchanged.  Only the country-normalised value is emitted.
  - supplier_name field: ORG entities are ranked by their position in the
    document; entities in the top third of page 0 receive a confidence boost
    (they are more likely the supplier header). PERSON entities at the top of
    the document are excluded from supplier_name candidates (they belong to
    requested_by / contacts).
  - All other fields: behaviour unchanged.

Substring guarantee: spaCy operates on parsed.full_text directly; entity
spans are substrings by construction (ent.text is doc.text[ent.start_char:ent.end_char]).
"""
from __future__ import annotations

import threading

import spacy

from src.services.extraction_v3.extractors.base import Extractor
from src.services.extraction_v3.schemas.candidate import Candidate
from src.services.extraction_v3.schemas.parsed_document import ParsedDocument
from src.services.extraction_v3.yaml_schema.loader import DocSchema
from src.services.extraction_v3.yaml_schema.registry import register_extractor

# Transformer-based model for best accuracy. Falls back to en_core_web_sm if
# the transformer model is not installed (e.g. CPU-only environments).
MODEL_NAME = "en_core_web_trf"

_nlp = None
_lock = threading.Lock()


def _get_nlp():
    global _nlp
    if _nlp is not None:
        return _nlp
    with _lock:
        if _nlp is None:
            try:
                _nlp = spacy.load(MODEL_NAME)
            except OSError:
                # Transformer model not installed — fall back to small CPU model.
                # In production, install en_core_web_trf for best accuracy.
                _nlp = spacy.load("en_core_web_sm")
    return _nlp


# ---------------------------------------------------------------------------
# City / region → country lookup table
# Covers the most common US cities/states/regions that spaCy classifies as GPE.
# Expand as needed based on actual document geography.
# ---------------------------------------------------------------------------
_CITY_REGION_TO_COUNTRY: dict[str, str] = {
    # US cities
    "las vegas": "United States",
    "new york": "United States",
    "los angeles": "United States",
    "chicago": "United States",
    "houston": "United States",
    "phoenix": "United States",
    "philadelphia": "United States",
    "san antonio": "United States",
    "san diego": "United States",
    "dallas": "United States",
    "san jose": "United States",
    "austin": "United States",
    "jacksonville": "United States",
    "fort worth": "United States",
    "columbus": "United States",
    "charlotte": "United States",
    "san francisco": "United States",
    "indianapolis": "United States",
    "seattle": "United States",
    "denver": "United States",
    "washington": "United States",
    "nashville": "United States",
    "oklahoma city": "United States",
    "el paso": "United States",
    "boston": "United States",
    "portland": "United States",
    "atlanta": "United States",
    "miami": "United States",
    "minneapolis": "United States",
    "tulsa": "United States",
    "raleigh": "United States",
    "omaha": "United States",
    "cleveland": "United States",
    "colorado springs": "United States",
    "virginia beach": "United States",
    "long beach": "United States",
    "tampa": "United States",
    "new orleans": "United States",
    "honolulu": "United States",
    "anaheim": "United States",
    "aurora": "United States",
    "santa ana": "United States",
    "corpus christi": "United States",
    "riverside": "United States",
    "lexington": "United States",
    "st. louis": "United States",
    "pittsburgh": "United States",
    "anchorage": "United States",
    "stockton": "United States",
    "cincinnati": "United States",
    "st. paul": "United States",
    "toledo": "United States",
    "greensboro": "United States",
    "newark": "United States",
    "plano": "United States",
    "henderson": "United States",
    "lincoln": "United States",
    "buffalo": "United States",
    "fort wayne": "United States",
    "jersey city": "United States",
    "chula vista": "United States",
    "orlando": "United States",
    "st. petersburg": "United States",
    "norfolk": "United States",
    "chandler": "United States",
    "laredo": "United States",
    "madison": "United States",
    "durham": "United States",
    "lubbock": "United States",
    "winston-salem": "United States",
    "garland": "United States",
    "glendale": "United States",
    "hialeah": "United States",
    "reno": "United States",
    "baton rouge": "United States",
    "irvine": "United States",
    "chesapeake": "United States",
    "scottsdale": "United States",
    "north las vegas": "United States",
    "fremont": "United States",
    "gilbert": "United States",
    "san bernardino": "United States",
    "birmingham": "United States",
    "rochester": "United States",
    "spokane": "United States",
    "des moines": "United States",
    "montgomery": "United States",
    "modesto": "United States",
    "fayetteville": "United States",
    "tacoma": "United States",
    "shreveport": "United States",
    "fontana": "United States",
    "moreno valley": "United States",
    "glendale": "United States",
    "akron": "United States",
    "yonkers": "United States",
    "augusta": "United States",
    "little rock": "United States",
    "columbus": "United States",
    "grand rapids": "United States",
    "ontario": "United States",  # city in CA
    "salt lake city": "United States",
    "huntington beach": "United States",
    "worcester": "United States",
    "knoxville": "United States",
    "providence": "United States",
    "tallahassee": "United States",
    "oxnard": "United States",
    "newport news": "United States",
    "huntsville": "United States",
    "aberdeen": "United States",
    # US states
    "alabama": "United States",
    "alaska": "United States",
    "arizona": "United States",
    "arkansas": "United States",
    "california": "United States",
    "colorado": "United States",
    "connecticut": "United States",
    "delaware": "United States",
    "florida": "United States",
    "georgia": "United States",
    "hawaii": "United States",
    "idaho": "United States",
    "illinois": "United States",
    "indiana": "United States",
    "iowa": "United States",
    "kansas": "United States",
    "kentucky": "United States",
    "louisiana": "United States",
    "maine": "United States",
    "maryland": "United States",
    "massachusetts": "United States",
    "michigan": "United States",
    "minnesota": "United States",
    "mississippi": "United States",
    "missouri": "United States",
    "montana": "United States",
    "nebraska": "United States",
    "nevada": "United States",
    "new hampshire": "United States",
    "new jersey": "United States",
    "new mexico": "United States",
    "new york": "United States",
    "north carolina": "United States",
    "north dakota": "United States",
    "ohio": "United States",
    "oklahoma": "United States",
    "oregon": "United States",
    "pennsylvania": "United States",
    "rhode island": "United States",
    "south carolina": "United States",
    "south dakota": "United States",
    "tennessee": "United States",
    "texas": "United States",
    "utah": "United States",
    "vermont": "United States",
    "virginia": "United States",
    "washington": "United States",
    "west virginia": "United States",
    "wisconsin": "United States",
    "wyoming": "United States",
    # UK cities
    "london": "United Kingdom",
    "manchester": "United Kingdom",
    "birmingham": "United Kingdom",
    "leeds": "United Kingdom",
    "glasgow": "United Kingdom",
    "sheffield": "United Kingdom",
    "bradford": "United Kingdom",
    "liverpool": "United Kingdom",
    "edinburgh": "United Kingdom",
    "bristol": "United Kingdom",
    "cardiff": "United Kingdom",
    "belfast": "United Kingdom",
    "leicester": "United Kingdom",
    "coventry": "United Kingdom",
    "nottingham": "United Kingdom",
    # UK / regions
    "england": "United Kingdom",
    "scotland": "United Kingdom",
    "wales": "United Kingdom",
    "northern ireland": "United Kingdom",
    # Australian cities
    "sydney": "Australia",
    "melbourne": "Australia",
    "brisbane": "Australia",
    "perth": "Australia",
    "adelaide": "Australia",
    "gold coast": "Australia",
    "canberra": "Australia",
    # Canadian cities
    "toronto": "Canada",
    "vancouver": "Canada",
    "montreal": "Canada",
    "calgary": "Canada",
    "edmonton": "Canada",
    "ottawa": "Canada",
    "winnipeg": "Canada",
    "quebec city": "Canada",
    # Known country names that spaCy tags as GPE — pass through unchanged
    "united states": "United States",
    "usa": "United States",
    "united kingdom": "United Kingdom",
    "uk": "United Kingdom",
    "australia": "Australia",
    "canada": "Canada",
    "india": "India",
    "china": "China",
    "germany": "Germany",
    "france": "France",
    "japan": "Japan",
    "brazil": "Brazil",
    "mexico": "Mexico",
    "south africa": "South Africa",
    "new zealand": "New Zealand",
    "singapore": "Singapore",
    "hong kong": "Hong Kong",
    "uae": "United Arab Emirates",
    "united arab emirates": "United Arab Emirates",
}


def _normalise_country(raw_gpe: str) -> str | None:
    """Map a spaCy GPE entity to a country name.

    Returns a country string if we can map it, or None if the GPE is not
    in our lookup table and doesn't look like a country on its own.
    """
    key = raw_gpe.strip().lower()
    return _CITY_REGION_TO_COUNTRY.get(key)


@register_extractor("spacy_ner")
class SpacyNERExtractor(Extractor):
    """Emit one Candidate per named entity that matches a field's ner_type_check.

    Only fields that list "spacy_ner" in their extractors list are considered.
    The spaCy model runs over parsed.full_text once and groups entities by label;
    each entity that matches a field's required NER label becomes one candidate.

    Confidence is fixed at 0.7; the L3 confidence-merger upgrades candidates
    that multiple extractors agree on and downgrades those that stand alone.
    """

    def produce_candidates(self, parsed: ParsedDocument, schema: DocSchema) -> list[Candidate]:
        # Collect only fields that opted into spacy_ner
        active = [f for f in schema.fields if "spacy_ner" in f.extractors]
        if not active:
            return []

        # Filter further to fields that actually have a NER type check
        active = [f for f in active if f.judge.ner_type_check != "none"]
        if not active:
            return []

        # Run NER over the full document text once
        nlp = _get_nlp()
        doc = nlp(parsed.full_text)

        # Group entities by spaCy label for O(1) lookup per field
        ents_by_label: dict[str, list] = {}
        for ent in doc.ents:
            ents_by_label.setdefault(ent.label_, []).append(ent)

        # Compute page 0 height for position-based heuristics
        page0_height = parsed.pages[0].height if parsed.pages else 792.0

        candidates: list[Candidate] = []
        for field in active:
            # YAML uses spaCy's literal label names (ORG, PERSON, GPE, LOC, …)
            spacy_label = field.judge.ner_type_check

            field_ents = ents_by_label.get(spacy_label, [])

            if field.name == "country":
                # Special handling: GPE → country normalisation
                seen_countries: set[str] = set()
                for ent in field_ents:
                    ent_text = ent.text.strip()
                    country = _normalise_country(ent_text)
                    if country is None:
                        continue
                    if country in seen_countries:
                        continue
                    seen_countries.add(country)
                    # Substring guarantee: the original ent_text must be in full_text
                    if ent_text not in parsed.full_text:
                        continue
                    bbox = _find_bbox_for_text(parsed, ent_text)
                    page_idx, b = bbox if bbox else (0, (0.0, 0.0, 0.0, 0.0))
                    candidates.append(Candidate(
                        field=field.name,
                        value=country,
                        page=page_idx,
                        bbox=b,
                        evidence_text=ent_text,
                        model="spacy_ner",
                        confidence=0.65,
                    ))
                continue

            if field.name == "supplier_name":
                # Prefer ORG entities; apply a context-based exclusion to avoid
                # picking up the buyer (Bill To) organisation.
                # Build a set of ORG entity texts that appear within 120 chars
                # after "bill to", "ship to", "customer", "client" in full_text.
                full_lower = parsed.full_text.lower()
                buyer_context_orgs: set[str] = set()
                for buyer_kw in ("bill to", "bill to:", "customer:", "client:"):
                    pos = full_lower.find(buyer_kw)
                    if pos >= 0:
                        snippet = parsed.full_text[pos:pos + 250].lower()
                        for ent in field_ents:
                            if ent.text.strip().lower() in snippet:
                                buyer_context_orgs.add(ent.text.strip())

                # Also find ORG entities near "vendor:", "from:", "supplier:", "seller:"
                supplier_context_orgs: set[str] = set()
                for supp_kw in ("vendor:", "vendor", "from:", "supplier:", "seller:", "billed from"):
                    pos = full_lower.find(supp_kw)
                    if pos >= 0:
                        snippet = parsed.full_text[pos:pos + 250].lower()
                        for ent in field_ents:
                            if ent.text.strip().lower() in snippet:
                                supplier_context_orgs.add(ent.text.strip())

                for ent in field_ents:
                    ent_text = ent.text.strip()
                    if not ent_text or ent_text not in parsed.full_text:
                        continue

                    is_buyer = ent_text in buyer_context_orgs
                    is_supplier = ent_text in supplier_context_orgs

                    bbox = _find_bbox_for_text(parsed, ent_text)
                    page_idx, b = bbox if bbox else (0, (0.0, 0.0, 0.0, 0.0))
                    in_header = (page_idx == 0 and b[1] < page0_height * 0.40)

                    # Confidence assignment:
                    # - Explicitly in supplier context (near Vendor:/From:): 0.90
                    # - In page header, not in buyer context: 0.80
                    # - In page header, but also in buyer context: 0.55 (ambiguous)
                    # - Elsewhere, not in buyer context: 0.65
                    # - Elsewhere, in buyer context: 0.45 (last resort — may be
                    #   the only org in doc; don't exclude entirely)
                    if is_supplier:
                        conf = 0.90
                    elif in_header and not is_buyer:
                        conf = 0.80
                    elif in_header and is_buyer:
                        conf = 0.55
                    elif not is_buyer:
                        conf = 0.65
                    else:
                        conf = 0.45  # buyer context but may be only option

                    candidates.append(Candidate(
                        field=field.name,
                        value=ent_text,
                        page=page_idx,
                        bbox=b,
                        evidence_text=ent_text,
                        model="spacy_ner",
                        confidence=conf,
                    ))
                continue

            # Default path for all other fields (requested_by → PERSON, etc.)
            for ent in field_ents:
                ent_text = ent.text.strip()
                if not ent_text:
                    continue
                # Substring guarantee
                if ent_text not in parsed.full_text:
                    continue

                bbox = _find_bbox_for_text(parsed, ent_text)
                page_idx, b = bbox if bbox else (0, (0.0, 0.0, 0.0, 0.0))

                candidates.append(Candidate(
                    field=field.name,
                    value=ent_text,
                    page=page_idx,
                    bbox=b,
                    evidence_text=ent_text,
                    model="spacy_ner",
                    confidence=0.7,
                ))

        return candidates


def _find_bbox_for_text(parsed: ParsedDocument, text: str):
    """Return (page_index, bbox) of the first token that contains `text`.

    Searches case-insensitively so short entities like "Ltd" match even if
    the token was extracted with mixed case from the PDF.
    Returns None when no matching token is found (caller falls back to page 0,
    zero-bbox — the candidate is still valid, just without precise location).
    """
    text_lower = text.lower().strip()
    for page in parsed.pages:
        for tok in page.tokens:
            if text_lower in tok.text.lower():
                return (page.index, tok.bbox)
    return None
