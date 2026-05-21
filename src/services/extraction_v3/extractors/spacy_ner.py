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
  - supplier_name field: ORG entities are filtered by position and context:
      * Carrier blocklist: known shipping carriers (UPS, FedEx, DHL, etc.)
        are dropped entirely — they are never the invoice issuer.
      * Header-band filter: only ORG entities in the top 30% of page 0 (the
        document masthead) OR explicitly in a "Vendor/From/Supplier/Issuer"
        labeled block are accepted. Entities appearing only below the header
        band and not near a vendor label are dropped rather than emitted at
        low confidence.
      * Carrier-context ORGs (near "SHIP VIA:", "CARRIER:", etc.) are
        dropped before the header-band check.
  - buyer_id field: looks in the "BILL TO" / "Sold To" / "Customer" block
    for ORG entities exclusively. If only PERSON entities are found in that
    block, no candidate is emitted (buyer_id remains NULL/residual — better
    than committing a person name as a buyer identifier).
  - All other fields: behaviour unchanged.

Substring guarantee: spaCy operates on parsed.full_text directly; entity
spans are substrings by construction (ent.text is doc.text[ent.start_char:ent.end_char]).
"""
from __future__ import annotations

import logging
import re
import threading

import spacy

from src.services.extraction_v3.extractors._carrier_blocklist import is_carrier
from src.services.extraction_v3.extractors.base import Extractor
from src.services.extraction_v3.schemas.candidate import Candidate
from src.services.extraction_v3.schemas.parsed_document import ParsedDocument
from src.services.extraction_v3.yaml_schema.loader import DocSchema
from src.services.extraction_v3.yaml_schema.registry import register_extractor

log = logging.getLogger(__name__)

# Transformer-based model for best accuracy. Falls back to en_core_web_sm if
# the transformer model is not installed (e.g. CPU-only environments).
# en_core_web_sm is used deliberately over en_core_web_trf: the transformer
# model misclassifies personal names as ORG in invoice/PO document layouts
# (e.g. "John Smith" → ORG). The small model is more conservative and correct
# for these structured financial documents.
MODEL_NAME = "en_core_web_sm"

_nlp = None
_lock = threading.Lock()
_call_lock = threading.Lock()  # serialize nlp() calls: en_core_web_sm internals not thread-safe


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


def _run_nlp(text: str):
    """Run spaCy NLP pipeline on text with a serialization lock.

    en_core_web_sm uses Cython internals that are not thread-safe when
    called concurrently from the extraction pipeline's ThreadPoolExecutor.
    This wrapper serializes calls to prevent non-deterministic NER results.
    """
    with _call_lock:
        return _get_nlp()(text)


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

    Handles trailing non-geographic noise like "New York Phone" by progressively
    stripping trailing words until a match is found.
    """
    s = raw_gpe.strip()
    words = s.split()
    # Try progressively shorter prefixes (strip trailing words one at a time)
    # so "New York Phone" → tries "New York Phone" then "New York" → matches.
    for i in range(len(words), 0, -1):
        candidate_key = " ".join(words[:i]).lower()
        result = _CITY_REGION_TO_COUNTRY.get(candidate_key)
        if result is not None:
            return result
    return None


# Suffixes that follow a company name but are NOT part of the name itself.
# spaCy often bundles "Attn", "Attn:", "Attention:" into the ORG span because
# they appear immediately after the company name with no punctuation boundary.
# The pattern matches:
#  - " Attn" at end-of-string  (e.g. "TechTonic Attn")
#  - " Attn: ..." with trailing content  (e.g. "TechTonic Attn: Mr. John Doe")
_ATTN_SUFFIX_RE = re.compile(
    r"\s+(?:Attn\.?|Attention|Att\.?|c/o|Care\s+of|Contact|RE:?)(?:[\s:].*)?\s*$",
    re.IGNORECASE | re.DOTALL,
)


def _clean_org_name(raw_org: str) -> str:
    """Strip attention/contact suffixes from an ORG entity name.

    "TechTonic Attn" → "TechTonic"
    "TechTonic Attn: Mr. John Doe" → "TechTonic"
    "ACME Ltd Attention: Finance Dept" → "ACME Ltd"
    Strips trailing punctuation after cleaning.
    """
    cleaned = _ATTN_SUFFIX_RE.sub("", raw_org).strip().rstrip(",;:-").strip()
    return cleaned if cleaned else raw_org


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

        # Run NER over the full document text once.
        # _run_nlp uses a serialization lock to prevent non-deterministic
        # results when called from concurrent threads (ThreadPoolExecutor).
        doc = _run_nlp(parsed.full_text)

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

            if spacy_label == "GPE":
                # Special handling for all GPE fields: normalise raw GPE entity
                # to a country name via the city/state/region lookup table.
                # Applies to both invoice.country and purchase_order.ship_to_country.
                seen_countries: set[str] = set()
                for ent in field_ents:
                    ent_text = ent.text.strip()
                    country = _normalise_country(ent_text)
                    if country is None:
                        continue
                    if country in seen_countries:
                        continue
                    seen_countries.add(country)
                    # Substring guarantee: the original ent_text must be in full_text.
                    # Note: value=country may NOT be a substring; evidence_text=ent_text is.
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
                # --- supplier_name: position-aware + carrier-filtered ORG extraction ---
                #
                # Strategy (in order of precedence):
                #   1. Known carrier blocklist: DROP immediately — carriers are never issuers.
                #   2. Carrier-context ORGs (near "SHIP VIA:", "CARRIER:"): DROP.
                #   3. Buyer-context ORGs (near "BILL TO:", "CUSTOMER:"): LOW or DROP.
                #   4. Explicitly vendor-context ORGs (near "FROM:", "VENDOR:", etc.): HIGH.
                #   5. Header-band ORGs (top 30% of page 0): MEDIUM-HIGH.
                #   6. Non-header, non-vendor ORGs: DROP (better NULL than wrong).
                #
                # The 30% threshold covers the typical supplier masthead on page 1
                # while excluding the BILL TO, SHIP TO, and line-item sections that
                # occupy the middle and lower portions of the page.
                full_lower = parsed.full_text.lower()

                # 2a. Carrier-context: ORGs appearing near shipping/carrier keywords
                #     are blocklisted regardless of position.
                carrier_context_orgs: set[str] = set()
                for carrier_kw in (
                    "ship via", "ship via:", "shipped via", "shipped by",
                    "carrier:", "carrier", "via:", "delivered by", "delivery by",
                    "shipping method", "shipping carrier",
                ):
                    pos = full_lower.find(carrier_kw)
                    if pos >= 0:
                        snippet = parsed.full_text[pos:pos + 200].lower()
                        for ent in field_ents:
                            if ent.text.strip().lower() in snippet:
                                carrier_context_orgs.add(ent.text.strip())

                # 2b. Buyer-context: ORGs in "BILL TO:", "CUSTOMER:", etc.
                buyer_context_orgs: set[str] = set()
                for buyer_kw in (
                    "bill to", "bill to:", "customer:", "client:",
                    "deliver to", "deliver to:", "delivered to:",
                    "ship to", "ship to:", "delivery address",
                ):
                    pos = full_lower.find(buyer_kw)
                    if pos >= 0:
                        snippet = parsed.full_text[pos:pos + 250].lower()
                        for ent in field_ents:
                            if ent.text.strip().lower() in snippet:
                                buyer_context_orgs.add(ent.text.strip())

                # 2c. Vendor-context: ORGs near explicit "FROM:", "VENDOR:", etc.
                supplier_context_orgs: set[str] = set()
                for supp_kw in (
                    "vendor:", "vendor", "from:", "supplier:", "seller:",
                    "billed from", "issued by", "issued from", "issuer:",
                    "invoice from", "bill from",
                ):
                    pos = full_lower.find(supp_kw)
                    if pos >= 0:
                        snippet = parsed.full_text[pos:pos + 250].lower()
                        for ent in field_ents:
                            if ent.text.strip().lower() in snippet:
                                supplier_context_orgs.add(ent.text.strip())

                # 2d. Table-context: ORGs inside markdown table rows (product
                #     descriptions / line items) — exclude these.
                table_context_orgs: set[str] = set()
                for ent in field_ents:
                    et = ent.text.strip()
                    if not et:
                        continue
                    for line in parsed.full_text.splitlines():
                        if "|" in line and et in line:
                            table_context_orgs.add(et)
                            break

                for ent in field_ents:
                    ent_text = ent.text.strip()
                    if not ent_text or ent_text not in parsed.full_text:
                        continue

                    # --- Carrier blocklist: drop known shipping carriers ---
                    # Checked before position logic; carriers are never issuers.
                    if is_carrier(ent_text):
                        continue

                    # --- Carrier-context: drop ORGs found near shipping keywords ---
                    if ent_text in carrier_context_orgs:
                        continue

                    # --- Table-context: skip product/line-item strings ---
                    if ent_text in table_context_orgs:
                        continue

                    # Strip "Attn:", "Attention:" suffixes (e.g. "TechTonic Attn" → "TechTonic").
                    cleaned_name = _clean_org_name(ent_text)
                    if not cleaned_name:
                        continue

                    # --- Carrier blocklist on cleaned name too ---
                    if is_carrier(cleaned_name):
                        continue

                    is_buyer = ent_text in buyer_context_orgs
                    is_supplier = ent_text in supplier_context_orgs

                    bbox = _find_bbox_for_text(parsed, ent_text)
                    bbox_known = bbox is not None
                    page_idx, b = bbox if bbox else (0, (0.0, 0.0, 0.0, 0.0))
                    # Header band: top 30% of page 0 (down from 40%; tighter = more precise)
                    in_header = (bbox_known and page_idx == 0 and b[1] < page0_height * 0.30)

                    # --- Strict position gate ---
                    # Only emit if: explicitly in vendor context OR in page header.
                    # Non-header, non-vendor ORGs are DROPPED (prefer NULL over wrong).
                    # Exception: if the only ORG in the doc is in buyer context but not
                    # in the header band, we still suppress it (buyer ≠ supplier).
                    if not is_supplier and not in_header:
                        # This ORG is neither near a vendor label nor in the masthead.
                        # Dropping it is safer than emitting it at low confidence.
                        continue

                    # --- Confidence assignment ---
                    # Vendor-context (near FROM:/VENDOR:): highest — explicit label
                    # Header-band only, not buyer context: strong positional evidence
                    # Header-band but also in buyer context: ambiguous — moderate
                    # Vendor-context overrides buyer context (explicit label wins)
                    if is_supplier:
                        conf = 0.90
                    elif in_header and not is_buyer:
                        conf = 0.80
                    elif in_header and is_buyer:
                        conf = 0.55
                    else:
                        # is_supplier is True (guaranteed by gate above), so this
                        # branch only fires if the position gate logic changes.
                        conf = 0.70

                    candidates.append(Candidate(
                        field=field.name,
                        value=cleaned_name,
                        page=page_idx,
                        bbox=b,
                        evidence_text=ent_text,
                        model="spacy_ner",
                        confidence=conf,
                    ))

                # Fallback: if spaCy found no ORG entities, try a company-suffix
                # regex scan.  This catches invented names like "UrbEdge Facilities
                # Management Ltd" that spaCy doesn't recognise as ORG.
                if not field_ents:
                    _emit_company_suffix_candidates(
                        parsed, field, candidates, buyer_context_orgs, page0_height
                    )

                continue

            if field.name == "buyer_id":
                # --- buyer_id: prefer ORG over PERSON in BILL TO block ---
                #
                # The BILL TO block structure is typically:
                #   BILL TO:
                #   <Company Name>   ← ORG → buyer_id
                #   <Contact Name>   ← PERSON → NOT buyer_id
                #   <Address>
                #   [optional shipping section starts after the address]
                #
                # Strategy:
                #   1. Locate the BILL TO / Sold To / Customer / Buyer block.
                #   2. Clip the window to the first shipping section marker
                #      (SHIP VIA:, CARRIER:, TERMS:) to avoid including carrier names.
                #   3. Within that clipped block, collect ORG entities.
                #   4. Apply the carrier blocklist — carriers are never buyers.
                #   5. If ORG found → emit the first ORG at high confidence.
                #   6. If no ORG found → emit nothing (buyer_id stays NULL).
                #      Never emit PERSON names for buyer_id.
                full_lower = parsed.full_text.lower()

                # Find the start of the buyer block
                buyer_block_start: int = -1
                for bill_kw in (
                    "bill to:", "bill to", "sold to:", "sold to",
                    "customer:", "buyer:", "client:", "billed to:", "invoice to:",
                ):
                    pos = full_lower.find(bill_kw)
                    if pos >= 0:
                        buyer_block_start = pos
                        break

                if buyer_block_start < 0:
                    # No BILL TO block found — cannot reliably identify the buyer org.
                    continue

                # Extract a window of text after the BILL TO keyword.
                # Use 300 chars as max; clip to the first shipping section marker
                # (SHIP VIA:, CARRIER:, TERMS:) to prevent carrier names from
                # leaking into the buyer block.
                raw_window = parsed.full_text[buyer_block_start:buyer_block_start + 300]
                raw_window_lower = raw_window.lower()

                # Find the earliest shipping/terms section boundary within the window
                clip_pos = len(raw_window)  # default: no clipping
                for boundary_kw in (
                    "ship via", "ship via:", "carrier:", "terms:", "shipped via",
                    "shipping method", "tracking", "delivery method",
                ):
                    bpos = raw_window_lower.find(boundary_kw)
                    if bpos >= 0 and bpos < clip_pos:
                        clip_pos = bpos

                buyer_block_text = raw_window[:clip_pos]
                buyer_block_lower = buyer_block_text.lower()

                # Collect ORG entities from ALL spaCy entities (not just field_ents
                # which are filtered to the field's ner_type_check=ORG)
                all_ents_by_label = ents_by_label  # already computed above

                # ORG entities in the clipped buyer block (carrier-filtered)
                org_ents_in_block: list = []
                for ent in all_ents_by_label.get("ORG", []):
                    et = ent.text.strip()
                    if not et:
                        continue
                    # Must be in the clipped buyer block
                    if et.lower() not in buyer_block_lower:
                        continue
                    # Must be a substring of the original full_text
                    if et not in parsed.full_text:
                        continue
                    # Apply carrier blocklist — carriers are never buyers
                    if is_carrier(et):
                        continue
                    org_ents_in_block.append(ent)

                if not org_ents_in_block:
                    # No ORG in BILL TO block — leave buyer_id as residual (NULL).
                    # Better NULL than a person name.
                    continue

                # Emit the first ORG found in the buyer block (leftmost / topmost)
                best_ent = org_ents_in_block[0]
                ent_text = best_ent.text.strip()
                cleaned_org = _clean_org_name(ent_text)
                if not cleaned_org or cleaned_org not in parsed.full_text:
                    # Cleaned name is no longer a substring — use raw for safety
                    cleaned_org = ent_text

                bbox = _find_bbox_for_text(parsed, ent_text)
                page_idx, b = bbox if bbox else (0, (0.0, 0.0, 0.0, 0.0))

                candidates.append(Candidate(
                    field=field.name,
                    value=cleaned_org,
                    page=page_idx,
                    bbox=b,
                    evidence_text=ent_text,
                    model="spacy_ner",
                    confidence=0.85,  # high confidence: explicit ORG in BILL TO block
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
                # Reject candidates that look like addresses or layout noise.
                # In procurement docs, spaCy often mis-tags title-cased UK
                # address fragments as PERSON and ALL-CAPS column headers as
                # ORG. Better NULL than a wrong value here — the context
                # layer below has full document text to fall back on.
                if _looks_like_address(ent_text):
                    log.debug("ner: dropping address-shaped %s candidate %r", field.name, ent_text)
                    continue
                if _looks_like_layout_noise(ent_text):
                    log.debug("ner: dropping layout-noise %s candidate %r", field.name, ent_text)
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


# Tokens that mark a string as a (UK/US) street/postal address rather than
# a person or organisation. Used to reject mis-tagged PERSON candidates
# like "Redkiln Way Horsham West Sussex" that spaCy emits for procurement
# docs where the buyer address dominates the masthead.
_ADDRESS_TOKEN_RE = re.compile(
    r"\b(?:Road|Rd|Street|St|Avenue|Ave|Lane|Ln|Way|Drive|Dr|Close|Crescent|"
    r"Court|Ct|Boulevard|Blvd|Place|Pl|Square|Sq|Terrace|Park|Plaza|Highway|"
    r"Hwy|Sussex|Yorkshire|Surrey|Middlesex|Manchester|Birmingham|Liverpool|"
    r"Leeds|Bristol|Sheffield|Edinburgh|Glasgow|London)\b",
    re.IGNORECASE,
)
# UK / US postcode shapes — anywhere in the string means it's an address.
_POSTCODE_RE = re.compile(
    r"\b(?:[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}|\d{5}(?:-\d{4})?)\b"
)


def _looks_like_address(text: str) -> bool:
    """True if the text contains UK/US address keywords or a postcode."""
    if not text or len(text) < 6:
        return False
    if _POSTCODE_RE.search(text):
        return True
    # Require at least one address-suffix word AND multi-word structure to
    # avoid rejecting a person named e.g. "Mr Park".
    if _ADDRESS_TOKEN_RE.search(text) and len(text.split()) >= 2:
        return True
    return False


# Common column-header / label tokens that spaCy mis-tags as ORG in markdown
# tables. Always lowercase keys — comparison is case-insensitive.
_LAYOUT_NOISE_WORDS: frozenset[str] = frozenset({
    "qty", "quantity", "total", "subtotal", "sub-total", "grand-total",
    "grand total", "tax", "vat", "gst", "discount", "amount", "balance",
    "description", "item", "items", "price", "cost", "rate", "unit",
    "net", "gross", "due", "paid", "payment", "currency", "invoice",
    "quote", "quotation", "po", "purchase order", "order", "date",
    "number", "no", "reference", "ref", "id", "from", "to", "by",
    "agency", "department", "billing", "shipping",
    # Section labels frequently emitted as ORG in docling output.
    "bill to", "ship to", "sold to", "remit to", "send to", "billing to",
    "recipient", "buyer", "seller", "vendor", "supplier", "customer",
})


def _looks_like_layout_noise(text: str) -> bool:
    """True if the candidate is a layout token, not a real entity.

    Rejects:
      - single ALL-CAPS short words ("QTY", "TOTAL")
      - any candidate whose canonical form (lowercased, trimmed) is a
        known column-header / section-label word
      - candidates ending in a digit-only suffix that looks like a row no
        (e.g. "Assurity Ltd 10")
    """
    if not text:
        return True
    s = text.strip()
    if not s:
        return True
    norm = re.sub(r"\s+", " ", s).strip().lower()
    if norm in _LAYOUT_NOISE_WORDS:
        return True
    # Single ALL-CAPS token of 2-5 chars (typical column header).
    if " " not in s and s.isupper() and 2 <= len(s) <= 5:
        return True
    # Trailing bare number ("Assurity Ltd 10") — leftover from table cells.
    if re.search(r"\b\d{1,4}$", s):
        return True
    return False


# Regex for company-suffix heuristic (catches names spaCy misses).
# Only matches names that TERMINATE with a recognised legal suffix — this
# prevents service descriptions like "Office Cleaning Services" (which merely
# contain a generic word) from being treated as company names.
_COMPANY_SUFFIX_RE = re.compile(
    r"([A-Z][A-Za-z0-9\-\. &']{2,60}?\s+"
    r"(?:Ltd|Limited|Inc|Incorporated|Corp|Corporation|LLC|LLP|GmbH|Pty|PLC|plc)"
    r"(?:\b|\.))",
    re.MULTILINE,
)


def _emit_company_suffix_candidates(
    parsed,
    field,
    candidates: list,
    buyer_context_orgs: set,
    page0_height: float,
) -> None:
    """Scan full_text for company-suffix patterns and emit low-confidence candidates.

    Used only when spaCy found zero ORG entities for supplier_name.
    Confidence is capped at 0.60 since the regex is less precise than NER.
    """
    seen: set[str] = set()
    for m in _COMPANY_SUFFIX_RE.finditer(parsed.full_text):
        name = m.group(1).strip()
        if name in seen:
            continue
        if name not in parsed.full_text:
            continue
        seen.add(name)

        is_buyer = name in buyer_context_orgs
        bbox = _find_bbox_for_text(parsed, name)
        page_idx, b = bbox if bbox else (0, (0.0, 0.0, 0.0, 0.0))
        in_header = page_idx == 0 and b[1] < page0_height * 0.40

        if is_buyer:
            conf = 0.40  # buyer context — very low but not excluded
        elif in_header:
            conf = 0.60  # header position, not buyer context → plausible supplier
        else:
            conf = 0.50  # found somewhere else in document

        candidates.append(Candidate(
            field=field.name,
            value=name,
            page=page_idx,
            bbox=b,
            evidence_text=name,
            model="spacy_ner",
            confidence=conf,
        ))


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
