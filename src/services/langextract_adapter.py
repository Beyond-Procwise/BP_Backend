"""Confidence-gated LangExtract fallback for narrative fields.

When the structural extractor + NLU passes leave certain narrative fields
unresolved (supplier_name, payment_terms, notes), this adapter wraps
Google's LangExtract library against our existing on-prem Ollama
deployment to recover them with character-offset grounding.

Strict scope — explicitly OUT of bounds:
- PK fields (invoice_id / po_id / quote_id) — structural extractor is 100%
- Numeric amounts (invoice_amount / tax_amount / total_amount_incl_tax)
- Dates (invoice_date / due_date / order_date / ...)
- Geography (country / region / ship_to_country / delivery_region)

These are reliably resolved by the structural pipeline and routing them
through an LLM would TRADE A WORKING LAYER FOR RISK.

The adapter never raises. On any failure (LangExtract import error,
Ollama unreachable, parse error, etc.) it returns {}. The retry driver
must continue to the next attempt as if this layer was a no-op.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, Optional

from src.services.structural_extractor.types import ExtractedValue

logger = logging.getLogger(__name__)


# Fields this adapter is allowed to resolve. Adding a new field here is a
# deliberate decision — make sure structural+derivation can't cover it
# first, and add a few-shot example below.
#
# Identity / narrative fields:
_LANGEXTRACT_ALLOWED_FIELDS = frozenset({
    # Original narrative set
    "supplier_name",
    "payment_terms",
    "incoterm",
    "notes",
    # Buyer / counterparty (recurring miss across ~15 docs in production)
    "buyer_id",
    "buyer_name",
    # Address fields — quotes have them, POs need delivery details
    "supplier_address",
    "buyer_address",
    "delivery_address_line1",
    "delivery_city",
    "postal_code",
})

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("PROCWISE_EXTRACTION_MODEL",
                          "BeyondProcwise/AgentNick:extract")


def _build_examples(doc_type: str):
    """Few-shot examples — kept verbatim as LangExtract requires.

    These come from real procurement documents we've seen. Each example
    must show ONLY allowed-fields so the model doesn't learn to surface
    PK/amounts/dates from this fallback.
    """
    try:
        import langextract as lx
    except ImportError:
        return None

    common = [
        # Example 1 — full invoice with all party + payment fields
        lx.data.ExampleData(
            text=(
                "From: Acme Corporation Ltd\n"
                "10 King Street, London, EC2A 3NW\n"
                "Bill To: Beta Holdings Ltd\n"
                "45 Market Street, Birmingham, B3 1AA\n"
                "Payment Terms: Net 30\n"
                "Incoterm: DDP"
            ),
            extractions=[
                lx.data.Extraction(extraction_class="supplier_name",
                                   extraction_text="Acme Corporation Ltd"),
                lx.data.Extraction(extraction_class="supplier_address",
                                   extraction_text="10 King Street, London, EC2A 3NW"),
                lx.data.Extraction(extraction_class="buyer_name",
                                   extraction_text="Beta Holdings Ltd"),
                lx.data.Extraction(extraction_class="buyer_address",
                                   extraction_text="45 Market Street, Birmingham, B3 1AA"),
                lx.data.Extraction(extraction_class="payment_terms",
                                   extraction_text="Net 30"),
                lx.data.Extraction(extraction_class="incoterm",
                                   extraction_text="DDP"),
            ],
        ),
        # Example 2 — verbose payment terms, supplier-only example
        lx.data.ExampleData(
            text=(
                "Issued by: TechWorld Solutions Ltd\n"
                "Payment must be made within 30 days of receiving the invoice."
            ),
            extractions=[
                lx.data.Extraction(extraction_class="supplier_name",
                                   extraction_text="TechWorld Solutions Ltd"),
                lx.data.Extraction(extraction_class="payment_terms",
                                   extraction_text=(
                                       "Payment must be made within 30 days "
                                       "of receiving the invoice."
                                   )),
            ],
        ),
        # Example 3 — Purchase Order with delivery details
        lx.data.ExampleData(
            text=(
                "PURCHASE ORDER PO-2024-0142\n"
                "Vendor: Greenfield Holdings Ltd\n"
                "Buyer: Assurity Ltd\n"
                "Ship To:\n"
                "Unit 12, Meridian Business Park\n"
                "Phoenix Way\n"
                "Leicester LE19 1WX\n"
            ),
            extractions=[
                lx.data.Extraction(extraction_class="supplier_name",
                                   extraction_text="Greenfield Holdings Ltd"),
                lx.data.Extraction(extraction_class="buyer_name",
                                   extraction_text="Assurity Ltd"),
                lx.data.Extraction(extraction_class="delivery_address_line1",
                                   extraction_text="Unit 12, Meridian Business Park"),
                lx.data.Extraction(extraction_class="delivery_city",
                                   extraction_text="Leicester"),
                lx.data.Extraction(extraction_class="postal_code",
                                   extraction_text="LE19 1WX"),
            ],
        ),
        # Example 4 — quote with "BILLED TO:" buyer address on wrapped lines
        # (address line wraps so phone and street appear merged in OCR text)
        lx.data.ExampleData(
            text=(
                "QUOTE\n"
                "ACME LTD Quote No. Q-001\n"
                "BILLED TO:\n"
                "Beta Holdings Ltd\n"
                "+44-7700-12345610 Redkiln Way, Horsham, West Sussex\n"
                "RH13 5QH, United Kingdom\n"
                "Item Monthly Cost Total\n"
                "Service A £100 £1200\n"
                "Total £1200\n"
                "PAYMENT INFORMATION\n"
                "Beta Bank, Account: 123\n"
                "Pay by: 5 July 2025 478 Branding Lane, SW16 7JD, London, UK\n"
            ),
            extractions=[
                lx.data.Extraction(extraction_class="supplier_name",
                                   extraction_text="ACME LTD"),
                lx.data.Extraction(extraction_class="supplier_address",
                                   extraction_text="478 Branding Lane, SW16 7JD, London, UK"),
                lx.data.Extraction(extraction_class="buyer_name",
                                   extraction_text="Beta Holdings Ltd"),
                lx.data.Extraction(extraction_class="buyer_address",
                                   extraction_text="Redkiln Way, Horsham, West Sussex RH13 5QH, United Kingdom"),
                lx.data.Extraction(extraction_class="delivery_city",
                                   extraction_text="Horsham"),
                lx.data.Extraction(extraction_class="postal_code",
                                   extraction_text="RH13 5QH"),
            ],
        ),
        # NOTE: LangExtract rejects ExampleData with extractions=[]
        # ("Source tokens and extraction tokens cannot be empty"). Negative
        # constraints carried in the prompt; grounding check enforces them.
    ]
    return common


def _build_prompt(fields_needed: list[str]) -> str:
    field_list = ", ".join(fields_needed)
    return (
        f"Extract these procurement document fields from the text below: {field_list}.\n"
        "Rules:\n"
        " - Use the exact text from the document — do NOT paraphrase.\n"
        " - Each extraction MUST appear verbatim in the source text.\n"
        " - If a field is not in the body, OMIT it (do NOT guess).\n"
        " - supplier_name is the COMPANY that ISSUED this document. Look for"
        " 'From', 'Vendor', or company letterhead at the TOP.\n"
        " - buyer_name is the COMPANY RECEIVING the document. Look for 'Bill To',"
        " 'Customer', 'Invoice To', 'Ship To', 'Sold To', 'Buyer'.\n"
        " - DO NOT use a person's name (e.g., 'Sophie Turner') as buyer_name."
        " The buyer is a COMPANY. Common labels that are NOT the buyer: 'Salesperson',"
        " 'Account Manager', 'Contact', 'Attention', 'Prepared By', 'Sales Rep'.\n"
        " - If the only human name on the page is under a 'Salesperson' or"
        " 'Account Manager' label, OMIT buyer_name rather than fabricate.\n"
        " - supplier_address / buyer_address: the full multi-line address block"
        " under the corresponding company name. Do not include the company name itself.\n"
        " - delivery_address_line1: street + first line of the ship-to / delivery address.\n"
        " - delivery_city: city only (single token, e.g. 'Birmingham', 'Leicester').\n"
        " - postal_code: the postcode (UK format like 'B3 1AA' or US ZIP).\n"
        " - payment_terms is the literal terms string from the document.\n"
        " - NEVER capture a filename, document-type label, address, or PO number"
        " as supplier_name or buyer_name.\n"
    )


def extract_low_confidence_fields(
    source_text: str,
    fields_needed: list[str],
    doc_type: str,
    *,
    attempt_no: int = 0,
) -> Dict[str, ExtractedValue]:
    """Recover narrative fields from `source_text` using LangExtract.

    Args:
        source_text: Full document text (already extracted by upstream parsers).
        fields_needed: Fields the retry driver is missing. Filtered down to
            the allowed-set before any LLM call — fields outside the set
            are silently dropped.
        doc_type: "Invoice" / "Purchase_Order" / "Quote".
        attempt_no: For tagging the returned ExtractedValue.attempt.

    Returns:
        Dict mapping field name → ExtractedValue. Empty on any error.
    """
    targets = [f for f in fields_needed if f in _LANGEXTRACT_ALLOWED_FIELDS]
    if not targets:
        return {}
    if not source_text or len(source_text) < 50:
        return {}

    try:
        import langextract as lx
        from langextract import factory
    except ImportError:
        logger.debug("langextract not installed — skipping fallback")
        return {}

    examples = _build_examples(doc_type)
    if not examples:
        return {}

    cfg = factory.ModelConfig(
        model_id=OLLAMA_MODEL,
        provider="OllamaLanguageModel",
        provider_kwargs={"model_url": OLLAMA_URL},
    )

    try:
        result = lx.extract(
            text_or_documents=source_text,
            prompt_description=_build_prompt(targets),
            examples=examples,
            config=cfg,
            format_type=lx.data.FormatType.JSON,
            temperature=0.0,
            extraction_passes=1,
            max_workers=1,
            use_schema_constraints=False,
            fence_output=False,
            show_progress=False,
            fetch_urls=False,
        )
    except Exception as exc:
        logger.debug("langextract.extract failed: %s", exc)
        return {}

    out: Dict[str, ExtractedValue] = {}
    for e in result.extractions:
        field = e.extraction_class
        if field not in _LANGEXTRACT_ALLOWED_FIELDS:
            continue
        if field in out:
            continue  # earlier hit wins
        text = (e.extraction_text or "").strip()
        if not text:
            continue
        # Grounding check: char_interval must be present AND the extracted
        # text must actually appear at that interval (defends against
        # hallucinated spans that the resolver mis-aligned).
        char_iv = getattr(e, "char_interval", None)
        if char_iv is None:
            logger.debug("langextract: %s ungrounded — dropping", field)
            continue
        try:
            spans = source_text[char_iv.start_pos:char_iv.end_pos]
            if text not in spans and spans not in text:
                logger.debug(
                    "langextract: %s text '%s' does not match interval span '%s'",
                    field, text, spans,
                )
                continue
        except Exception:
            continue

        out[field] = ExtractedValue(
            value=text,
            provenance="extracted",
            anchor_text=text,
            anchor_ref=None,  # char-offset; no bbox available from langextract
            source="langextract",
            confidence=0.85,
            attempt=attempt_no or 5,
        )
    if out:
        logger.info(
            "[langextract] resolved %d field(s): %s",
            len(out), list(out.keys()),
        )
    return out
