import re

from src.services.structural_extractor.derivation import rule

UK_POSTCODE_RE = re.compile(r"\b[A-Z]{1,2}[0-9][A-Z0-9]?\s*[0-9][A-Z]{2}\b")
US_ZIP_RE = re.compile(r"\b\d{5}(-\d{4})?\b")


@rule("country_from_postcode", "country", ["_address_text"])
def _country(inputs):
    text = str(inputs["_address_text"] or "")
    if UK_POSTCODE_RE.search(text):
        return "United Kingdom"
    if US_ZIP_RE.search(text):
        return "United States"
    return None


@rule("region_from_address", "region", ["_address_text"])
def _region(inputs):
    text = str(inputs["_address_text"] or "")
    # Simplified: look for common UK regions
    for region in ["West Sussex", "East Sussex", "Greater London", "Surrey",
                   "Hampshire", "Kent", "Essex", "Yorkshire", "Lancashire"]:
        if region in text:
            return region
    return None


@rule("invoice_status_default", "invoice_status", ["invoice_id"])
def _inv_status(inputs):
    return "Issued"


@rule("po_status_default", "po_status", ["po_id"])
def _po_status(inputs):
    return "Open"


@rule("currency_from_symbol_inv", "currency", ["_amount_text"])
def _curr_from_symbol(inputs):
    text = str(inputs.get("_amount_text", "") or "")
    symbol_map = {"£": "GBP", "$": "USD", "€": "EUR", "¥": "JPY", "₹": "INR"}
    for sym, code in symbol_map.items():
        if sym in text:
            return code
    return None
