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
    # Collapse all whitespace (including newlines) to single spaces so that
    # regions split across line breaks (e.g. "Horsham West\nSussex RH13 5QH")
    # still match against the canonical "West Sussex" region name.
    text_lower = re.sub(r"\s+", " ", str(inputs["_address_text"] or "").lower())
    # Case-insensitive match: handles "West Sussex", "WEST SUSSEX", "west sussex".
    # Return value is the canonical capitalization.
    UK_REGIONS = [
        "West Sussex", "East Sussex", "West Midlands", "East Midlands",
        "Greater London", "Greater Manchester", "Merseyside",
        "South Yorkshire", "West Yorkshire", "North Yorkshire", "East Yorkshire",
        "Surrey", "Hampshire", "Kent", "Essex", "Yorkshire", "Lancashire",
        "Cheshire", "Norfolk", "Suffolk", "Devon", "Cornwall", "Somerset",
        "Dorset", "Wiltshire", "Gloucestershire", "Warwickshire",
        "Leicestershire", "Nottinghamshire", "Derbyshire", "Staffordshire",
        "Berkshire", "Oxfordshire", "Hertfordshire", "Buckinghamshire",
        "Bedfordshire", "Cambridgeshire", "Northamptonshire",
        "Lincolnshire", "Rutland", "Shropshire", "Herefordshire",
        "Worcestershire", "Cumbria", "Northumberland", "Durham",
        "Tyne and Wear", "Middlesex", "London",
    ]
    for region in UK_REGIONS:
        if region.lower() in text_lower:
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
