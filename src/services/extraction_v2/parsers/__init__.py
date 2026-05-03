"""Deterministic parsers for the extraction_v2 pipeline.

Contract:
    - Accepts raw input (str / int / None / etc.)
    - Returns the typed value (Money, IsoDate, ParsedAddress, ...) or None
    - NEVER raises — abstention is encoded as None
"""
from .addresses import ParsedAddress, parse_address
from .amounts import parse_amount
from .currency import parse_currency
from .dates import parse_date
from .postcodes import parse_postcode

__all__ = [
    "ParsedAddress", "parse_address",
    "parse_amount",
    "parse_currency",
    "parse_date",
    "parse_postcode",
]
