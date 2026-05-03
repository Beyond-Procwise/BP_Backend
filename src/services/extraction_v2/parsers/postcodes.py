"""Postcode parser — wraps Postcode construction, never raises."""
from __future__ import annotations

from typing import Optional

from src.services.extraction_v2.types import InvalidValue, Postcode


def parse_postcode(raw) -> Optional[Postcode]:
    """Parse `raw` into a Postcode, or return None if unparseable."""
    if raw is None or raw == "":
        return None
    if not isinstance(raw, str):
        return None
    try:
        return Postcode(raw)
    except InvalidValue:
        return None
