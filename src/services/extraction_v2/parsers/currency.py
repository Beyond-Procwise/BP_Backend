"""Currency parser — wraps Currency construction, never raises."""
from __future__ import annotations

from typing import Optional

from src.services.extraction_v2.types import Currency, InvalidValue


def parse_currency(raw) -> Optional[Currency]:
    """Parse `raw` into a Currency code, or return None if unparseable."""
    if raw is None or raw == "":
        return None
    try:
        return Currency(raw)
    except InvalidValue:
        return None
