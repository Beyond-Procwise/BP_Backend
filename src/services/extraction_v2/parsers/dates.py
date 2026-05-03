"""Date parser — try-many-formats wrapper that never raises."""
from __future__ import annotations

from typing import Optional

from src.services.extraction_v2.types import InvalidValue, IsoDate


def parse_date(raw) -> Optional[IsoDate]:
    """Parse `raw` into an IsoDate, or return None if unparseable.

    Wraps :class:`IsoDate` construction with a try/except so callers
    can use this in chains without exception handling.
    """
    if raw is None or raw == "":
        return None
    try:
        return IsoDate(raw)
    except InvalidValue:
        return None
