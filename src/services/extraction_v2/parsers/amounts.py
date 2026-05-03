"""Amount parser — wraps Money construction, never raises."""
from __future__ import annotations

from typing import Optional

from src.services.extraction_v2.types import InvalidValue, Money


def parse_amount(raw) -> Optional[Money]:
    """Parse `raw` into a Money, or return None if unparseable."""
    if raw is None or raw == "":
        return None
    try:
        return Money(raw)
    except InvalidValue:
        return None
