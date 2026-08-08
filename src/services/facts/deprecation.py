"""One-release deprecation shim over ``calculation_details``.

Phase 1b makes the structured columns on ``proc.bp_opportunity`` the system of
record. ``calculation_details`` is still written for one release so nothing
breaks mid-migration, but it is no longer authoritative.

Every read goes through here, and every fall back to the JSONB is logged with
the key and the opportunity id. That log is the retirement criterion: when it
stops firing in production, the JSONB column can go. "We think nothing reads it
any more" is not evidence, and a shim nobody can measure cannot be retired.

Phase 0 measured the whole surface: 2 producers, 4 reader groups, 1 store, and
no API or render path reads ``calculation_details`` at all.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional

logger = logging.getLogger(__name__)

#: calculation_details key -> the structured column that now supersedes it.
#:
#: Aliases matter: the benchmark detector writes ``actual_price`` while the
#: column is ``unit_price``. Without the alias the shim would fall back forever
#: and the deprecation log would never go quiet, so the column would look unused
#: and the JSONB would look indispensable — exactly backwards.
_KEY_TO_COLUMN: Dict[str, str] = {
    "currency": "currency",
    "amount_native": "amount_native",
    "unit_price": "unit_price",
    "actual_price": "unit_price",
    "quantity": "quantity",
    "uom": "uom",
    "unit_of_measure": "uom",
    "uom_normalised": "uom_normalised",
    "fx_rate": "fx_rate",
    "fx_rate_date": "fx_rate_date",
    "value_basis": "value_basis",
}


def _get(record: Any, name: str) -> Any:
    """Read a field from either a dict row or a finding object."""
    if record is None:
        return None
    if isinstance(record, Mapping):
        return record.get(name)
    return getattr(record, name, None)


def _identify(record: Any) -> str:
    for key in ("opportunity_ref_id", "opportunity_id"):
        value = _get(record, key)
        if value:
            return str(value)
    return "<unidentified>"


def read_calculation_detail(record: Any, key: str, default: Optional[Any] = None) -> Any:
    """Return ``key`` for ``record``, preferring the structured column.

    Falls back to ``calculation_details[key]`` and logs the hit. A NULL column
    is NOT treated as authoritative: an INDETERMINATE row has NULL columns by
    design, and letting NULL win would silently drop values the JSONB still
    holds during the shim release.
    """
    column = _KEY_TO_COLUMN.get(key)
    if column is not None:
        value = _get(record, column)
        if value is not None:
            return value

    details = _get(record, "calculation_details")
    if isinstance(details, Mapping) and key in details:
        value = details.get(key)
        logger.warning(
            "calculation_details fallback: key=%s opportunity=%s "
            "(column=%s unavailable) -- this JSONB read is deprecated",
            key, _identify(record), column or "<none>",
        )
        return value

    return default
