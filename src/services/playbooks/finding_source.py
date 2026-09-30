"""Normalise a row from either finding store into one shape.

Two stores, two vocabularies. ``proc.bp_detection_finding.rule_id`` holds triage
check codes (``cumulative_total``, ``line_arithmetic``); ``proc.bp_opportunity
.detector_type`` holds opportunity detector names (``Duplicate Invoice
Recovery``). They are separate namespaces and unifying them is its own piece of
work. This module keeps them apart -- a playbook matches within one source and
never across -- while giving the selector a single ``Finding`` to reason about.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

DETECTION_FINDING = "detection_finding"
OPPORTUNITY = "opportunity"

SOURCES: Tuple[str, ...] = (DETECTION_FINDING, OPPORTUNITY)

#: The columns a playbook of each source may match on. Closed, for the same
#: reason services.actions.ACTIONS is closed: a key that matches no column is a
#: playbook that silently never fires, and that failure looks exactly like the
#: playbook working and nothing having matched yet.
MATCH_FIELDS: Dict[str, Tuple[str, ...]] = {
    DETECTION_FINDING: ("rule_id", "category", "severity", "doc_type", "blocks_promotion"),
    OPPORTUNITY: ("detector_type", "supplier_id", "category_id"),
}

#: The primary key column per source. The two disagree on type: finding_id is
#: BIGINT, opportunity_id is VARCHAR. Both become text in a Finding.
_ID_FIELD: Dict[str, str] = {
    DETECTION_FINDING: "finding_id",
    OPPORTUNITY: "opportunity_id",
}

#: Open work per source. A detection finding is open by status; an opportunity
#: is open until it is retired.
OPEN_SQL: Dict[str, str] = {
    DETECTION_FINDING: (
        "SELECT finding_id, rule_id, category, severity, doc_type, "
        "       blocks_promotion, deal_id "
        "  FROM proc.bp_detection_finding "
        " WHERE status = 'open' "
        "   AND finding_id > %s "
        " ORDER BY finding_id "
        " LIMIT %s"
    ),
    OPPORTUNITY: (
        "SELECT opportunity_id, detector_type, supplier_id, category_id, deal_id "
        "  FROM proc.bp_opportunity "
        " WHERE retired_at IS NULL "
        "   AND opportunity_id > %s "
        " ORDER BY opportunity_id "
        " LIMIT %s"
    ),
}


@dataclass(frozen=True)
class Finding:
    """One open finding, in the only shape the selector sees."""

    source: str
    finding_id: str
    deal_id: Optional[str]
    attrs: Dict[str, Any]


def _require_source(source: str) -> str:
    if source not in MATCH_FIELDS:
        raise ValueError(
            f"{source!r} is not a finding source. Known sources: "
            f"{', '.join(SOURCES)}."
        )
    return source


def canonical(value: Any) -> Optional[str]:
    """One comparable form for a value from either side of a match.

    A playbook's trigger_match arrives as JSON, where a boolean column's value
    may have been authored as ``true`` or as ``"true"``, and a severity may have
    been typed ``Critical``. The database gives us ``True`` and ``'critical'``.
    Comparing those raw produces a playbook that never fires and no error to
    say so, so both sides are folded here before they are compared.

    ``None`` stays ``None``: an absent value is not the string "none", and must
    not match anything.
    """

    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).strip().lower()


def normalise(source: str, row: Mapping[str, Any]) -> Finding:
    """A database row from ``source`` as a :class:`Finding`."""

    _require_source(source)
    id_field = _ID_FIELD[source]
    raw_id = row.get(id_field)
    if raw_id is None:
        raise ValueError(f"{source} row has no {id_field}")
    return Finding(
        source=source,
        finding_id=str(raw_id),
        deal_id=(str(row["deal_id"]) if row.get("deal_id") is not None else None),
        attrs={field: row.get(field) for field in MATCH_FIELDS[source]},
    )


def validate_trigger_match(source: str, match: Mapping[str, Any]) -> Dict[str, Any]:
    """Return ``match`` if every key is a real match field for ``source``.

    Raises ``ValueError`` otherwise, rather than storing it. The detection
    registry this codebase removed bound configuration to detectors by
    accumulating aliases until something matched; four of five bindings were
    silently wrong for months. An unknown key here is the same failure in a new
    table, so it is refused at the door.
    """

    _require_source(source)
    allowed = MATCH_FIELDS[source]
    unknown = [key for key in match if key not in allowed]
    if unknown:
        raise ValueError(
            f"{', '.join(sorted(unknown))} "
            f"{'is not a match field' if len(unknown) == 1 else 'are not match fields'} "
            f"for {source}. Allowed: {', '.join(allowed)}."
        )
    null_keys = [key for key, value in match.items() if value is None]
    if null_keys:
        raise ValueError(
            f"{', '.join(sorted(null_keys))} may not be null. A null match value "
            "would mean 'fires only when this column is unset', which is never "
            "what an author means. Omit the key to ignore the column."
        )
    return dict(match)
