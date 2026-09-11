"""The three rules a critique may never break.

These REFUSE. They do not repair. A critique quietly corrected is a critique
nobody knows was wrong, and the whole value of the critic is that its reasoning
can be inspected.
"""
from __future__ import annotations

from typing import Any, Dict, List

VERDICTS = ("VALID", "VALID_REFRAMED", "INVALID", "UNASSESSED", "DUPLICATE")

#: Verdicts that assert a finding is actionable, and so carry the value ceiling
#: and the no-blocking-gap rule.
_AFFIRMATIVE = ("VALID", "VALID_REFRAMED")


def check_invariants(critique: Dict[str, Any]) -> List[str]:
    """Return a list of violation messages. Empty means the critique is well-formed."""
    violations: List[str] = []

    verdict = str(critique.get("verdict") or "")
    if verdict not in VERDICTS:
        violations.append(f"unknown verdict {verdict!r}")

    value = critique.get("value") or {}
    proposed = value.get("detector_proposed")
    addressable = value.get("critic_addressable")
    gaps = critique.get("gaps") or []
    blocking = [g for g in gaps if g.get("blocking")]

    if verdict in _AFFIRMATIVE:
        if (isinstance(proposed, (int, float))
                and isinstance(addressable, (int, float))
                and addressable > proposed):
            violations.append(
                f"{verdict} carries {addressable} which is above the detector's "
                f"{proposed}; a critic may reduce a value, never raise it"
            )
        if blocking:
            violations.append(
                f"{verdict} carries a blocking gap ({blocking[0].get('gap_id')}); "
                "a verdict cannot be affirmative while something blocks it"
            )

    if verdict == "UNASSESSED" and not blocking:
        violations.append(
            "UNASSESSED must carry at least one blocking gap; if nothing blocks "
            "the decision then it was decidable, and 'we could not say' needs a reason"
        )

    return violations
