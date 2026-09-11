"""Shadow mode for the critic, in the dialect services/guardrail.py established.

Three rules carried over verbatim, because each was paid for:

  * Enrolment is PER DETECTOR, never a global boolean. A global switch is
    fail-open, which is the defect P0 existed to fix.
  * Every enrolment carries an "until". An enrolment without one is not
    honoured, so shadow mode cannot become the permanent state by nobody
    getting round to it.
  * Some things can never be suppressed, and that list lives in code rather
    than config -- a list in a row can be edited, and the point of these is
    that they cannot.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

#: A finding past 'identified' has a human acting on it. A model changing its
#: mind must not pull it out from under them. Spelled exactly as
#: opportunity_store._STAGES spells them; a test pins that.
NEVER_SUPPRESS_STAGES = ("negotiation", "agreed", "realised")


def _expiry(entry: Dict[str, Any]) -> Optional[datetime]:
    raw = entry.get("until")
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        logger.error("shadow enrolment for %r has an unreadable expiry %r",
                     entry.get("detector"), raw)
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def is_shadowed(detector_type: Optional[str], thresholds) -> bool:
    """Whether this detector's verdicts are observed but not acted on."""
    if not detector_type:
        return False
    now = datetime.now(timezone.utc)
    for entry in getattr(thresholds, "shadow_detectors", ()) or ():
        if str(entry.get("detector") or "") != str(detector_type):
            continue
        expiry = _expiry(entry)
        if expiry and expiry > now:
            return True
    return False


def may_suppress(finding: Dict[str, Any], thresholds) -> Tuple[bool, str]:
    """Whether an INVALID verdict on this finding may remove it from the page."""
    stage = str(finding.get("stage") or "identified").lower()
    if stage in NEVER_SUPPRESS_STAGES:
        return False, (
            f"finding is at stage {stage}: a human is already acting on it and it "
            "can never be suppressed by the critic"
        )
    if is_shadowed(finding.get("detector_type"), thresholds):
        return False, (
            f"detector {finding.get('detector_type')!r} is in shadow mode: the "
            "verdict is recorded, the finding stands"
        )
    return True, "enforcing"


def shadow_status(thresholds) -> Dict[str, Any]:
    """What is enrolled and until when. Surfaced in /health.

    A control that is off must be visible, not something you discover by
    reading code.
    """
    now = datetime.now(timezone.utc)
    enrolled = []
    for entry in getattr(thresholds, "shadow_detectors", ()) or ():
        expiry = _expiry(entry)
        enrolled.append({
            "detector": entry.get("detector"),
            "until": entry.get("until"),
            "active": bool(expiry and expiry > now),
        })
    return {
        "enrolled": enrolled,
        "never_suppressed_stages": list(NEVER_SUPPRESS_STAGES),
    }


def health_status(policy_engine: Any) -> Dict[str, Any]:
    """shadow_status for /health, refusing to report an outage as "none enrolled".

    Every governance read in this codebase is fail-open: an unreadable policy
    comes back looking exactly like an empty one. load_thresholds marks a policy
    it could not resolve with ``source=None``, and that is reported as
    unavailable -- because "nothing is in shadow" is the one answer a reader
    would act on, and it must not be invented by an outage.
    """
    from .governed import load_thresholds

    thresholds = load_thresholds(policy_engine)
    if thresholds.source is None:
        return {"error": "unavailable"}
    return shadow_status(thresholds)
