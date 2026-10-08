"""The send-time question: do the facts this draft rests on still say what they said?

A draft is written at one moment and sent at another, and the supplier's offer, the
contact or the currency can change in between. The assurance record keeps the exact
row each fact came from; this re-reads those rows.

Never raises. In ``shadow`` mode a change is reported and the send proceeds; in
``enforce`` mode the caller refuses. If the family cannot be loaded the mode stored
on the draft's own record is used, so an outage cannot quietly turn an enforced
family into a shadowed one.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .assure import recheck_facts
from .family import FamilyConfigUnavailable, load_family

logger = logging.getLogger(__name__)


def recheck_for_send(conn: Any, draft: Dict[str, Any], policy_engine: Optional[Any]) -> Dict[str, Any]:
    assurance = draft.get("assurance")
    if not isinstance(assurance, dict) or not assurance.get("facts"):
        return {"checked": False, "reason": "draft carries no assured facts", "changed": []}
    stored_mode = assurance.get("mode") or "shadow"
    try:
        family = load_family(f"email_family_{assurance.get('family_id')}", policy_engine)
    except FamilyConfigUnavailable as exc:
        logger.error("send-time fact re-check could not load its family: %s", exc)
        return {"checked": False, "mode": stored_mode, "changed": [],
                "reason": f"family unreadable: {exc}"}
    try:
        changed = recheck_facts(conn, family, assurance)
    except Exception as exc:  # noqa: BLE001
        logger.exception("send-time fact re-check failed")
        return {"checked": False, "mode": family.mode, "changed": [],
                "reason": f"re-read failed: {type(exc).__name__}"}
    return {"checked": True, "mode": family.mode, "changed": changed}
