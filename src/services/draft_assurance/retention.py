"""How long raw email text is kept, and the purge that enforces it.

Raw text is the sent text and its diff (``bp_draft_sent_text``) and the model's own draft
(``bp_draft_capture.draft_text``). Derived features (scores, classes, changed figures, hashes, the facts a
draft rested on) are NOT touched here and outlive the text.

The period is governed in ``proc.bp_policy`` (``EmailTextRetention``). A missing, non-numeric or non-positive
value makes ``load_rules`` raise and ``raw_text_days`` return ``None``, and a ``None`` period means NO raw text
is stored at all: the layer never keeps text on a period it made up.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)
SLUG = "email_text_retention"


class RetentionRulesUnavailable(RuntimeError):
    """The retention period is missing or unusable. Nothing is stored and nothing is purged on a guess."""


def load_rules(policy_engine: Any) -> Dict[str, int]:
    if policy_engine is None:
        raise RetentionRulesUnavailable("no policy engine available")
    try:
        policy = policy_engine.get_policy(SLUG)
    except Exception as exc:  # noqa: BLE001 - an outage is not "no retention"
        raise RetentionRulesUnavailable(f"policy store unreadable: {exc}") from exc
    rules = ((policy or {}).get("details") or {}).get("rules") if isinstance(policy, dict) else None
    if not isinstance(rules, dict):
        raise RetentionRulesUnavailable("no retention rules are defined")
    days = rules.get("raw_text_days")
    if isinstance(days, bool) or not isinstance(days, (int, float)) or days <= 0 or int(days) != days:
        raise RetentionRulesUnavailable("raw_text_days is missing or not a positive whole number")
    return {"raw_text_days": int(days)}


def raw_text_days(policy_engine: Any) -> Optional[int]:
    """The period in days, or ``None`` (store no raw text) if it cannot be read. Never raises."""

    try:
        return load_rules(policy_engine)["raw_text_days"]
    except RetentionRulesUnavailable as exc:
        logger.warning("raw email text will not be stored: %s", exc)
        return None


def purge_expired(conn: Any, days: Any, now: Optional[datetime] = None) -> Dict[str, int]:
    """Delete sent text older than ``days`` and blank the model's draft text older than ``days``.

    Blanking keeps the row (and its hash, facts and scores) and stamps ``text_expired_at``. Idempotent.
    """

    if isinstance(days, bool) or not isinstance(days, int) or days <= 0:
        raise ValueError("the retention period must be a positive whole number of days")
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)
    with conn.cursor() as cur:
        cur.execute("DELETE FROM email_agent.bp_draft_sent_text WHERE stored_at < %s", (cutoff,))
        deleted = cur.rowcount
        cur.execute("UPDATE email_agent.bp_draft_capture SET draft_text = '', text_expired_at = %s "
                    "WHERE captured_at < %s AND text_expired_at IS NULL", (now, cutoff))
        blanked = cur.rowcount
    return {"sent_text_deleted": int(deleted), "draft_text_blanked": int(blanked)}
