"""proc.bp_policy_observation -- what the gate decided, including what it did not do.

Sixteen of the nineteen live policies carry no ``applies_to``, so the gate cannot
see them. Giving them one turns dormant rules into refusals, and nobody knows
what would be refused, to whom, or how often. Turning them on to find out is a
guess with a blast radius.

This is the record that makes it a measurement instead. Every decision the gate
reaches is written here -- allow and deny alike -- and for actions explicitly
enrolled in shadow mode the denial is recorded and then not applied.

WHY ALLOWS ARE RECORDED TOO

Because "we observed no denials" and "we were not observing" are the same
observation otherwise, and the second is the one that gets acted on by mistake.
A reader needs to be able to see the traffic that passed as well as the traffic
that would not have.

WHY THIS WRITE IS SYNCHRONOUS

``egress_log`` buffers because it sits under every model inference. This sits
under ``guardrail.authorize``, which has four call sites and is reached once per
irreversible action. A row per decision is not a load problem here, and a
buffered write that could be dropped would undermine the one property shadow
mode is bought for: ``record`` returning False is what makes the gate fall back
to enforcing, so the caller has to know whether the write happened.

WHAT A FAILED WRITE MEANS

``False``, not an exception. The gate must never crash on its own bookkeeping.
But a shadowed decision whose observation could not be written is not allowed
through: no record, no shadow -- see ``guardrail.authorize``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DDL = """
CREATE TABLE IF NOT EXISTS proc.bp_policy_observation (
    observation_id    BIGSERIAL PRIMARY KEY,
    observed_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    action            TEXT        NOT NULL,
    action_class      TEXT,
    principal_subject TEXT,
    role              TEXT,
    verdict           TEXT        NOT NULL,
    would_have_denied BOOLEAN     NOT NULL,
    shadowed          BOOLEAN     NOT NULL DEFAULT false,
    policy_id         TEXT,
    policy_name       TEXT,
    policy_version    INTEGER,
    reason            TEXT,
    evidence          JSONB
);

CREATE INDEX IF NOT EXISTS ix_bp_policy_observation_observed
    ON proc.bp_policy_observation (observed_at DESC);
CREATE INDEX IF NOT EXISTS ix_bp_policy_observation_action
    ON proc.bp_policy_observation (action, observed_at DESC);
CREATE INDEX IF NOT EXISTS ix_bp_policy_observation_shadowed
    ON proc.bp_policy_observation (shadowed) WHERE shadowed;
"""

_INSERT = """
INSERT INTO proc.bp_policy_observation
    (observed_at, action, action_class, principal_subject, role, verdict,
     would_have_denied, shadowed, policy_id, policy_name, policy_version,
     reason, evidence)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""


def record(
    *,
    action: str,
    action_class: Optional[str] = None,
    principal_subject: Optional[str] = None,
    role: Optional[str] = None,
    would_have_denied: bool = False,
    shadowed: bool = False,
    policy_id: Optional[Any] = None,
    policy_name: Optional[str] = None,
    policy_version: Optional[int] = None,
    reason: Optional[str] = None,
    evidence: Optional[Dict[str, Any]] = None,
) -> bool:
    """Write one observation. Returns whether it was written.

    Never raises: the gate must not fail on its own bookkeeping. The boolean is
    the point of the return value -- a shadowed decision is only allowed through
    when this said True.
    """

    try:
        from src.services.db import get_conn

        params = (
            datetime.now(timezone.utc),
            str(action),
            action_class,
            principal_subject,
            role,
            "deny" if would_have_denied else "allow",
            bool(would_have_denied),
            bool(shadowed),
            None if policy_id is None else str(policy_id),
            policy_name,
            policy_version,
            reason,
            json.dumps(evidence or {}, default=str),
        )
        with get_conn() as conn:
            conn.autocommit = False
            cur = conn.cursor()
            try:
                cur.execute(_INSERT, params)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return True
    except Exception as exc:  # noqa: BLE001 - bookkeeping must not break the gate
        logger.error("policy_observation.record(%s) failed: %s", action, exc)
        return False


def ensure_table() -> bool:
    """Create the table if it does not exist. For deploys and for the report."""

    try:
        from src.services.db import get_conn

        with get_conn() as conn:
            conn.autocommit = True
            conn.cursor().execute(DDL)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("policy_observation.ensure_table failed: %s", exc)
        return False
