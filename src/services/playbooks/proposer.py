"""Write one proposal for one finding, and say so in the audit log.

A proposal is a recommendation awaiting a person. Writing one starts nothing:
the run_id stays NULL until somebody approves it through the endpoint.

Idempotency is the unique index ux_bp_playbook_proposal_finding, not a SELECT
first -- two sweeps overlapping would both find nothing and both insert. The
index is keyed on (playbook_id, finding_source, finding_id) and deliberately
NOT on version, so editing a playbook does not re-raise a proposal for a
finding already queued.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional, Sequence

from src.services import agent_actions
from src.services.db import get_conn

from .finding_source import Finding
from .selector import Selection
from .store import Playbook

logger = logging.getLogger(__name__)

PHASE = "playbook"

_INSERT = """
    INSERT INTO proc.bp_playbook_proposal
        (playbook_id, finding_source, finding_id, deal_id, evidence)
    VALUES (%s, %s, %s, %s, %s::jsonb)
    ON CONFLICT DO NOTHING
    RETURNING proposal_id
"""


def propose(
    finding: Finding,
    selection: Selection,
    *,
    conn: Optional[Any] = None,
) -> Optional[int]:
    """Queue ``selection`` for ``finding``. Returns the new proposal_id.

    Returns ``None`` when this finding already has a proposal from this
    playbook, which is the ordinary case on every sweep after the first.
    """

    params = (
        selection.playbook.playbook_id,
        finding.source,
        finding.finding_id,
        finding.deal_id,
        json.dumps(selection.evidence, default=str),
    )
    if conn is not None:
        proposal_id = _insert(conn, params)
        audit_conn = conn
    else:
        with get_conn() as own:
            proposal_id = _insert(own, params)
        audit_conn = None

    if proposal_id is None:
        # Already queued. Auditing it again would write one row per open
        # finding per sweep, saying nothing happened.
        return None

    # On the sweep's own connection. record_action opens a fresh one when it is
    # given none, and a catch-all playbook proposes on every open finding --
    # ~5,000 short-lived connections to a shared cluster in one scheduler tick.
    agent_actions.record_action(
        conn=audit_conn,
        phase=PHASE,
        action_type="playbook.propose",
        agent="PlaybookProposer",
        deal_id=finding.deal_id,
        status="proposed",
        summary=(
            f"{selection.playbook.playbook_name!r} proposed for "
            f"{finding.source} {finding.finding_id}"
        ),
        details={
            "proposal_id": proposal_id,
            "playbook_id": selection.playbook.playbook_id,
            "agent_workflow_id": selection.playbook.agent_workflow_id,
            "finding_source": finding.source,
            "finding_id": finding.finding_id,
            "evidence": selection.evidence,
        },
    )
    return proposal_id


def _insert(conn: Any, params: tuple) -> Optional[int]:
    cursor = conn.cursor()
    try:
        cursor.execute(_INSERT, params)
        row = cursor.fetchone()
        return int(row[0]) if row else None
    finally:
        cursor.close()


def record_ambiguous(finding: Finding, tied: Sequence[Playbook]) -> None:
    """Record that two or more playbooks tied, so nothing was proposed.

    The selector has already said so at ERROR. This puts it in the same event
    log as the proposals, because "no proposal appeared for this finding" is a
    question someone will ask of the audit trail, not of the log files.
    """

    names = ", ".join(f"{pb.playbook_name!r} (id {pb.playbook_id})" for pb in tied)
    agent_actions.record_action(
        phase=PHASE,
        action_type="playbook.ambiguous",
        agent="PlaybookProposer",
        deal_id=finding.deal_id,
        status="skipped",
        summary=(
            f"{len(tied)} playbooks tie for {finding.source} "
            f"{finding.finding_id}: {names}. Nothing proposed."
        ),
        details={
            "finding_source": finding.source,
            "finding_id": finding.finding_id,
            "playbook_ids": [pb.playbook_id for pb in tied],
            "playbook_names": [pb.playbook_name for pb in tied],
        },
    )
