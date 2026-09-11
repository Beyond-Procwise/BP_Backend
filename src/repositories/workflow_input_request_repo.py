# src/repositories/workflow_input_request_repo.py
"""What the workflow asked the human, and what the human answered.

A HITL run is only trustworthy if you can see what a person supplied and when.
"""

from __future__ import annotations

import json
import logging
from datetime import timedelta
from typing import Any, Dict, List, Optional

from services.db import get_conn

logger = logging.getLogger(__name__)

# A run claimed for execution transitions to 'executing' and is excluded from
# future claims (see claim_for_execution) so it can never run twice. But if
# the worker that claimed it is killed mid-run — OOM, segfault, deploy
# restart, a hung LLM/GPU call that gets reaped — the row is left stuck at
# 'executing' forever, with no in-process exception ever firing to mark it
# 'failed' (which IS reclaimable). Past this window, an 'executing' run is
# treated as abandoned and becomes reclaimable again.
#
# Correctness trade-off (deliberate, not accidental): a run that is
# GENUINELY still executing past this window COULD be reclaimed and executed
# a second time — for HITL workflows that dispatch real emails, that means a
# second email. 30 minutes is chosen because it is far longer than any real
# run of this system's LLM/GPU agents is expected to take, making that
# double-execution scenario implausible while still bounding how long a
# hard-crashed run stays wedged with no recovery path.
STALE_EXECUTING_WINDOW = timedelta(minutes=30)

DDL = """
CREATE SCHEMA IF NOT EXISTS proc;

CREATE TABLE IF NOT EXISTS proc.bp_workflow_input_request (
    request_id       BIGSERIAL PRIMARY KEY,
    workflow_id       TEXT NOT NULL,          -- the RUN id
    node_name        TEXT NOT NULL,
    agent_slug       TEXT NOT NULL,
    required_field   TEXT NOT NULL,
    field_type       TEXT,
    prompt           TEXT,
    status           TEXT NOT NULL DEFAULT 'pending',
    answer           JSONB,
    answered_by      TEXT,
    requested_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    answered_at      TIMESTAMPTZ,
    -- The numeric proc.bp_agent_workflow this run was started from. Lets a
    -- run be resolved back to its saved workflow by lookup instead of by
    -- parsing the run_id string (run_id format is an implementation detail
    -- of the router, not a contract this table should depend on).
    agent_workflow_id BIGINT
);

CREATE INDEX IF NOT EXISTS ix_bp_workflow_input_request_open
    ON proc.bp_workflow_input_request (workflow_id, status);

-- One row per RUN (not per question). This is what makes a resume trustworthy:
--   * ``payload`` is the run's ORIGINAL POST /{id}/run body, persisted before the
--     run ever halts, so it survives a process restart between run and resume
--     and is never silently dropped when the human's answers are merged back in.
--   * ``status`` is the run's execution state, and the transition into
--     'executing' is claimed with a single atomic UPDATE (see
--     ``claim_for_execution``) so a replayed or concurrent final answer cannot
--     make the workflow execute twice.
CREATE TABLE IF NOT EXISTS proc.bp_workflow_run (
    run_id            TEXT PRIMARY KEY,
    agent_workflow_id BIGINT,
    payload           JSONB NOT NULL DEFAULT '{}'::jsonb,
    status            TEXT NOT NULL DEFAULT 'pending',
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- The subject of the person who STARTED the run, or NULL when nobody did.
    -- A run that stops to ask a question is resumed by whoever answers, so the
    -- starter has to be kept here or it is lost at the pause.
    initiated_by      TEXT
);
"""

# Older deployments of these tables predate the agent_workflow_id and
# initiated_by columns.
_MIGRATE = """
ALTER TABLE proc.bp_workflow_input_request
    ADD COLUMN IF NOT EXISTS agent_workflow_id BIGINT;
ALTER TABLE proc.bp_workflow_run
    ADD COLUMN IF NOT EXISTS initiated_by TEXT;
"""


def ensure_schema() -> None:
    """Ensure ``proc.bp_workflow_input_request`` and ``proc.bp_workflow_run`` exist."""

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(DDL)
        cur.execute(_MIGRATE)
        cur.close()


def create_run(
    run_id: str, *, agent_workflow_id: Optional[int], payload: Dict[str, Any],
    status: str = "pending", initiated_by: Optional[str] = None,
) -> None:
    """Persist the run's original payload (and starting status) exactly once.

    Idempotent: if the run row already exists (e.g. this is called again on
    the same run_id) the existing payload/status are left untouched — the
    ORIGINAL payload must never be overwritten by a later call. The same holds
    for ``initiated_by``: whoever started the run stays who started it.
    """
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO proc.bp_workflow_run
                   (run_id, agent_workflow_id, payload, status, initiated_by)
                   VALUES (%s, %s, %s::jsonb, %s, %s)
               ON CONFLICT (run_id) DO NOTHING""",
            (run_id, agent_workflow_id, json.dumps(payload or {}), status, initiated_by),
        )
        cur.close()


def initiator_for(run_id: str) -> Optional[str]:
    """Who started ``run_id``: a principal's subject, or None when nobody did."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT initiated_by FROM proc.bp_workflow_run WHERE run_id = %s",
            (run_id,),
        )
        row = cur.fetchone()
        cur.close()
    return (row[0] or None) if row else None


def get_run(run_id: str) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT run_id, agent_workflow_id, payload, status
                 FROM proc.bp_workflow_run WHERE run_id = %s""",
            (run_id,),
        )
        r = cur.fetchone()
        cur.close()
        if not r:
            return None
        payload = r[2]
        if isinstance(payload, str):
            payload = json.loads(payload)
        return {
            "run_id": r[0], "agent_workflow_id": r[1],
            "payload": payload or {}, "status": r[3],
        }


def payload_for(run_id: str) -> Dict[str, Any]:
    """The run's ORIGINAL payload, as supplied on ``POST /{id}/run`` — never the
    human's answers, which are merged in separately by the caller."""
    run = get_run(run_id)
    return run["payload"] if run else {}


def claim_for_execution(run_id: str) -> bool:
    """Atomically claim this run for execution.

    A single conditional UPDATE — not a Python check followed by an UPDATE,
    which would race — so that when a final answer is replayed, or two
    answers to the last two outstanding questions land concurrently, only
    ONE caller ever transitions the run to 'executing' and therefore only
    one caller ever executes the workflow. Everyone else must treat the run
    as already in flight (or already finished) and must not execute again.

    A run that has been stuck at 'executing' for longer than
    STALE_EXECUTING_WINDOW is also claimable — see that constant's docstring
    for why, and for the correctness trade-off this reopens. This stays a
    SINGLE atomic conditional UPDATE ... RETURNING (services/db.py sets
    conn.autocommit = True, so a separate Python-side staleness check
    followed by an UPDATE would reintroduce the exact double-execution race
    this claim exists to prevent).
    """
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """UPDATE proc.bp_workflow_run
                  SET status = 'executing', updated_at = now()
                WHERE run_id = %s
                  AND (status NOT IN ('executing', 'completed')
                       OR (status = 'executing'
                           AND updated_at < now() - %s::interval))
            RETURNING run_id""",
            (run_id, f"{STALE_EXECUTING_WINDOW.total_seconds()} seconds"),
        )
        won = cur.fetchone() is not None
        cur.close()
        return won


def finish_run(run_id: str, status: str) -> None:
    """Record the terminal state ('completed' or 'failed') once execution ends."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """UPDATE proc.bp_workflow_run SET status = %s, updated_at = now()
                WHERE run_id = %s""",
            (status, run_id),
        )
        cur.close()


def raise_requests(
    run_id: str, requests: List[Any], *, agent_workflow_id: Optional[int] = None
) -> None:
    """Persist the questions this run needs answered (orchestration.elicitation.InputRequest).

    ``agent_workflow_id`` records which saved workflow this run belongs to, so
    the run can later be resolved back to it via ``workflow_id_for`` rather
    than by parsing the run_id string.
    """
    if not requests:
        return
    with get_conn() as conn:
        cur = conn.cursor()
        for r in requests:
            cur.execute(
                """INSERT INTO proc.bp_workflow_input_request
                       (workflow_id, node_name, agent_slug, required_field, field_type,
                        prompt, agent_workflow_id)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (run_id, r.node_id, r.agent_slug, r.required_field, r.field_type, r.prompt,
                 agent_workflow_id),
            )
        cur.close()


def workflow_id_for(run_id: str) -> Optional[int]:
    """The numeric proc.bp_agent_workflow id this run was started from, if known.

    Resolved from persisted state (the agent_workflow_id recorded by
    ``raise_requests``), never by parsing the run_id string.
    """
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT agent_workflow_id FROM proc.bp_workflow_input_request
                WHERE workflow_id = %s AND agent_workflow_id IS NOT NULL
                ORDER BY request_id LIMIT 1""",
            (run_id,),
        )
        r = cur.fetchone()
        cur.close()
        return int(r[0]) if r and r[0] is not None else None


def open_requests(run_id: str) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT request_id, node_name, agent_slug, required_field, field_type, prompt
                 FROM proc.bp_workflow_input_request
                WHERE workflow_id = %s AND status = 'pending'
                ORDER BY request_id""",
            (run_id,),
        )
        rows = cur.fetchall()
        cur.close()
        return [
            {"request_id": r[0], "node_name": r[1], "agent_slug": r[2],
             "required_field": r[3], "field_type": r[4], "prompt": r[5]}
            for r in rows
        ]


def request_run_id(request_id: int) -> Optional[str]:
    """Which run (``workflow_id``) this request belongs to, or None if the
    request_id does not exist at all. Lets a caller tell "this request_id
    was never raised" apart from "it belongs to a DIFFERENT run" apart from
    "it belongs to THIS run" -- three distinct cases that used to collapse
    into one silent no-op/blind-update."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT workflow_id FROM proc.bp_workflow_input_request WHERE request_id = %s",
            (request_id,),
        )
        r = cur.fetchone()
        cur.close()
        return r[0] if r else None


def answer(run_id: str, request_id: int, answer: Any, answered_by: str) -> bool:
    """Record the human's answer -- scoped to BOTH this request AND this run.

    ``proc.bp_workflow_input_request`` is the HITL audit trail: what a person
    was asked and what they answered. Without the ``workflow_id`` in the
    WHERE clause, any caller could answer (and silently rewrite the
    answered_at/answer/status of) a request belonging to a completely
    different, possibly already-completed, run just by guessing/reusing a
    request_id. Scoping the UPDATE makes that structurally impossible.

    Also scoped to ``status = 'pending'``: a request that has already been
    answered is left untouched (idempotent no-op on replay) rather than
    silently overwritten -- an already-completed run's trail must be
    immutable. The caller is responsible for telling "no-op because already
    answered" apart from "no-op because request_id doesn't belong to this
    run at all" (see ``request_run_id``) and responding accordingly.

    Returns True iff a row was actually updated.
    """
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """UPDATE proc.bp_workflow_input_request
                  SET answer = %s::jsonb, answered_by = %s, answered_at = now(), status = 'answered'
                WHERE request_id = %s AND workflow_id = %s AND status = 'pending'""",
            (json.dumps(answer), answered_by, request_id, run_id),
        )
        updated = cur.rowcount > 0
        cur.close()
        return updated


def answers_for(run_id: str) -> Dict[str, Any]:
    """{required_field: answer} for everything the human has already supplied."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT required_field, answer FROM proc.bp_workflow_input_request
                WHERE workflow_id = %s AND status = 'answered'""",
            (run_id,),
        )
        rows = cur.fetchall()
        cur.close()
        # psycopg2 auto-decodes JSONB columns into native Python objects, so `ans`
        # here is already the answer value (e.g. a str, not JSON-encoded text).
        # Do NOT json.loads() it again — that double-decode breaks on plain strings.
        return {field: ans for field, ans in rows}
