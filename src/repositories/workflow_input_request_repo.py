# src/repositories/workflow_input_request_repo.py
"""What the workflow asked the human, and what the human answered.

A HITL run is only trustworthy if you can see what a person supplied and when.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from services.db import get_conn

logger = logging.getLogger(__name__)

DDL = """
CREATE SCHEMA IF NOT EXISTS proc;

CREATE TABLE IF NOT EXISTS proc.bp_workflow_input_request (
    request_id     BIGSERIAL PRIMARY KEY,
    workflow_id    TEXT NOT NULL,          -- the RUN id
    node_name      TEXT NOT NULL,
    agent_slug     TEXT NOT NULL,
    required_field TEXT NOT NULL,
    field_type     TEXT,
    prompt         TEXT,
    status         TEXT NOT NULL DEFAULT 'pending',
    answer         JSONB,
    answered_by    TEXT,
    requested_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    answered_at    TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS ix_bp_workflow_input_request_open
    ON proc.bp_workflow_input_request (workflow_id, status);
"""


def ensure_schema() -> None:
    """Ensure the ``proc.bp_workflow_input_request`` table exists."""

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(DDL)
        cur.close()


def raise_requests(run_id: str, requests: List[Any]) -> None:
    """Persist the questions this run needs answered (orchestration.elicitation.InputRequest)."""
    if not requests:
        return
    with get_conn() as conn:
        cur = conn.cursor()
        for r in requests:
            cur.execute(
                """INSERT INTO proc.bp_workflow_input_request
                       (workflow_id, node_name, agent_slug, required_field, field_type, prompt)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (run_id, r.node_id, r.agent_slug, r.required_field, r.field_type, r.prompt),
            )
        cur.close()


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


def answer(request_id: int, answer: Any, answered_by: str) -> None:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """UPDATE proc.bp_workflow_input_request
                  SET answer = %s::jsonb, answered_by = %s, answered_at = now(), status = 'answered'
                WHERE request_id = %s""",
            (json.dumps(answer), answered_by, request_id),
        )
        cur.close()


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
