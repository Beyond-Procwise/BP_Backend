# src/repositories/agent_run_repo.py
"""What a created agent was told, what it did, and what it said — one row per run.

Before this a reasoning run left its answer and the names of the tools it used.
What it had been told (its instructions and the policies in front of it at that
moment), which model answered, and what each tool returned were gone the moment
the run ended, so nobody could later show why an agent said what it said.

The whole system prompt is stored as it was sent, not a pointer to the prompt
and policy rows it was built from: those rows change, and the record has to
show what the agent saw then.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from services.db import get_conn

logger = logging.getLogger(__name__)

DDL = """
CREATE SCHEMA IF NOT EXISTS proc;

CREATE TABLE IF NOT EXISTS proc.bp_agent_run (
    run_id         TEXT PRIMARY KEY,
    agent_slug     TEXT NOT NULL,
    workflow_id    TEXT,
    user_id        TEXT,
    model          TEXT,
    system_prompt  TEXT NOT NULL,
    system_sha256  TEXT NOT NULL,
    task           TEXT NOT NULL,
    tools_offered  JSONB NOT NULL,
    calls          JSONB NOT NULL,
    answer         TEXT,
    outputs        JSONB,
    rounds         INTEGER NOT NULL DEFAULT 0,
    error          TEXT,
    started_at     TIMESTAMPTZ NOT NULL,
    finished_at    TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_bp_agent_run_agent
    ON proc.bp_agent_run (agent_slug, started_at DESC);
CREATE INDEX IF NOT EXISTS ix_bp_agent_run_workflow
    ON proc.bp_agent_run (workflow_id);
"""

# A tool can return thousands of rows. The record keeps this much of each
# result, and the hash of all of it, and says when it kept less than the whole.
RESULT_MAX_CHARS = 20000


def ensure_schema() -> None:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(DDL)
        cur.close()


def new_run_id() -> str:
    return uuid.uuid4().hex


def _call_record(call: Any) -> Dict[str, Any]:
    try:
        text = json.dumps(call.result, ensure_ascii=False, default=str, sort_keys=True)
    except Exception:  # noqa: BLE001 - a result that will not serialise is still recorded
        text = str(call.result)
    return {
        "name": call.name,
        "arguments": call.arguments,
        "ok": call.ok,
        "error": call.error,
        "duration_ms": getattr(call, "duration_ms", 0),
        "result": text[:RESULT_MAX_CHARS],
        "result_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "result_truncated": len(text) > RESULT_MAX_CHARS,
    }


def record(
    *,
    run_id: str,
    agent_slug: str,
    workflow_id: Optional[str],
    user_id: Optional[str],
    model: Optional[str],
    system_prompt: str,
    task: str,
    tools_offered: List[str],
    calls: List[Any],
    answer: Optional[str],
    rounds: int,
    error: Optional[str],
    started_at: datetime,
    finished_at: datetime,
    outputs: Optional[Dict[str, Any]] = None,
) -> None:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO proc.bp_agent_run
                   (run_id, agent_slug, workflow_id, user_id, model, system_prompt,
                    system_sha256, task, tools_offered, calls, answer, outputs, rounds,
                    error, started_at, finished_at)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s::jsonb,
                       %s, %s, %s, %s)""",
            (
                run_id, agent_slug, workflow_id, user_id, model or None, system_prompt,
                hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(), task,
                json.dumps(list(tools_offered)),
                json.dumps([_call_record(c) for c in calls], default=str),
                answer or None,
                json.dumps(outputs, default=str) if outputs is not None else None,
                int(rounds or 0), error or None, started_at, finished_at,
            ),
        )
        cur.close()


_COLUMNS = (
    "run_id", "agent_slug", "workflow_id", "user_id", "model", "system_prompt",
    "system_sha256", "task", "tools_offered", "calls", "answer", "outputs", "rounds",
    "error", "started_at", "finished_at",
)


def _row(r, columns=_COLUMNS) -> Dict[str, Any]:
    out = dict(zip(columns, r))
    for key in ("tools_offered", "calls", "outputs"):
        if isinstance(out.get(key), str):
            out[key] = json.loads(out[key])
    for key in ("started_at", "finished_at"):
        if out.get(key) is not None:
            out[key] = out[key].isoformat()
    return out


def get(run_id: str) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM proc.bp_agent_run WHERE run_id = %s", (run_id,)
        )
        row = cur.fetchone()
        cur.close()
    return _row(row) if row else None


_LIST_COLUMNS = (
    "run_id", "agent_slug", "workflow_id", "user_id", "model", "rounds", "error",
    "started_at", "finished_at",
)


def list_for_agent(agent_slug: str, limit: int = 50) -> List[Dict[str, Any]]:
    """The runs of one agent, newest first — without the prompt, task and results."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            f"""SELECT {', '.join(_LIST_COLUMNS)} FROM proc.bp_agent_run
                 WHERE agent_slug = %s ORDER BY started_at DESC LIMIT %s""",
            (agent_slug, max(1, min(int(limit), 200))),
        )
        rows = cur.fetchall()
        cur.close()
    return [_row(r, _LIST_COLUMNS) for r in rows]
