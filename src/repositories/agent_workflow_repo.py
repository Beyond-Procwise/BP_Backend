# src/repositories/agent_workflow_repo.py
"""Saved agent workflows — the graphs a user drew on the canvas.

Until now the canvas had nowhere to go: Save deep-copied into a JS array and died
on page reload. This is where a drawn workflow actually lives.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from services.db import get_conn

logger = logging.getLogger(__name__)

DDL = """
CREATE SCHEMA IF NOT EXISTS proc;

CREATE TABLE IF NOT EXISTS proc.bp_agent_workflow (
    workflow_id  BIGSERIAL PRIMARY KEY,
    name         TEXT NOT NULL,
    description  TEXT,
    graph        JSONB NOT NULL,
    entry_node   TEXT NOT NULL,
    is_active    BOOLEAN NOT NULL DEFAULT TRUE,
    created_by   TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_bp_agent_workflow_active
    ON proc.bp_agent_workflow (is_active, updated_at DESC);
"""


def ensure_schema() -> None:
    """Ensure the ``proc.bp_agent_workflow`` table exists."""

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(DDL)
        cur.close()


def _row(r) -> Dict[str, Any]:
    graph = r[3]
    if isinstance(graph, str):
        graph = json.loads(graph)
    return {
        "workflow_id": r[0], "name": r[1], "description": r[2], "graph": graph,
        "entry_node": r[4], "created_by": r[5],
        "created_at": r[6].isoformat() if r[6] else None,
        "updated_at": r[7].isoformat() if r[7] else None,
    }


_SELECT = """SELECT workflow_id, name, description, graph, entry_node, created_by,
                    created_at, updated_at
               FROM proc.bp_agent_workflow"""


def create(name: str, graph: Dict[str, Any], entry_node: str,
           description: str = "", created_by: Optional[str] = None) -> int:
    """``created_by`` is who saved it (a principal's subject), or None. It
    defaulted to "system", and the canvas never passed one, so every saved
    workflow claimed to be created by somebody called "system"."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO proc.bp_agent_workflow (name, description, graph, entry_node, created_by)
               VALUES (%s, %s, %s::jsonb, %s, %s) RETURNING workflow_id""",
            (name, description, json.dumps(graph), entry_node, created_by),
        )
        wid = cur.fetchone()[0]
        cur.close()
        return int(wid)


def get(workflow_id: int) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(_SELECT + " WHERE workflow_id = %s AND is_active", (workflow_id,))
        r = cur.fetchone()
        cur.close()
        return _row(r) if r else None


def list_active() -> List[Dict[str, Any]]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(_SELECT + " WHERE is_active ORDER BY updated_at DESC")
        rows = cur.fetchall()
        cur.close()
        return [_row(r) for r in rows]


def update(workflow_id: int, *, name: Optional[str] = None,
           graph: Optional[Dict[str, Any]] = None,
           entry_node: Optional[str] = None,
           description: Optional[str] = None) -> None:
    sets, vals = [], []
    if name is not None:
        sets.append("name = %s"); vals.append(name)
    if description is not None:
        sets.append("description = %s"); vals.append(description)
    if graph is not None:
        sets.append("graph = %s::jsonb"); vals.append(json.dumps(graph))
    if entry_node is not None:
        sets.append("entry_node = %s"); vals.append(entry_node)
    if not sets:
        return
    sets.append("updated_at = now()")
    vals.append(workflow_id)
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(f"UPDATE proc.bp_agent_workflow SET {', '.join(sets)} WHERE workflow_id = %s", vals)
        cur.close()


def soft_delete(workflow_id: int) -> None:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE proc.bp_agent_workflow SET is_active = FALSE, updated_at = now() WHERE workflow_id = %s",
            (workflow_id,),
        )
        cur.close()
