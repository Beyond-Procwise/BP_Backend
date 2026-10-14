# src/repositories/agent_group_repo.py
"""Saved agent groups — the sets of agents a user keeps putting on the canvas together.

Deliberately NOT proc.bp_agent_workflow. That table's graph is validated as a runnable
DAG (exactly one entry node, no cycles, see workflow_compiler.validate_saved_graph); a
selection of agents almost never satisfies that, and does not need to — a group is
stamped back onto a canvas, never compiled and never run.

Positions are stored RELATIVE to the selection's top-left, so a group is a shape rather
than a location, and links reference members by index so a stamped copy needs no id
rewriting.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from services.db import get_conn

logger = logging.getLogger(__name__)

DDL = """
CREATE SCHEMA IF NOT EXISTS proc;

CREATE TABLE IF NOT EXISTS proc.bp_agent_group (
    group_id    BIGSERIAL PRIMARY KEY,
    name        TEXT NOT NULL,
    members     JSONB NOT NULL,
    links       JSONB NOT NULL DEFAULT '[]'::jsonb,
    is_active   BOOLEAN NOT NULL DEFAULT TRUE,
    created_by  TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_bp_agent_group_active
    ON proc.bp_agent_group (is_active, updated_at DESC);
"""


def ensure_schema() -> None:
    """Ensure the ``proc.bp_agent_group`` table exists."""

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(DDL)
        cur.close()


def _json(value):
    return json.loads(value) if isinstance(value, str) else value


def _row(r) -> Dict[str, Any]:
    return {
        "group_id": r[0], "name": r[1],
        "members": _json(r[2]), "links": _json(r[3]),
        "created_by": r[4],
        "created_at": r[5].isoformat() if r[5] else None,
        "updated_at": r[6].isoformat() if r[6] else None,
    }


_SELECT = """SELECT group_id, name, members, links, created_by, created_at, updated_at
               FROM proc.bp_agent_group"""


def create(name: str, members: List[Dict[str, Any]], links: List[Dict[str, Any]],
           created_by: Optional[str] = None) -> int:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO proc.bp_agent_group (name, members, links, created_by)
               VALUES (%s, %s::jsonb, %s::jsonb, %s) RETURNING group_id""",
            (name, json.dumps(members), json.dumps(links), created_by),
        )
        gid = cur.fetchone()[0]
        cur.close()
        return int(gid)


def list_active() -> List[Dict[str, Any]]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(_SELECT + " WHERE is_active ORDER BY updated_at DESC")
        rows = cur.fetchall()
        cur.close()
        return [_row(r) for r in rows]


def get(group_id: int) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(_SELECT + " WHERE group_id = %s AND is_active", (group_id,))
        r = cur.fetchone()
        cur.close()
        return _row(r) if r else None


def update(group_id: int, *, name: Optional[str] = None,
           members: Optional[List[Dict[str, Any]]] = None,
           links: Optional[List[Dict[str, Any]]] = None) -> None:
    sets, vals = [], []
    if name is not None:
        sets.append("name = %s"); vals.append(name)
    if members is not None:
        sets.append("members = %s::jsonb"); vals.append(json.dumps(members))
    if links is not None:
        sets.append("links = %s::jsonb"); vals.append(json.dumps(links))
    if not sets:
        return
    sets.append("updated_at = now()")
    vals.append(group_id)
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            f"UPDATE proc.bp_agent_group SET {', '.join(sets)} WHERE group_id = %s",
            tuple(vals),
        )
        cur.close()


def soft_delete(group_id: int) -> None:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE proc.bp_agent_group SET is_active = FALSE, updated_at = now() WHERE group_id = %s",
            (group_id,),
        )
        cur.close()
