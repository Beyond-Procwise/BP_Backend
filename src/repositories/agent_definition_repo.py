# src/repositories/agent_definition_repo.py
"""Created agents — the ones a person made in the workspace.

They used to be appended to ``agent_definitions.json``, a file tracked in git.
A redeploy replaced the file and the agents with it; nothing recorded who made
one or what it looked like before it was changed; and two server processes
writing the file at once could lose an agent between them.

The built-in agents stay in the file, where a release ships them. A created
agent is a row here, and every change to it is a row in the version table.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from services.db import get_conn

logger = logging.getLogger(__name__)

DDL = """
CREATE SCHEMA IF NOT EXISTS proc;

CREATE TABLE IF NOT EXISTS proc.bp_agent_definition (
    slug          TEXT PRIMARY KEY,
    definition    JSONB NOT NULL,
    derived_from  TEXT NOT NULL,
    version       INTEGER NOT NULL DEFAULT 1,
    status        SMALLINT NOT NULL DEFAULT 1,      -- 1 live, 0 deleted
    created_by    TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_by    TEXT,
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_bp_agent_definition_status
    ON proc.bp_agent_definition (status);

CREATE TABLE IF NOT EXISTS proc.bp_agent_definition_version (
    slug        TEXT NOT NULL,
    version     INTEGER NOT NULL,
    change      TEXT NOT NULL
                CHECK (change IN ('created', 'updated', 'published', 'deleted', 'imported')),
    definition  JSONB NOT NULL,
    changed_by  TEXT,
    changed_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (slug, version)
);
"""

# How long one read of the table is reused. A write in this process clears it
# at once; this is how long ANOTHER process can go without seeing that write.
CACHE_SECONDS = 30.0

_lock = threading.Lock()
_cached: Optional[Tuple[List[Dict[str, Any]], Set[str]]] = None
_cached_at = 0.0


class AgentExists(Exception):
    """A live created agent already has this slug."""


class AgentNotFound(Exception):
    """No live created agent has this slug."""


def ensure_schema() -> None:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(DDL)
        cur.close()


def invalidate() -> None:
    global _cached, _cached_at
    with _lock:
        _cached = None
        _cached_at = 0.0


def _as_dict(value: Any) -> Dict[str, Any]:
    return json.loads(value) if isinstance(value, str) else dict(value or {})


def _read() -> Tuple[List[Dict[str, Any]], Set[str]]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT slug, definition, status FROM proc.bp_agent_definition ORDER BY created_at, slug"
        )
        rows = cur.fetchall()
        cur.close()
    live: List[Dict[str, Any]] = []
    deleted: Set[str] = set()
    for slug, definition, status in rows:
        if status == 1:
            entry = _as_dict(definition)
            entry["slug"] = slug          # the key is the row's, whatever the document says
            live.append(entry)
        else:
            deleted.add(slug)
    return live, deleted


def snapshot() -> Tuple[List[Dict[str, Any]], Set[str]]:
    """``(live created agents, slugs that were deleted)``.

    If the table cannot be read, the last good read is served: a database blip
    must not make every created agent vanish from a running system. With no
    good read to fall back on, the error is raised for the caller to decide.
    """
    global _cached, _cached_at
    with _lock:
        if _cached is not None and time.monotonic() - _cached_at < CACHE_SECONDS:
            return _cached
    try:
        fresh = _read()
    except Exception:
        with _lock:
            if _cached is not None:
                logger.error(
                    "created agents could not be read; serving the last good read",
                    exc_info=True,
                )
                return _cached
        raise
    with _lock:
        _cached, _cached_at = fresh, time.monotonic()
    return fresh


def _write(sql: str, params: tuple) -> Optional[int]:
    """Run one statement that changes the agent row and records the version in
    the same breath, and return the new version (None if no row was changed).

    One statement, not two: connections here commit each statement on its own,
    so an agent row written apart from its history row could be left without it.
    """
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(sql, params)
        row = cur.fetchone()
        cur.close()
    invalidate()
    return int(row[0]) if row else None


_RECORD = """
INSERT INTO proc.bp_agent_definition_version (slug, version, change, definition, changed_by)
SELECT slug, version, %s, definition, %s FROM changed
RETURNING version
"""


def create(slug: str, definition: Dict[str, Any], *, derived_from: str,
           by: Optional[str]) -> int:
    """Store a new created agent. A slug that was deleted may be used again;
    its history carries on from where it stopped."""
    version = _write(
        """
        WITH changed AS (
            INSERT INTO proc.bp_agent_definition AS d
                (slug, definition, derived_from, created_by, updated_by)
            VALUES (%s, %s::jsonb, %s, %s, %s)
            ON CONFLICT (slug) DO UPDATE
               SET definition = EXCLUDED.definition,
                   derived_from = EXCLUDED.derived_from,
                   status = 1,
                   version = d.version + 1,
                   created_by = EXCLUDED.created_by,
                   created_at = now(),
                   updated_by = EXCLUDED.updated_by,
                   updated_at = now()
             WHERE d.status = 0
            RETURNING slug, version, definition
        )""" + _RECORD,
        (slug, json.dumps(definition), derived_from, by, by, "created", by),
    )
    if version is None:
        raise AgentExists(slug)
    return version


def update(slug: str, definition: Dict[str, Any], *, by: Optional[str]) -> int:
    version = _write(
        """
        WITH changed AS (
            UPDATE proc.bp_agent_definition
               SET definition = %s::jsonb, version = version + 1,
                   updated_by = %s, updated_at = now()
             WHERE slug = %s AND status = 1
            RETURNING slug, version, definition
        )""" + _RECORD,
        (json.dumps(definition), by, slug, "updated", by),
    )
    if version is None:
        raise AgentNotFound(slug)
    return version


def publish(slug: str, definition: Dict[str, Any], *, by: Optional[str]) -> int:
    """Store the published form of a draft, recorded as a publication and not
    as an ordinary change, so the history shows who released the agent."""
    version = _write(
        """
        WITH changed AS (
            UPDATE proc.bp_agent_definition
               SET definition = %s::jsonb, version = version + 1,
                   updated_by = %s, updated_at = now()
             WHERE slug = %s AND status = 1
            RETURNING slug, version, definition
        )""" + _RECORD,
        (json.dumps(definition), by, slug, "published", by),
    )
    if version is None:
        raise AgentNotFound(slug)
    return version


def delete(slug: str, definition: Dict[str, Any], *, derived_from: str,
           by: Optional[str]) -> int:
    """Mark a created agent deleted. The row stays, so its history stays, and so
    a copy of the agent still sitting in the catalogue file stays hidden."""
    version = _write(
        """
        WITH changed AS (
            INSERT INTO proc.bp_agent_definition AS d
                (slug, definition, derived_from, status, created_by, updated_by)
            VALUES (%s, %s::jsonb, %s, 0, NULL, %s)
            ON CONFLICT (slug) DO UPDATE
               SET status = 0, version = d.version + 1,
                   updated_by = EXCLUDED.updated_by, updated_at = now()
             WHERE d.status = 1
            RETURNING slug, version, definition
        )""" + _RECORD,
        (slug, json.dumps(definition), derived_from, by, "deleted", by),
    )
    if version is None:
        raise AgentNotFound(slug)
    return version


def import_from_file(entries: List[Dict[str, Any]]) -> List[str]:
    """Take in the created agents an older version left in the catalogue file.

    Only a slug this table has never seen is taken: one that is here already,
    live or deleted, is this table's to answer for. Returns the slugs taken in.
    """
    taken: List[str] = []
    for entry in entries:
        slug = entry.get("slug")
        derived_from = entry.get("derived_from")
        if not slug or not derived_from:
            continue
        by = entry.get("created_by")
        by = None if by in (None, "", "workspace") else str(by)
        version = _write(
            """
            WITH changed AS (
                INSERT INTO proc.bp_agent_definition
                    (slug, definition, derived_from, created_by, updated_by)
                VALUES (%s, %s::jsonb, %s, %s, %s)
                ON CONFLICT (slug) DO NOTHING
                RETURNING slug, version, definition
            )""" + _RECORD,
            (slug, json.dumps(entry), derived_from, by, by, "imported", by),
        )
        if version is not None:
            taken.append(slug)
    return taken


def install_as_source() -> List[str]:
    """Make this table where the catalogue's created agents come from.

    Called once at startup, BEFORE the agent registry is built. Created agents
    an older version left in the catalogue file are taken in first, so none is
    lost by the move. Returns the slugs taken in.
    """
    from agents import definitions

    ensure_schema()
    taken = import_from_file(
        [e for e in definitions._load_file() if e.get("derived_from")]
    )
    definitions.set_created_agent_source(snapshot)
    return taken


def history(slug: str) -> List[Dict[str, Any]]:
    """Every recorded change to one created agent, newest first."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT version, change, definition, changed_by, changed_at
                 FROM proc.bp_agent_definition_version
                WHERE slug = %s ORDER BY version DESC""",
            (slug,),
        )
        rows = cur.fetchall()
        cur.close()
    return [
        {
            "version": r[0], "change": r[1], "definition": _as_dict(r[2]),
            "changed_by": r[3], "changed_at": r[4].isoformat() if r[4] else None,
        }
        for r in rows
    ]
