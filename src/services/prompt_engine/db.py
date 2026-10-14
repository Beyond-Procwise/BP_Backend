"""One kept-open database connection per thread, for the prompt engine.

A render is called on every agent step, and ``services.db.get_conn`` opens a
new connection for every statement: measured at about 100 ms each against the
shared database, which made a single pointer lookup cost 200 ms. Here a thread
keeps its connection, so a statement costs a round trip and nothing more.

Rules it keeps:
  * autocommit, like ``get_conn``: each statement stands alone.
  * a connection found closed is reopened before the statement is sent.
  * a READ that fails on a broken connection is retried once on a new one. A
    WRITE is never retried here: it may have been applied before the
    connection broke, and writing it twice is the caller's decision to make.
  * under pytest without the live-database switch it defers to ``get_conn`` and
    its in-memory stand-in, so the unit suite opens no real connection.
"""
from __future__ import annotations

import threading
from typing import Any, List, Optional

from services import db as _shared

_local = threading.local()


def _connect() -> Any:
    dsn = _shared._pg_dsn()
    conn = _shared.psycopg2.connect(dsn)
    conn.autocommit = True
    return conn


def _connection() -> Any:
    conn = getattr(_local, "conn", None)
    if conn is None or conn.closed:
        conn = _local.conn = _connect()
    return conn


def close() -> None:
    """Close this thread's connection. The next statement opens a new one."""
    conn = getattr(_local, "conn", None)
    _local.conn = None
    if conn is not None:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass


def _run(conn: Any, sql: str, params: tuple) -> List[tuple]:
    cur = conn.cursor()
    try:
        cur.execute(sql, params)
        return cur.fetchall() if cur.description else []
    finally:
        cur.close()


def query(sql: str, params: tuple = (), *, write: bool = False) -> List[tuple]:
    """Run one statement and return its rows."""
    dsn: Optional[str] = _shared._pg_dsn()
    if _shared._use_fake_conn(dsn) or not dsn or _shared.psycopg2 is None:
        with _shared.get_conn() as conn:
            return _run(conn, sql, params)
    try:
        return _run(_connection(), sql, params)
    except (_shared.psycopg2.OperationalError, _shared.psycopg2.InterfaceError):
        close()
        if write:
            raise
        return _run(_connection(), sql, params)


def available() -> bool:
    """False in a process that has no real database: the unit suite, whose
    in-memory stand-in does not hold the registry's tables."""
    dsn = _shared._pg_dsn()
    return bool(dsn) and _shared.psycopg2 is not None and not _shared._use_fake_conn(dsn)
