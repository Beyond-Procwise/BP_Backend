"""Shared plumbing for the sell-side services.

Every write function here does `except Exception: conn.rollback(); raise` around a
multi-statement transaction, and `record_outcome` takes a `SELECT ... FOR UPDATE`
lock to serialise the quote state machine. Both of those depend on the connection
actually being transactional: under `conn.autocommit = True` (what `db.get_conn()`
hands out), `rollback()` is a no-op and a `FOR UPDATE` lock is released the instant
its statement completes, not held for the transaction. `dict_cursor` refuses an
autocommit connection outright so that failure mode is caught at the first cursor
use rather than silently producing non-atomic writes and unenforced locks; callers
that don't already have a transactional connection should get one from
`transactional_conn()` below.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import psycopg2.extras


class NotFound(LookupError):
    """The row asked for does not exist. HTTP 404."""


class StateConflict(ValueError):
    """The row exists but is in a state that forbids this. HTTP 409."""


def dict_cursor(conn: Any):
    if getattr(conn, "autocommit", False):
        raise RuntimeError("sell-side services need a transactional connection "
                           "(autocommit is on); use sell_side._db.transactional_conn()")
    return conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)


@contextmanager
def transactional_conn():
    """get_conn() with autocommit OFF. get_conn() opens autocommit connections, under which
    rollback is a no-op and FOR UPDATE locks end with the statement."""
    from src.services.db import get_conn
    with get_conn() as conn:
        conn.autocommit = False
        try:
            yield conn
        finally:
            try:
                conn.rollback()   # end any transaction the caller left open; no-op after commit
            except Exception:
                pass
