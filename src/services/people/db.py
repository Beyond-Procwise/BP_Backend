"""One explicit transaction on an AUTOCOMMIT connection (get_conn()): commit, or roll back on error."""
from __future__ import annotations

from contextlib import contextmanager


@contextmanager
def tx(conn):
    prev = getattr(conn, "autocommit", False)
    if prev:
        conn.autocommit = False
    try:
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        if prev:
            conn.autocommit = True
