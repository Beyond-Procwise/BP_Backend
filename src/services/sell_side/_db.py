"""Shared plumbing for the sell-side services."""
from __future__ import annotations

from typing import Any

import psycopg2.extras


class NotFound(LookupError):
    """The row asked for does not exist. HTTP 404."""


class StateConflict(ValueError):
    """The row exists but is in a state that forbids this. HTTP 409."""


def dict_cursor(conn: Any):
    return conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
