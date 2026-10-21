"""The two reads behind the contract tools. Each returns rows or raises StoreUnavailable."""
from __future__ import annotations

from typing import Any, List, Sequence

from src.services.db import get_conn


class StoreUnavailable(RuntimeError):
    """The database could not be read; the caller must say so, not report "none found"."""


def rows(sql: str, params: Sequence[Any] = ()) -> List[tuple]:
    try:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute(sql, tuple(params))
            return list(cur.fetchall())
    except Exception as exc:  # noqa: BLE001 - any failure to read is an outage
        raise StoreUnavailable(str(exc)) from exc
