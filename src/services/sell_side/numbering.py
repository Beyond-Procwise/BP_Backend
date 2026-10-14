"""The next number for a customer document, with none missing (QDA B6).

One counter per document type per year (proc.bp_doc_number_counter). `allocate` takes the
next value on the caller's transaction and returns it; nothing is committed here. That is
the whole of the gap-free guarantee:

  * the row is locked from the moment it is incremented until the caller's transaction
    ends, so two issues at once queue and take consecutive values;
  * if the caller rolls back -- the issue failed for any reason -- the increment rolls back
    with it and the next issue is given that same number.

So it must be called inside the transaction that issues the document, and never on an
autocommit connection, where the increment would commit by itself and a failed issue would
burn its number. `sell_side._db.dict_cursor` already refuses an autocommit connection.

The year is the database's, read in the same transaction that stamps issued_at, so a quote
issued at midnight on New Year's Eve cannot be numbered in one year and dated in the other.
"""
from __future__ import annotations

from typing import Any, Tuple


def allocate(cur: Any, doc_type: str) -> Tuple[int, int]:
    """(year, n): the next number for `doc_type` this year. Uncommitted."""
    cur.execute(
        "INSERT INTO proc.bp_doc_number_counter (doc_type, year, last_value) "
        "VALUES (%s, EXTRACT(YEAR FROM now())::int, 1) "
        "ON CONFLICT (doc_type, year) DO UPDATE "
        "SET last_value = proc.bp_doc_number_counter.last_value + 1, "
        "    last_modified_date = now() "
        "RETURNING year, last_value", (doc_type,))
    row = cur.fetchone()
    return int(row["year"]), int(row["last_value"])


def formatted(prefix: str, year: int, n: int) -> str:
    """Q-2026-0007. Four digits by default; a longer series simply grows."""
    return f"{prefix}-{year}-{n:04d}"
