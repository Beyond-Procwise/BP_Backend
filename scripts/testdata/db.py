"""Connections and bulk loading for the test-data generator.

Credentials come from config.settings.Settings, matching the pattern in
src/services/uicanvas_bridge.py. Only the database name varies.
"""
from __future__ import annotations

import csv
import io
from typing import Any, Iterable, Sequence

import psycopg2
import psycopg2.extensions
import psycopg2.extras

from config.settings import Settings
from scripts.testdata.guards import assert_safe_target

_COPY_BUFFER_ROWS = 5000

# Read json and jsonb back as the raw text the server sent. The default adapter
# decodes them to Python dicts, and str(dict) is Python repr -- single quotes,
# True/False -- which COPY then rejects as invalid JSON. Passing the text
# through untouched is also what "copy reference data verbatim" should mean.
psycopg2.extras.register_default_json(globally=True, loads=lambda text: text)
psycopg2.extras.register_default_jsonb(globally=True, loads=lambda text: text)


def connect(dbname: str, *, autocommit: bool = False) -> psycopg2.extensions.connection:
    """Open a connection. Reading a live database is allowed; writing is not."""
    settings = Settings()
    conn = psycopg2.connect(
        host=settings.db_host,
        dbname=dbname,
        user=settings.db_user,
        password=settings.db_password,
        port=settings.db_port,
        connect_timeout=30,
    )
    conn.autocommit = autocommit
    return conn


def database_exists(dbname: str) -> bool:
    conn = connect("postgres", autocommit=True)
    try:
        with conn.cursor() as cur:
            cur.execute("select 1 from pg_database where datname = %s", (dbname,))
            return cur.fetchone() is not None
    finally:
        conn.close()


def create_database(dbname: str, *, drop_first: bool = False) -> None:
    """Create the target database. Refuses live names, with or without drop_first."""
    assert_safe_target(dbname)
    conn = connect("postgres", autocommit=True)
    try:
        with conn.cursor() as cur:
            if drop_first:
                cur.execute(
                    "select pg_terminate_backend(pid) from pg_stat_activity "
                    "where datname = %s and pid <> pg_backend_pid()",
                    (dbname,),
                )
                cur.execute(f'drop database if exists "{dbname}"')
            cur.execute("select 1 from pg_database where datname = %s", (dbname,))
            if cur.fetchone() is None:
                cur.execute(f'create database "{dbname}"')
    finally:
        conn.close()


def _array_literal(values: Sequence[Any]) -> str:
    """Render a Python sequence as a Postgres array literal.

    psycopg2 hands array columns back as Python lists, whose str() is '[1, 2]'.
    Postgres wants '{1,2}', with every element quoted so that commas, quotes and
    backslashes inside a text element survive the round trip.
    """
    parts: list[str] = []
    for value in values:
        if value is None:
            parts.append("NULL")
            continue
        if isinstance(value, (list, tuple)):
            parts.append(_array_literal(value))
            continue
        text = str(value).replace("\\", "\\\\").replace('"', '\\"')
        parts.append(f'"{text}"')
    return "{" + ",".join(parts) + "}"


def _render(value: Any) -> Any:
    """Convert a psycopg2-decoded value into something COPY ... CSV accepts."""
    if value is None:
        return "\\N"
    if isinstance(value, (list, tuple)):
        return _array_literal(value)
    if isinstance(value, dict):
        # Only reachable if a caller passes a dict directly; json/jsonb read
        # through connect() already arrive as text.
        import json

        return json.dumps(value)
    return value


def copy_rows(
    conn: psycopg2.extensions.connection,
    schema: str,
    table: str,
    columns: Sequence[str],
    rows: Iterable[Sequence[Any]],
) -> int:
    """Bulk-insert via COPY FROM STDIN. Returns the number of rows written.

    None becomes SQL NULL. Buffered so a multi-hundred-thousand-row table does
    not have to be materialised in memory as one string.
    """
    column_list = ", ".join(f'"{c}"' for c in columns)
    copy_sql = (
        f'copy "{schema}"."{table}" ({column_list}) '
        f"from stdin with (format csv, null '\\N')"
    )

    written = 0
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    pending = 0

    def flush() -> None:
        nonlocal pending
        if not pending:
            return
        buffer.seek(0)
        with conn.cursor() as cur:
            cur.copy_expert(copy_sql, buffer)
        buffer.seek(0)
        buffer.truncate(0)
        pending = 0

    for row in rows:
        writer.writerow([_render(value) for value in row])
        written += 1
        pending += 1
        if pending >= _COPY_BUFFER_ROWS:
            flush()
    flush()
    return written
