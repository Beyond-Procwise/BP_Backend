"""Refuse to touch production. Nothing in this package writes to a live database.

The refusal list is deliberately not overridable. If a future caller genuinely
needs to write to a live database, that is a different tool, not a flag here.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

LIVE_DATABASES = frozenset({"bp_sqldb", "uicanvas", "ses", "postgres", "rdsadmin"})


class UnsafeTargetError(RuntimeError):
    """Raised when an operation would touch a live database."""


def assert_safe_target(name: Any) -> None:
    """Raise unless `name` is a non-empty, non-live database name."""
    if not isinstance(name, str) or not name.strip():
        raise UnsafeTargetError(f"target database name is empty: {name!r}")
    normalised = name.strip().lower()
    if normalised in LIVE_DATABASES:
        raise UnsafeTargetError(
            f"the test-data generator refuses to target the live database {normalised!r}. "
            f"Use bp_testdb or uicanvas_test."
        )


def snapshot_counts(dbnames: Sequence[str]) -> dict[str, int]:
    """Row counts for every base table in `dbnames`, keyed '<db>.<schema>.<table>'."""
    from scripts.testdata.db import connect

    counts: dict[str, int] = {}
    for dbname in dbnames:
        conn = connect(dbname)
        try:
            with conn.cursor() as cur:
                # No parameters are bound here, so '%' is a literal and must not
                # be doubled -- psycopg2 only unescapes '%%' when args are passed.
                cur.execute(
                    """
                    select table_schema, table_name
                    from information_schema.tables
                    where table_type = 'BASE TABLE'
                      and table_schema not in ('information_schema', 'pg_catalog')
                      and table_schema not like 'pg_temp%'
                      and table_schema not like 'pg_toast%'
                    order by table_schema, table_name
                    """
                )
                relations = cur.fetchall()
                for schema, table in relations:
                    try:
                        cur.execute(f'select count(*) from "{schema}"."{table}"')
                        counts[f"{dbname}.{schema}.{table}"] = cur.fetchone()[0]
                    except Exception:
                        conn.rollback()
        finally:
            conn.close()
    return counts


def assert_live_unchanged(
    before: Mapping[str, int], after: Mapping[str, int]
) -> None:
    """Raise if any live table's row count moved between the two snapshots."""
    differences: list[str] = []
    for key in sorted(set(before) | set(after)):
        was = before.get(key)
        now = after.get(key)
        if was != now:
            differences.append(f"  {key}: {was} -> {now}")
    if differences:
        raise UnsafeTargetError(
            "live database row counts changed during the build:\n" + "\n".join(differences)
        )
