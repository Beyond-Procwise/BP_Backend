"""Clone live database structure into the test databases.

Structure only: pg_dump --schema-only. No business data crosses over here;
reference data is copied separately and explicitly in reference.py.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

from config.settings import Settings
from scripts.testdata.db import connect, create_database
from scripts.testdata.guards import assert_safe_target

SCHEMA_PAIRS: tuple[tuple[str, str], ...] = (
    ("bp_sqldb", "bp_testdb"),
    ("uicanvas", "uicanvas_test"),
)


@dataclass(frozen=True)
class CloneReport:
    tables: int
    views: int
    functions: int
    triggers: int


def _pg_env(settings: Settings) -> dict[str, str]:
    env = dict(os.environ)
    env["PGPASSWORD"] = settings.db_password
    return env


def clone_schema(
    source_db: str, target_db: str, *, drop_first: bool = False
) -> CloneReport:
    """Reproduce source_db's structure in target_db. Refuses live targets."""
    assert_safe_target(target_db)
    settings = Settings()

    create_database(target_db, drop_first=drop_first)

    dump = subprocess.run(
        [
            "pg_dump",
            "--schema-only",
            "--no-owner",
            "--no-privileges",
            "-h", settings.db_host,
            "-p", str(settings.db_port),
            "-U", settings.db_user,
            "-d", source_db,
        ],
        env=_pg_env(settings),
        capture_output=True,
        text=True,
        check=True,
    )

    restore = subprocess.run(
        [
            "psql",
            "-h", settings.db_host,
            "-p", str(settings.db_port),
            "-U", settings.db_user,
            "-d", target_db,
            "-v", "ON_ERROR_STOP=0",
            "-f", "-",
        ],
        input=dump.stdout,
        env=_pg_env(settings),
        capture_output=True,
        text=True,
    )
    if restore.returncode not in (0, 3):
        raise RuntimeError(f"schema restore into {target_db} failed:\n{restore.stderr}")

    return _describe(target_db)


def _describe(dbname: str) -> CloneReport:
    conn = connect(dbname)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "select count(*) from information_schema.tables "
                "where table_type = 'BASE TABLE' "
                "and table_schema not in ('information_schema', 'pg_catalog')"
            )
            tables = cur.fetchone()[0]

            cur.execute(
                "select count(*) from information_schema.views "
                "where table_schema not in ('information_schema', 'pg_catalog')"
            )
            views = cur.fetchone()[0]

            cur.execute(
                "select count(*) from pg_proc p join pg_namespace n on n.oid = p.pronamespace "
                "where n.nspname not in ('information_schema', 'pg_catalog')"
            )
            functions = cur.fetchone()[0]

            cur.execute("select count(*) from pg_trigger where not tgisinternal")
            triggers = cur.fetchone()[0]
    finally:
        conn.close()
    return CloneReport(tables=tables, views=views, functions=functions, triggers=triggers)
