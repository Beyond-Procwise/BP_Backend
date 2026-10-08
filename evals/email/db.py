"""A throwaway Postgres for the evals, loaded with the real tables and the real migrations.

Never the shared databases. ``EMAIL_EVAL_DSN`` points at an already-running server (CI uses a
service container); otherwise a ``postgres:16-alpine`` container is started on a free port and
removed afterwards. With neither Docker nor a DSN there is nothing to run against, and that is an
error for CI and a skip for a developer's pytest run.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Optional

HERE = Path(__file__).parent
REPO = HERE.parents[1]
SQL = REPO / "deploy" / "sql"

#: The real migrations, in the order they must run. The evals therefore also prove they load.
MIGRATIONS: List[str] = [
    "2026-10-07_email_agent_capture.sql",
    "2026-10-08_email_agent_capture_v2.sql",
    "2026-10-07_email_family_negotiation_counter.sql",
    "2026-10-07_email_family_free_prompt.sql",
    "2026-10-08_email_family_v2.sql",
    "2026-10-08_email_tone_rules.sql",
    "2026-10-08_email_assurance_prompts.sql",
    "2026-10-09_email_agent_learning.sql",
    "2026-10-09_email_agent_roles.sql",
]


class NoDatabase(RuntimeError):
    """Neither EMAIL_EVAL_DSN nor a usable Docker daemon."""


def _docker_ok() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode == 0
    except Exception:  # noqa: BLE001
        return False


@contextmanager
def database() -> Iterator[str]:
    """Yield a DSN for an empty, disposable Postgres."""

    dsn = os.environ.get("EMAIL_EVAL_DSN")
    if dsn:
        yield dsn
        return
    if not _docker_ok():
        raise NoDatabase("set EMAIL_EVAL_DSN, or run where Docker is available")
    name = "email-eval-" + uuid.uuid4().hex[:8]
    subprocess.run(["docker", "run", "-d", "--rm", "--name", name, "-e", "POSTGRES_PASSWORD=eval",
                    "-p", "127.0.0.1::5432", "postgres:16-alpine"], check=True, capture_output=True)
    try:
        port = subprocess.run(["docker", "port", name, "5432/tcp"], check=True, capture_output=True, text=True
                              ).stdout.strip().splitlines()[0].rsplit(":", 1)[1]
        dsn = f"postgresql://postgres:eval@127.0.0.1:{port}/postgres"
        import psycopg2
        for _ in range(60):
            try:
                psycopg2.connect(dsn, connect_timeout=2).close()
                break
            except Exception:  # noqa: BLE001 - still starting
                time.sleep(0.5)
        else:
            raise NoDatabase("the eval database did not start")
        yield dsn
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def existing_schema() -> bool:
    """True when EMAIL_EVAL_EXISTING_SCHEMA=1: the database already holds the real tables (and the real
    policy rows), as in the bp_sqldb rehearsal, so nothing is generated or seeded here."""
    return os.environ.get("EMAIL_EVAL_EXISTING_SCHEMA", "").strip() in ("1", "true")


def load(conn, skip=(), generate=None) -> None:
    """Create the tables from schema.sql, seed the two authority policies, run the real migrations.

    ``skip`` names migrations to leave out, for a test that needs a database a migration has never touched.
    With EMAIL_EVAL_EXISTING_SCHEMA=1 the tables and authority policies are NOT created: they come from the
    database itself (a restored copy of the real schema), and only the migrations are applied.
    ``generate=True`` forces the generated tables regardless (for a probe database built from nothing).
    """

    conn.autocommit = True
    with conn.cursor() as cur:
        if generate if generate is not None else not existing_schema():
            # Start clean. In CI the golden runner and the pytest session share ONE server, so the second
            # load found the first's tables ("relation bp_policy already exists"). Only ever a throwaway
            # database reaches this branch: the existing-schema rehearsal path above never drops anything.
            cur.execute("DROP SCHEMA IF EXISTS proc, email_agent CASCADE")
            cur.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public")
            cur.execute((HERE / "schema.sql").read_text())
            cur.execute((HERE / "seed_policies.sql").read_text())
        for name in MIGRATIONS:
            if name not in skip:
                cur.execute((SQL / name).read_text())


def rollback_all(conn) -> None:
    """Run every rollback in reverse order (used to prove the migrations are reversible)."""

    conn.autocommit = True
    with conn.cursor() as cur:
        for name in reversed(MIGRATIONS):
            cur.execute((SQL / name.replace(".sql", "_rollback.sql")).read_text())
