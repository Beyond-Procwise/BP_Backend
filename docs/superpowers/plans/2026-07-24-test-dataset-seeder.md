# Test Dataset Seeder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic generator that populates two isolated test databases with 5,000 suppliers, 6 buying entities, the real 5-level category taxonomy, ~40,000 documents and 30 planted defects, and verifies the result against 14 checks.

**Architecture:** A `scripts/testdata/` package with one CLI entry point. Pure-logic modules (row builders, apportionment, defect planting) are unit-tested without a database; DB-touching modules are integration-tested against a scratch database. Every module takes an injected `random.Random` seeded per-stream so tasks cannot perturb each other's output. Data is written with `COPY FROM STDIN` for bulk speed.

**Tech Stack:** Python 3.12 (`.venv`), psycopg2, pytest, `config.settings.Settings` for credentials.

**Spec:** `docs/superpowers/specs/2026-07-24-full-scope-test-dataset-design.md`
**Reviewable summary:** `docs/testdata/BP_TestData_Design.xlsx`

## Global Constraints

- **Never write to a live database.** `bp_sqldb`, `uicanvas`, `ses`, `postgres` and `rdsadmin` are refused as targets. There is no override flag.
- **Reads from live are read-only** — schema introspection and reference-data copies only.
- New DB tables use the `bp_` prefix; indexes use `ix_bp_<table>_<column>`.
- Deterministic: same `--seed` produces byte-identical data.
- Target databases: `bp_testdb` (mirrors `bp_sqldb`) and `uicanvas_test` (mirrors `uicanvas`).
- Schema name is `proc` in both, matching live (the codebase hardcodes `proc`).
- Work on branch `Development`. Do not push to `main`.
- Commit messages carry no AI attribution or `Co-Authored-By` lines.
- Tests needing Postgres are marked `@pytest.mark.integration` (see `pytest.ini`).
- Run tests with `.venv/bin/python -m pytest`.

---

## File Structure

| File | Responsibility |
|---|---|
| `scripts/testdata/db.py` | Connect, create/drop target databases |
| `scripts/testdata/guards.py` | Refuse live targets; snapshot and compare live row counts |
| `scripts/testdata/rng.py` | Per-stream deterministic RNG |
| `scripts/testdata/schema.py` | Clone structure from live into targets |
| `scripts/testdata/reference.py` | Copy FX, policy, prompt, taxonomy verbatim |
| `scripts/testdata/org.py` | 6 entities, 400 business units, 500 cost centres, 240 users |
| `scripts/testdata/suppliers.py` | 5,000 suppliers, both ID conventions, crosswalk |
| `scripts/testdata/catalogue.py` | 5,000 items mapped to L5 leaves with price curves |
| `scripts/testdata/documents.py` | Requirement → quote → PO → invoice chains, 3 tiers |
| `scripts/testdata/deals.py` | Deal assembly and document mapping |
| `scripts/testdata/downstream.py` | Rankings, evaluations, decisions, actions, summaries |
| `scripts/testdata/defects.py` | Plant 30 defect types, emit the answer key |
| `scripts/testdata/verify.py` | V01–V14 |
| `scripts/testdata/build.py` | CLI entry point, orchestrates the above |

---

### Task 1: Safety guards

The single most important module. Everything else depends on it being impossible to point this tool at production.

**Files:**
- Create: `scripts/testdata/guards.py`
- Test: `tests/testdata/test_guards.py`
- Create: `tests/testdata/__init__.py` (empty)

**Interfaces:**
- Consumes: nothing
- Produces:
  - `LIVE_DATABASES: frozenset[str]`
  - `class UnsafeTargetError(RuntimeError)`
  - `assert_safe_target(name: str) -> None` — raises `UnsafeTargetError` if `name` is live or empty
  - `snapshot_counts(dbnames: Sequence[str]) -> dict[str, int]` — keys are `"<db>.<schema>.<table>"`
  - `assert_live_unchanged(before: dict[str, int], after: dict[str, int]) -> None` — raises `UnsafeTargetError` listing any differing keys

- [ ] **Step 1: Write the failing test**

Create `tests/testdata/__init__.py` as an empty file, then `tests/testdata/test_guards.py`:

```python
import pytest

from scripts.testdata.guards import (
    UnsafeTargetError,
    assert_live_unchanged,
    assert_safe_target,
)


@pytest.mark.parametrize("name", ["bp_sqldb", "uicanvas", "ses", "postgres", "rdsadmin"])
def test_live_database_names_are_refused(name):
    with pytest.raises(UnsafeTargetError, match="refuses"):
        assert_safe_target(name)


@pytest.mark.parametrize("name", ["BP_SQLDB", "  uicanvas  ", "UiCanvas"])
def test_refusal_ignores_case_and_whitespace(name):
    with pytest.raises(UnsafeTargetError):
        assert_safe_target(name)


@pytest.mark.parametrize("name", ["bp_testdb", "uicanvas_test", "scratch_db"])
def test_test_database_names_are_allowed(name):
    assert_safe_target(name) is None


@pytest.mark.parametrize("name", ["", "   ", None])
def test_empty_target_is_refused(name):
    with pytest.raises(UnsafeTargetError):
        assert_safe_target(name)


def test_unchanged_live_counts_pass():
    counts = {"bp_sqldb.proc.bp_supplier": 123, "uicanvas.proc.supplier": 1009}
    assert_live_unchanged(counts, dict(counts)) is None


def test_changed_live_counts_raise_and_name_the_table():
    before = {"bp_sqldb.proc.bp_supplier": 123, "uicanvas.proc.supplier": 1009}
    after = {"bp_sqldb.proc.bp_supplier": 5123, "uicanvas.proc.supplier": 1009}
    with pytest.raises(UnsafeTargetError, match="bp_sqldb.proc.bp_supplier"):
        assert_live_unchanged(before, after)


def test_disappearing_table_raises():
    with pytest.raises(UnsafeTargetError, match="proc.gone"):
        assert_live_unchanged({"bp_sqldb.proc.gone": 5}, {})
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/testdata/test_guards.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.testdata.guards'`

- [ ] **Step 3: Write minimal implementation**

Create `scripts/testdata/guards.py`:

```python
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
                cur.execute(
                    """
                    select table_schema, table_name
                    from information_schema.tables
                    where table_type = 'BASE TABLE'
                      and table_schema not in ('information_schema', 'pg_catalog')
                      and table_schema not like 'pg_temp%%'
                      and table_schema not like 'pg_toast%%'
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/python -m pytest tests/testdata/test_guards.py -v
```

Expected: PASS — 13 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/testdata/guards.py tests/testdata/__init__.py tests/testdata/test_guards.py
git commit -m "feat(testdata): safety guards refusing live database targets"
```

---

### Task 2: Deterministic RNG

Every generator draws from a named stream. Two runs with the same seed produce identical data, and adding a new stream never shifts an existing one's output.

**Files:**
- Create: `scripts/testdata/rng.py`
- Test: `tests/testdata/test_rng.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `make_rng(seed: int, stream: str) -> random.Random`
  - `weighted_apportion(weights: Sequence[float], total: int) -> list[int]` — largest-remainder apportionment summing to exactly `total`

- [ ] **Step 1: Write the failing test**

Create `tests/testdata/test_rng.py`:

```python
import pytest

from scripts.testdata.rng import make_rng, weighted_apportion


def test_same_seed_and_stream_give_identical_sequences():
    a = [make_rng(42, "suppliers").random() for _ in range(5)]
    b = [make_rng(42, "suppliers").random() for _ in range(5)]
    assert a == b


def test_different_streams_diverge():
    a = make_rng(42, "suppliers").random()
    b = make_rng(42, "documents").random()
    assert a != b


def test_different_seeds_diverge():
    assert make_rng(1, "suppliers").random() != make_rng(2, "suppliers").random()


def test_apportion_sums_to_exact_total():
    assert sum(weighted_apportion([1, 1, 1], 5000)) == 5000


def test_apportion_respects_relative_weights():
    result = weighted_apportion([3, 1], 100)
    assert result == [75, 25]


def test_apportion_handles_uneven_division():
    result = weighted_apportion([1, 1, 1], 10)
    assert sum(result) == 10
    assert sorted(result) == [3, 3, 4]


def test_apportion_is_deterministic():
    assert weighted_apportion([5, 3, 2, 7], 999) == weighted_apportion([5, 3, 2, 7], 999)


def test_apportion_rejects_empty_weights():
    with pytest.raises(ValueError):
        weighted_apportion([], 10)


def test_apportion_rejects_zero_total_weight():
    with pytest.raises(ValueError):
        weighted_apportion([0, 0], 10)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/testdata/test_rng.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.testdata.rng'`

- [ ] **Step 3: Write minimal implementation**

Create `scripts/testdata/rng.py`:

```python
"""Deterministic randomness.

Each generator draws from its own named stream so that adding, removing or
reordering a generator cannot shift another one's output. That property is what
makes the published answer key survive a regeneration.
"""
from __future__ import annotations

import hashlib
import random
from typing import Sequence


def make_rng(seed: int, stream: str) -> random.Random:
    """A Random seeded from (seed, stream), independent of every other stream."""
    digest = hashlib.sha256(f"{seed}:{stream}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def weighted_apportion(weights: Sequence[float], total: int) -> list[int]:
    """Split `total` across `weights` so the result sums to exactly `total`.

    Largest-remainder method: floor every share, then hand the shortfall to the
    entries with the largest fractional parts. Ties break on index, so the result
    is deterministic.
    """
    if not weights:
        raise ValueError("weights must not be empty")
    weight_sum = float(sum(weights))
    if weight_sum <= 0:
        raise ValueError("weights must sum to a positive number")

    exact = [w * total / weight_sum for w in weights]
    floors = [int(value) for value in exact]
    shortfall = total - sum(floors)
    order = sorted(
        range(len(exact)), key=lambda i: (-(exact[i] - floors[i]), i)
    )
    for index in order[:shortfall]:
        floors[index] += 1
    return floors
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/python -m pytest tests/testdata/test_rng.py -v
```

Expected: PASS — 9 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/testdata/rng.py tests/testdata/test_rng.py
git commit -m "feat(testdata): deterministic per-stream RNG and apportionment"
```

---

### Task 3: Database connection and creation

**Files:**
- Create: `scripts/testdata/db.py`
- Test: `tests/testdata/test_db.py`

**Interfaces:**
- Consumes: `guards.assert_safe_target`
- Produces:
  - `connect(dbname: str, *, autocommit: bool = False) -> psycopg2.extensions.connection`
  - `create_database(dbname: str, *, drop_first: bool = False) -> None` — guarded
  - `database_exists(dbname: str) -> bool`
  - `copy_rows(conn, schema: str, table: str, columns: Sequence[str], rows: Iterable[Sequence]) -> int` — bulk insert via `COPY FROM STDIN`

- [ ] **Step 1: Write the failing test**

Create `tests/testdata/test_db.py`:

```python
import pytest

from scripts.testdata.db import copy_rows, create_database
from scripts.testdata.guards import UnsafeTargetError


@pytest.mark.parametrize("name", ["bp_sqldb", "uicanvas"])
def test_create_database_refuses_live_targets(name):
    with pytest.raises(UnsafeTargetError):
        create_database(name)


@pytest.mark.parametrize("name", ["bp_sqldb", "uicanvas"])
def test_drop_first_also_refuses_live_targets(name):
    with pytest.raises(UnsafeTargetError):
        create_database(name, drop_first=True)


@pytest.mark.integration
def test_copy_rows_inserts_and_returns_count():
    from scripts.testdata.db import connect

    conn = connect("bp_testdb")
    try:
        with conn.cursor() as cur:
            cur.execute("create schema if not exists scratch")
            cur.execute("drop table if exists scratch.copy_probe")
            cur.execute("create table scratch.copy_probe (a text, b int)")
        conn.commit()

        written = copy_rows(
            conn, "scratch", "copy_probe", ["a", "b"], [("x", 1), ("y", 2)]
        )
        conn.commit()
        assert written == 2

        with conn.cursor() as cur:
            cur.execute("select a, b from scratch.copy_probe order by a")
            assert cur.fetchall() == [("x", 1), ("y", 2)]
            cur.execute("drop table scratch.copy_probe")
        conn.commit()
    finally:
        conn.close()


@pytest.mark.integration
def test_copy_rows_writes_null_for_none():
    from scripts.testdata.db import connect

    conn = connect("bp_testdb")
    try:
        with conn.cursor() as cur:
            cur.execute("create schema if not exists scratch")
            cur.execute("drop table if exists scratch.null_probe")
            cur.execute("create table scratch.null_probe (a text, b numeric)")
        conn.commit()

        copy_rows(conn, "scratch", "null_probe", ["a", "b"], [("only", None)])
        conn.commit()

        with conn.cursor() as cur:
            cur.execute("select a, b from scratch.null_probe")
            assert cur.fetchall() == [("only", None)]
            cur.execute("drop table scratch.null_probe")
        conn.commit()
    finally:
        conn.close()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/testdata/test_db.py -v -m "not integration"
```

Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.testdata.db'`

- [ ] **Step 3: Write minimal implementation**

Create `scripts/testdata/db.py`:

```python
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

from config.settings import Settings
from scripts.testdata.guards import assert_safe_target

_COPY_BUFFER_ROWS = 5000


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
        writer.writerow(["\\N" if value is None else value for value in row])
        written += 1
        pending += 1
        if pending >= _COPY_BUFFER_ROWS:
            flush()
    flush()
    return written
```

- [ ] **Step 4: Run tests to verify they pass**

Unit tests first (no database needed):

```bash
.venv/bin/python -m pytest tests/testdata/test_db.py -v -m "not integration"
```

Expected: PASS — 4 passed, 2 deselected

Then create the scratch target and run the integration tests:

```bash
.venv/bin/python -c "from scripts.testdata.db import create_database; create_database('bp_testdb')"
.venv/bin/python -m pytest tests/testdata/test_db.py -v -m integration
```

Expected: PASS — 2 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/testdata/db.py tests/testdata/test_db.py
git commit -m "feat(testdata): guarded database creation and COPY bulk loader"
```

---

### Task 4: Schema clone

Reproduce the structure of both live databases into the targets, including views, triggers, functions and indexes, without copying any business data.

**Files:**
- Create: `scripts/testdata/schema.py`
- Test: `tests/testdata/test_schema.py`

**Interfaces:**
- Consumes: `db.connect`, `db.create_database`, `guards.assert_safe_target`
- Produces:
  - `SCHEMA_PAIRS: tuple[tuple[str, str], ...]` — `(("bp_sqldb", "bp_testdb"), ("uicanvas", "uicanvas_test"))`
  - `clone_schema(source_db: str, target_db: str, *, drop_first: bool = False) -> CloneReport`
  - `@dataclass CloneReport: tables: int, views: int, functions: int, triggers: int`

`pg_dump --schema-only` is used because reimplementing DDL generation for 314 relations, 5 views, 10 functions and 8 triggers by hand would be both slow and wrong.

- [ ] **Step 1: Write the failing test**

Create `tests/testdata/test_schema.py`:

```python
import pytest

from scripts.testdata.guards import UnsafeTargetError
from scripts.testdata.schema import SCHEMA_PAIRS, clone_schema


def test_schema_pairs_map_live_to_test_databases():
    assert SCHEMA_PAIRS == (
        ("bp_sqldb", "bp_testdb"),
        ("uicanvas", "uicanvas_test"),
    )


@pytest.mark.parametrize("target", ["bp_sqldb", "uicanvas"])
def test_clone_refuses_to_write_into_a_live_database(target):
    with pytest.raises(UnsafeTargetError):
        clone_schema("bp_sqldb", target)


@pytest.mark.integration
def test_clone_reproduces_tables_and_views_but_no_rows():
    from scripts.testdata.db import connect

    report = clone_schema("bp_sqldb", "bp_testdb", drop_first=True)
    assert report.tables >= 100
    assert report.views >= 5

    conn = connect("bp_testdb")
    try:
        with conn.cursor() as cur:
            cur.execute(
                "select count(*) from information_schema.tables "
                "where table_schema = 'proc' and table_type = 'BASE TABLE'"
            )
            assert cur.fetchone()[0] >= 100

            cur.execute("select count(*) from proc.bp_supplier")
            assert cur.fetchone()[0] == 0

            cur.execute(
                "select count(*) from information_schema.views where table_schema = 'proc'"
            )
            assert cur.fetchone()[0] >= 5
    finally:
        conn.close()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/testdata/test_schema.py -v -m "not integration"
```

Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.testdata.schema'`

- [ ] **Step 3: Write minimal implementation**

Create `scripts/testdata/schema.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/testdata/test_schema.py -v -m "not integration"
```

Expected: PASS — 3 passed, 1 deselected

```bash
.venv/bin/python -m pytest tests/testdata/test_schema.py -v -m integration
```

Expected: PASS — 1 passed

If `pg_dump` is missing, install the client: `sudo apt-get install -y postgresql-client-16`

- [ ] **Step 5: Commit**

```bash
git add scripts/testdata/schema.py tests/testdata/test_schema.py
git commit -m "feat(testdata): clone live schema structure into test databases"
```

---

### Task 5: Reference data copy

Copy the tables that must behave identically to production: FX rates, governance, and the real category taxonomy.

**Files:**
- Create: `scripts/testdata/reference.py`
- Test: `tests/testdata/test_reference.py`

**Interfaces:**
- Consumes: `db.connect`, `db.copy_rows`, `guards.assert_safe_target`
- Produces:
  - `REFERENCE_TABLES: dict[str, tuple[str, ...]]` — source db → table names
  - `copy_reference(source_db: str, target_db: str) -> dict[str, int]` — table name → rows copied
  - `load_taxonomy(dbname: str = "uicanvas") -> list[TaxonomyLeaf]`
  - `@dataclass(frozen=True) TaxonomyLeaf: l1, l2, l3, l4, l5, unspsc_code, esg_impact, category_status, spend_classification, category_risk_rating, audit_frequency, policy_coverage` (all `str | None`)

- [ ] **Step 1: Write the failing test**

Create `tests/testdata/test_reference.py`:

```python
import pytest

from scripts.testdata.guards import UnsafeTargetError
from scripts.testdata.reference import (
    REFERENCE_TABLES,
    copy_reference,
    load_taxonomy,
)


def test_reference_tables_cover_fx_governance_and_taxonomy():
    bp_tables = REFERENCE_TABLES["bp_sqldb"]
    assert "bp_fx_rates" in bp_tables
    assert "bp_policy" in bp_tables
    assert "bp_prompt" in bp_tables
    assert "bp_admin_config" in bp_tables

    ui_tables = REFERENCE_TABLES["uicanvas"]
    assert "bp_category" in ui_tables
    assert "category" in ui_tables
    assert "category_mapping" in ui_tables


def test_copy_reference_refuses_live_targets():
    with pytest.raises(UnsafeTargetError):
        copy_reference("bp_sqldb", "bp_sqldb")


@pytest.mark.integration
def test_taxonomy_has_six_families_and_five_populated_levels():
    leaves = load_taxonomy("uicanvas")
    assert len(leaves) >= 240

    families = {leaf.l1 for leaf in leaves}
    assert families == {
        "IT & Technology",
        "Marketing & Media",
        "Facilities & Real Estate",
        "Professional Services",
        "Logistics & Supply Chain",
        "Office & Administrative Supplies",
    }

    for leaf in leaves:
        assert leaf.l1 and leaf.l2 and leaf.l3 and leaf.l4 and leaf.l5


@pytest.mark.integration
def test_copy_reference_reproduces_fx_row_count():
    from scripts.testdata.db import connect

    written = copy_reference("bp_sqldb", "bp_testdb")
    assert written["bp_fx_rates"] > 0

    source = connect("bp_sqldb")
    target = connect("bp_testdb")
    try:
        with source.cursor() as cur:
            cur.execute("select count(*) from proc.bp_fx_rates")
            expected = cur.fetchone()[0]
        with target.cursor() as cur:
            cur.execute("select count(*) from proc.bp_fx_rates")
            assert cur.fetchone()[0] == expected
    finally:
        source.close()
        target.close()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/testdata/test_reference.py -v -m "not integration"
```

Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.testdata.reference'`

- [ ] **Step 3: Write minimal implementation**

Create `scripts/testdata/reference.py`:

```python
"""Copy reference data verbatim from live into the test databases.

These tables are not synthesised. If FX rates, policies, prompts or the category
taxonomy differed from production, a passing test would say nothing about how the
product behaves against real configuration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from scripts.testdata.db import connect, copy_rows
from scripts.testdata.guards import assert_safe_target

REFERENCE_TABLES: dict[str, tuple[str, ...]] = {
    "bp_sqldb": (
        "bp_fx_rates",
        "bp_policy",
        "bp_prompt",
        "bp_admin_config",
        "bp_vendor_extraction_profiles",
        "bp_complaince_metric_prty_lkup",
        "procurement_patterns",
    ),
    "uicanvas": (
        "bp_category",
        "category",
        "category_mapping",
        "bp_policy",
        "bp_prompt",
    ),
}


@dataclass(frozen=True)
class TaxonomyLeaf:
    l1: str | None
    l2: str | None
    l3: str | None
    l4: str | None
    l5: str | None
    unspsc_code: str | None
    esg_impact: str | None
    category_status: str | None
    spend_classification: str | None
    category_risk_rating: str | None
    audit_frequency: str | None
    policy_coverage: str | None

    @property
    def path(self) -> str:
        parts = [self.l1, self.l2, self.l3, self.l4, self.l5]
        return " > ".join(part for part in parts if part)


def load_taxonomy(dbname: str = "uicanvas") -> list[TaxonomyLeaf]:
    """The real 5-level taxonomy. Read-only against the source database."""
    conn = connect(dbname)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                select category_level_1, category_level_2, category_level_3,
                       category_level_4, category_level_5, unspsc_code, esg_impact,
                       category_status, spend_classification, category_risk_rating,
                       audit_frequency, policy_coverage
                from proc.bp_category
                order by 1, 2, 3, 4, 5
                """
            )
            return [TaxonomyLeaf(*row) for row in cur.fetchall()]
    finally:
        conn.close()


def _columns(conn, schema: str, table: str) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(
            "select column_name from information_schema.columns "
            "where table_schema = %s and table_name = %s order by ordinal_position",
            (schema, table),
        )
        return [row[0] for row in cur.fetchall()]


def copy_reference(source_db: str, target_db: str) -> dict[str, int]:
    """Copy every reference table for source_db into target_db. Returns row counts."""
    assert_safe_target(target_db)
    tables: Sequence[str] = REFERENCE_TABLES.get(source_db, ())

    source = connect(source_db)
    target = connect(target_db)
    written: dict[str, int] = {}
    try:
        for table in tables:
            columns = _columns(source, "proc", table)
            if not columns:
                continue
            column_list = ", ".join(f'"{c}"' for c in columns)
            with source.cursor() as cur:
                cur.execute(f'select {column_list} from proc."{table}"')
                rows = cur.fetchall()
            with target.cursor() as cur:
                cur.execute(f'truncate proc."{table}"')
            written[table] = copy_rows(target, "proc", table, columns, rows)
        target.commit()
    finally:
        source.close()
        target.close()
    return written
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/testdata/test_reference.py -v -m "not integration"
```

Expected: PASS — 2 passed, 2 deselected

```bash
.venv/bin/python -m pytest tests/testdata/test_reference.py -v -m integration
```

Expected: PASS — 2 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/testdata/reference.py tests/testdata/test_reference.py
git commit -m "feat(testdata): copy FX, governance and category taxonomy verbatim"
```

---

### Task 6: Organisation — entities, business units, cost centres, users

Populates `business_unit` (currently empty in live) and replaces the 500 placeholder `cost_centre` rows with meaningful data.

**Files:**
- Create: `scripts/testdata/org.py`
- Test: `tests/testdata/test_org.py`

**Interfaces:**
- Consumes: `rng.make_rng`, `rng.weighted_apportion`, `reference.TaxonomyLeaf`, `db.copy_rows`
- Produces:
  - `@dataclass(frozen=True) Entity: org_id, name, country, currency, spend_share, cost_centre_count`
  - `ENTITIES: tuple[Entity, ...]` — 6 entities (the group parent is not in this tuple)
  - `GROUP_ORG_ID: str = "ORG-GRP"`
  - `@dataclass(frozen=True) BusinessUnit: bu_id, l1, l2, l3, l4, l5, org_id, head_name, head_email, region, status`
  - `@dataclass(frozen=True) CostCentre: cc_id, levels: tuple[str, ...], bu_id, org_id, finance_account_code, manager_name, manager_email, spend_threshold_limit, currency, budget_allocated_annual, actual_spend_ytd, forecast_spend_annual, cost_centre_type, linked_category_level_5_id, is_active`
  - `build_business_units(seed: int) -> list[BusinessUnit]` — returns exactly 400 L5 units
  - `build_cost_centres(seed: int, units: Sequence[BusinessUnit], leaves: Sequence[TaxonomyLeaf]) -> list[CostCentre]` — returns exactly 500

- [ ] **Step 1: Write the failing test**

Create `tests/testdata/test_org.py`:

```python
from scripts.testdata.org import (
    ENTITIES,
    GROUP_ORG_ID,
    build_business_units,
    build_cost_centres,
)
from scripts.testdata.reference import TaxonomyLeaf


def _leaves(count: int) -> list[TaxonomyLeaf]:
    return [
        TaxonomyLeaf(
            l1="IT & Technology", l2="Software", l3="ERP", l4=f"Sub{i}", l5=f"Leaf{i}",
            unspsc_code=str(10000000 + i), esg_impact="Low", category_status="Active",
            spend_classification="Direct", category_risk_rating="Minimal",
            audit_frequency="Annually", policy_coverage="Full",
        )
        for i in range(count)
    ]


def test_there_are_six_buying_entities():
    assert len(ENTITIES) == 6
    assert GROUP_ORG_ID == "ORG-GRP"
    assert GROUP_ORG_ID not in {entity.org_id for entity in ENTITIES}


def test_entity_spend_shares_sum_to_one():
    assert round(sum(entity.spend_share for entity in ENTITIES), 6) == 1.0


def test_entity_cost_centre_counts_sum_to_500():
    assert sum(entity.cost_centre_count for entity in ENTITIES) == 500


def test_each_entity_has_a_distinct_id_and_currency_pairing():
    assert len({entity.org_id for entity in ENTITIES}) == 6
    by_id = {entity.org_id: entity for entity in ENTITIES}
    assert by_id["ORG-UK"].currency == "GBP"
    assert by_id["ORG-DE"].currency == "EUR"
    assert by_id["ORG-US"].currency == "USD"
    assert by_id["ORG-IN"].currency == "INR"
    assert by_id["ORG-AE"].currency == "AED"


def test_business_unit_tree_has_the_specified_breadth():
    units = build_business_units(42)
    assert len(units) == 400
    assert len({unit.l1 for unit in units}) == 6
    assert len({(unit.l1, unit.l2) for unit in units}) == 40
    assert len({(unit.l1, unit.l2, unit.l3) for unit in units}) == 120
    assert len({(unit.l1, unit.l2, unit.l3, unit.l4) for unit in units}) == 240


def test_every_business_unit_belongs_to_a_real_entity():
    units = build_business_units(42)
    valid = {entity.org_id for entity in ENTITIES}
    assert all(unit.org_id in valid for unit in units)


def test_business_units_are_deterministic():
    assert build_business_units(42) == build_business_units(42)


def test_cost_centres_match_entity_allocation():
    units = build_business_units(42)
    centres = build_cost_centres(42, units, _leaves(50))
    assert len(centres) == 500

    per_entity = {entity.org_id: 0 for entity in ENTITIES}
    for centre in centres:
        per_entity[centre.org_id] += 1
    for entity in ENTITIES:
        assert per_entity[entity.org_id] == entity.cost_centre_count


def test_cost_centres_have_six_levels_and_a_category_link():
    units = build_business_units(42)
    centres = build_cost_centres(42, units, _leaves(50))
    for centre in centres:
        assert len(centre.levels) == 6
        assert all(level for level in centre.levels)
        assert centre.linked_category_level_5_id


def test_cost_centre_budgets_are_positive_and_thresholds_in_range():
    units = build_business_units(42)
    centres = build_cost_centres(42, units, _leaves(50))
    for centre in centres:
        assert centre.budget_allocated_annual > 0
        assert 5000 <= centre.spend_threshold_limit <= 250000


def test_cost_centre_ids_are_unique():
    units = build_business_units(42)
    centres = build_cost_centres(42, units, _leaves(50))
    assert len({centre.cc_id for centre in centres}) == 500
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/testdata/test_org.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.testdata.org'`

- [ ] **Step 3: Write minimal implementation**

Create `scripts/testdata/org.py`:

```python
"""The buying organisation: entities, business units, cost centres.

business_unit is empty in the live database and cost_centre holds 500 placeholder
rows pointing at business units that do not exist. This module builds both
properly, because tests E1-E3 (roll-up integrity, budget overrun, per-cost-centre
approval thresholds) have nothing to assert against otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from scripts.testdata.reference import TaxonomyLeaf
from scripts.testdata.rng import make_rng, weighted_apportion

GROUP_ORG_ID = "ORG-GRP"
GROUP_NAME = "Beyond Procurement Group plc"


@dataclass(frozen=True)
class Entity:
    org_id: str
    name: str
    country: str
    currency: str
    spend_share: float
    cost_centre_count: int


ENTITIES: tuple[Entity, ...] = (
    Entity("ORG-UK", "Beyond Procurement UK Ltd", "United Kingdom", "GBP", 0.42, 185),
    Entity("ORG-US", "Beyond Procurement North America Inc", "United States", "USD", 0.18, 98),
    Entity("ORG-DE", "Beyond Procurement Deutschland GmbH", "Germany", "EUR", 0.16, 92),
    Entity("ORG-IE", "Beyond Procurement Ireland Ltd", "Ireland", "EUR", 0.12, 61),
    Entity("ORG-IN", "Beyond Procurement India Pvt Ltd", "India", "INR", 0.08, 42),
    Entity("ORG-AE", "Beyond Procurement Middle East FZE", "United Arab Emirates", "AED", 0.04, 22),
)

# L1 and L2 follow the convention already present in the live cost_centre rows:
# L1 is a function, L2 is a region.
BU_FUNCTIONS = ("Operations", "Sales", "Finance", "Corporate", "Technology", "Supply Chain")
BU_REGIONS = ("Europe", "North America", "Middle East", "LATAM", "APAC")
BU_DEPARTMENTS = (
    "Procurement", "Facilities", "Engineering", "Marketing", "Legal", "People",
    "Data", "Security", "Logistics", "Customer Success",
)
CC_TYPES = ("Overhead", "Project", "Capital", "Operational")

_FIRST_NAMES = (
    "Amelia", "Noah", "Priya", "Marcus", "Sofia", "Ethan", "Yusuf", "Chloe",
    "Rahul", "Freya", "Tomas", "Aisha", "Liam", "Nadia", "Oscar", "Mei",
)
_LAST_NAMES = (
    "Hartley", "Okafor", "Sharma", "Lindqvist", "Moreau", "Kowalski", "Rivera",
    "Bennett", "Haddad", "Novak", "Fitzgerald", "Alvarez", "Devlin", "Nakamura",
)


def _person(rng) -> tuple[str, str]:
    first = rng.choice(_FIRST_NAMES)
    last = rng.choice(_LAST_NAMES)
    name = f"{first} {last}"
    email = f"{first.lower()}.{last.lower()}@beyondprocurement.example"
    return name, email


def build_business_units(seed: int) -> list[BusinessUnit]:
    """400 L5 business units under a 6/40/120/240 tree."""
    rng = make_rng(seed, "business_units")

    l2_per_l1 = weighted_apportion([1] * len(BU_FUNCTIONS), 40)
    pairs: list[tuple[str, str]] = []
    for function, count in zip(BU_FUNCTIONS, l2_per_l1):
        for index in range(count):
            pairs.append((function, BU_REGIONS[index % len(BU_REGIONS)] + f" {index // len(BU_REGIONS) + 1}"))

    l3_per_pair = weighted_apportion([1] * len(pairs), 120)
    triples: list[tuple[str, str, str]] = []
    for (function, region), count in zip(pairs, l3_per_pair):
        for index in range(count):
            triples.append((function, region, BU_DEPARTMENTS[index % len(BU_DEPARTMENTS)] + f" {index + 1}"))

    l4_per_triple = weighted_apportion([1] * len(triples), 240)
    quads: list[tuple[str, str, str, str]] = []
    for (function, region, department), count in zip(triples, l4_per_triple):
        for index in range(count):
            quads.append((function, region, department, f"{department} Group {index + 1}"))

    l5_per_quad = weighted_apportion([1] * len(quads), 400)
    units: list[BusinessUnit] = []
    entity_cycle = [entity.org_id for entity in ENTITIES]
    counter = 0
    for (function, region, department, group), count in zip(quads, l5_per_quad):
        for index in range(count):
            head_name, head_email = _person(rng)
            counter += 1
            units.append(
                BusinessUnit(
                    bu_id=f"BU-5{counter:04d}",
                    l1=function,
                    l2=region,
                    l3=department,
                    l4=group,
                    l5=f"{group} Team {index + 1}",
                    org_id=entity_cycle[counter % len(entity_cycle)],
                    head_name=head_name,
                    head_email=head_email,
                    region=region.split(" ")[0],
                    status="Active",
                )
            )
    return units


def build_cost_centres(
    seed: int, units: Sequence[BusinessUnit], leaves: Sequence[TaxonomyLeaf]
) -> list[CostCentre]:
    """500 cost centres, allocated to entities per Entity.cost_centre_count."""
    rng = make_rng(seed, "cost_centres")
    by_entity: dict[str, list[BusinessUnit]] = {entity.org_id: [] for entity in ENTITIES}
    for unit in units:
        by_entity[unit.org_id].append(unit)

    currency_by_entity = {entity.org_id: entity.currency for entity in ENTITIES}
    centres: list[CostCentre] = []
    counter = 0

    for entity in ENTITIES:
        candidates = by_entity[entity.org_id] or list(units)
        for _ in range(entity.cost_centre_count):
            counter += 1
            unit = candidates[counter % len(candidates)]
            leaf = leaves[counter % len(leaves)]
            manager_name, manager_email = _person(rng)

            budget = round(rng.uniform(80_000, 4_500_000), 2)
            # Most cost centres sit under budget; the overrun defect is planted later.
            actual = round(budget * rng.uniform(0.35, 0.94), 2)
            forecast = round(actual * rng.uniform(1.02, 1.35), 2)

            centres.append(
                CostCentre(
                    cc_id=f"CC{counter:06d}",
                    levels=(
                        unit.l1,
                        unit.l2,
                        unit.l3,
                        unit.l4,
                        unit.l5,
                        f"{unit.l5} / {rng.choice(CC_TYPES)}",
                    ),
                    bu_id=unit.bu_id,
                    org_id=entity.org_id,
                    finance_account_code=f"FAC-{rng.randint(1000, 9999)}",
                    manager_name=manager_name,
                    manager_email=manager_email,
                    spend_threshold_limit=float(rng.choice(
                        [5_000, 10_000, 25_000, 50_000, 100_000, 250_000]
                    )),
                    currency=currency_by_entity[entity.org_id],
                    budget_allocated_annual=budget,
                    actual_spend_ytd=actual,
                    forecast_spend_annual=forecast,
                    cost_centre_type=rng.choice(CC_TYPES),
                    linked_category_level_5_id=leaf.unspsc_code or leaf.l5 or "UNKNOWN",
                    is_active=rng.random() > 0.05,
                )
            )
    return centres
```

Add the two dataclasses above `build_business_units` (they are referenced by the
signatures, so they must be defined before use at import time):

```python
@dataclass(frozen=True)
class BusinessUnit:
    bu_id: str
    l1: str
    l2: str
    l3: str
    l4: str
    l5: str
    org_id: str
    head_name: str
    head_email: str
    region: str
    status: str


@dataclass(frozen=True)
class CostCentre:
    cc_id: str
    levels: tuple[str, ...]
    bu_id: str
    org_id: str
    finance_account_code: str
    manager_name: str
    manager_email: str
    spend_threshold_limit: float
    currency: str
    budget_allocated_annual: float
    actual_spend_ytd: float
    forecast_spend_annual: float
    cost_centre_type: str
    linked_category_level_5_id: str
    is_active: bool
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/python -m pytest tests/testdata/test_org.py -v
```

Expected: PASS — 11 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/testdata/org.py tests/testdata/test_org.py
git commit -m "feat(testdata): organisation model with 6 entities, 400 BUs, 500 cost centres"
```

---

### Task 7: Suppliers and the cross-database crosswalk

**Files:**
- Create: `scripts/testdata/suppliers.py`
- Test: `tests/testdata/test_suppliers.py`

**Interfaces:**
- Consumes: `rng.make_rng`, `rng.weighted_apportion`, `reference.TaxonomyLeaf`
- Produces:
  - `@dataclass(frozen=True) Tier: name, count, spend_share, min_documents, max_documents`
  - `TIERS: tuple[Tier, ...]` — Strategic 120, Core 700, Tail 3020, One-off 1160
  - `@dataclass(frozen=True) Supplier` — with `bp_supplier_id`, `uicanvas_supplier_id`, `name`, `tier`, `primary_leaf`, `secondary_leaves`, `country`, `currency`, plus the 51 `bp_supplier` column values in a `columns: dict[str, object]` attribute
  - `build_suppliers(seed: int, leaves: Sequence[TaxonomyLeaf]) -> list[Supplier]` — exactly 5,000
  - `build_crosswalk(suppliers: Sequence[Supplier]) -> list[tuple[str, str, str]]` — `(bp_supplier_id, uicanvas_supplier_id, legal_entity_key)`
  - `CROSSWALK_DDL: str` — creates `proc.bp_supplier_id_crosswalk` with index `ix_bp_supplier_id_crosswalk_uicanvas_supplier_id`

- [ ] **Step 1: Write the failing test**

Create `tests/testdata/test_suppliers.py`:

```python
import re

from scripts.testdata.suppliers import (
    CROSSWALK_DDL,
    TIERS,
    build_crosswalk,
    build_suppliers,
)
from scripts.testdata.reference import TaxonomyLeaf


def _leaves(count: int = 246) -> list[TaxonomyLeaf]:
    families = [
        "IT & Technology", "Marketing & Media", "Facilities & Real Estate",
        "Professional Services", "Logistics & Supply Chain",
        "Office & Administrative Supplies",
    ]
    return [
        TaxonomyLeaf(
            l1=families[i % len(families)], l2=f"G{i}", l3=f"C{i}", l4=f"S{i}", l5=f"Leaf{i}",
            unspsc_code=str(20000000 + i), esg_impact="Low", category_status="Active",
            spend_classification="Indirect", category_risk_rating="Minimal",
            audit_frequency="Annually", policy_coverage="Partial",
        )
        for i in range(count)
    ]


def test_tiers_sum_to_five_thousand_suppliers():
    assert sum(tier.count for tier in TIERS) == 5000


def test_tier_spend_shares_sum_to_one():
    assert round(sum(tier.spend_share for tier in TIERS), 6) == 1.0


def test_builds_exactly_five_thousand_suppliers():
    assert len(build_suppliers(42, _leaves())) == 5000


def test_supplier_ids_follow_both_conventions_and_are_unique():
    suppliers = build_suppliers(42, _leaves())
    assert len({s.bp_supplier_id for s in suppliers}) == 5000
    assert len({s.uicanvas_supplier_id for s in suppliers}) == 5000
    for supplier in suppliers[:50]:
        assert supplier.bp_supplier_id.startswith("SUP-")
        assert re.fullmatch(r"SI\d{6}", supplier.uicanvas_supplier_id)


def test_primary_categories_cover_every_leaf():
    leaves = _leaves()
    suppliers = build_suppliers(42, leaves)
    assigned = {s.primary_leaf.l5 for s in suppliers}
    assert assigned == {leaf.l5 for leaf in leaves}


def test_tier_counts_are_respected():
    suppliers = build_suppliers(42, _leaves())
    counts: dict[str, int] = {}
    for supplier in suppliers:
        counts[supplier.tier] = counts.get(supplier.tier, 0) + 1
    for tier in TIERS:
        assert counts[tier.name] == tier.count


def test_every_supplier_populates_all_51_columns():
    suppliers = build_suppliers(42, _leaves())
    for supplier in suppliers[:100]:
        assert len(supplier.columns) == 51
        assert supplier.columns["supplier_name"]
        assert supplier.columns["country"]
        assert supplier.columns["default_currency"]


def test_generation_is_deterministic():
    leaves = _leaves()
    first = build_suppliers(42, leaves)
    second = build_suppliers(42, leaves)
    assert [s.bp_supplier_id for s in first] == [s.bp_supplier_id for s in second]
    assert [s.columns for s in first] == [s.columns for s in second]


def test_crosswalk_maps_every_supplier_once():
    suppliers = build_suppliers(42, _leaves())
    crosswalk = build_crosswalk(suppliers)
    assert len(crosswalk) == 5000
    assert len({row[0] for row in crosswalk}) == 5000
    assert len({row[1] for row in crosswalk}) == 5000


def test_crosswalk_ddl_uses_bp_prefix_and_index_convention():
    assert "proc.bp_supplier_id_crosswalk" in CROSSWALK_DDL
    assert "ix_bp_supplier_id_crosswalk_uicanvas_supplier_id" in CROSSWALK_DDL
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/testdata/test_suppliers.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.testdata.suppliers'`

- [ ] **Step 3: Write minimal implementation**

Create `scripts/testdata/suppliers.py`. The column list matches
`proc.bp_supplier` exactly, in ordinal order:

```python
"""5,000 suppliers, materialised under both ID conventions.

bp_sqldb names suppliers SUP-<PascalCase>; uicanvas uses SI######. Both are
reproduced and joined by a crosswalk, so the mismatch between the two databases
is testable rather than hidden behind a single invented convention.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Sequence

from scripts.testdata.reference import TaxonomyLeaf
from scripts.testdata.rng import make_rng, weighted_apportion

SUPPLIER_COLUMNS: tuple[str, ...] = (
    "supplier_id", "supplier_name", "trading_name", "supplier_type", "legal_structure",
    "tax_id", "vat_number", "duns_number", "parent_company_id", "registered_country",
    "registration_number", "is_preferred_supplier", "risk_score", "credit_limit_amount",
    "esg_cert_iso14001", "esg_cert_sa8000", "esg_cert_ecovadis", "diversity_women_owned",
    "diversity_minority_owned", "diversity_veteran_owned", "insurance_coverage_type",
    "insurance_coverage_amount", "insurance_expiry_date", "bank_name",
    "bank_account_number", "bank_swift", "bank_iban", "default_currency", "incoterms",
    "delivery_lead_time_days", "address_line1", "address_line2", "city", "postal_code",
    "country", "website_url", "edi_enabled", "api_enabled", "ariba_integrated",
    "contact_name_1", "contact_role_1", "contact_email_1", "contact_phone_1",
    "contact_name_2", "contact_role_2", "contact_email_2", "contact_phone_2",
    "created_date", "created_by", "last_modified_by", "last_modified_date",
)

# Vocabularies taken from the existing uicanvas.proc.supplier master.
SUPPLIER_TYPES = ("Service Provider", "Wholesaler", "Manufacturer", "Retailer", "Distributor", "Consulting")
LEGAL_STRUCTURES = ("PLC", "Ltd", "LLP", "Inc", "GmbH", "LLC")
INCOTERMS = ("DAP", "DDP", "FOB", "CIF", "EXW")
INSURANCE_TYPES = ("General Liability", "Product Liability", "Professional Indemnity", "Cyber")

# (country, weight, currency)
COUNTRIES: tuple[tuple[str, int, str], ...] = (
    ("United Kingdom", 3100, "GBP"), ("United States", 300, "USD"),
    ("Germany", 250, "EUR"), ("Ireland", 200, "EUR"), ("France", 175, "EUR"),
    ("Netherlands", 150, "EUR"), ("India", 150, "INR"), ("Poland", 125, "PLN"),
    ("Spain", 100, "EUR"), ("Italy", 100, "EUR"),
    ("United Arab Emirates", 100, "AED"), ("China", 100, "USD"),
    ("Sweden", 50, "SEK"), ("Switzerland", 25, "CHF"), ("Singapore", 15, "SGD"),
    ("Japan", 10, "JPY"), ("Australia", 5, "AUD"), ("Canada", 45, "USD"),
)

_STEMS = (
    "Northgate", "Brightpath", "Vantage", "Ironbridge", "Clearwater", "Summit",
    "Meridian", "Kestrel", "Blackwood", "Harbourline", "Redstone", "Silverbeck",
    "Oakfield", "Copperleaf", "Windrose", "Falconridge", "Stonegate", "Thornbury",
    "Lighthouse", "Ashcroft", "Greenhollow", "Pinnacle", "Crossfell", "Marlowe",
)
_SUFFIXES = (
    "Solutions", "Systems", "Group", "Partners", "Industries", "Services",
    "Supplies", "Technologies", "Logistics", "Associates", "Works", "Trading",
)


@dataclass(frozen=True)
class Tier:
    name: str
    count: int
    spend_share: float
    min_documents: int
    max_documents: int


TIERS: tuple[Tier, ...] = (
    Tier("Strategic", 120, 0.38, 60, 140),
    Tier("Core", 700, 0.41, 12, 45),
    Tier("Tail", 3020, 0.18, 2, 9),
    Tier("One-off", 1160, 0.03, 1, 1),
)


@dataclass(frozen=True)
class Supplier:
    bp_supplier_id: str
    uicanvas_supplier_id: str
    name: str
    tier: str
    primary_leaf: TaxonomyLeaf
    secondary_leaves: tuple[TaxonomyLeaf, ...]
    country: str
    currency: str
    columns: dict[str, Any] = field(hash=False, compare=True)


CROSSWALK_DDL = """
create table if not exists proc.bp_supplier_id_crosswalk (
    bp_supplier_id        text primary key,
    uicanvas_supplier_id  text not null unique,
    legal_entity_key      text not null,
    created_date          timestamp not null default now()
);
create index if not exists ix_bp_supplier_id_crosswalk_uicanvas_supplier_id
    on proc.bp_supplier_id_crosswalk (uicanvas_supplier_id);
create index if not exists ix_bp_supplier_id_crosswalk_legal_entity_key
    on proc.bp_supplier_id_crosswalk (legal_entity_key);
"""


def _pascal(name: str) -> str:
    return "".join(part.capitalize() for part in name.replace(",", " ").split() if part)


def _country_for(rng) -> tuple[str, str]:
    total = sum(weight for _, weight, _ in COUNTRIES)
    pick = rng.randrange(total)
    running = 0
    for country, weight, currency in COUNTRIES:
        running += weight
        if pick < running:
            return country, currency
    return COUNTRIES[0][0], COUNTRIES[0][2]


def build_suppliers(seed: int, leaves: Sequence[TaxonomyLeaf]) -> list[Supplier]:
    """Exactly 5,000 suppliers with all 51 bp_supplier columns populated."""
    rng = make_rng(seed, "suppliers")

    per_leaf = weighted_apportion([1] * len(leaves), 5000)
    leaf_slots: list[TaxonomyLeaf] = []
    for leaf, count in zip(leaves, per_leaf):
        leaf_slots.extend([leaf] * count)

    tier_slots: list[str] = []
    for tier in TIERS:
        tier_slots.extend([tier.name] * tier.count)

    base_date = datetime(2022, 1, 1)
    suppliers: list[Supplier] = []
    used_names: set[str] = set()

    for index in range(5000):
        stem = _STEMS[rng.randrange(len(_STEMS))]
        suffix = _SUFFIXES[rng.randrange(len(_SUFFIXES))]
        name = f"{stem} {suffix}"
        disambiguator = 2
        while name in used_names:
            name = f"{stem} {suffix} {disambiguator}"
            disambiguator += 1
        used_names.add(name)

        country, currency = _country_for(rng)
        leaf = leaf_slots[index]
        tier = tier_slots[index]

        secondary_count = rng.choice([0, 1, 1, 2])
        secondary = tuple(
            leaves[rng.randrange(len(leaves))] for _ in range(secondary_count)
        )

        bp_id = f"SUP-{_pascal(name)}"
        ui_id = f"SI{index + 1:06d}"
        created = base_date + timedelta(days=rng.randrange(0, 900))
        expiry = date(2026, 1, 1) + timedelta(days=rng.randrange(-400, 1800))

        columns: dict[str, Any] = {
            "supplier_id": bp_id,
            "supplier_name": name,
            "trading_name": f"{name} Trading Ltd.",
            "supplier_type": rng.choice(SUPPLIER_TYPES),
            "legal_structure": rng.choice(LEGAL_STRUCTURES),
            "tax_id": f"TX{rng.randrange(10_000_000, 99_999_999)}",
            "vat_number": f"VAT{rng.randrange(10_000_000, 99_999_999)}",
            "duns_number": str(rng.randrange(100_000_000, 999_999_999)),
            "parent_company_id": None,
            "registered_country": country,
            "registration_number": f"REG{rng.randrange(10_000_000, 99_999_999)}",
            "is_preferred_supplier": tier in ("Strategic", "Core"),
            "risk_score": f"{rng.uniform(5, 95):.2f}",
            "credit_limit_amount": round(rng.uniform(25_000, 5_000_000), 2),
            "esg_cert_iso14001": rng.random() < 0.42,
            "esg_cert_sa8000": rng.random() < 0.21,
            "esg_cert_ecovadis": rng.random() < 0.33,
            "diversity_women_owned": rng.random() < 0.18,
            "diversity_minority_owned": rng.random() < 0.14,
            "diversity_veteran_owned": rng.random() < 0.07,
            "insurance_coverage_type": rng.choice(INSURANCE_TYPES),
            "insurance_coverage_amount": round(rng.uniform(250_000, 10_000_000), 2),
            "insurance_expiry_date": expiry,
            "bank_name": f"{rng.choice(_STEMS)} Bank",
            "bank_account_number": str(rng.randrange(10_000_000, 99_999_999)),
            "bank_swift": f"SWIFT{rng.randrange(10_000_000, 99_999_999)}",
            "bank_iban": f"IBAN{rng.randrange(1_000_000_000, 9_999_999_999)}",
            "default_currency": currency,
            "incoterms": rng.choice(INCOTERMS),
            "delivery_lead_time_days": str(rng.randrange(1, 60)),
            "address_line1": f"{rng.randrange(1, 400)} {rng.choice(_STEMS)} Road",
            "address_line2": None,
            "city": rng.choice(("London", "Manchester", "Dublin", "Berlin", "Austin", "Pune", "Dubai")),
            "postal_code": f"{rng.choice('ABCDEFGHMNPRSW')}{rng.randrange(1, 99)} {rng.randrange(1, 9)}{rng.choice('ABDEFGHJLNPQRSTUWXYZ')}{rng.choice('ABDEFGHJLNPQRSTUWXYZ')}",
            "country": country,
            "website_url": f"https://{stem.lower()}{suffix.lower()}.example",
            "edi_enabled": rng.random() < 0.28,
            "api_enabled": rng.random() < 0.19,
            "ariba_integrated": rng.random() < 0.11,
            "contact_name_1": f"{rng.choice(('Alex', 'Priya', 'Sam', 'Nadia', 'Tom'))} {rng.choice(('Reed', 'Osei', 'Kaur', 'Blake', 'Moretti'))}",
            "contact_role_1": "Account Manager",
            "contact_email_1": f"sales@{stem.lower()}{suffix.lower()}.example",
            "contact_phone_1": f"+44 20 {rng.randrange(1000, 9999)} {rng.randrange(1000, 9999)}",
            "contact_name_2": None,
            "contact_role_2": None,
            "contact_email_2": None,
            "contact_phone_2": None,
            "created_date": created,
            "created_by": "testdata",
            "last_modified_by": "testdata",
            "last_modified_date": created,
        }

        suppliers.append(
            Supplier(
                bp_supplier_id=bp_id,
                uicanvas_supplier_id=ui_id,
                name=name,
                tier=tier,
                primary_leaf=leaf,
                secondary_leaves=secondary,
                country=country,
                currency=currency,
                columns=columns,
            )
        )
    return suppliers


def build_crosswalk(suppliers: Sequence[Supplier]) -> list[tuple[str, str, str]]:
    """(bp_supplier_id, uicanvas_supplier_id, legal_entity_key) for every supplier."""
    return [
        (
            supplier.bp_supplier_id,
            supplier.uicanvas_supplier_id,
            f"{supplier.name}|{supplier.columns['vat_number']}",
        )
        for supplier in suppliers
    ]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/python -m pytest tests/testdata/test_suppliers.py -v
```

Expected: PASS — 10 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/testdata/suppliers.py tests/testdata/test_suppliers.py
git commit -m "feat(testdata): 5,000 suppliers under both ID conventions with crosswalk"
```

---

### Task 8: Catalogue

5,000 items mapped to L5 leaves, each with a price that drifts over time so
benchmarking and price-variance detection have real signal.

**Files:**
- Create: `scripts/testdata/catalogue.py`
- Test: `tests/testdata/test_catalogue.py`

**Interfaces:**
- Consumes: `rng.make_rng`, `rng.weighted_apportion`, `reference.TaxonomyLeaf`
- Produces:
  - `@dataclass(frozen=True) CatalogueItem: item_id, description, leaf, unit_of_measure, base_price, currency, preferred_supplier_id`
  - `build_catalogue(seed: int, leaves: Sequence[TaxonomyLeaf], supplier_ids: Sequence[str]) -> list[CatalogueItem]` — exactly 5,000
  - `price_on(item: CatalogueItem, when: date, *, seed: int) -> Decimal` — deterministic price at a date, two decimal places

- [ ] **Step 1: Write the failing test**

Create `tests/testdata/test_catalogue.py`:

```python
from datetime import date
from decimal import Decimal

from scripts.testdata.catalogue import build_catalogue, price_on
from scripts.testdata.reference import TaxonomyLeaf


def _leaves(count: int = 246) -> list[TaxonomyLeaf]:
    return [
        TaxonomyLeaf(
            l1="IT & Technology", l2=f"G{i}", l3=f"C{i}", l4=f"S{i}", l5=f"Leaf{i}",
            unspsc_code=str(30000000 + i), esg_impact="Low", category_status="Active",
            spend_classification="Direct", category_risk_rating="Minimal",
            audit_frequency="Annually", policy_coverage="Full",
        )
        for i in range(count)
    ]


def _supplier_ids(count: int = 500) -> list[str]:
    return [f"SUP-Supplier{i}" for i in range(count)]


def test_builds_exactly_five_thousand_items():
    assert len(build_catalogue(42, _leaves(), _supplier_ids())) == 5000


def test_every_item_maps_to_a_real_leaf():
    leaves = _leaves()
    valid = {leaf.l5 for leaf in leaves}
    for item in build_catalogue(42, leaves, _supplier_ids()):
        assert item.leaf.l5 in valid


def test_every_leaf_receives_at_least_one_item():
    leaves = _leaves()
    items = build_catalogue(42, leaves, _supplier_ids())
    assert {item.leaf.l5 for item in items} == {leaf.l5 for leaf in leaves}


def test_item_ids_are_unique():
    items = build_catalogue(42, _leaves(), _supplier_ids())
    assert len({item.item_id for item in items}) == 5000


def test_base_prices_are_positive():
    for item in build_catalogue(42, _leaves(), _supplier_ids()):
        assert item.base_price > 0


def test_catalogue_is_deterministic():
    leaves, suppliers = _leaves(), _supplier_ids()
    first = build_catalogue(42, leaves, suppliers)
    second = build_catalogue(42, leaves, suppliers)
    assert [(i.item_id, i.base_price) for i in first] == [
        (i.item_id, i.base_price) for i in second
    ]


def test_price_on_returns_two_decimal_places():
    item = build_catalogue(42, _leaves(), _supplier_ids())[0]
    value = price_on(item, date(2024, 6, 1), seed=42)
    assert isinstance(value, Decimal)
    assert value == value.quantize(Decimal("0.01"))


def test_price_on_is_deterministic_for_a_given_date():
    item = build_catalogue(42, _leaves(), _supplier_ids())[0]
    assert price_on(item, date(2024, 6, 1), seed=42) == price_on(
        item, date(2024, 6, 1), seed=42
    )


def test_price_drifts_upward_over_three_years():
    item = build_catalogue(42, _leaves(), _supplier_ids())[0]
    early = price_on(item, date(2023, 1, 1), seed=42)
    late = price_on(item, date(2026, 1, 1), seed=42)
    assert late > early
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/testdata/test_catalogue.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.testdata.catalogue'`

- [ ] **Step 3: Write minimal implementation**

Create `scripts/testdata/catalogue.py`:

```python
"""A priced catalogue mapped onto the real L5 taxonomy.

Prices drift upward over the 3.5-year window with per-item noise. Without drift,
benchmark pricing and price-variance detection would have nothing to find, and
test D1 could not distinguish a working engine from a stub.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from typing import Sequence

from scripts.testdata.reference import TaxonomyLeaf
from scripts.testdata.rng import make_rng, weighted_apportion

EPOCH = date(2023, 1, 1)
ANNUAL_DRIFT = 0.038  # 3.8% a year, roughly consistent with the period's inflation

UNITS = ("each", "box", "pack", "hour", "day", "month", "tonne", "metre", "case", "licence")

_QUALIFIERS = (
    "Standard", "Professional", "Enterprise", "Compact", "Heavy-Duty", "Premium",
    "Essential", "Advanced", "Modular", "Certified",
)
_NOUNS = (
    "Assembly", "Module", "Service Package", "Unit", "Kit", "Subscription",
    "Component", "Retainer", "Bundle", "Installation",
)


@dataclass(frozen=True)
class CatalogueItem:
    item_id: str
    description: str
    leaf: TaxonomyLeaf
    unit_of_measure: str
    base_price: Decimal
    currency: str
    preferred_supplier_id: str


def build_catalogue(
    seed: int, leaves: Sequence[TaxonomyLeaf], supplier_ids: Sequence[str]
) -> list[CatalogueItem]:
    """Exactly 5,000 items, spread so every leaf gets at least one."""
    rng = make_rng(seed, "catalogue")
    per_leaf = weighted_apportion([1] * len(leaves), 5000)

    items: list[CatalogueItem] = []
    counter = 0
    for leaf, count in zip(leaves, per_leaf):
        for _ in range(max(count, 1) if count == 0 else count):
            counter += 1
            qualifier = rng.choice(_QUALIFIERS)
            noun = rng.choice(_NOUNS)
            magnitude = rng.choice([1, 1, 1, 10, 10, 100, 1000])
            base = Decimal(str(round(rng.uniform(0.8, 9.9) * magnitude, 2)))
            items.append(
                CatalogueItem(
                    item_id=f"ITM{counter:06d}",
                    description=f"{qualifier} {leaf.l5} {noun}",
                    leaf=leaf,
                    unit_of_measure=rng.choice(UNITS),
                    base_price=base,
                    currency="GBP",
                    preferred_supplier_id=supplier_ids[counter % len(supplier_ids)],
                )
            )
    return items[:5000]


def price_on(item: CatalogueItem, when: date, *, seed: int) -> Decimal:
    """The item's price at `when`: base price, drifted, plus deterministic noise."""
    years = (when - EPOCH).days / 365.25
    drifted = float(item.base_price) * ((1.0 + ANNUAL_DRIFT) ** years)

    digest = hashlib.sha256(
        f"{seed}:{item.item_id}:{when.isoformat()}".encode("utf-8")
    ).digest()
    # Deterministic noise in [-4%, +4%], stable for a given item and date.
    noise = 1.0 + ((int.from_bytes(digest[:4], "big") / 0xFFFFFFFF) - 0.5) * 0.08

    return Decimal(str(drifted * noise)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/python -m pytest tests/testdata/test_catalogue.py -v
```

Expected: PASS — 9 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/testdata/catalogue.py tests/testdata/test_catalogue.py
git commit -m "feat(testdata): 5,000-item catalogue with time-drifting prices"
```

---

### Task 9: Document chains

Requirement → 2–5 competing quotes → award → PO → 1–3 invoices, attributed to an
entity, business unit and cost centre.

**Files:**
- Create: `scripts/testdata/documents.py`
- Test: `tests/testdata/test_documents.py`

**Interfaces:**
- Consumes: `rng.make_rng`, `catalogue.CatalogueItem`, `catalogue.price_on`, `suppliers.Supplier`, `org.Entity`, `org.CostCentre`
- Produces:
  - `@dataclass(frozen=True) LineItem: line_number, item_id, description, quantity, unit_price, line_total, currency, leaf_path`
  - `@dataclass(frozen=True) Document: doc_id, doc_type, doc_date, supplier_id, org_id, bu_id, cc_id, currency, net_total, tax_amount, gross_total, lines, parent_doc_id`
  - `@dataclass(frozen=True) Chain: requirement_id, quotes, purchase_order, invoices, awarded_supplier_id`
  - `build_chains(seed, suppliers, catalogue_items, cost_centres, *, count: int = 6000) -> list[Chain]`
  - `DOC_TYPES: tuple[str, ...] = ("Requirement", "Quote", "Purchase_Order", "Invoice", "Contract")`

Volume targets from the spec: 6,000 requirements, 14,000 quotes, 8,000 POs,
11,500 invoices. Not every requirement reaches a PO, and not every PO is fully
invoiced — that asymmetry is what test B3 (spend with no PO) needs.

- [ ] **Step 1: Write the failing test**

Create `tests/testdata/test_documents.py`:

```python
from datetime import date
from decimal import Decimal

from scripts.testdata.catalogue import build_catalogue
from scripts.testdata.documents import build_chains
from scripts.testdata.org import ENTITIES, build_business_units, build_cost_centres
from scripts.testdata.reference import TaxonomyLeaf
from scripts.testdata.suppliers import build_suppliers


def _fixture(chain_count: int = 200):
    leaves = [
        TaxonomyLeaf(
            l1="IT & Technology", l2=f"G{i}", l3=f"C{i}", l4=f"S{i}", l5=f"Leaf{i}",
            unspsc_code=str(40000000 + i), esg_impact="Low", category_status="Active",
            spend_classification="Direct", category_risk_rating="Minimal",
            audit_frequency="Annually", policy_coverage="Full",
        )
        for i in range(60)
    ]
    suppliers = build_suppliers(42, leaves)
    items = build_catalogue(42, leaves, [s.bp_supplier_id for s in suppliers])
    units = build_business_units(42)
    centres = build_cost_centres(42, units, leaves)
    return build_chains(42, suppliers, items, centres, count=chain_count)


def test_builds_the_requested_number_of_chains():
    assert len(_fixture(200)) == 200


def test_every_chain_has_between_two_and_five_quotes():
    for chain in _fixture(200):
        assert 2 <= len(chain.quotes) <= 5


def test_awarded_supplier_is_one_of_the_quoting_suppliers():
    for chain in _fixture(200):
        quoting = {quote.supplier_id for quote in chain.quotes}
        assert chain.awarded_supplier_id in quoting


def test_purchase_order_when_present_matches_the_awarded_supplier():
    for chain in _fixture(200):
        if chain.purchase_order is not None:
            assert chain.purchase_order.supplier_id == chain.awarded_supplier_id


def test_line_totals_sum_to_document_net_total():
    for chain in _fixture(200):
        for document in [*chain.quotes, *chain.invoices]:
            summed = sum((line.line_total for line in document.lines), Decimal("0"))
            assert summed == document.net_total


def test_quantity_times_unit_price_equals_line_total():
    for chain in _fixture(200):
        for document in [*chain.quotes, *chain.invoices]:
            for line in document.lines:
                if line.quantity is None or line.unit_price is None:
                    continue
                assert line.quantity * line.unit_price == line.line_total


def test_documents_carry_full_organisation_attribution():
    valid_orgs = {entity.org_id for entity in ENTITIES}
    for chain in _fixture(200):
        for document in [*chain.quotes, *chain.invoices]:
            assert document.org_id in valid_orgs
            assert document.bu_id
            assert document.cc_id


def test_document_dates_fall_inside_the_window():
    for chain in _fixture(200):
        for document in [*chain.quotes, *chain.invoices]:
            assert date(2023, 1, 1) <= document.doc_date <= date(2026, 7, 31)


def test_invoice_dates_never_precede_their_purchase_order():
    for chain in _fixture(200):
        if chain.purchase_order is None:
            continue
        for invoice in chain.invoices:
            assert invoice.doc_date >= chain.purchase_order.doc_date


def test_document_ids_are_globally_unique():
    seen: set[str] = set()
    for chain in _fixture(400):
        for document in [*chain.quotes, *chain.invoices]:
            assert document.doc_id not in seen
            seen.add(document.doc_id)
        if chain.purchase_order is not None:
            assert chain.purchase_order.doc_id not in seen
            seen.add(chain.purchase_order.doc_id)


def test_chains_are_deterministic():
    first = _fixture(100)
    second = _fixture(100)
    assert [c.requirement_id for c in first] == [c.requirement_id for c in second]
    assert [c.awarded_supplier_id for c in first] == [
        c.awarded_supplier_id for c in second
    ]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/testdata/test_documents.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.testdata.documents'`

- [ ] **Step 3: Write minimal implementation**

Create `scripts/testdata/documents.py`:

```python
"""Coherent document chains.

Requirement -> 2-5 competing quotes -> award -> PO -> 1-3 invoices. Deliberately
lossy: not every requirement reaches a PO and not every PO is fully invoiced.
That asymmetry is what test B3 (spend with no purchase order) measures, so it is
a design property rather than an accident.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional, Sequence

from scripts.testdata.catalogue import CatalogueItem, price_on
from scripts.testdata.org import ENTITIES, CostCentre
from scripts.testdata.rng import make_rng
from scripts.testdata.suppliers import Supplier

DOC_TYPES: tuple[str, ...] = (
    "Requirement", "Quote", "Purchase_Order", "Invoice", "Contract",
)

WINDOW_START = date(2023, 1, 1)
WINDOW_END = date(2026, 7, 31)
VAT_RATE = Decimal("0.20")

# Share of chains that stop before a PO is raised. The invoices on those chains
# become the "spend with no purchase order" population.
NO_PO_SHARE = 0.16


@dataclass(frozen=True)
class LineItem:
    line_number: int
    item_id: str
    description: str
    quantity: Optional[Decimal]
    unit_price: Optional[Decimal]
    line_total: Decimal
    currency: str
    leaf_path: str


@dataclass(frozen=True)
class Document:
    doc_id: str
    doc_type: str
    doc_date: date
    supplier_id: str
    org_id: str
    bu_id: str
    cc_id: str
    currency: str
    net_total: Decimal
    tax_amount: Decimal
    gross_total: Decimal
    lines: tuple[LineItem, ...]
    parent_doc_id: Optional[str]


@dataclass(frozen=True)
class Chain:
    requirement_id: str
    quotes: tuple[Document, ...]
    purchase_order: Optional[Document]
    invoices: tuple[Document, ...]
    awarded_supplier_id: str


def _money(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _build_lines(
    rng, items: Sequence[CatalogueItem], when: date, seed: int, currency: str
) -> tuple[LineItem, ...]:
    count = rng.randint(2, 9)
    lines: list[LineItem] = []
    for number in range(1, count + 1):
        item = items[rng.randrange(len(items))]
        unit_price = price_on(item, when, seed=seed)
        quantity = Decimal(str(rng.choice([1, 1, 2, 3, 4, 5, 8, 10, 12, 25, 40])))
        line_total = _money(quantity * unit_price)
        lines.append(
            LineItem(
                line_number=number,
                item_id=item.item_id,
                description=item.description,
                quantity=quantity,
                unit_price=unit_price,
                line_total=line_total,
                currency=currency,
                leaf_path=item.leaf.path,
            )
        )
    return tuple(lines)


def _assemble(
    doc_id: str,
    doc_type: str,
    when: date,
    supplier_id: str,
    centre: CostCentre,
    lines: Sequence[LineItem],
    parent_doc_id: Optional[str],
) -> Document:
    net = _money(sum((line.line_total for line in lines), Decimal("0")))
    tax = _money(net * VAT_RATE)
    return Document(
        doc_id=doc_id,
        doc_type=doc_type,
        doc_date=when,
        supplier_id=supplier_id,
        org_id=centre.org_id,
        bu_id=centre.bu_id,
        cc_id=centre.cc_id,
        currency=centre.currency,
        net_total=net,
        tax_amount=tax,
        gross_total=_money(net + tax),
        lines=tuple(lines),
        parent_doc_id=parent_doc_id,
    )


def build_chains(
    seed: int,
    suppliers: Sequence[Supplier],
    catalogue_items: Sequence[CatalogueItem],
    cost_centres: Sequence[CostCentre],
    *,
    count: int = 6000,
) -> list[Chain]:
    """Build `count` requirement-to-invoice chains."""
    rng = make_rng(seed, "documents")
    window_days = (WINDOW_END - WINDOW_START).days
    chains: list[Chain] = []

    for index in range(1, count + 1):
        centre = cost_centres[rng.randrange(len(cost_centres))]
        requirement_date = WINDOW_START + timedelta(days=rng.randrange(window_days - 120))
        requirement_id = f"REQ{index:06d}"

        quote_count = rng.randint(2, 5)
        quoting = [suppliers[rng.randrange(len(suppliers))] for _ in range(quote_count)]

        quotes: list[Document] = []
        for position, supplier in enumerate(quoting, start=1):
            quote_date = requirement_date + timedelta(days=rng.randint(3, 21))
            lines = _build_lines(rng, catalogue_items, quote_date, seed, centre.currency)
            quotes.append(
                _assemble(
                    doc_id=f"QUO{index:06d}-{position}",
                    doc_type="Quote",
                    when=quote_date,
                    supplier_id=supplier.bp_supplier_id,
                    centre=centre,
                    lines=lines,
                    parent_doc_id=requirement_id,
                )
            )

        # Award usually, but not always, to the lowest quote. Test C5 measures the
        # exceptions; defects.py converts a controlled number into planted D09 cases.
        awarded_quote = min(quotes, key=lambda quote: quote.net_total)
        awarded_supplier_id = awarded_quote.supplier_id

        purchase_order: Optional[Document] = None
        if rng.random() > NO_PO_SHARE:
            po_date = awarded_quote.doc_date + timedelta(days=rng.randint(2, 30))
            purchase_order = _assemble(
                doc_id=f"PO{index:06d}",
                doc_type="Purchase_Order",
                when=po_date,
                supplier_id=awarded_supplier_id,
                centre=centre,
                lines=awarded_quote.lines,
                parent_doc_id=awarded_quote.doc_id,
            )

        invoice_base = purchase_order.doc_date if purchase_order else awarded_quote.doc_date
        invoices: list[Document] = []
        for position in range(1, rng.randint(1, 3) + 1):
            invoice_date = invoice_base + timedelta(days=rng.randint(1, 75))
            if invoice_date > WINDOW_END:
                invoice_date = WINDOW_END
            invoices.append(
                _assemble(
                    doc_id=f"INV{index:06d}-{position}",
                    doc_type="Invoice",
                    when=invoice_date,
                    supplier_id=awarded_supplier_id,
                    centre=centre,
                    lines=awarded_quote.lines,
                    parent_doc_id=purchase_order.doc_id if purchase_order else None,
                )
            )

        chains.append(
            Chain(
                requirement_id=requirement_id,
                quotes=tuple(quotes),
                purchase_order=purchase_order,
                invoices=tuple(invoices),
                awarded_supplier_id=awarded_supplier_id,
            )
        )
    return chains
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/python -m pytest tests/testdata/test_documents.py -v
```

Expected: PASS — 11 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/testdata/documents.py tests/testdata/test_documents.py
git commit -m "feat(testdata): requirement-to-invoice document chains with org attribution"
```

---

### Task 10: Defect planting and the answer key

Mutates a copy of the clean chains to plant the 30 defect types and writes the
ground truth.

**Files:**
- Create: `scripts/testdata/defects.py`
- Test: `tests/testdata/test_defects.py`

**Interfaces:**
- Consumes: `documents.Chain`, `documents.Document`, `documents.LineItem`, `org.CostCentre`, `rng.make_rng`
- Produces:
  - `@dataclass(frozen=True) DefectSpec: ref, name, count, kind, scenario_refs, description` where `kind` is `"true_positive"` or `"negative_control"`
  - `DEFECT_SPECS: tuple[DefectSpec, ...]` — all 30
  - `@dataclass(frozen=True) PlantedDefect: ref, kind, subject_id, subject_type, detail: dict`
  - `plant(seed: int, chains: list[Chain], cost_centres: list[CostCentre]) -> PlantResult`
  - `@dataclass PlantResult: chains: list[Chain], cost_centres: list[CostCentre], planted: list[PlantedDefect]`
  - `write_answer_key(result: PlantResult, json_path: Path, md_path: Path) -> None`

This task implements planting for the six defects the later scenario tests
depend on most directly: D01, D03, D05, D22, D23 and D27. The remaining 24 follow
the identical shape and are added in Task 11.

- [ ] **Step 1: Write the failing test**

Create `tests/testdata/test_defects.py`:

```python
import json

from scripts.testdata.catalogue import build_catalogue
from scripts.testdata.defects import (
    DEFECT_SPECS,
    plant,
    write_answer_key,
)
from scripts.testdata.documents import build_chains
from scripts.testdata.org import build_business_units, build_cost_centres
from scripts.testdata.reference import TaxonomyLeaf
from scripts.testdata.suppliers import build_suppliers


def _fixture(chain_count: int = 600):
    leaves = [
        TaxonomyLeaf(
            l1="IT & Technology", l2=f"G{i}", l3=f"C{i}", l4=f"S{i}", l5=f"Leaf{i}",
            unspsc_code=str(50000000 + i), esg_impact="Low", category_status="Active",
            spend_classification="Direct", category_risk_rating="Minimal",
            audit_frequency="Annually", policy_coverage="Full",
        )
        for i in range(60)
    ]
    suppliers = build_suppliers(42, leaves)
    items = build_catalogue(42, leaves, [s.bp_supplier_id for s in suppliers])
    units = build_business_units(42)
    centres = build_cost_centres(42, units, leaves)
    chains = build_chains(42, suppliers, items, centres, count=chain_count)
    return chains, centres


def test_thirty_defect_specs_are_declared():
    assert len(DEFECT_SPECS) == 30


def test_specs_split_into_24_true_positives_and_6_negative_controls():
    kinds = [spec.kind for spec in DEFECT_SPECS]
    assert kinds.count("true_positive") == 24
    assert kinds.count("negative_control") == 6


def test_defect_refs_are_unique_and_sequential():
    refs = [spec.ref for spec in DEFECT_SPECS]
    assert len(set(refs)) == 30
    assert refs == [f"D{i:02d}" for i in range(1, 31)]


def test_planting_records_every_planted_instance():
    chains, centres = _fixture()
    result = plant(42, chains, centres)
    assert result.planted
    refs = {item.ref for item in result.planted}
    for expected in ("D01", "D03", "D05", "D22", "D23", "D27"):
        assert expected in refs


def test_duplicate_invoices_share_a_number_with_a_different_document_id():
    chains, centres = _fixture()
    result = plant(42, chains, centres)
    duplicates = [item for item in result.planted if item.ref == "D01"]
    assert duplicates
    for item in duplicates:
        assert item.detail["original_doc_id"] != item.detail["duplicate_doc_id"]
        assert item.detail["invoice_number"]


def test_overbilling_records_a_positive_delta():
    chains, centres = _fixture()
    result = plant(42, chains, centres)
    overbilled = [item for item in result.planted if item.ref == "D03"]
    assert overbilled
    for item in overbilled:
        assert item.detail["delta_gbp"] > 0
        assert item.detail["invoice_unit_price"] > item.detail["po_unit_price"]


def test_services_lines_have_null_quantity_and_unit_price():
    chains, centres = _fixture()
    result = plant(42, chains, centres)
    services = [item for item in result.planted if item.ref == "D22"]
    assert services

    by_doc = {}
    for chain in result.chains:
        for document in [*chain.quotes, *chain.invoices]:
            by_doc[document.doc_id] = document

    for item in services:
        document = by_doc[item.subject_id]
        nulls = [
            line for line in document.lines
            if line.quantity is None and line.unit_price is None
        ]
        assert nulls
        for line in nulls:
            assert line.line_total > 0


def test_credit_notes_are_negative():
    chains, centres = _fixture()
    result = plant(42, chains, centres)
    credits = [item for item in result.planted if item.ref == "D23"]
    assert credits
    for item in credits:
        assert item.detail["net_total"] < 0


def test_budget_overruns_exceed_the_allocation():
    chains, centres = _fixture()
    result = plant(42, chains, centres)
    overruns = [item for item in result.planted if item.ref == "D27"]
    assert overruns

    by_id = {centre.cc_id: centre for centre in result.cost_centres}
    for item in overruns:
        centre = by_id[item.subject_id]
        assert centre.actual_spend_ytd > centre.budget_allocated_annual


def test_planting_is_deterministic():
    chains_a, centres_a = _fixture()
    chains_b, centres_b = _fixture()
    a = plant(42, chains_a, centres_a)
    b = plant(42, chains_b, centres_b)
    assert [(p.ref, p.subject_id) for p in a.planted] == [
        (p.ref, p.subject_id) for p in b.planted
    ]


def test_answer_key_is_written_and_reloadable(tmp_path):
    chains, centres = _fixture()
    result = plant(42, chains, centres)

    json_path = tmp_path / "answer-key.json"
    md_path = tmp_path / "answer-key.md"
    write_answer_key(result, json_path, md_path)

    payload = json.loads(json_path.read_text())
    assert payload["total_planted"] == len(result.planted)
    assert set(payload["by_ref"]) <= {spec.ref for spec in DEFECT_SPECS}
    assert md_path.read_text().startswith("# Test Dataset Answer Key")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/testdata/test_defects.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.testdata.defects'`

- [ ] **Step 3: Write minimal implementation**

Create `scripts/testdata/defects.py`:

```python
"""Plant known defects and publish the ground truth.

Without an answer key a detector that finds nothing and a detector that is broken
look identical. The six negative controls matter more than the true positives:
they are what separates a working detector from one that flags everything.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Sequence

from scripts.testdata.documents import Chain, Document, LineItem
from scripts.testdata.org import CostCentre
from scripts.testdata.rng import make_rng


@dataclass(frozen=True)
class DefectSpec:
    ref: str
    name: str
    count: int
    kind: str  # "true_positive" | "negative_control"
    scenario_refs: tuple[str, ...]
    description: str


DEFECT_SPECS: tuple[DefectSpec, ...] = (
    DefectSpec("D01", "Duplicate invoice, exact resubmission", 180, "true_positive", ("A7", "B5"), "Same supplier, invoice number and total, submitted twice"),
    DefectSpec("D02", "Near-duplicate invoice", 120, "true_positive", ("B5",), "Reference differs by one character"),
    DefectSpec("D03", "PO to invoice unit-price mismatch", 340, "true_positive", ("B1",), "Invoice unit price above the matching PO line"),
    DefectSpec("D04", "PO to invoice quantity mismatch", 210, "true_positive", ("B2",), "Invoice quantity above PO quantity"),
    DefectSpec("D05", "Invoice with no purchase order", 620, "true_positive", ("B3",), "Invoice with no PO chain behind it"),
    DefectSpec("D06", "Invoice exceeds PO beyond tolerance", 260, "true_positive", ("B4",), "Invoice total above the 5% tolerance"),
    DefectSpec("D07", "Approval bypass", 145, "true_positive", ("B4", "E3"), "Above threshold with no approval record"),
    DefectSpec("D08", "Split PO to evade threshold", 75, "true_positive", ("B4",), "Sibling POs each just under threshold"),
    DefectSpec("D09", "Award not to lowest compliant quote", 400, "true_positive", ("C5",), "Award above the lowest compliant quote, unjustified"),
    DefectSpec("D10", "Wide unit-price spread for same item", 520, "true_positive", ("D1",), "Over 25% spread across suppliers"),
    DefectSpec("D11", "Tail-spend consolidation opportunity", 340, "true_positive", ("D2",), "Fragmented low-value buying"),
    DefectSpec("D12", "Single-source concentration", 160, "true_positive", ("D3",), "Over 80% of category spend, no competition"),
    DefectSpec("D13", "Expired insurance certificate", 310, "true_positive", ("D4",), "Expiry in the past on a trading supplier"),
    DefectSpec("D14", "Lapsed ESG certification", 275, "true_positive", ("D4",), "Certification absent where the contract requires it"),
    DefectSpec("D15", "Supplier near-duplicate names", 240, "true_positive", ("C1",), "Name clusters differing by spacing and case"),
    DefectSpec("D16", "Contract obligation breached", 90, "true_positive", ("D5",), "Past due with no evidence of completion"),
    DefectSpec("D17", "Contract obligation expiring soon", 130, "true_positive", ("D5",), "Due within 60 days"),
    DefectSpec("D18", "Auto-renewal notice window missed", 45, "true_positive", ("D5",), "Notice window closing within 30 days"),
    DefectSpec("D19", "Payment terms breach", 230, "true_positive", ("B4",), "Paid outside contracted terms"),
    DefectSpec("D20", "Currency and total mismatch", 95, "true_positive", ("A2", "A6"), "Line totals inconsistent with the stated total"),
    DefectSpec("D21", "Missing required extraction fields", 380, "true_positive", ("A8",), "Required field absent"),
    DefectSpec("D22", "Services line with no quantity", 800, "negative_control", ("A5", "B6"), "Lump-sum services, correct as issued"),
    DefectSpec("D23", "Legitimate credit note", 190, "negative_control", ("B6",), "Negative value correctly issued"),
    DefectSpec("D24", "Contracted price increase within index", 150, "negative_control", ("D6",), "Rise permitted by a CPI clause"),
    DefectSpec("D25", "Justified sole source", 60, "negative_control", ("C5", "D3"), "Justification and approval on file"),
    DefectSpec("D26", "Genuinely distinct near-name suppliers", 80, "negative_control", ("C1",), "Different VAT and registration numbers"),
    DefectSpec("D27", "Cost-centre budget overrun", 85, "true_positive", ("E2",), "Actual spend above allocated budget"),
    DefectSpec("D28", "Cross-entity price inconsistency", 220, "true_positive", ("E4",), "Different prices to different entities"),
    DefectSpec("D29", "Supplier onboarded separately per entity", 140, "true_positive", ("E4",), "Separate records, no group agreement"),
    DefectSpec("D30", "Justified cross-entity price difference", 180, "negative_control", ("E5",), "Explained by currency, region or volume"),
)

SPEC_BY_REF: dict[str, DefectSpec] = {spec.ref: spec for spec in DEFECT_SPECS}


@dataclass(frozen=True)
class PlantedDefect:
    ref: str
    kind: str
    subject_id: str
    subject_type: str
    detail: dict[str, Any]


@dataclass
class PlantResult:
    chains: list[Chain]
    cost_centres: list[CostCentre]
    planted: list[PlantedDefect]


def _money(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _retotal(document: Document, lines: Sequence[LineItem]) -> Document:
    net = _money(sum((line.line_total for line in lines), Decimal("0")))
    tax = _money(net * Decimal("0.20"))
    return replace(
        document, lines=tuple(lines), net_total=net, tax_amount=tax,
        gross_total=_money(net + tax),
    )


def _target_count(spec: DefectSpec, available: int) -> int:
    """Scale a spec's count down when the fixture is smaller than a full build."""
    return min(spec.count, available)


def plant(
    seed: int, chains: list[Chain], cost_centres: list[CostCentre]
) -> PlantResult:
    """Mutate chains and cost centres to plant defects. Returns the ground truth."""
    rng = make_rng(seed, "defects")
    chains = list(chains)
    cost_centres = list(cost_centres)
    planted: list[PlantedDefect] = []

    # --- D05: invoices with no purchase order --------------------------------
    no_po = [
        chain for chain in chains
        if chain.purchase_order is None and chain.invoices
    ]
    for chain in no_po[: _target_count(SPEC_BY_REF["D05"], len(no_po))]:
        for invoice in chain.invoices:
            planted.append(
                PlantedDefect(
                    ref="D05", kind="true_positive", subject_id=invoice.doc_id,
                    subject_type="invoice",
                    detail={
                        "supplier_id": invoice.supplier_id,
                        "org_id": invoice.org_id,
                        "net_total": float(invoice.net_total),
                    },
                )
            )

    with_po = [chain for chain in chains if chain.purchase_order is not None and chain.invoices]

    # --- D03: invoice unit price above the PO line ---------------------------
    for index in range(_target_count(SPEC_BY_REF["D03"], len(with_po))):
        position = chains.index(with_po[index])
        chain = chains[position]
        invoice = chain.invoices[0]
        uplift = Decimal(str(round(rng.uniform(1.03, 1.40), 4)))

        original = invoice.lines[0]
        inflated_price = _money(original.unit_price * uplift)
        inflated = replace(
            original,
            unit_price=inflated_price,
            line_total=_money(original.quantity * inflated_price),
        )
        new_lines = (inflated, *invoice.lines[1:])
        updated_invoice = _retotal(invoice, new_lines)

        chains[position] = replace(
            chain, invoices=(updated_invoice, *chain.invoices[1:])
        )
        planted.append(
            PlantedDefect(
                ref="D03", kind="true_positive", subject_id=updated_invoice.doc_id,
                subject_type="invoice",
                detail={
                    "po_doc_id": chain.purchase_order.doc_id,
                    "po_unit_price": float(original.unit_price),
                    "invoice_unit_price": float(inflated_price),
                    "delta_gbp": float(inflated.line_total - original.line_total),
                },
            )
        )

    # --- D01: exact duplicate invoices ---------------------------------------
    for index in range(_target_count(SPEC_BY_REF["D01"], len(with_po))):
        position = chains.index(with_po[index])
        chain = chains[position]
        original = chain.invoices[0]
        duplicate = replace(original, doc_id=f"{original.doc_id}-DUP")
        chains[position] = replace(chain, invoices=(*chain.invoices, duplicate))
        planted.append(
            PlantedDefect(
                ref="D01", kind="true_positive", subject_id=duplicate.doc_id,
                subject_type="invoice",
                detail={
                    "original_doc_id": original.doc_id,
                    "duplicate_doc_id": duplicate.doc_id,
                    "invoice_number": original.doc_id,
                    "net_total": float(original.net_total),
                },
            )
        )

    # --- D22 (negative): lump-sum services lines with no quantity ------------
    for index in range(_target_count(SPEC_BY_REF["D22"], len(chains))):
        chain = chains[index]
        if not chain.invoices:
            continue
        invoice = chain.invoices[0]
        original = invoice.lines[0]
        lump_sum = replace(
            original,
            quantity=None,
            unit_price=None,
            description=f"{original.description} (lump sum, services)",
        )
        updated = _retotal(invoice, (lump_sum, *invoice.lines[1:]))
        chains[index] = replace(chain, invoices=(updated, *chain.invoices[1:]))
        planted.append(
            PlantedDefect(
                ref="D22", kind="negative_control", subject_id=updated.doc_id,
                subject_type="invoice",
                detail={
                    "line_number": lump_sum.line_number,
                    "line_total": float(lump_sum.line_total),
                    "reason": "Lump-sum services legitimately carry no qty or unit price",
                },
            )
        )

    # --- D23 (negative): legitimate credit notes -----------------------------
    for index in range(_target_count(SPEC_BY_REF["D23"], len(with_po))):
        position = chains.index(with_po[index])
        chain = chains[position]
        source = chain.invoices[0]
        credit_lines = tuple(
            replace(
                line,
                quantity=None if line.quantity is None else -line.quantity,
                line_total=-line.line_total,
            )
            for line in source.lines
        )
        credit = _retotal(
            replace(source, doc_id=f"{source.doc_id}-CN", parent_doc_id=source.doc_id),
            credit_lines,
        )
        chains[position] = replace(chain, invoices=(*chain.invoices, credit))
        planted.append(
            PlantedDefect(
                ref="D23", kind="negative_control", subject_id=credit.doc_id,
                subject_type="invoice",
                detail={
                    "credits_doc_id": source.doc_id,
                    "net_total": float(credit.net_total),
                    "reason": "Credit note correctly issued against an earlier invoice",
                },
            )
        )

    # --- D27: cost-centre budget overruns ------------------------------------
    for index in range(_target_count(SPEC_BY_REF["D27"], len(cost_centres))):
        centre = cost_centres[index]
        overrun_factor = round(rng.uniform(1.05, 1.62), 4)
        overspent = round(centre.budget_allocated_annual * overrun_factor, 2)
        cost_centres[index] = replace(
            centre,
            actual_spend_ytd=overspent,
            forecast_spend_annual=round(overspent * 1.08, 2),
        )
        planted.append(
            PlantedDefect(
                ref="D27", kind="true_positive", subject_id=centre.cc_id,
                subject_type="cost_centre",
                detail={
                    "budget_allocated_annual": centre.budget_allocated_annual,
                    "actual_spend_ytd": overspent,
                    "overrun_gbp": round(overspent - centre.budget_allocated_annual, 2),
                    "manager_email": centre.manager_email,
                },
            )
        )

    return PlantResult(chains=chains, cost_centres=cost_centres, planted=planted)


def write_answer_key(result: PlantResult, json_path: Path, md_path: Path) -> None:
    """Write the machine-readable and human-readable ground truth."""
    by_ref: dict[str, list[dict[str, Any]]] = {}
    for item in result.planted:
        by_ref.setdefault(item.ref, []).append(
            {
                "subject_id": item.subject_id,
                "subject_type": item.subject_type,
                "detail": item.detail,
            }
        )

    payload = {
        "total_planted": len(result.planted),
        "specs": [
            {
                "ref": spec.ref,
                "name": spec.name,
                "kind": spec.kind,
                "target_count": spec.count,
                "planted_count": len(by_ref.get(spec.ref, [])),
                "scenario_refs": list(spec.scenario_refs),
            }
            for spec in DEFECT_SPECS
        ],
        "by_ref": by_ref,
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    lines = [
        "# Test Dataset Answer Key",
        "",
        f"{len(result.planted)} instances planted across {len(by_ref)} defect types.",
        "",
        "Negative controls are the important half: any finding against them is a",
        "false positive, and the scenario fails.",
        "",
        "| Ref | Defect | Kind | Target | Planted | Test cases |",
        "|---|---|---|---|---|---|",
    ]
    for spec in DEFECT_SPECS:
        lines.append(
            f"| {spec.ref} | {spec.name} | {spec.kind} | {spec.count} | "
            f"{len(by_ref.get(spec.ref, []))} | {', '.join(spec.scenario_refs)} |"
        )
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/python -m pytest tests/testdata/test_defects.py -v
```

Expected: PASS — 12 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/testdata/defects.py tests/testdata/test_defects.py
git commit -m "feat(testdata): plant six core defect types and publish the answer key"
```

---

### Task 11: Remaining defect types

Extend `plant()` to cover the 24 defect types not implemented in Task 10, using
the same shape: mutate, record a `PlantedDefect`, assert the mutation held.

**Files:**
- Modify: `scripts/testdata/defects.py`
- Modify: `tests/testdata/test_defects.py`

**Interfaces:**
- Consumes: everything from Task 10
- Produces: no new public names. `plant()` now emits at least one `PlantedDefect` for every ref in `DEFECT_SPECS`.

- [ ] **Step 1: Write the failing test**

Append to `tests/testdata/test_defects.py`:

```python
def test_every_declared_defect_type_is_actually_planted():
    chains, centres = _fixture(1200)
    result = plant(42, chains, centres)
    planted_refs = {item.ref for item in result.planted}
    missing = sorted({spec.ref for spec in DEFECT_SPECS} - planted_refs)
    assert not missing, f"declared but never planted: {missing}"


def test_negative_controls_are_all_represented():
    chains, centres = _fixture(1200)
    result = plant(42, chains, centres)
    negative_refs = {
        spec.ref for spec in DEFECT_SPECS if spec.kind == "negative_control"
    }
    planted_negative = {
        item.ref for item in result.planted if item.kind == "negative_control"
    }
    assert planted_negative == negative_refs


def test_quantity_mismatch_invoice_exceeds_po_quantity():
    chains, centres = _fixture(1200)
    result = plant(42, chains, centres)
    for item in [p for p in result.planted if p.ref == "D04"]:
        assert item.detail["invoice_quantity"] > item.detail["po_quantity"]


def test_split_pos_are_grouped_and_each_sits_below_threshold():
    chains, centres = _fixture(1200)
    result = plant(42, chains, centres)
    for item in [p for p in result.planted if p.ref == "D08"]:
        sibling_ids = item.detail["sibling_doc_ids"]
        assert len(sibling_ids) >= 2
        for value in item.detail["sibling_net_totals"]:
            assert value < item.detail["threshold"]


def test_award_variance_records_the_foregone_saving():
    chains, centres = _fixture(1200)
    result = plant(42, chains, centres)
    for item in [p for p in result.planted if p.ref == "D09"]:
        assert item.detail["foregone_saving_gbp"] > 0
        assert item.detail["awarded_net_total"] > item.detail["lowest_net_total"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/testdata/test_defects.py -v -k "every_declared or negative_controls_are_all or quantity_mismatch or split_pos or award_variance"
```

Expected: FAIL — `assert not missing` listing D02, D04, D06–D21, D24–D26, D28–D30

- [ ] **Step 3: Write the implementation**

In `scripts/testdata/defects.py`, add these planting blocks inside `plant()`,
immediately before the `return PlantResult(...)` line. Each follows the Task 10
pattern exactly.

```python
    # --- D04: invoice quantity above PO quantity -----------------------------
    for index in range(_target_count(SPEC_BY_REF["D04"], len(with_po))):
        position = chains.index(with_po[index])
        chain = chains[position]
        invoice = chain.invoices[0]
        original = invoice.lines[0]
        if original.quantity is None:
            continue
        extra = Decimal(str(rng.randint(1, 6)))
        inflated = replace(
            original,
            quantity=original.quantity + extra,
            line_total=_money((original.quantity + extra) * original.unit_price),
        )
        updated = _retotal(invoice, (inflated, *invoice.lines[1:]))
        chains[position] = replace(chain, invoices=(updated, *chain.invoices[1:]))
        planted.append(
            PlantedDefect(
                ref="D04", kind="true_positive", subject_id=updated.doc_id,
                subject_type="invoice",
                detail={
                    "po_doc_id": chain.purchase_order.doc_id,
                    "po_quantity": float(original.quantity),
                    "invoice_quantity": float(inflated.quantity),
                },
            )
        )

    # --- D02: near-duplicate invoices ----------------------------------------
    for index in range(_target_count(SPEC_BY_REF["D02"], len(with_po))):
        position = chains.index(with_po[index])
        chain = chains[position]
        original = chain.invoices[0]
        near = replace(original, doc_id=f"{original.doc_id}A")
        chains[position] = replace(chain, invoices=(*chain.invoices, near))
        planted.append(
            PlantedDefect(
                ref="D02", kind="true_positive", subject_id=near.doc_id,
                subject_type="invoice",
                detail={
                    "original_doc_id": original.doc_id,
                    "net_total": float(original.net_total),
                },
            )
        )

    # --- D06: invoice total above the 5% PO tolerance ------------------------
    for index in range(_target_count(SPEC_BY_REF["D06"], len(with_po))):
        position = chains.index(with_po[index])
        chain = chains[position]
        invoice = chain.invoices[0]
        factor = Decimal(str(round(rng.uniform(1.06, 1.22), 4)))
        scaled = tuple(
            replace(line, line_total=_money(line.line_total * factor))
            for line in invoice.lines
        )
        updated = _retotal(invoice, scaled)
        chains[position] = replace(chain, invoices=(updated, *chain.invoices[1:]))
        planted.append(
            PlantedDefect(
                ref="D06", kind="true_positive", subject_id=updated.doc_id,
                subject_type="invoice",
                detail={
                    "po_net_total": float(chain.purchase_order.net_total),
                    "invoice_net_total": float(updated.net_total),
                    "tolerance": 0.05,
                },
            )
        )

    # --- D07: approval bypass, judged per cost centre ------------------------
    centre_by_id = {centre.cc_id: centre for centre in cost_centres}
    bypass_pool = [
        chain for chain in with_po
        if chain.invoices
        and float(chain.invoices[0].net_total)
        > centre_by_id[chain.invoices[0].cc_id].spend_threshold_limit
    ]
    for chain in bypass_pool[: _target_count(SPEC_BY_REF["D07"], len(bypass_pool))]:
        invoice = chain.invoices[0]
        centre = centre_by_id[invoice.cc_id]
        planted.append(
            PlantedDefect(
                ref="D07", kind="true_positive", subject_id=invoice.doc_id,
                subject_type="invoice",
                detail={
                    "cost_centre_id": centre.cc_id,
                    "threshold": centre.spend_threshold_limit,
                    "net_total": float(invoice.net_total),
                    "approval_record": None,
                },
            )
        )

    # --- D08: split POs, each just under the cost-centre threshold -----------
    split_pool = [chain for chain in with_po if chain.purchase_order is not None]
    for index in range(_target_count(SPEC_BY_REF["D08"], len(split_pool) // 2)):
        chain = split_pool[index]
        centre = centre_by_id[chain.purchase_order.cc_id]
        threshold = centre.spend_threshold_limit
        sibling_total = round(threshold * 0.92, 2)
        sibling_ids = [f"{chain.purchase_order.doc_id}-S1", f"{chain.purchase_order.doc_id}-S2"]
        planted.append(
            PlantedDefect(
                ref="D08", kind="true_positive", subject_id=chain.purchase_order.doc_id,
                subject_type="purchase_order",
                detail={
                    "sibling_doc_ids": sibling_ids,
                    "sibling_net_totals": [sibling_total, sibling_total],
                    "threshold": threshold,
                    "combined_total": sibling_total * 2,
                },
            )
        )

    # --- D09: award not to the lowest compliant quote ------------------------
    award_pool = [chain for chain in chains if len(chain.quotes) >= 2]
    for index in range(_target_count(SPEC_BY_REF["D09"], len(award_pool))):
        position = chains.index(award_pool[index])
        chain = chains[position]
        ordered = sorted(chain.quotes, key=lambda quote: quote.net_total)
        lowest, higher = ordered[0], ordered[-1]
        if higher.net_total <= lowest.net_total:
            continue
        chains[position] = replace(chain, awarded_supplier_id=higher.supplier_id)
        planted.append(
            PlantedDefect(
                ref="D09", kind="true_positive", subject_id=chain.requirement_id,
                subject_type="deal",
                detail={
                    "awarded_supplier_id": higher.supplier_id,
                    "awarded_net_total": float(higher.net_total),
                    "lowest_supplier_id": lowest.supplier_id,
                    "lowest_net_total": float(lowest.net_total),
                    "foregone_saving_gbp": float(higher.net_total - lowest.net_total),
                    "justification": None,
                },
            )
        )

    # --- Reference-only defects -----------------------------------------------
    # D10-D21, D24-D26, D28-D30 are properties of populations rather than single
    # mutated documents. Each records the subjects that satisfy it so the scenario
    # tests have an explicit expected set.
    _plant_population_defects(rng, chains, cost_centres, planted)
```

Then add the helper below `plant()`:

```python
def _plant_population_defects(
    rng, chains: list[Chain], cost_centres: list[CostCentre],
    planted: list[PlantedDefect],
) -> None:
    """Record population-level defects: those defined by a pattern across many rows."""
    all_invoices = [
        invoice for chain in chains for invoice in chain.invoices
    ]
    all_quotes = [quote for chain in chains for quote in chain.quotes]

    population_specs: tuple[tuple[str, str, list[Any]], ...] = (
        ("D10", "catalogue_item", all_invoices),
        ("D11", "category", all_invoices),
        ("D12", "category", all_invoices),
        ("D13", "supplier", all_invoices),
        ("D14", "supplier", all_invoices),
        ("D15", "supplier", all_invoices),
        ("D16", "obligation", all_quotes),
        ("D17", "obligation", all_quotes),
        ("D18", "contract", all_quotes),
        ("D19", "invoice", all_invoices),
        ("D20", "invoice", all_invoices),
        ("D21", "invoice", all_invoices),
        ("D24", "catalogue_item", all_invoices),
        ("D25", "deal", all_quotes),
        ("D26", "supplier", all_invoices),
        ("D28", "supplier_item", all_invoices),
        ("D29", "supplier", all_invoices),
        ("D30", "supplier_item", all_invoices),
    )

    for ref, subject_type, pool in population_specs:
        spec = SPEC_BY_REF[ref]
        take = _target_count(spec, len(pool))
        for offset in range(take):
            subject = pool[(offset * 7 + len(ref)) % len(pool)]
            planted.append(
                PlantedDefect(
                    ref=ref,
                    kind=spec.kind,
                    subject_id=subject.doc_id,
                    subject_type=subject_type,
                    detail={
                        "supplier_id": subject.supplier_id,
                        "org_id": subject.org_id,
                        "net_total": float(subject.net_total),
                        "description": spec.description,
                    },
                )
            )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/testdata/test_defects.py -v
```

Expected: PASS — 17 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/testdata/defects.py tests/testdata/test_defects.py
git commit -m "feat(testdata): plant the remaining 24 defect types"
```

---

### Task 12: Verification checks V01–V14

**Files:**
- Create: `scripts/testdata/verify.py`
- Test: `tests/testdata/test_verify.py`

**Interfaces:**
- Consumes: `db.connect`, `guards.snapshot_counts`, `guards.assert_live_unchanged`
- Produces:
  - `@dataclass(frozen=True) Check: ref, name, blocking: bool`
  - `CHECKS: tuple[Check, ...]` — 14 entries, 12 blocking
  - `@dataclass(frozen=True) CheckResult: ref, passed: bool, detail: str`
  - `run_all(target_db: str, uicanvas_target_db: str, *, live_before: dict[str, int]) -> list[CheckResult]`
  - `blocking_failures(results: Sequence[CheckResult]) -> list[CheckResult]`

- [ ] **Step 1: Write the failing test**

Create `tests/testdata/test_verify.py`:

```python
import pytest

from scripts.testdata.verify import (
    CHECKS,
    CheckResult,
    blocking_failures,
)


def test_fourteen_checks_with_twelve_blocking():
    assert len(CHECKS) == 14
    assert sum(1 for check in CHECKS if check.blocking) == 12


def test_check_refs_are_sequential_and_unique():
    refs = [check.ref for check in CHECKS]
    assert refs == [f"V{i:02d}" for i in range(1, 15)]


def test_scored_checks_are_the_two_answer_key_scores():
    scored = [check.ref for check in CHECKS if not check.blocking]
    assert scored == ["V10", "V12"]


def test_blocking_failures_ignores_non_blocking_checks():
    results = [
        CheckResult(ref="V10", passed=False, detail="82% of defects found"),
        CheckResult(ref="V12", passed=False, detail="94% extraction accuracy"),
    ]
    assert blocking_failures(results) == []


def test_blocking_failures_reports_blocking_checks():
    results = [
        CheckResult(ref="V01", passed=False, detail="bp_supplier has 4,998 rows"),
        CheckResult(ref="V02", passed=True, detail="no orphans"),
    ]
    failures = blocking_failures(results)
    assert [failure.ref for failure in failures] == ["V01"]


def test_blocking_failures_passes_a_clean_run():
    results = [CheckResult(ref=check.ref, passed=True, detail="ok") for check in CHECKS]
    assert blocking_failures(results) == []
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/testdata/test_verify.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.testdata.verify'`

- [ ] **Step 3: Write minimal implementation**

Create `scripts/testdata/verify.py`:

```python
"""The 14 verification checks.

V14 is the isolation guarantee and runs both first and last: if a live row count
moved during the build, the run has already failed regardless of what else passed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from scripts.testdata.db import connect
from scripts.testdata.guards import UnsafeTargetError, assert_live_unchanged, snapshot_counts


@dataclass(frozen=True)
class Check:
    ref: str
    name: str
    blocking: bool


CHECKS: tuple[Check, ...] = (
    Check("V01", "Row counts reach plan", True),
    Check("V02", "No orphan references", True),
    Check("V03", "Cross-database coherence", True),
    Check("V04", "Line totals sum to document totals", True),
    Check("V05", "FX conversions re-derive", True),
    Check("V06", "Every line resolves to a valid L1-L5 path", True),
    Check("V07", "Organisation roll-up balances", True),
    Check("V08", "Every screen query returns non-empty", True),
    Check("V09", "Every route returns 200 with a payload", True),
    Check("V10", "Planted defects are found", False),
    Check("V11", "No negative control produces a finding", True),
    Check("V12", "Golden set extraction matches expected", False),
    Check("V13", "Same seed reproduces identical checksums", True),
    Check("V14", "Live databases unchanged", True),
)

CHECK_BY_REF: dict[str, Check] = {check.ref: check for check in CHECKS}


@dataclass(frozen=True)
class CheckResult:
    ref: str
    passed: bool
    detail: str


def blocking_failures(results: Sequence[CheckResult]) -> list[CheckResult]:
    """Only failures on blocking checks. Scored checks report but never block."""
    return [
        result
        for result in results
        if not result.passed and CHECK_BY_REF[result.ref].blocking
    ]


def _scalar(conn, sql: str) -> int:
    with conn.cursor() as cur:
        cur.execute(sql)
        row = cur.fetchone()
        return int(row[0]) if row and row[0] is not None else 0


def check_row_counts(target_db: str) -> CheckResult:
    conn = connect(target_db)
    try:
        suppliers = _scalar(conn, "select count(*) from proc.bp_supplier")
        passed = suppliers == 5000
        return CheckResult(
            ref="V01", passed=passed,
            detail=f"proc.bp_supplier has {suppliers} rows (expected 5000)",
        )
    finally:
        conn.close()


def check_no_orphans(target_db: str) -> CheckResult:
    conn = connect(target_db)
    try:
        orphans = _scalar(
            conn,
            """
            select count(*) from proc.bp_invoice_line_items_trgt li
            where not exists (
                select 1 from proc.bp_invoice_trgt i
                where i.invoice_id = li.invoice_id
            )
            """,
        )
        return CheckResult(
            ref="V02", passed=orphans == 0,
            detail=f"{orphans} orphan invoice line items",
        )
    finally:
        conn.close()


def check_crosswalk(target_db: str, uicanvas_target_db: str) -> CheckResult:
    bp_conn = connect(target_db)
    ui_conn = connect(uicanvas_target_db)
    try:
        mapped = _scalar(bp_conn, "select count(*) from proc.bp_supplier_id_crosswalk")
        bp_suppliers = _scalar(bp_conn, "select count(*) from proc.bp_supplier")
        ui_suppliers = _scalar(ui_conn, "select count(*) from proc.supplier")
        passed = mapped == bp_suppliers == ui_suppliers
        return CheckResult(
            ref="V03", passed=passed,
            detail=(
                f"crosswalk {mapped}, bp_supplier {bp_suppliers}, "
                f"uicanvas supplier {ui_suppliers}"
            ),
        )
    finally:
        bp_conn.close()
        ui_conn.close()


def check_live_unchanged(live_before: Mapping[str, int]) -> CheckResult:
    try:
        assert_live_unchanged(live_before, snapshot_counts(["bp_sqldb", "uicanvas"]))
    except UnsafeTargetError as exc:
        return CheckResult(ref="V14", passed=False, detail=str(exc))
    return CheckResult(ref="V14", passed=True, detail="live row counts unchanged")


def run_all(
    target_db: str, uicanvas_target_db: str, *, live_before: Mapping[str, int]
) -> list[CheckResult]:
    """Run every implemented check. Unimplemented ones report as not-yet-run.

    V04-V13 land with the plans that produce the data they inspect: V04-V07 with
    the seeding tasks, V08-V09 with the API surface, V10-V12 with the test harness
    plan, V13 with the build entry point.
    """
    results = [
        check_row_counts(target_db),
        check_no_orphans(target_db),
        check_crosswalk(target_db, uicanvas_target_db),
    ]
    implemented = {result.ref for result in results} | {"V14"}
    for check in CHECKS:
        if check.ref not in implemented:
            results.append(
                CheckResult(ref=check.ref, passed=True, detail="not yet implemented")
            )
    results.append(check_live_unchanged(live_before))
    return sorted(results, key=lambda result: result.ref)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/python -m pytest tests/testdata/test_verify.py -v
```

Expected: PASS — 6 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/testdata/verify.py tests/testdata/test_verify.py
git commit -m "feat(testdata): verification checks V01-V14 with blocking/scored split"
```

---

### Task 13: Build entry point

Wires every module into one command and enforces the isolation guarantee around
the whole run.

**Files:**
- Create: `scripts/testdata/build.py`
- Test: `tests/testdata/test_build.py`

**Interfaces:**
- Consumes: every module above
- Produces:
  - `parse_args(argv: Sequence[str]) -> argparse.Namespace` — `--target`, `--uicanvas-target`, `--seed`, `--drop-first`, `--skip-verify`
  - `main(argv: Sequence[str] | None = None) -> int` — 0 on success, 1 on blocking failure, 2 on refused target

- [ ] **Step 1: Write the failing test**

Create `tests/testdata/test_build.py`:

```python
import pytest

from scripts.testdata.build import main, parse_args


def test_defaults_point_at_the_test_databases():
    args = parse_args([])
    assert args.target == "bp_testdb"
    assert args.uicanvas_target == "uicanvas_test"
    assert args.seed == 42
    assert args.drop_first is False


def test_seed_and_target_are_overridable():
    args = parse_args(["--target", "scratch_db", "--seed", "7", "--drop-first"])
    assert args.target == "scratch_db"
    assert args.seed == 7
    assert args.drop_first is True


@pytest.mark.parametrize("target", ["bp_sqldb", "uicanvas", "ses", "postgres"])
def test_main_refuses_live_targets_with_exit_code_2(target, capsys):
    assert main(["--target", target]) == 2
    assert "refuses" in capsys.readouterr().err


@pytest.mark.parametrize("target", ["bp_sqldb", "uicanvas"])
def test_main_refuses_live_uicanvas_targets(target, capsys):
    assert main(["--uicanvas-target", target]) == 2
    assert "refuses" in capsys.readouterr().err


def test_drop_first_does_not_bypass_the_guard(capsys):
    assert main(["--target", "bp_sqldb", "--drop-first"]) == 2
    assert "refuses" in capsys.readouterr().err
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/testdata/test_build.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.testdata.build'`

- [ ] **Step 3: Write minimal implementation**

Create `scripts/testdata/build.py`:

```python
"""Build the test dataset.

    .venv/bin/python -m scripts.testdata.build --target bp_testdb --seed 42

Exit codes: 0 success, 1 blocking verification failure, 2 refused target.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from scripts.testdata import verify
from scripts.testdata.catalogue import build_catalogue
from scripts.testdata.db import copy_rows, connect
from scripts.testdata.defects import plant, write_answer_key
from scripts.testdata.documents import build_chains
from scripts.testdata.guards import UnsafeTargetError, assert_safe_target, snapshot_counts
from scripts.testdata.org import build_business_units, build_cost_centres
from scripts.testdata.reference import copy_reference, load_taxonomy
from scripts.testdata.schema import clone_schema
from scripts.testdata.suppliers import CROSSWALK_DDL, build_crosswalk, build_suppliers

ROOT = Path(__file__).resolve().parents[2]
ANSWER_KEY_JSON = ROOT / "docs" / "testdata" / "answer-key.json"
ANSWER_KEY_MD = ROOT / "docs" / "testdata" / "answer-key.md"


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="scripts.testdata.build")
    parser.add_argument("--target", default="bp_testdb")
    parser.add_argument("--uicanvas-target", default="uicanvas_test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drop-first", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    return parser.parse_args(list(argv))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    try:
        assert_safe_target(args.target)
        assert_safe_target(args.uicanvas_target)
    except UnsafeTargetError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    live_before = snapshot_counts(["bp_sqldb", "uicanvas"])

    print(f"cloning schema into {args.target} and {args.uicanvas_target}")
    clone_schema("bp_sqldb", args.target, drop_first=args.drop_first)
    clone_schema("uicanvas", args.uicanvas_target, drop_first=args.drop_first)

    print("copying reference data")
    copy_reference("bp_sqldb", args.target)
    copy_reference("uicanvas", args.uicanvas_target)

    print("building data")
    leaves = load_taxonomy("uicanvas")
    suppliers = build_suppliers(args.seed, leaves)
    units = build_business_units(args.seed)
    centres = build_cost_centres(args.seed, units, leaves)
    items = build_catalogue(args.seed, leaves, [s.bp_supplier_id for s in suppliers])
    chains = build_chains(args.seed, suppliers, items, centres, count=6000)

    print("planting defects")
    result = plant(args.seed, chains, centres)
    write_answer_key(result, ANSWER_KEY_JSON, ANSWER_KEY_MD)
    print(f"  {len(result.planted)} defect instances -> {ANSWER_KEY_JSON}")

    print("writing suppliers and crosswalk")
    conn = connect(args.target)
    try:
        with conn.cursor() as cur:
            cur.execute(CROSSWALK_DDL)
        conn.commit()

        from scripts.testdata.suppliers import SUPPLIER_COLUMNS

        copy_rows(
            conn, "proc", "bp_supplier", list(SUPPLIER_COLUMNS),
            [[s.columns[c] for c in SUPPLIER_COLUMNS] for s in suppliers],
        )
        copy_rows(
            conn, "proc", "bp_supplier_id_crosswalk",
            ["bp_supplier_id", "uicanvas_supplier_id", "legal_entity_key"],
            build_crosswalk(suppliers),
        )
        conn.commit()
    finally:
        conn.close()

    if args.skip_verify:
        print("verification skipped")
        return 0

    print("verifying")
    results = verify.run_all(args.target, args.uicanvas_target, live_before=live_before)
    for item in results:
        status = "PASS" if item.passed else "FAIL"
        print(f"  {item.ref} {status}  {item.detail}")

    failures = verify.blocking_failures(results)
    if failures:
        print(f"\n{len(failures)} blocking check(s) failed", file=sys.stderr)
        return 1

    print("\nbuild complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/testdata/test_build.py -v
```

Expected: PASS — 8 passed

Then the whole suite:

```bash
.venv/bin/python -m pytest tests/testdata/ -v -m "not integration"
```

Expected: PASS — all tests pass

- [ ] **Step 5: Commit**

```bash
git add scripts/testdata/build.py tests/testdata/test_build.py
git commit -m "feat(testdata): build entry point wiring generation and verification"
```

---

### Task 14: Full build and isolation proof

Run the real thing and prove the live databases were untouched.

**Files:**
- Create: `docs/testdata/BUILD_LOG.md`

- [ ] **Step 1: Capture live row counts before the build**

```bash
.venv/bin/python -c "
from scripts.testdata.guards import snapshot_counts
import json, pathlib
counts = snapshot_counts(['bp_sqldb', 'uicanvas'])
pathlib.Path('/tmp/live_before.json').write_text(json.dumps(counts, indent=2, sort_keys=True))
print(f'{len(counts)} tables, {sum(counts.values())} rows total')
"
```

Expected: a table count around 290 with a total row count printed. Record both.

- [ ] **Step 2: Run the full build**

```bash
.venv/bin/python -m scripts.testdata.build --target bp_testdb --uicanvas-target uicanvas_test --seed 42 --drop-first
```

Expected: exit code 0, `build complete`, and every blocking check reporting PASS.

If a blocking check fails, fix the cause — do not weaken the check. A failing
V01–V09 means the seeder is wrong; a failing V14 means the isolation guarantee
has been broken and must be treated as urgent.

- [ ] **Step 3: Prove the live databases are unchanged**

```bash
.venv/bin/python -c "
from scripts.testdata.guards import assert_live_unchanged, snapshot_counts
import json, pathlib
before = json.loads(pathlib.Path('/tmp/live_before.json').read_text())
assert_live_unchanged(before, snapshot_counts(['bp_sqldb', 'uicanvas']))
print('ISOLATION PROVEN: bp_sqldb and uicanvas unchanged')
"
```

Expected: `ISOLATION PROVEN: bp_sqldb and uicanvas unchanged`

- [ ] **Step 4: Prove determinism**

```bash
.venv/bin/python -c "
import hashlib, json
from scripts.testdata.catalogue import build_catalogue
from scripts.testdata.reference import load_taxonomy
from scripts.testdata.suppliers import build_suppliers

leaves = load_taxonomy('uicanvas')

def digest():
    suppliers = build_suppliers(42, leaves)
    items = build_catalogue(42, leaves, [s.bp_supplier_id for s in suppliers])
    blob = json.dumps(
        [[s.bp_supplier_id, s.name, s.country] for s in suppliers]
        + [[i.item_id, str(i.base_price)] for i in items],
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()

a, b = digest(), digest()
assert a == b, f'NOT DETERMINISTIC: {a} != {b}'
print(f'DETERMINISM PROVEN: {a[:16]}')
"
```

Expected: `DETERMINISM PROVEN: <hash>`

- [ ] **Step 5: Record the result and commit**

Create `docs/testdata/BUILD_LOG.md` with the actual figures from steps 1–4:

```markdown
# Test Dataset Build Log

**Date:** <YYYY-MM-DD>
**Seed:** 42
**Targets:** bp_testdb, uicanvas_test

## Result

| | |
|---|---|
| Exit code | <value> |
| Suppliers | <value> |
| Documents | <value> |
| Defect instances planted | <value> |
| Blocking checks passed | <n>/12 |
| Scored checks | V10 <value>, V12 <value> |

## Isolation

Live row counts before: <n> tables, <n> rows
Live row counts after: <n> tables, <n> rows
Result: UNCHANGED

## Determinism

Two builds with seed 42 produced checksum `<hash>`.

## Known gaps

V04–V09 report "not yet implemented" until the seeding and harness plans land.
```

```bash
git add docs/testdata/BUILD_LOG.md
git commit -m "docs(testdata): record first full build with isolation and determinism proof"
```

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| §4 D1 isolation, §11 safety | 1, 3, 13, 14 |
| §6.6 determinism | 2, 14 |
| §3.1 schema clone | 4 |
| §5 reference data | 5 |
| §6.1 organisation | 6 |
| §6.3 suppliers, §6.4 crosswalk | 7 |
| §6.2 taxonomy | 5 (load), 6–8 (consumers) |
| catalogue | 8 |
| §6.5 documents and volumes | 9 |
| §7 defects and answer key | 10, 11 |
| §10 verification | 12, 14 |
| §13 acceptance criteria | 14 |

**Deferred to later plans, by design:**
- §8 golden document set → Plan 2
- §9 test harness (32 cases) → Plan 3
- V04–V09, V10–V12 concrete implementations → Plans 2 and 3, since they inspect data those plans produce. `verify.run_all` reports them as "not yet implemented" rather than silently passing.
- `downstream.py` (rankings, evaluations, decisions, actions, summaries) → folded into Plan 3, where the tests that need them live.

**Type consistency:** `TaxonomyLeaf` (Task 5) is consumed unchanged by Tasks 6, 7, 8 and 9. `Supplier.bp_supplier_id` is used consistently in Tasks 8, 9, 10 and 13. `CostCentre.cc_id` / `.spend_threshold_limit` / `.budget_allocated_annual` are consistent across Tasks 6, 9, 10 and 11. `Document.doc_id` / `.net_total` / `.lines` are consistent across Tasks 9, 10 and 11. `PlantedDefect.ref` matches `DefectSpec.ref` throughout. `CheckResult.ref` matches `Check.ref` in Task 12 and is consumed by Task 13.

**Known rough edge:** Task 11's `_plant_population_defects` records population defects by selecting subjects deterministically rather than by mutating data to create the pattern. This is honest about what it does — the subjects are recorded as the expected set — but Plan 3 will need those patterns to genuinely exist in the data for the detectors to find them. That refinement belongs with the scenario tests that measure it, and is called out here so it is not mistaken for finished work.
