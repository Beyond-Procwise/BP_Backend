"""Exactly one ACTIVE governance row per (type, name).

Every resolver in the platform reads governance the same way: match on type + name, filter to
status = 1, take one row. PromptEngine, PolicyEngine, the ask-persona lookup in
model_selector, summary_agent's persona, and the AgentNick get_prompt/get_policy tools all do
this. None of them can express "there were two candidates" — a second active row does not
error, it silently changes which prompt governs the assistant or which threshold is enforced,
and the choice comes down to physical row order.

That was tolerable while the only writers were migrations. It stopped being tolerable when the
Prompts and Policies admin screens were repointed off uicanvas.proc.prompt|policy (which no
agent read) onto these tables: the screens can now create and edit governance, so the
duplicate is a click away rather than a hypothetical.

The gateway checks for a clash before writing and returns a 409, but a check-then-write in
application code is not a guarantee — two concurrent edits both pass the check. The partial
unique index is the guarantee, and these tests assert it is present and that it still permits
the version history the update path depends on.
"""

import os

import psycopg2
import pytest
from dotenv import load_dotenv

load_dotenv()

TABLES = [
    ("bp_prompt", "prompt_type", "prompt_name", "prompts_status", "ux_bp_prompt_active_type_name"),
    ("bp_policy", "policy_type", "policy_name", "policy_status", "ux_bp_policy_active_type_name"),
]


@pytest.fixture(scope="module")
def conn():
    try:
        c = psycopg2.connect(
            host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT", 5432),
            dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"), connect_timeout=8,
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"bp_sqldb not reachable: {exc}")
    yield c
    c.close()


@pytest.mark.parametrize("table,type_col,name_col,status_col,index_name", TABLES)
def test_partial_unique_index_exists(conn, table, type_col, name_col, status_col, index_name):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT indexdef FROM pg_indexes WHERE schemaname='proc' AND indexname=%s",
            (index_name,),
        )
        row = cur.fetchone()
    assert row, (
        f"{index_name} is missing. Without it the admin screens can create two active "
        f"{table} rows for one (type, name), and which one governs becomes row-order luck."
    )
    definition = row[0].lower()
    assert "unique" in definition
    assert type_col in definition and name_col in definition
    # Partial, not total: superseded rows ARE the version history, and the update path relies
    # on being able to keep them.
    assert "where" in definition and f"{status_col} = 1" in definition


@pytest.mark.parametrize("table,type_col,name_col,status_col,index_name", TABLES)
def test_no_duplicate_active_rows_today(conn, table, type_col, name_col, status_col, index_name):
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT {type_col}, {name_col}, COUNT(*) FROM proc.{table} "
            f"WHERE {status_col} = 1 GROUP BY 1, 2 HAVING COUNT(*) > 1"
        )
        dupes = cur.fetchall()
    assert not dupes, f"ambiguous active governance in proc.{table}: {dupes}"


@pytest.mark.parametrize("table,type_col,name_col,status_col,index_name", TABLES)
def test_index_rejects_a_second_active_row(conn, table, type_col, name_col, status_col, index_name):
    """Prove it actually bites, rather than trusting the definition string."""
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT {type_col}, {name_col} FROM proc.{table} WHERE {status_col} = 1 LIMIT 1"
        )
        existing = cur.fetchone()
        if not existing:
            pytest.skip(f"proc.{table} has no active rows to collide with")
        ptype, pname = existing

        payload = "prompts_desc" if table == "bp_prompt" else "policy_details"
        with pytest.raises(psycopg2.errors.UniqueViolation):
            cur.execute(
                f"INSERT INTO proc.{table} ({type_col}, {name_col}, {payload}, {status_col}) "
                f"VALUES (%s, %s, '{{}}'::jsonb, 1)",
                (ptype, pname),
            )
    conn.rollback()


@pytest.mark.parametrize("table,type_col,name_col,status_col,index_name", TABLES)
def test_index_still_allows_superseded_history(conn, table, type_col, name_col, status_col, index_name):
    """An edit writes a new active row and demotes the old one — history must remain legal."""
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT {type_col}, {name_col} FROM proc.{table} WHERE {status_col} = 1 LIMIT 1"
        )
        existing = cur.fetchone()
        if not existing:
            pytest.skip(f"proc.{table} has no active rows")
        ptype, pname = existing

        payload = "prompts_desc" if table == "bp_prompt" else "policy_details"
        cur.execute(
            f"INSERT INTO proc.{table} ({type_col}, {name_col}, {payload}, {status_col}) "
            f"VALUES (%s, %s, '{{}}'::jsonb, 0)",
            (ptype, pname),
        )
        assert cur.rowcount == 1, "a superseded duplicate must still be insertable"
    conn.rollback()
