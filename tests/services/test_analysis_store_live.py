"""Live integration tests for analysis_store against the real database.

These tests run against the live database to verify idempotency guarantees.
They are skipped cleanly if no database is available.
"""
import os
import uuid

import psycopg2
import pytest
from dotenv import load_dotenv

from src.services import analysis_store


# Load environment variables from .env
load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", 5432))
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# Skip the entire module if database credentials are missing or connection fails
_DB_AVAILABLE = False
_SKIP_REASON = ""

if all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD]):
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
        )
        conn.close()
        _DB_AVAILABLE = True
    except Exception as e:
        _SKIP_REASON = f"Could not connect to database: {e}"
else:
    _SKIP_REASON = "Database environment variables not set"

if not _DB_AVAILABLE:
    pytest.skip(_SKIP_REASON, allow_module_level=True)


def get_db_conn():
    """Create and return a database connection."""
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


def test_start_is_idempotent_preserves_chosen_name():
    """Prove idempotency: a second call returns the same id and preserves the name.

    This is the core safety property: COALESCE prevents a second call with name=None
    from wiping out the user's chosen name.
    """
    session_id = f"pytest-{uuid.uuid4().hex[:12]}"
    conn = get_db_conn()
    try:
        cur = conn.cursor()

        # First call: user provides a chosen name
        analysis_id_1 = analysis_store.start(
            session_id=session_id,
            name="Chosen name",
            mode="new",
            conn=conn,
        )

        # Second call: sweep fallback provides no name
        analysis_id_2 = analysis_store.start(
            session_id=session_id,
            name=None,
            mode="new",
            conn=conn,
        )

        # Both calls must return the same id
        assert analysis_id_1 == analysis_id_2, "Second call must return the same analysis_id"

        # Exactly one row must exist for this session_id
        cur.execute("SELECT COUNT(*) FROM proc.bp_analysis WHERE session_id = %s", (session_id,))
        count = cur.fetchone()[0]
        assert count == 1, f"Expected exactly 1 row for session_id, got {count}"

        # The name must still be the user's chosen name (not NULL)
        cur.execute("SELECT name FROM proc.bp_analysis WHERE session_id = %s", (session_id,))
        row = cur.fetchone()
        assert row is not None, "Row not found"
        assert row[0] == "Chosen name", f"Name was overwritten; expected 'Chosen name', got {row[0]!r}"

    finally:
        # Always clean up: delete the test row
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM proc.bp_analysis WHERE session_id = %s", (session_id,))
            conn.commit()
        except Exception as e:
            # If cleanup fails, log but don't raise - test ran, cleanup is best-effort
            print(f"Cleanup failed for {session_id}: {e}")
        finally:
            conn.close()


def test_start_writes_all_columns():
    """Verify that all input columns are written to the database."""
    session_id = f"pytest-{uuid.uuid4().hex[:12]}"
    conn = get_db_conn()
    try:
        cur = conn.cursor()

        analysis_id = analysis_store.start(
            session_id=session_id,
            name="Integration test",
            mode="amend",
            created_by="test@example.com",
            conn=conn,
        )

        # Verify all columns
        cur.execute(
            "SELECT session_id, name, mode, created_by FROM proc.bp_analysis WHERE analysis_id = %s",
            (analysis_id,),
        )
        row = cur.fetchone()
        assert row is not None, "Row not found"
        assert row[0] == session_id
        assert row[1] == "Integration test"
        assert row[2] == "amend"
        assert row[3] == "test@example.com"

    finally:
        # Always clean up
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM proc.bp_analysis WHERE session_id = %s", (session_id,))
            conn.commit()
        except Exception as e:
            print(f"Cleanup failed for {session_id}: {e}")
        finally:
            conn.close()
