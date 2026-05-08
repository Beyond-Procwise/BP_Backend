"""Pytest fixtures for extraction_v3 tests."""
from pathlib import Path
import sys
import os
import pytest
import psycopg2

# Ensure project root is in path for config imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from config.settings import Settings


MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "scripts/migrations"
V3_MIGRATIONS = sorted(MIGRATIONS_DIR.glob("2026-05-*-*.sql"))


@pytest.fixture(scope="session", autouse=True)
def apply_v3_migrations():
    """Apply v3 migrations before any tests in this directory run.

    Migrations are idempotent (CREATE TABLE IF NOT EXISTS) so re-applying
    on every test session is safe and ensures a clean test environment.
    """
    s = Settings()
    with psycopg2.connect(
        host=s.db_host,
        dbname=s.db_name,
        user=s.db_user,
        password=s.db_password,
        port=s.db_port,
    ) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            for path in V3_MIGRATIONS:
                cur.execute(path.read_text())
    yield
