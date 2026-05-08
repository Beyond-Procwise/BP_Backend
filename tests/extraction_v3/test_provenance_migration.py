"""Tests for bp_extraction_provenance_v3 table schema migration."""
from __future__ import annotations

import os
import sys
import psycopg2
from dotenv import load_dotenv

# Load .env file from project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


def get_db_conn() -> psycopg2.extensions.connection:
    """
    Create a psycopg2 connection using environment variables.
    Follows the pattern used by the project (see config/settings.py).
    """
    db_host = os.environ.get("DB_HOST", "localhost")
    db_name = os.environ.get("DB_NAME", "postgres")
    db_user = os.environ.get("DB_USER", "postgres")
    db_password = os.environ.get("DB_PASSWORD", "")
    db_port = int(os.environ.get("DB_PORT", "5432"))

    conn = psycopg2.connect(
        host=db_host,
        dbname=db_name,
        user=db_user,
        password=db_password,
        port=db_port,
    )
    conn.autocommit = True
    return conn


def test_provenance_v3_table_exists():
    """Verify that proc.bp_extraction_provenance_v3 table exists with correct schema."""
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_schema='proc' AND table_name='bp_extraction_provenance_v3'
                ORDER BY ordinal_position
            """)
            cols = {name: dtype for name, dtype in cur.fetchall()}

        expected = {
            "provenance_id":    "bigint",
            "doc_type":         "text",
            "doc_pk":           "text",
            "field_path":       "text",
            "value":            "text",
            "page":             "integer",
            "bbox_x0":          "real",
            "bbox_y0":          "real",
            "bbox_x1":          "real",
            "bbox_y1":          "real",
            "evidence_text":    "text",
            "model":            "text",
            "model_confidence": "real",
            "judge_actions":    "jsonb",
            "final_confidence": "real",
            "extracted_at":     "timestamp with time zone",
            "pipeline_version": "text",
        }

        for col, dtype in expected.items():
            actual = cols.get(col)
            assert actual == dtype, (
                f"Column {col}: expected {dtype!r}, got {actual!r}"
            )

    finally:
        conn.close()
