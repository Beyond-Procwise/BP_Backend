"""Tests for bp_extraction_provenance_v3 table schema migration."""
from __future__ import annotations

import sys
import os
import psycopg2

# Ensure project root is in path for config imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from config.settings import Settings


def get_db_conn() -> psycopg2.extensions.connection:
    """
    Create a psycopg2 connection using Settings configuration.
    Follows the canonical pattern used by the project (see config/settings.py).
    """
    s = Settings()
    conn = psycopg2.connect(
        host=s.db_host,
        dbname=s.db_name,
        user=s.db_user,
        password=s.db_password,
        port=s.db_port,
    )
    conn.autocommit = True
    return conn


def test_provenance_v3_table_exists():
    """Verify that proc.bp_extraction_provenance_v3 table exists with correct schema and nullability."""
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema='proc' AND table_name='bp_extraction_provenance_v3'
                ORDER BY ordinal_position
            """)
            cols = {name: (dtype, nullable) for name, dtype, nullable in cur.fetchall()}

        expected = {
            # (data_type, is_nullable)
            "provenance_id":    ("bigint", "NO"),
            "doc_type":         ("text", "NO"),
            "doc_pk":           ("text", "NO"),
            "field_path":       ("text", "NO"),
            "value":            ("text", "NO"),
            "page":             ("integer", "NO"),
            "bbox_x0":          ("real", "NO"),
            "bbox_y0":          ("real", "NO"),
            "bbox_x1":          ("real", "NO"),
            "bbox_y1":          ("real", "NO"),
            "evidence_text":    ("text", "NO"),
            "model":            ("text", "NO"),
            "model_confidence": ("real", "NO"),
            "judge_actions":    ("jsonb", "YES"),     # nullable
            "final_confidence": ("real", "NO"),
            "extracted_at":     ("timestamp with time zone", "NO"),
            "pipeline_version": ("text", "NO"),
        }

        for col, (expected_dtype, expected_nullable) in expected.items():
            actual = cols.get(col)
            assert actual is not None, f"Column {col} missing from table"
            actual_dtype, actual_nullable = actual
            assert actual_dtype == expected_dtype, (
                f"Column {col}: expected type {expected_dtype!r}, got {actual_dtype!r}"
            )
            assert actual_nullable == expected_nullable, (
                f"Column {col}: expected nullable={expected_nullable!r}, got {actual_nullable!r}"
            )

    finally:
        conn.close()


def test_unique_constraint_present():
    """Verify that UNIQUE constraint exists on bp_extraction_provenance_v3."""
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT constraint_name
                FROM information_schema.table_constraints
                WHERE table_schema='proc'
                  AND table_name='bp_extraction_provenance_v3'
                  AND constraint_type='UNIQUE'
            """)
            names = [r[0] for r in cur.fetchall()]
        assert names, "UNIQUE constraint missing on bp_extraction_provenance_v3"
    finally:
        conn.close()


def test_index_present():
    """Verify that idx_provenance_v3_doc index exists on bp_extraction_provenance_v3."""
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT indexname FROM pg_indexes
                WHERE schemaname='proc' AND tablename='bp_extraction_provenance_v3'
            """)
            names = {r[0] for r in cur.fetchall()}
        assert "idx_provenance_v3_doc" in names, f"missing idx_provenance_v3_doc, got {names}"
    finally:
        conn.close()
