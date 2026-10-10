"""Stage 5 tables exist with the shape the code relies on. Needs PROCWISE_TEST_LIVE_DB=1."""
import os
import uuid

import psycopg2
import pytest

from services.db import get_conn

pytestmark = pytest.mark.skipif(os.getenv("PROCWISE_TEST_LIVE_DB") != "1", reason="live DB required")


def _cols(cur, table):
    cur.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='proc' AND table_name=%s",
                (table,))
    return {r[0] for r in cur.fetchall()}


def test_tables_and_columns():
    with get_conn() as conn:
        cur = conn.cursor()
        assert _cols(cur, "bp_agent_policy_suggestion") >= {
            "suggestion_id", "policy_key", "change_key", "policy_version", "kind", "proposal", "stats",
            "confident", "status", "n_at_action", "acted_by", "acted_at", "saved_version", "created_at", "updated_at"}
        assert _cols(cur, "bp_agent_policy_second_review") >= {
            "policy_key", "version", "suggestion_id", "reviewed_by", "reviewed_at", "note"}
        assert _cols(cur, "bp_policy_firing_reversal") >= {"firing_id", "reversed_by", "reversed_at", "reason"}


def test_one_suggestion_per_policy_and_change():
    key = f"TST-{uuid.uuid4().hex[:6]}a"
    ins = ("INSERT INTO proc.bp_agent_policy_suggestion (policy_key, change_key, policy_version, kind, proposal, "
           "stats, confident) VALUES (%s, 'threshold:args.amount:gt:500', 1, 'threshold', '{}', '{}', true)")
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(ins, (key,))
        try:
            with pytest.raises(psycopg2.errors.UniqueViolation):
                cur.execute(ins, (key,))
        finally:
            cur.execute("DELETE FROM proc.bp_agent_policy_suggestion WHERE policy_key = %s", (key,))


def test_a_reversal_is_append_only():
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("INSERT INTO proc.bp_policy_firing (policy_key, policy_version, checkpoint, action_name, outcome, "
                    "result) VALUES (%s, 1, 'tool.call.before', 'tst', 'approve', 'approved') RETURNING firing_id",
                    (f"TST-{uuid.uuid4().hex[:6]}a",))
        fid = cur.fetchone()[0]
        cur.execute("INSERT INTO proc.bp_policy_firing_reversal (firing_id, reversed_by, reason) VALUES (%s,'t','r')", (fid,))
        with pytest.raises(psycopg2.Error, match="append-only"):
            cur.execute("DELETE FROM proc.bp_policy_firing_reversal WHERE firing_id = %s", (fid,))
        with pytest.raises(psycopg2.Error, match="append-only"):
            cur.execute("UPDATE proc.bp_policy_firing_reversal SET reason = 'x' WHERE firing_id = %s", (fid,))
