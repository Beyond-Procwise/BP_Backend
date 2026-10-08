"""Pack (a) must work on its own: it is applied first, and pack (b) (tone rules, prompts, steering settings) waits for live verification.

The first version of the split put the `steering` column in pack (b) while the capture code writes to it, so applying (a) alone would
have made every capture silently fail. Nothing had ever applied (a) alone. This does, on a fresh database, and runs the real flow.
"""

import json
from datetime import datetime, timedelta, timezone

import psycopg2
import pytest

from evals.email import db as evaldb, runner
from src.services.draft_assurance import capture, learning, metrics, retention, steering, sweep

NOW = datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc)
DRAFT = "<p>Thank you for your latest offer of 47.50 GBP. We would like to propose 44.80 GBP for this order. Could you confirm by 30 October 2026?</p>"
ASSURANCE = {"family_id": "negotiation_counter", "family_version": 1, "mode": "shadow", "status": "verified", "facts": {}, "conflicts": [],
             "reasoned": {}, "assumptions": [], "violations": [], "repaired": False, "carried_unverified": {}, "unverified_figures": [],
             "assumption_items": [], "family_source": "declared", "ready": True, "clarification": {},
             "accountability": {"initiated_by": "NegotiationAgent", "kind": "agent"},
             "steering": {"status": "off", "reason": "no steering rules", "tone": [], "style_rule_ids": [], "exemplar_ids": [], "exemplar_scope": "none"}}


def test_the_groups_partition_the_migrations_exactly():
    assert set(evaldb.PACK_A) | set(evaldb.PACK_B) | set(evaldb.ROLES) == set(evaldb.MIGRATIONS)
    assert not (set(evaldb.PACK_A) & set(evaldb.PACK_B)) and not (set(evaldb.PACK_A) & set(evaldb.ROLES)) and not (set(evaldb.PACK_B) & set(evaldb.ROLES))
    assert len(evaldb.PACK_A) == 11 and len(evaldb.PACK_B) == 3 and len(evaldb.ROLES) == 1


@pytest.fixture
def pack_a_only(eval_dsn):
    """A database built from nothing with pack (a) alone: no tone rules, no prompts, no steering settings, no roles."""
    admin = psycopg2.connect(eval_dsn)
    admin.autocommit = True
    admin.cursor().execute("DROP DATABASE IF EXISTS pack_a_probe")
    admin.cursor().execute("CREATE DATABASE pack_a_probe")
    conn = psycopg2.connect(eval_dsn.rsplit("/", 1)[0] + "/pack_a_probe")
    conn.autocommit = True
    evaldb.load(conn, skip=evaldb.PACK_B + evaldb.ROLES, generate=True)
    yield conn
    conn.close()
    admin.cursor().execute("DROP DATABASE IF EXISTS pack_a_probe")


def test_pack_a_alone_contains_no_pack_b_rows(pack_a_only):
    with pack_a_only.cursor() as cur:
        cur.execute("SELECT policy_name FROM proc.bp_policy WHERE created_by = 'email_assurance_migration' ORDER BY 1")
        names = [r[0] for r in cur.fetchall()]
        cur.execute("SELECT count(*) FROM proc.bp_prompt WHERE prompt_name IN ('email_family_classify','email_brief_plan','email_draft_judge')")
        prompts = cur.fetchone()[0]
        # Roles are CLUSTER-wide, so whether they exist says nothing about this database (they do after a full-pack rehearsal).
        # What pack (a) must not do is GRANT anything here: with the roles present, they hold no privilege on a (a)-only database.
        cur.execute("SELECT rolname FROM pg_roles WHERE rolname IN ('email_agent_reader', 'email_agent_writer')")
        granted = []
        for (role,) in cur.fetchall():
            cur.execute("SELECT has_schema_privilege(%s, 'email_agent', 'USAGE') OR has_table_privilege(%s, 'email_agent.bp_draft_capture', 'SELECT')", (role, role))
            granted.append(cur.fetchone()[0])
    assert names == ["EmailDraftSweepRules", "EmailFamily_free_prompt", "EmailFamily_human_written", "EmailFamily_negotiation_counter",
                     "EmailFamily_rfq_batch", "EmailLearningRules", "EmailTextRetention"] and prompts == 0
    assert not any(granted)


def test_capture_send_sweep_retention_learning_and_metrics_all_work_with_pack_a_alone(pack_a_only):
    conn = pack_a_only
    engine = runner.make_engine(conn)
    cid = capture.record_draft(conn, {"unique_id": "U-A", "workflow_id": "wf", "supplier_id": "S-1", "body": DRAFT,
                                      "metadata": {"intent": "NEGOTIATION_COUNTER"}, "assurance": ASSURANCE})
    assert cid, "capture failed with pack (a) alone: a column the code writes is missing from pack (a)"
    assert capture.record_sent(conn, "U-A", DRAFT.replace("44.80", "40.00"), reviewed_by="boss", sent_by="u1",
                               retention_days=retention.raw_text_days(engine))
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM email_agent.bp_draft_outcome")
        assert cur.fetchone()[0] == 1
        cur.execute("SELECT count(*) FROM email_agent.bp_draft_sent_text")
        assert cur.fetchone()[0] == 1
        cur.execute("SELECT steering ->> 'status' FROM email_agent.bp_draft_capture")
        assert cur.fetchone()[0] == "off"
    # an unsent, quiet draft the product confirms was not sent is swept
    capture.record_draft(conn, {"unique_id": "U-B", "workflow_id": "wf", "supplier_id": "S-1", "body": DRAFT, "metadata": {}, "assurance": ASSURANCE})
    with conn.cursor() as cur:
        cur.execute("UPDATE email_agent.bp_draft_capture SET captured_at = %s WHERE unique_id = 'U-B'", (NOW - timedelta(days=30),))
        cur.execute("INSERT INTO proc.draft_rfq_emails (rfq_id, subject, body, unique_id, sent) VALUES ('R','s','b','U-B',false)")
    assert sweep.sweep(conn, conn, sweep.load_rules(engine), now=NOW)["abandoned"] == 1
    assert retention.purge_expired(conn, retention.load_rules(engine)["raw_text_days"], now=NOW + timedelta(days=200))["sent_text_deleted"] == 1
    assert learning.run_learning(conn, engine, now=NOW)                       # the job runs on pack (a)'s settings alone
    rows = metrics.by_family(conn, bucket="month")
    assert rows and sum(r["drafts"] for r in rows) == 2


def test_with_pack_a_alone_nothing_steers_and_the_prompt_is_unchanged(pack_a_only):
    s = steering.resolve(runner.make_engine(pack_a_only), None, family_id="negotiation_counter", author="nick", tone=None, directives=None)
    assert s.status == "off" and s.block() == ""


def test_pack_b_applies_on_top_of_a_and_rolls_back_leaving_a_intact(pack_a_only):
    conn = pack_a_only
    with conn.cursor() as cur:
        for name in evaldb.PACK_B:
            cur.execute((evaldb.SQL / name).read_text())
        cur.execute("SELECT count(*) FROM proc.bp_policy WHERE created_by = 'email_assurance_migration'")
        assert cur.fetchone()[0] == 9
        for name in reversed(evaldb.PACK_B):
            cur.execute((evaldb.SQL / name.replace(".sql", "_rollback.sql")).read_text())
        cur.execute("SELECT count(*) FROM proc.bp_policy WHERE created_by = 'email_assurance_migration'")
        assert cur.fetchone()[0] == 7
        cur.execute("SELECT count(*) FROM information_schema.columns WHERE table_schema = 'email_agent' AND column_name = 'steering'")
        assert cur.fetchone()[0] == 1                                          # the column belongs to (a) and stays
