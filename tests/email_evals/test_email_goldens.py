"""The golden cases, one pytest test each, against a throwaway Postgres.

Skipped (not failed) when there is no Docker daemon and no EMAIL_EVAL_DSN, so a laptop without
either still runs the rest of the suite; CI provides a Postgres service and so always runs them.
"""

import json

import pytest

from evals.email import db, runner

CASES = runner.load_cases()


@pytest.fixture(scope="session")
def eval_run(eval_db):
    yield eval_db, runner.run_all(eval_db)


@pytest.mark.parametrize("case", CASES, ids=[f"{c['family']}/{c['id']}" for c in CASES])
def test_golden_case(eval_run, case):
    _, report = eval_run
    result = next(r for r in report["results"] if r["id"] == case["id"] and r["family"] == case["family"])
    failed = [f"{c['check']}: {c.get('detail')}" for c in result["checks"] if not c["ok"]]
    assert not failed, "\n".join(failed)


def test_every_family_meets_its_baseline(eval_run):
    _, report = eval_run
    baseline = json.loads((runner.HERE / "baseline.json").read_text())
    assert runner.compare_to_baseline(report, baseline) == []


def test_the_migrations_apply_roll_back_and_reapply(eval_run):
    """The evals load the real migration files, so this also proves the rollback scripts work."""
    import psycopg2
    conn, _ = eval_run
    cur = conn.cursor()
    try:
        db.rollback_all(conn)
        cur.execute("SELECT count(*) FROM proc.bp_policy WHERE created_by = 'email_assurance_migration'")
        assert cur.fetchone()[0] == 0
        cur.execute("SELECT count(*) FROM information_schema.schemata WHERE schema_name = 'email_agent'")
        assert cur.fetchone()[0] == 0
        cur.execute("SELECT count(*) FROM proc.bp_prompt WHERE prompt_name IN ('email_family_classify', 'email_brief_plan', 'email_draft_judge')")
        assert cur.fetchone()[0] == 0        # ours only: against a restored copy the table also holds the real prompts
    finally:                                   # leave the shared eval database as the cases expect it
        for name in db.MIGRATIONS:
            cur.execute((db.SQL / name).read_text())
        runner.snapshot(conn)
    cur.execute("SELECT count(*) FROM proc.bp_policy WHERE created_by = 'email_assurance_migration'")
    assert cur.fetchone()[0] == 6          # counter, free_prompt, rfq_batch, human_written, tone rules, learning rules


def test_every_family_that_has_cases_has_a_baseline_and_at_least_ten_cases():
    baseline = json.loads((runner.HERE / "baseline.json").read_text())["families"]
    counts = {}
    for c in CASES:
        counts[c["family"]] = counts.get(c["family"], 0) + 1
    assert set(counts) == set(baseline)
    assert all(n >= 10 for n in counts.values()), counts           # the spec's floor for a family
    assert all(baseline[f]["min_cases"] <= n for f, n in counts.items())


def test_the_required_scenarios_are_each_covered_for_every_family_that_can_have_them():
    """Missing record, multiple match, thread-vs-database conflict and changed-before-send, by case id."""
    ids = {c["id"] for c in CASES if c["family"] == "negotiation_counter"}
    for needle in ("missing-record", "multiple-matching-records", "thread-vs-postgres-conflict",
                   "changed-before-send-shadow", "changed-before-send-enforce"):
        assert any(needle in i for i in ids), needle


# --- the scoring must not be able to lie (no database needed) -------------------------------------------

def test_a_wrong_expectation_is_reported_failed():
    out = runner.evaluate({"a.b": 2, "a.c": {"has": "x"}, "a.d": {"contains": "zz"}}, {"a": {"b": 1, "c": ["y"], "d": "abc"}})
    assert [c["ok"] for c in out] == [False, False, False]
    assert all("detail" in c for c in out)


def test_a_right_expectation_passes_and_a_missing_path_is_null_not_a_pass():
    ctx = {"a": {"b": 1, "c": ["x"], "n": [{"k": 1}, {"k": 2}]}}
    ok = runner.evaluate({"a.b": 1, "a.c": {"has": "x"}, "a.n[*].k": {"has": 2}, "a.zzz": {"null": True}}, ctx)
    assert all(c["ok"] for c in ok)
    assert runner.evaluate({"a.zzz": 5}, ctx)[0]["ok"] is False          # absent never equals a value


def test_has_and_lacks_refuse_to_pass_on_a_non_list():
    assert runner.check("x", {"has": "x"})[0] is False
    assert runner.check(None, {"lacks": "x"})[0] is True and runner.check(["x"], {"lacks": "x"})[0] is False


def test_an_unknown_operator_is_an_error_not_a_pass():
    with pytest.raises(ValueError):
        runner.check(1, {"approximately": 1})


def test_a_case_that_crashes_counts_as_failed_not_skipped(monkeypatch):
    class Conn:
        autocommit = True
        def cursor(self):
            class C:
                def __enter__(s): return s
                def __exit__(s, *a): return False
                def execute(s, *a): pass
            return C()
    monkeypatch.setattr(runner, "run_case", lambda case, conn, base: (_ for _ in ()).throw(RuntimeError("boom")))
    report = runner.run_all(Conn(), "free_prompt")
    f = report["families"]["free_prompt"]
    assert f["passed"] == 0 and f["cases"] >= 10 and f["pass_rate"] == 0.0
    assert "boom" in report["results"][0]["checks"][0]["detail"]


def test_the_baseline_blocks_a_lower_pass_rate_removed_cases_and_unbaselined_families():
    baseline = {"families": {"f": {"min_cases": 5, "min_pass_rate": 1.0}}}
    good = {"families": {"f": {"cases": 5, "pass_rate": 1.0}}}
    assert runner.compare_to_baseline(good, baseline) == []
    assert any("pass rate" in p for p in runner.compare_to_baseline({"families": {"f": {"cases": 5, "pass_rate": 0.8}}}, baseline))
    assert any("removed" in p for p in runner.compare_to_baseline({"families": {"f": {"cases": 4, "pass_rate": 1.0}}}, baseline))
    assert any("no baseline" in p for p in runner.compare_to_baseline({"families": {"f": {"cases": 5, "pass_rate": 1.0}, "g": {"cases": 1, "pass_rate": 1.0}}}, baseline))
    assert any("no cases ran" in p for p in runner.compare_to_baseline({"families": {}}, baseline))
