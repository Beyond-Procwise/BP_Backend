# Task 1 report - enforcement migration
- Pre-check: none of the 3 tables / 6 bp_decision columns existed in bp_testdb or bp_sqldb. bp_decision's 20 columns were identical in both DBs and are asserted in the test.
- Files: deploy/sql/2026-10-10_bp_agent_policy_enforcement.sql (+ _rollback.sql), tests/migrations/test_2026_10_10_bp_agent_policy_enforcement.py.
- Trigger trg_bp_policy_firing_guard (function proc.bp_policy_firing_guard): DELETE refused; UPDATE only if OLD.result='paused_for_approval' and to_jsonb(NEW)-cols = to_jsonb(OLD)-cols for decision_id, decided_level, decided_by, decided_at, reason, result.
- Applied twice to bp_testdb then bp_sqldb (second run: only "already exists" notices).
- Green: 12 passed. RED proof: dropped trigger in bp_testdb only -> test_firing_trigger_is_append_only[bp_testdb] FAILED (1 failed, 11 passed); re-applied -> 12 passed.
- tests/engines + tests/approvals (fake DB): 303 passed, 1 skipped before and after.
- Note: a resolved row (result approved etc.) is frozen; a paused row may move to any valid result value (CHECK limits it).
