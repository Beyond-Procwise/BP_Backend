# Agent Policy Governance, Stage 5 (Learning) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When people keep approving the same kind of case under a "Needs approval" policy, the product suggests loosening that policy to its owner, with the evidence; nothing is ever applied without the owner's save, the normal activate step and (in Finance) a second reviewer.

**Architecture:**
- **Evidence is the firing log.** `proc.bp_policy_firing` already records every approve firing with its matched values, who decided, and when (stage 3). Learning only reads it. Reversals get one small append-only table beside it.
- **Three pure modules, one database module.** `learning_stats` (Wilson bound, the four statistics, the guardrails), `learning_patterns` (which parts of a condition can be loosened, and what the evidence supports), `learning_draft` (turn a suggestion into an unsaved form). `learning` is the only one that touches the database: it reads decisions, runs the hourly scan, stores suggestions and serves the card and the queue.
- **A suggestion is a row, never a change.** `proc.bp_agent_policy_suggestion`, one row per (policy, change). "Draft this change" returns a form and writes nothing. The suggestion is marked saved only when a person's save actually contains the change.
- **The second reviewer is checked at activation, by code, from the condition itself.** Not from a marker the browser sends: a loosening a suggestion pointed at needs a second review whether it was drafted from the card or typed by hand.

**Tech Stack:** Python 3 / FastAPI / psycopg2 (BP_Backend), NestJS + serverless yml (gateway), vanilla `engine.js` + vitest pure modules (UI), PostgreSQL (`bp_testdb`, `bp_sqldb`).

**Spec:** `specs/2026-10-08-agent-policy-governance-brief.md` §6, §7 (`learning.eligible`) and §8 test 19; `specs/2026-10-08-agent-policy-governance-design.md` §2 (rulings: second reviewer, defaults) and §4 row 5. The design note takes precedence over the brief where they differ.

**Repositories (all on a new branch `agent-policy-stage5`):**
- **BP:** `/home/muthu/PycharmProjects/BP_Backend/.claude/worktrees/agent-policy-stage5`, created from `origin/Development`. Every BP path below is relative to this folder.
- **Gateway:** `/home/muthu/PycharmProjects/beyond-procwaise-Api-worktrees/agent-policy-stage5/beyond_procwaise_api`, from `origin/spendiq-ui`.
- **UI:** `/home/muthu/PycharmProjects/beyond_procwise_ui-worktrees/agent-policy-stage5`, from `origin/spendiq-ui`.

**Commands used throughout.** Paste this once per shell:

```bash
BP=/home/muthu/PycharmProjects/BP_Backend/.claude/worktrees/agent-policy-stage5
GW=/home/muthu/PycharmProjects/beyond-procwaise-Api-worktrees/agent-policy-stage5/beyond_procwaise_api
UI=/home/muthu/PycharmProjects/beyond_procwise_ui-worktrees/agent-policy-stage5
# BP pytest: .env (bp_testdb), no GPU, src on the path. LIVE=1 turns on the live-DB tests.
bptest() { (cd "$BP" && set -a && . ./.env && set +a && export PROCWISE_TEST_LIVE_DB="${LIVE:-}" \
  && PYTHONPATH=.:src CUDA_VISIBLE_DEVICES="" /home/muthu/PycharmProjects/BP_Backend/venv/bin/python -m pytest "$@"); }
# psql against one database: bppsql <dbname> -f <file>  or  bppsql <dbname> -Atc "<sql>"
bppsql() { local db="$1"; shift; (cd "$BP" && set -a && . ./.env && set +a \
  && PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "${DB_PORT:-5432}" -U "$DB_USER" -d "$db" -v ON_ERROR_STOP=1 "$@"); }
```

(The worktree has no `.env`: copy it from `/home/muthu/PycharmProjects/BP_Backend/.env`, never commit it. It prints `supplierconnect: command not found` from line 27; harmless. Never print secret values.)

## Global Constraints

Carried from stages 1–4 (`specs/2026-10-09-conflict-history-and-precedent-plan.md`, "Global Constraints"). Every task's requirements include this section.

**Where you work**
- Run every BP command from the BP worktree. Never `cd` into `/home/muthu/PycharmProjects/BP_Backend` itself: it is a shared checkout that other sessions edit and that the running service executes.
- Python interpreter: `/home/muthu/PycharmProjects/BP_Backend/venv/bin/python`. Load `.env` first. `CUDA_VISIBLE_DEVICES=""`. `PYTHONPATH=.:src` (modules import as `services.…`, `api.…`, `repositories.…`).
- DB tests need `PROCWISE_TEST_LIVE_DB=1`; without it pytest uses a fake DB. Live tests run on `bp_testdb` only.
- `get_conn()` is AUTOCOMMIT: a rollback is a no-op unless `conn.autocommit = False` is set first. Every multi-statement write runs in `approvals._tx(conn)`.
- `bp_testdb` and `bp_sqldb` are SHARED with other sessions. Only additive, idempotent DDL/DML from this plan. Never DROP/TRUNCATE/DELETE anything that is not yours.
- Test data is scoped: policy keys `TST-<hex><letter>` (they deliberately do not match the real key pattern `^[A-Z]{3}-[0-9]{4,}$`), names "TST …". Suggestion and second-review rows a test makes are removed by that test. Firing rows and reversal rows are append-only BY DESIGN and stay.

**Commits**
- `git add` explicit paths only; never `git add -A` / `git add .`; never `git stash`.
- `.superpowers/` is NOT gitignored: never commit anything under it. `docs/` is gitignored: specs go under `specs/`.
- End every commit message with a blank line and then `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` (name the model that wrote the commit).
- Do not push. Do not merge. Never dispatch subagents yourself. No implementer on a Haiku model.

**Rules that bind**
- **Nothing learning produces becomes Active on its own** (brief §2, §6). No code path in this plan writes a policy version. The only writers of `bp_agent_policy_version` remain `repo.create_draft`, `repo.save_version` and `repo.retire`, called by a person's request.
- **Learning only ever suggests loosening an `approve` policy.** Never for `block`, never for a policy whose business area is marked "Never suggest changes" (read at scan time, not only from the compiled `learning.eligible`), never tightening.
- **The new value sits inside the range the evidence covers, never beyond it.**
- **Thresholds are company settings**, read from `settings.load_settings(conn)["learning"]`: `min_decisions` 30, `min_days` 30, `min_approvers` 3, `wilson_lower` 0.85, `median_seconds_floor` 30, `not_yet_more` 30, `dismiss_more` 30. The keys already exist in `settings.DEFAULTS`; add none. A setting that is not a number means no suggestion for that policy and a logged warning, never a guess.
- **Only decisions by people count.** Timeouts, `system:*` deciders (`system:timeout`, `system:group`, `system:precedent`, `system:retired`) and still-paused rows are excluded from every statistic.
- **One evaluator:** `conditions.to_engine` + `policy_condition.evaluate`; numbers are read with `policy_condition._number`, as the condition evaluator reads them. Example results are computed by `conditions.reviewer_view`, never written by hand.
- **Names differ on purpose:** "extraction confidence" (stage 1) and "suggestion confidence" (this stage) never share a label, field name or CSS class.
- `proc.bp_policy` is not written, altered or migrated. New tables use the `bp_` prefix, indexes `ix_bp_*`.
- Migrations live in `deploy/sql/` with a `_rollback.sql`, are additive and idempotent, and are applied to BOTH `bp_testdb` and `bp_sqldb`. A rollback is never run against shared data without a user ruling.
- **Every write endpoint** audits via `_require(p, <role>, "agent_policy.<verb>", {...})` before acting (it calls `agent_actions.record_action_or_fail`).
- **Gateway only.** Screens call the gateway (`apApi`), never the backend, never `fetch(`/`axios` directly.
- **UI:** `engine.js` changes are new `apLn*` functions plus minimal wiring. `policyEdit`, `policyDelete` and `openFormModal` stay byte-identical (the stage 4 contract test guards them). New UI reuses the existing card, tag and two-step confirm styles; no new visual look.
- **The stage 4 guard** `tests/agent_policy/test_conflict_live_gate.py::test_no_live_conflict_is_stage3_byte_for_byte` must still pass after every task. This stage does not touch the gate.
- **Never call the real model in tests.** Nothing in this stage calls a model at all.
- **Prove the guard fails:** where a step says so, break the code on purpose, capture the red output, restore it, and capture green. Put both in your report.
- **Deployment is not part of this plan.** No task restarts `procwise.service`, `bp-gateway` or `bp-ui`, or touches the shared checkouts, ports :8000/:3000/:3001 or Ollama.

## Review Focus

The inputs the brief is silent on that are most likely to bite. Each has a test in the task named.

1. **One human decision recorded on several rows.** A repeated agent call is linked to the same approval case, and one decision settles every linked row. Counting rows would turn 6 decisions into 30. Expected: one decision per approval case. (Task 5)
2. **The driving value is hidden.** A sensitive input is stored as `•••` in the log. Expected: those decisions are not evidence, and no suggestion is made on that input; nothing crashes. (Tasks 3, 5)
3. **A rejection in the middle of the range.** 40 approvals up to 800, one rejection at 650. Expected: the suggestion stops below what the evidence supports (or is not made); it never jumps the rejection because the total looks good. (Task 3)
4. **The policy changed while the card was open.** Someone saved a newer version, or the owner already raised the limit by hand. Expected: "Draft this change" answers that the policy has changed; the card disappears on the next scan; nothing stale is drafted. (Tasks 6, 5)
5. **The owner reviews their own loosening.** In a second-reviewer area the same person saves the draft and tries to record the review, or types the suggested number by hand without using the card. Expected: refused in the first case; the second review is still required in the second. (Task 8)

## Decisions taken in planning (the user may overrule; reviewers may challenge)

1. **Which conditions can be loosened.** Only a part of the condition that every match must satisfy (the condition itself, or inside nested `all`), never under `any` or `not`, and never `tool.name` (that is the action, not a limit). A number compared with `gt/gte/lt/lte` is a *threshold*; an `in` list with at least two values is a *list*. `not_in`, `eq`, `ne`, `exists` are not analysed in this release.
2. **Bands.** For a threshold, the bands are "everything from the current limit up to X", for each X people actually decided on. The suggestion is the furthest X whose band passes every guardrail, rounded toward the current limit to two significant figures when the rounded band still passes (837.42 → 830). For a list, the band is one listed value.
3. **Which firings are evidence.** Firings of every live version whose condition is identical to the current live condition, so an unrelated edit (owner, wording) does not throw 30 days of evidence away. A changed condition starts from zero.
4. **"Edited" is always 0.** Approve-with-changes is not in this version (user ruling). The statistic and its guardrail are kept so the rule holds the day it exists.
5. **Median decision time** is `decided_at − created_at` of the firing row. The ruling says "at least 30 s", so exactly 30 passes.
6. **Reversals are recorded by a person.** Nothing in this product knows that a refund was clawed back. An Approver marks a past approved firing "Reversed" with a reason, from the policy's activity list. It is append-only: a reversal cannot be undone, and it permanently disqualifies the bands that contain it for the current condition. Stored in its own table; stage 3's unused `bp_policy_firing.reversal_of` column stays unused, because a reversal row inside the firing log would be counted by every reader as a firing.
7. **Not enough yet.** A card is raised only when every guardrail holds. If the evidence later weakens (a rejection, a reversal), the card stays, shows "Not enough yet" and the draft button is off.
8. **The situation sentence is updated by code, not by a model.** When the old number appears exactly once in the situation (and likewise in the two messages), it is replaced. Otherwise the text is left alone and the form says which wording to check. The confirmation is cleared either way, so the owner must read it.
9. **New boundary examples carry `agentExpected` equal to the computed result.** No agent proposed them; leaving it blank would lower *extraction* confidence for a reason that is not true.
10. **Roles.** Read: Viewer. Draft this change, Not yet, Dismiss: Buyer (the role that edits drafts). Mark a reversal, record a second review: Approver.
11. **Second review.** Required when the policy's business area has the second-reviewer setting on AND the version being activated loosens the live condition in the direction any suggestion for that live condition points (open, snoozed, dismissed or saved). The reviewer needs the Approver role, must not be the person who saved the draft, and must not be linked to the policy's owner in the decider map. The review is of one saved draft version; activation must carry the same condition as that version.
12. **No notifications in this stage.** The card and the queue are the surface. (Stage 3's notification table needs a firing or a decision to hang on.)
13. **The scan** runs hourly on its own scheduler lane, on unless `AGENT_POLICY_LEARNING_SCAN=off`. It reads live policies only, so the stale test drafts in `bp_testdb` do not matter.

## User rulings (2026-10-10)

- **Q1: yes.** `GET /agent-policies/suggestions` and `GET /agent-policies/suggestions/{id}/draft` are added to the output-safety pass-through list, on the same terms as the rulings of 2026-10-08 and Q5 (exact method and path, 2xx only; errors still scrubbed). Task 6 Step 5 adds both entries.
- **Q2: yes.** An Approver marks an approved request as reversed by hand, with a reason. It cannot be undone (decision 6).
- **Q3: yes.** The second review also applies when the owner types the suggested loosening by hand (decision 11).
- **Execution:** subagent-driven, one implementer and one reviewer per task.

## File map

**BP_Backend**
| File | Responsibility |
|---|---|
| `deploy/sql/2026-10-15_bp_agent_policy_learning.sql` (+ `_rollback.sql`) | Three tables: suggestion, second review, firing reversal. |
| `src/services/agent_policy/learning_stats.py` (new) | `Decided`, `wilson_lower`, `stats`, `failed_guardrails`. Pure. |
| `src/services/agent_policy/learning_patterns.py` (new) | `eligible_leaves`, `propose`, `targets`, `applied`, `group_for`. Pure. |
| `src/services/agent_policy/learning_draft.py` (new) | `build(form, proposal)` → unsaved form. Pure. |
| `src/services/agent_policy/learning.py` (new) | Reads decisions, `scan`, cards, `act`, `draft`, `note_save`, `record_reversal`. |
| `src/services/agent_policy/second_review.py` (new) | `activation_problems`, `record`, `status`. |
| `src/repositories/agent_policy_repo.py` (modify) | `save_version`: second-review check on activate, `learning.note_save` after a save; `get_policy`: `suggestions`, `secondReview`. |
| `src/api/routers/agent_policies.py` (modify) | Six endpoints. |
| `src/services/agent_policy/approval_views.py` (modify) | `policy_firings` adds `reversed`. |
| `src/services/backend_scheduler.py` (modify) | Hourly learning scan job. |
| `src/api/main.py` (modify, ruling Q1) | Two pass-through entries. |
| `tests/agent_policy/test_learning_*.py` (new) | One file per module; `_live` suffix for DB tests. |

**Gateway and UI:** see Tasks 9–11.

---

### Task 1: Tables for suggestions, second reviews and reversals

**Files:**
- Create: `deploy/sql/2026-10-15_bp_agent_policy_learning.sql`
- Create: `deploy/sql/2026-10-15_bp_agent_policy_learning_rollback.sql`
- Test: `tests/agent_policy/test_learning_tables_live.py`

**Interfaces:**
- Produces: tables `proc.bp_agent_policy_suggestion`, `proc.bp_agent_policy_second_review`, `proc.bp_policy_firing_reversal`, with exactly the columns below. Later tasks use these names.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run it to see it fail**

Run: `LIVE=1 bptest tests/agent_policy/test_learning_tables_live.py -q`
Expected: FAIL, `relation "proc.bp_agent_policy_suggestion" does not exist` (or an empty column set).

- [ ] **Step 3: Write the migration**

`deploy/sql/2026-10-15_bp_agent_policy_learning.sql`:

```sql
-- Agent policy governance, stage 5 (learning). Additive and idempotent.
BEGIN;

-- One row per (policy, change). policy_key has no foreign key, like bp_policy_firing: the scan
-- must be able to keep a row about a policy whatever happens to it, and tests use keys that
-- are not real policies.
CREATE TABLE IF NOT EXISTS proc.bp_agent_policy_suggestion (
    suggestion_id  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    policy_key     TEXT NOT NULL,
    change_key     TEXT NOT NULL,
    policy_version INTEGER NOT NULL,
    kind           TEXT NOT NULL CHECK (kind IN ('threshold','list')),
    proposal       JSONB NOT NULL,
    stats          JSONB NOT NULL,
    confident      BOOLEAN NOT NULL,
    status         TEXT NOT NULL DEFAULT 'open'
        CHECK (status IN ('open','not_yet','dismissed','saved','superseded')),
    n_at_action    INTEGER,
    acted_by       TEXT,
    acted_at       TIMESTAMPTZ,
    saved_version  INTEGER,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_bp_agent_policy_suggestion_change UNIQUE (policy_key, change_key)
);
CREATE INDEX IF NOT EXISTS ix_bp_agent_policy_suggestion_status
    ON proc.bp_agent_policy_suggestion (status, policy_key);
COMMENT ON TABLE proc.bp_agent_policy_suggestion IS
    'A change learning suggests to a policy owner. Never applied by code: only a person''s save changes a policy.';

CREATE TABLE IF NOT EXISTS proc.bp_agent_policy_second_review (
    policy_key    TEXT NOT NULL,
    version       INTEGER NOT NULL,
    suggestion_id BIGINT,
    reviewed_by   TEXT NOT NULL,
    reviewed_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    note          TEXT,
    PRIMARY KEY (policy_key, version)
);
COMMENT ON TABLE proc.bp_agent_policy_second_review IS
    'A second person reviewed this saved draft version of a policy that loosens it as learning suggested.';

-- A person's record that an approved action was later undone (a refund clawed back).
CREATE TABLE IF NOT EXISTS proc.bp_policy_firing_reversal (
    firing_id   BIGINT PRIMARY KEY REFERENCES proc.bp_policy_firing (firing_id),
    reversed_by TEXT NOT NULL,
    reversed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    reason      TEXT NOT NULL CHECK (btrim(reason) <> '')
);

CREATE OR REPLACE FUNCTION proc.bp_policy_firing_reversal_guard() RETURNS trigger AS $fn$
BEGIN
    RAISE EXCEPTION 'bp_policy_firing_reversal is append-only: % refused', TG_OP;
END;
$fn$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_bp_policy_firing_reversal_guard ON proc.bp_policy_firing_reversal;
CREATE TRIGGER trg_bp_policy_firing_reversal_guard
    BEFORE UPDATE OR DELETE ON proc.bp_policy_firing_reversal
    FOR EACH ROW EXECUTE FUNCTION proc.bp_policy_firing_reversal_guard();

COMMIT;
```

`deploy/sql/2026-10-15_bp_agent_policy_learning_rollback.sql`:

```sql
-- Rollback of 2026-10-15_bp_agent_policy_learning.sql. Loses every suggestion, second review and
-- reversal record. Never run against shared data without a user ruling.
BEGIN;
DROP TABLE IF EXISTS proc.bp_policy_firing_reversal;
DROP FUNCTION IF EXISTS proc.bp_policy_firing_reversal_guard();
DROP TABLE IF EXISTS proc.bp_agent_policy_second_review;
DROP TABLE IF EXISTS proc.bp_agent_policy_suggestion;
COMMIT;
```

- [ ] **Step 4: Apply to both databases, twice each (idempotence), and announce it**

Tell the controller before running: "Applying additive DDL `2026-10-15_bp_agent_policy_learning.sql` to bp_testdb and bp_sqldb (three new tables, nothing altered)."

```bash
for db in bp_testdb bp_sqldb; do
  bppsql $db -f deploy/sql/2026-10-15_bp_agent_policy_learning.sql
  bppsql $db -f deploy/sql/2026-10-15_bp_agent_policy_learning.sql
  bppsql $db -Atc "SELECT count(*) FROM information_schema.tables WHERE table_schema='proc' AND table_name IN ('bp_agent_policy_suggestion','bp_agent_policy_second_review','bp_policy_firing_reversal')"
done
```
Expected: no error on either run; `3` printed for each database. Do NOT run the rollback.

- [ ] **Step 5: Run the test to see it pass**

Run: `LIVE=1 bptest tests/agent_policy/test_learning_tables_live.py -q`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add deploy/sql/2026-10-15_bp_agent_policy_learning.sql deploy/sql/2026-10-15_bp_agent_policy_learning_rollback.sql tests/agent_policy/test_learning_tables_live.py
git commit -m "feat(agent-policy): tables for learning suggestions, second reviews and reversals"
```

---

### Task 2: The statistics and the guardrails

**Files:**
- Create: `src/services/agent_policy/learning_stats.py`
- Test: `tests/agent_policy/test_learning_stats.py`

**Interfaces:**
- Produces:
  - `Decided(firing_id: int, values: dict, approved: bool, decided_by: str, asked_at: datetime, decided_at: datetime, reversed: bool = False, edited: bool = False)` (frozen dataclass)
  - `wilson_lower(k: int, n: int) -> float`
  - `stats(group: Sequence[Decided]) -> dict` with keys `decisions, approved, wilsonLower, approvers, days, edited, reversed, medianSeconds`
  - `failed_guardrails(s: dict, cfg: dict) -> list[str]` (empty = every guardrail holds). Names: `min_decisions, min_days, min_approvers, wilson_lower, edited_or_reversed, median_seconds_floor`.

- [ ] **Step 1: Write the failing tests**

```python
"""The four statistics and the guardrails (brief §6.2, §6.3). No database."""
from datetime import datetime, timedelta, timezone

import pytest

from services.agent_policy import learning_stats as LS
from services.agent_policy.settings import DEFAULTS

CFG = DEFAULTS["learning"]
T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def group(n=30, approved=None, approvers=3, days=30, seconds=45, reversed_at=(), edited_at=()):
    """n decisions spread evenly over `days`, by `approvers` people, each taking `seconds`."""
    approved = n if approved is None else approved
    out = []
    for i in range(n):
        asked = T0 + timedelta(days=days * i / (n - 1) if n > 1 else 0)
        out.append(LS.Decided(firing_id=i, values={"args.amount": 600 + i}, approved=i < approved,
                              decided_by=f"person-{i % approvers}", asked_at=asked,
                              decided_at=asked + timedelta(seconds=seconds),
                              reversed=i in reversed_at, edited=i in edited_at))
    return out


@pytest.mark.parametrize("k,n,expect", [(30, 30, 0.8865), (29, 30, 0.8333), (0, 30, 0.0), (58, 60, 0.8864)])
def test_wilson_lower_bound(k, n, expect):
    assert round(LS.wilson_lower(k, n), 4) == expect


def test_wilson_of_nothing_is_zero():
    assert LS.wilson_lower(0, 0) == 0.0


def test_stats():
    s = LS.stats(group(n=30, approved=29, approvers=4, days=45, seconds=90))
    assert s == {"decisions": 30, "approved": 29, "wilsonLower": 0.8333, "approvers": 4, "days": 45,
                 "edited": 0, "reversed": 0, "medianSeconds": 90.0}


def test_stats_of_an_empty_group():
    assert LS.stats([])["decisions"] == 0 and LS.stats([])["medianSeconds"] is None


def test_the_same_person_in_different_letter_case_is_one_approver():
    g = group(n=4, approvers=1)
    g[1] = LS.Decided(**{**g[1].__dict__, "decided_by": "PERSON-0 "})
    assert LS.stats(g)["approvers"] == 1


def test_every_guardrail_holds_on_the_ruled_defaults():
    assert LS.failed_guardrails(LS.stats(group()), CFG) == []


@pytest.mark.parametrize("kwargs,failed", [
    (dict(n=29), ["min_decisions"]),
    (dict(days=29), ["min_days"]),
    (dict(approvers=2), ["min_approvers"]),
    (dict(approved=29), ["wilson_lower"]),
    (dict(reversed_at=(3,)), ["edited_or_reversed"]),
    (dict(edited_at=(3,)), ["edited_or_reversed"]),
    (dict(seconds=29), ["median_seconds_floor"]),
])
def test_each_guardrail_alone(kwargs, failed):
    assert LS.failed_guardrails(LS.stats(group(**kwargs)), CFG) == failed


def test_exactly_on_each_threshold_passes():
    # 30 decisions, 30 days, 3 approvers, median exactly 30 s: "at least" (design §2, Defaults)
    assert LS.failed_guardrails(LS.stats(group(n=30, days=30, approvers=3, seconds=30)), CFG) == []


def test_a_setting_that_is_not_a_number_raises_and_never_guesses():
    with pytest.raises(TypeError):
        LS.failed_guardrails(LS.stats(group()), {**CFG, "min_decisions": "thirty"})
```

(`n=29` at 100% approved has Wilson 0.883, so only `min_decisions` fails. `days=29` with 30 evenly spread decisions spans exactly 29 days.)

- [ ] **Step 2: Run to see them fail**

Run: `bptest tests/agent_policy/test_learning_stats.py -q`
Expected: FAIL, `ImportError: cannot import name 'learning_stats'`.

- [ ] **Step 3: Write the module**

```python
"""The statistics behind a learning suggestion, and the guardrails (brief §6.2, §6.3).

Pure: no database, no settings lookup. A group is the decisions people made on one band of
values under one policy condition. Timeouts and system decisions never reach this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from math import sqrt
from statistics import median
from typing import Any, Dict, List, Optional, Sequence

Z95 = 1.959963984540054   # two-sided 95%


@dataclass(frozen=True)
class Decided:
    firing_id: int
    values: Dict[str, Any] = field(hash=False)
    approved: bool = False
    decided_by: str = ""
    asked_at: Optional[datetime] = None
    decided_at: Optional[datetime] = None
    reversed: bool = False
    edited: bool = False


def wilson_lower(k: int, n: int) -> float:
    """Lower bound of the 95% Wilson score interval for k approvals out of n decisions."""
    if n <= 0:
        return 0.0
    p, z2 = k / n, Z95 * Z95
    return (p + z2 / (2 * n) - Z95 * sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / (1 + z2 / n)


def stats(group: Sequence[Decided]) -> Dict[str, Any]:
    n = len(group)
    if not n:
        return {"decisions": 0, "approved": 0, "wilsonLower": 0.0, "approvers": 0, "days": 0,
                "edited": 0, "reversed": 0, "medianSeconds": None}
    k = sum(1 for d in group if d.approved)
    when = [d.decided_at for d in group]
    return {
        "decisions": n,
        "approved": k,
        "wilsonLower": round(wilson_lower(k, n), 4),
        "approvers": len({str(d.decided_by).strip().lower() for d in group}),
        "days": (max(when) - min(when)).days,
        "edited": sum(1 for d in group if d.edited),
        "reversed": sum(1 for d in group if d.reversed),
        "medianSeconds": float(median((d.decided_at - d.asked_at).total_seconds() for d in group)),
    }


def _num(cfg: Dict[str, Any], key: str) -> float:
    value = cfg[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"learning setting {key!r} is not a number")
    return value


def failed_guardrails(s: Dict[str, Any], cfg: Dict[str, Any]) -> List[str]:
    """The guardrails that do not hold, in the brief's order. Empty = raise the suggestion."""
    failed: List[str] = []
    if s["decisions"] < _num(cfg, "min_decisions"):
        failed.append("min_decisions")
    if s["days"] < _num(cfg, "min_days"):
        failed.append("min_days")
    if s["approvers"] < _num(cfg, "min_approvers"):
        failed.append("min_approvers")
    if s["wilsonLower"] < _num(cfg, "wilson_lower"):
        failed.append("wilson_lower")
    if s["edited"] or s["reversed"]:
        failed.append("edited_or_reversed")
    floor = _num(cfg, "median_seconds_floor")
    if s["medianSeconds"] is None or s["medianSeconds"] < floor:
        failed.append("median_seconds_floor")
    return failed
```

- [ ] **Step 4: Run to see them pass**

Run: `bptest tests/agent_policy/test_learning_stats.py -q`
Expected: all pass.

- [ ] **Step 5: Prove the Wilson guard fails.** Temporarily change `failed_guardrails` to compare `s["approved"] / s["decisions"]` instead of `s["wilsonLower"]`. Run the tests: `test_each_guardrail_alone[kwargs3-failed3]` must go red (29/30 = 0.967 passes a plain rate). Capture the output, restore the line, re-run green.

- [ ] **Step 6: Commit**

```bash
git add src/services/agent_policy/learning_stats.py tests/agent_policy/test_learning_stats.py
git commit -m "feat(agent-policy): the learning statistics and guardrails, with the Wilson lower bound"
```

---

### Task 3: What the evidence supports loosening

**Files:**
- Create: `src/services/agent_policy/learning_patterns.py`
- Test: `tests/agent_policy/test_learning_patterns.py`

**Interfaces:**
- Consumes: `learning_stats.Decided`, `stats`, `failed_guardrails`; `policy_condition._number`; `conditions.leaves`.
- Produces:
  - `Leaf(kind: str, field: str, op: str, value: Any)`; `eligible_leaves(condition) -> list[Leaf]`
  - `propose(condition, decided: Sequence[Decided], cfg: dict) -> list[dict]`. A proposal is
    `{"kind": "threshold", "field", "op", "from", "to", "changeKey", "stats"}` or
    `{"kind": "list", "field", "op": "in", "remove", "changeKey", "stats"}`.
    `changeKey` is `threshold:<field>:<op>:<from>` or `list:<field>:<value>`.
  - `group_for(proposal, decided) -> list[Decided]` (the band a stored proposal is about)
  - `targets(proposal, condition) -> bool` (the condition still has the state the proposal starts from)
  - `applied(proposal, condition) -> bool` (the condition is looser than that state, in the proposal's direction)

- [ ] **Step 1: Write the failing tests**

```python
"""Which parts of a condition can be loosened and what the evidence supports. No database."""
from datetime import datetime, timedelta, timezone

from services.agent_policy import learning_patterns as LP
from services.agent_policy import learning_stats as LS
from services.agent_policy.enforcement import MASK
from services.agent_policy.settings import DEFAULTS

CFG = DEFAULTS["learning"]
T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
COND = {"all": [{"field": "tool.name", "op": "in", "value": ["refund.issue", "credit.issue"]},
                {"field": "args.amount", "op": "gt", "value": 500}]}


def decided(values, approved=True, n_people=3, days=40, field="args.amount"):
    """One decision per value, spread over `days`, by `n_people` people, 60 s each."""
    out, n = [], len(values)
    for i, v in enumerate(values):
        asked = T0 + timedelta(days=days * i / max(n - 1, 1))
        ok = approved if isinstance(approved, bool) else approved[i]
        out.append(LS.Decided(firing_id=i, values={field: v, "tool.name": "refund.issue"}, approved=ok,
                              decided_by=f"p{i % n_people}", asked_at=asked, decided_at=asked + timedelta(seconds=60)))
    return out


def test_eligible_leaves_are_thresholds_and_lists_every_match_must_satisfy():
    cond = {"all": [COND["all"][0], COND["all"][1],
                    {"field": "customer.country", "op": "in", "value": ["FR", "DE"]},
                    {"any": [{"field": "args.qty", "op": "gt", "value": 5}]},
                    {"not": {"field": "args.x", "op": "lt", "value": 1}},
                    {"field": "args.flag", "op": "eq", "value": True},
                    {"field": "args.one", "op": "in", "value": ["only"]}]}
    assert [(l.kind, l.field) for l in LP.eligible_leaves(cond)] == [
        ("threshold", "args.amount"), ("list", "customer.country")]


def test_a_single_leaf_condition_is_eligible():
    assert [l.field for l in LP.eligible_leaves({"field": "args.amount", "op": "gte", "value": 500})] == ["args.amount"]


def test_a_boolean_or_text_threshold_is_not_a_threshold():
    assert LP.eligible_leaves({"field": "a", "op": "gt", "value": True}) == []
    assert LP.eligible_leaves({"field": "a", "op": "gt", "value": "high"}) == []


def test_raise_the_limit_to_what_people_approved_rounded_inside_the_evidence():
    values = [510 + 10 * i for i in range(33)] + [837.42]          # 510 … 830, then 837.42
    [p] = LP.propose(COND, decided(values), CFG)
    assert (p["kind"], p["field"], p["op"], p["from"], p["to"]) == ("threshold", "args.amount", "gt", 500, 830)
    assert p["changeKey"] == "threshold:args.amount:gt:500"
    assert p["stats"]["decisions"] == 33 and p["to"] <= max(values)


def test_never_beyond_the_evidence_when_rounding_would_drop_below_the_guardrails():
    values = [801 + i * 0.1 for i in range(30)]                    # all between 801 and 803.9
    [p] = LP.propose(COND, decided(values), CFG)
    assert p["to"] == 803.9, "800 would leave no evidence; the furthest decided value is used"


def test_a_rejection_in_the_range_stops_the_suggestion_below_it():
    values = sorted([510 + 5 * i for i in range(40)] + [650.5] * 1)
    approved = [v != 650.5 for v in values]
    props = LP.propose(COND, decided(values, approved=approved), CFG)
    # 28 decisions sit below 650.5 (too few); every band that includes the rejection has a
    # Wilson bound under 0.85 until far more approvals pile up. With 41 decisions: 40/41 → 0.874.
    assert props == [] or all(p["stats"]["wilsonLower"] >= 0.85 for p in props)
    assert LP.propose(COND, decided(values[:30], approved=approved[:30]), CFG) == []


def test_too_little_evidence_proposes_nothing():
    assert LP.propose(COND, decided([600] * 29), CFG) == []


def test_masked_and_missing_values_are_not_evidence():
    ds = decided([MASK] * 40) + decided([None] * 5)
    assert LP.propose(COND, ds, CFG) == []


def test_text_numbers_are_read_as_the_evaluator_reads_them():
    [p] = LP.propose(COND, decided([str(600 + i) for i in range(30)]), CFG)
    assert p["to"] == 629, "rounding to 620 would leave 21 decisions, so the furthest decided value stands"


def test_a_lower_bound_is_loosened_downward():
    cond = {"field": "args.margin", "op": "lt", "value": 10}
    [p] = LP.propose(cond, decided([9.5 - 0.1 * i for i in range(30)], field="args.margin"), CFG)
    assert p["op"] == "lt" and p["from"] == 10 and 6.6 <= p["to"] < 10


def test_a_list_value_people_always_approve():
    cond = {"all": [{"field": "customer.country", "op": "in", "value": ["FR", "DE"]}]}
    ds = decided(["FR"] * 30, field="customer.country") + decided(["DE"] * 5, field="customer.country")
    [p] = LP.propose(cond, ds, CFG)
    assert (p["kind"], p["remove"], p["changeKey"]) == ("list", "FR", "list:customer.country:FR")


def test_targets_and_applied():
    p = {"kind": "threshold", "field": "args.amount", "op": "gt", "from": 500, "to": 830}
    looser = {"all": [COND["all"][0], {"field": "args.amount", "op": "gt", "value": 700}]}
    tighter = {"all": [COND["all"][0], {"field": "args.amount", "op": "gt", "value": 400}]}
    gone = {"all": [COND["all"][0]]}
    assert LP.targets(p, COND) and not LP.targets(p, looser)
    assert LP.applied(p, looser) and not LP.applied(p, COND) and not LP.applied(p, tighter)
    assert not LP.applied(p, gone), "dropping the limit makes the policy match MORE, which is not a loosening"
    q = {"kind": "list", "field": "c", "op": "in", "remove": "FR"}
    has = {"field": "c", "op": "in", "value": ["FR", "DE"]}
    assert LP.targets(q, has) and not LP.applied(q, has)
    assert LP.applied(q, {"field": "c", "op": "in", "value": ["DE"]})


def test_group_for_is_the_band_the_proposal_is_about():
    p = {"kind": "threshold", "field": "args.amount", "op": "gt", "from": 500, "to": 600}
    assert [d.values["args.amount"] for d in LP.group_for(p, decided([550, 600, 601]))] == [550, 600]
```

- [ ] **Step 2: Run to see them fail**

Run: `bptest tests/agent_policy/test_learning_patterns.py -q`
Expected: FAIL, `ImportError`.

- [ ] **Step 3: Write the module**

```python
"""What the evidence supports loosening in a policy's condition (brief §6.2, §6.4).

Pure. A threshold is loosened by moving its number toward what people approved; a list by
removing a value people always approve. Only parts of the condition that every match must
satisfy are considered: under `any` or `not`, changing one part does not mean one thing.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from typing import Any, Dict, List, Optional, Sequence

from services import policy_condition as pc
from services.agent_policy.learning_stats import Decided, failed_guardrails, stats

UP = ("gt", "gte")       # the policy matches above the number: loosening raises it
DOWN = ("lt", "lte")     # the policy matches below the number: loosening lowers it


@dataclass(frozen=True)
class Leaf:
    kind: str
    field: str
    op: str
    value: Any


def _number(value: Any) -> Optional[Decimal]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return pc._number(value, "learning")
    except pc.ConditionError:
        return None


def _plain(d: Decimal):
    return int(d) if d == d.to_integral_value() else float(d)


def _must(cond: Any) -> List[Dict[str, Any]]:
    """The leaves every match must satisfy: the condition itself, or inside nested `all`."""
    if not isinstance(cond, dict):
        return []
    if "all" in cond:
        return [leaf for c in cond["all"] if isinstance(cond["all"], list) for leaf in _must(c)]
    if "any" in cond or "not" in cond:
        return []
    return [cond]


def eligible_leaves(condition: Any) -> List[Leaf]:
    out: List[Leaf] = []
    for leaf in _must(condition):
        fld, op, value = leaf.get("field"), leaf.get("op"), leaf.get("value")
        if not fld or fld == "tool.name":
            continue
        if op in UP + DOWN and isinstance(value, (int, float)) and not isinstance(value, bool):
            out.append(Leaf("threshold", fld, op, value))
        elif op == "in" and isinstance(value, list) and len(value) >= 2:
            out.append(Leaf("list", fld, op, value))
    return out


def _nice(c: Decimal, up: bool) -> Decimal:
    """c at two significant figures, rounded toward the current limit (never past the evidence)."""
    if c == 0:
        return c
    step = Decimal(1).scaleb(c.adjusted() - 1)
    return c.quantize(step, rounding=ROUND_FLOOR if up else ROUND_CEILING)


def _band(points, frm: Decimal, to: Decimal, up: bool) -> List[Decided]:
    return [d for v, d in points if (frm <= v <= to if up else to <= v <= frm)]


def _threshold(leaf: Leaf, decided: Sequence[Decided], cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    frm, up = _number(leaf.value), leaf.op in UP
    points = [(v, d) for d in decided if (v := _number(d.values.get(leaf.field))) is not None
              and (v >= frm if up else v <= frm)]
    best = None
    for cut in sorted({v for v, _ in points}, reverse=not up):
        if not failed_guardrails(stats(_band(points, frm, cut, up)), cfg):
            best = cut                       # the furthest band that holds
    if best is None:
        return None
    nice = _nice(best, up)
    if nice != best and (nice > frm if up else nice < frm) \
            and not failed_guardrails(stats(_band(points, frm, nice, up)), cfg):
        best = nice
    if best == frm:
        return None
    return {"kind": "threshold", "field": leaf.field, "op": leaf.op, "from": _plain(frm), "to": _plain(best),
            "changeKey": f"threshold:{leaf.field}:{leaf.op}:{_plain(frm)}",
            "stats": stats(_band(points, frm, best, up))}


def _same(a: Any, b: Any) -> bool:
    return not isinstance(a, (list, dict, bool)) and a is not None and str(a) == str(b)


def _list(leaf: Leaf, decided: Sequence[Decided], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for value in leaf.value:
        group = [d for d in decided if _same(d.values.get(leaf.field), value)]
        s = stats(group)
        if not failed_guardrails(s, cfg):
            out.append({"kind": "list", "field": leaf.field, "op": "in", "remove": value,
                        "changeKey": f"list:{leaf.field}:{value}", "stats": s})
    return out


def propose(condition: Any, decided: Sequence[Decided], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for leaf in eligible_leaves(condition):
        if leaf.kind == "threshold":
            p = _threshold(leaf, decided, cfg)
            out.extend([p] if p else [])
        else:
            out.extend(_list(leaf, decided, cfg))
    return out


def group_for(proposal: Dict[str, Any], decided: Sequence[Decided]) -> List[Decided]:
    fld = proposal["field"]
    if proposal["kind"] == "list":
        return [d for d in decided if _same(d.values.get(fld), proposal["remove"])]
    frm, to, up = _number(proposal["from"]), _number(proposal["to"]), proposal["op"] in UP
    points = [(v, d) for d in decided if (v := _number(d.values.get(fld))) is not None]
    return _band(points, frm, to, up)


def _leaf_of(proposal: Dict[str, Any], condition: Any) -> Optional[Leaf]:
    for leaf in eligible_leaves(condition):
        if leaf.field == proposal["field"] and leaf.op == proposal["op"] and leaf.kind == proposal["kind"]:
            return leaf
    return None


def targets(proposal: Dict[str, Any], condition: Any) -> bool:
    leaf = _leaf_of(proposal, condition)
    if leaf is None:
        return False
    if proposal["kind"] == "list":
        return any(_same(v, proposal["remove"]) for v in leaf.value)
    return _number(leaf.value) == _number(proposal["from"])


def applied(proposal: Dict[str, Any], condition: Any) -> bool:
    """The condition is looser than the proposal's starting state, in the proposal's direction."""
    if proposal["kind"] == "list":
        leaf = next((l for l in _must(condition) if l.get("field") == proposal["field"] and l.get("op") == "in"
                     and isinstance(l.get("value"), list)), None)
        return leaf is not None and not any(_same(v, proposal["remove"]) for v in leaf["value"])
    leaf = _leaf_of(proposal, condition)
    if leaf is None:
        return False
    now, frm = _number(leaf.value), _number(proposal["from"])
    return now > frm if proposal["op"] in UP else now < frm
```

- [ ] **Step 4: Run to see them pass**

Run: `bptest tests/agent_policy/test_learning_patterns.py -q`
Expected: all pass. If `test_a_rejection_in_the_range…` fails on its second assertion, the band logic is jumping the rejection: fix the code, not the test.

- [ ] **Step 5: Prove the "inside the evidence" guard fails.** Temporarily make `_nice` round away from the limit (swap `ROUND_FLOOR`/`ROUND_CEILING`). `test_raise_the_limit_…` must go red (`to` becomes 840, above every decided value). Capture, restore, re-run green.

- [ ] **Step 6: Commit**

```bash
git add src/services/agent_policy/learning_patterns.py tests/agent_policy/test_learning_patterns.py
git commit -m "feat(agent-policy): what the evidence supports loosening, never beyond what people decided"
```

---

### Task 4: Turn a suggestion into an unsaved draft form

**Files:**
- Create: `src/services/agent_policy/learning_draft.py`
- Test: `tests/agent_policy/test_learning_draft.py`

**Interfaces:**
- Consumes: a proposal (Task 3) and a policy form (shape: `tests/agent_policy/fixtures.py::FORM_EXAMPLE`).
- Produces:
  - `plain(proposal, form) -> str` e.g. `"Raise the limit on Refund amount from 500 USD to 830 USD?"`, `"Lower the limit on …"`, `"Stop asking for approval when Customer country is FR?"`
  - `evidence(proposal) -> str` e.g. `"33 of 33 requests between 500 USD and 830 USD were approved, by 3 different people over 40 days."`
  - `build(form, proposal) -> {"form": dict, "changeNote": str, "checkWording": list[str]} | None` (None when the form no longer has the state the proposal starts from, or when a list would be left empty). Never mutates its input.

- [ ] **Step 1: Write the failing tests**

```python
"""'Draft this change' (brief §6.4): an unsaved form, built by code. No database."""
import copy

from services.agent_policy import conditions, learning_draft as LD
from tests.agent_policy.fixtures import FORM_EXAMPLE, SETTINGS

P = {"kind": "threshold", "field": "args.amount", "op": "gt", "from": 500, "to": 830,
     "changeKey": "threshold:args.amount:gt:500",
     "stats": {"decisions": 33, "approved": 33, "wilsonLower": 0.8957, "approvers": 3, "days": 40,
               "edited": 0, "reversed": 0, "medianSeconds": 60.0}}


def confirmed():
    f = copy.deepcopy(FORM_EXAMPLE)
    f["checked"] = {"by": "u1", "at": "2026-10-01T00:00:00+00:00"}
    return f


def test_plain_words_and_evidence():
    assert LD.plain(P, FORM_EXAMPLE) == "Raise the limit on Refund amount from 500 USD to 830 USD?"
    assert LD.evidence(P, FORM_EXAMPLE) == ("33 of 33 requests between 500 USD and 830 USD were approved, "
                                            "by 3 different people over 40 days.")


def test_the_draft_changes_the_condition_the_wording_and_clears_the_confirmation():
    before = confirmed()
    out = LD.build(before, P)
    f = out["form"]
    assert f["hidden"]["condition"]["all"][1] == {"field": "args.amount", "op": "gt", "value": 830}
    assert f["situation"] == "The agent is about to issue a refund or credit above $830."
    assert f["messageForAgent"] == "Refunds over $830 need Finance approval."
    assert f["checked"] is None
    assert out["changeNote"].startswith("Suggested by learning: Raise the limit on Refund amount from 500 USD to 830 USD?")
    assert "33 of 33" in out["changeNote"]
    assert out["checkWording"] == ["messageForPerson"], "it mentions no 500, so a person must read it"
    assert before == confirmed(), "build never edits the form it was given"


def test_the_examples_sit_on_the_new_boundary_and_are_computed_by_code():
    f = LD.build(confirmed(), P)["form"]
    amounts = [e["input"]["args.amount"] for e in f["examples"]]
    assert {829, 830, 831} <= set(amounts) and not {499, 500, 501} & set(amounts)
    assert 3 <= len(f["examples"]) <= 6 and not any(e["flipped"] for e in f["examples"])
    rows = conditions.reviewer_view(f, SETTINGS)
    by_amount = {r["input"]["args.amount"]: r["computed"] for r in rows if r["input"]["tool.name"] == "refund.issue"}
    assert (by_amount[829], by_amount[830], by_amount[831]) == ("none", "none", "approve")
    assert all(r["agent_expected"] == r["computed"] for r in rows)


def test_decimal_limits_step_by_a_hundredth():
    p = {**P, "to": 803.9}
    amounts = {e["input"]["args.amount"] for e in LD.build(confirmed(), p)["form"]["examples"]}
    assert {803.89, 803.9, 803.91} <= amounts


def test_wording_with_no_single_match_is_left_alone_and_flagged():
    f = confirmed()
    f["situation"] = "Refunds above the limit need approval."
    out = LD.build(f, P)
    assert out["form"]["situation"] == f["situation"] and "situation" in out["checkWording"]
    f["situation"] = "Above $500, or 500 in credits."
    assert "situation" in LD.build(f, P)["checkWording"]


def test_a_number_with_a_thousands_comma_is_replaced_in_the_same_style():
    f = confirmed()
    f["hidden"]["condition"]["all"][1]["value"] = 10000
    f["situation"] = "The agent is about to issue a refund above $10,000."
    out = LD.build(f, {**P, "from": 10000, "to": 12500})
    assert out["form"]["situation"] == "The agent is about to issue a refund above $12,500."


def test_a_list_value_is_removed():
    f = confirmed()
    f["hidden"]["condition"] = {"all": [{"field": "tool.name", "op": "in", "value": ["refund.issue"]},
                                        {"field": "customer.country", "op": "in", "value": ["FR", "DE"]}]}
    f["hidden"]["inputs"].append({"name": "Customer country", "field": "customer.country", "type": "string",
                                  "from": "action", "showApprover": True, "sensitive": False})
    p = {"kind": "list", "field": "customer.country", "op": "in", "remove": "FR",
         "changeKey": "list:customer.country:FR", "stats": P["stats"]}
    assert LD.plain(p, f) == "Stop asking for approval when Customer country is FR?"
    out = LD.build(f, p)
    assert out["form"]["hidden"]["condition"]["all"][1]["value"] == ["DE"]
    assert out["form"]["checked"] is None and "situation" in out["checkWording"]


def test_a_form_that_has_moved_on_gives_no_draft():
    f = confirmed()
    f["hidden"]["condition"]["all"][1]["value"] = 700
    assert LD.build(f, P) is None
```

- [ ] **Step 2: Run to see them fail**

Run: `bptest tests/agent_policy/test_learning_draft.py -q`
Expected: FAIL, `ImportError`.

- [ ] **Step 3: Write the module**

```python
"""'Draft this change' (brief §6.4): the next version as an unsaved form, built by code.

Pure. The condition is changed, the wording follows where that is unambiguous, the examples
are moved to the new boundary and the confirmation is cleared. Nothing is saved here: only
the owner's save, and then the normal activate step, changes a policy.
"""
from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Optional

from services.agent_policy import learning_patterns as patterns
from services.agent_policy.conditions import example_result
from services.agent_policy.learning_patterns import UP

WORDING = ("situation", "messageForAgent", "messageForPerson")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _input(form: Dict[str, Any], fld: str) -> Dict[str, Any]:
    for i in (form.get("hidden") or {}).get("inputs") or []:
        if i.get("field") == fld:
            return i
    return {}


def _amount(value: Any, unit: Optional[str]) -> str:
    return f"{value} {unit}" if unit else str(value)


def plain(proposal: Dict[str, Any], form: Dict[str, Any]) -> str:
    spec = _input(form, proposal["field"])
    name = spec.get("name") or proposal["field"]
    if proposal["kind"] == "list":
        return f"Stop asking for approval when {name} is {proposal['remove']}?"
    verb = "Raise" if proposal["op"] in UP else "Lower"
    unit = spec.get("unit")
    return f"{verb} the limit on {name} from {_amount(proposal['from'], unit)} to {_amount(proposal['to'], unit)}?"


def evidence(proposal: Dict[str, Any], form: Dict[str, Any]) -> str:
    s, spec = proposal["stats"], _input(form, proposal["field"])
    if proposal["kind"] == "list":
        what = f"requests where {spec.get('name') or proposal['field']} is {proposal['remove']}"
    else:
        lo, hi = sorted([proposal["from"], proposal["to"]])
        what = f"requests between {_amount(lo, spec.get('unit'))} and {_amount(hi, spec.get('unit'))}"
    return (f"{s['approved']} of {s['decisions']} {what} were approved, "
            f"by {s['approvers']} different people over {s['days']} days.")


def _swap_number(text: Any, old: Any, new: Any) -> Optional[str]:
    """`text` with its one mention of `old` replaced by `new`; None unless there is exactly one."""
    if not isinstance(text, str):
        return None
    hits = [m for m in _NUMBER.finditer(text) if float(m.group().replace(",", "")) == float(old)]
    if len(hits) != 1:
        return None
    m = hits[0]
    shown = f"{new:,}" if "," in m.group() else str(new)
    return text[:m.start()] + shown + text[m.end():]


def _replace_leaf(cond: Any, proposal: Dict[str, Any]) -> None:
    if not isinstance(cond, dict):
        return
    if "all" in cond:
        for c in cond["all"]:
            _replace_leaf(c, proposal)
        return
    if cond.get("field") != proposal["field"] or cond.get("op") != proposal["op"]:
        return
    if proposal["kind"] == "list":
        cond["value"] = [v for v in cond["value"] if str(v) != str(proposal["remove"])]
    else:
        cond["value"] = proposal["to"]


def _examples(form: Dict[str, Any], proposal: Dict[str, Any], on_missing: str) -> List[Dict[str, Any]]:
    fld, old = proposal["field"], form.get("examples") or []
    kept = old
    if proposal["kind"] == "threshold":
        whole = isinstance(proposal["from"], int) and isinstance(proposal["to"], int)
        step = 1 if whole else 0.01
        near = lambda v, c: isinstance(v, (int, float)) and abs(v - c) <= step * 1.5   # noqa: E731
        kept = [e for e in old if not near((e.get("input") or {}).get(fld), proposal["from"])]
        base = dict(((kept or old or [{}])[0].get("input")) or {})
        # the base must be an input the policy applies to, or the boundary would show nothing
        for e in old:
            if e.get("agentExpected") not in (None, "none"):
                base = dict(e.get("input") or {})
                break
        trio = [round(proposal["to"] + d * step, 2) if not whole else proposal["to"] + d for d in (-1, 0, 1)]
        kept = kept[:3] + [{"input": {**base, fld: v}} for v in trio]
    cond, outcome = form["hidden"]["condition"], form.get("outcome")
    return [{"input": e.get("input") or {}, "flipped": False,
             "agentExpected": example_result(cond, outcome, e.get("input") or {}, on_missing)} for e in kept]


def build(form: Dict[str, Any], proposal: Dict[str, Any], on_missing: str = "fail_closed") -> Optional[Dict[str, Any]]:
    condition = (form.get("hidden") or {}).get("condition")
    if not patterns.targets(proposal, condition):
        return None
    out = copy.deepcopy(form)
    _replace_leaf(out["hidden"]["condition"], proposal)
    check: List[str] = []
    for key in WORDING:
        if proposal["kind"] == "threshold":
            swapped = _swap_number(out.get(key), proposal["from"], proposal["to"])
            if swapped is None:
                check.append(key)
            else:
                out[key] = swapped
        else:
            check.append(key)
    out["examples"] = _examples(out, proposal, on_missing)
    out["checked"] = None
    note = f"Suggested by learning: {plain(proposal, form)} Evidence: {evidence(proposal, form)}"
    out["changeNote"] = note
    return {"form": out, "changeNote": note, "checkWording": check}
```

(`evidence` takes the form for the input's name and unit; both functions are called with the *current* form, before the change.)

- [ ] **Step 4: Run to see them pass**

Run: `bptest tests/agent_policy/test_learning_draft.py -q`
Expected: all pass. The examples test is the one that matters: results come from `conditions.reviewer_view`, the same code the form and the activation check use.

- [ ] **Step 5: Commit**

```bash
git add src/services/agent_policy/learning_draft.py tests/agent_policy/test_learning_draft.py
git commit -m "feat(agent-policy): a suggestion becomes an unsaved draft form, built by code"
```

---

### Task 5: Read the decisions, scan, and store suggestions

**Files:**
- Create: `src/services/agent_policy/learning.py`
- Modify: `src/services/backend_scheduler.py` (add the job beside `_register_agent_policy_conflict_scan_job`, and one call where line ~450 registers that job)
- Test: `tests/agent_policy/test_learning_scan_live.py`, `tests/agent_policy/test_learning_job.py`

**Interfaces:**
- Consumes: Tasks 1–4; `approvals._tx(conn)`; `settings.load_settings(conn)`.
- Produces (in `learning.py`):
  - `Target(key: str, live_version: int, form: dict, versions: list[int])` (the live form and every live version with the identical condition)
  - `decisions(cur, key: str, versions: list[int]) -> list[Decided]`
  - `targets(cur) -> list[Target]` (every live, eligible `approve` policy)
  - `scan(conn, now: datetime, *, only: Optional[list[Target]] = None) -> dict` → `{"policies", "raised", "updated", "reopened", "weakened", "superseded", "errors"}`
  - `MAX_DECISIONS = 5000`

**Behaviour, exactly:**

1. `decisions` reads people's decisions only, one per approval case:

```python
_DECIDED_SQL = """
    SELECT DISTINCT ON (COALESCE(f.decision_id, -f.firing_id))
           f.firing_id, f.matched_values, f.result, f.decided_by, f.created_at, f.decided_at,
           EXISTS (SELECT 1 FROM proc.bp_policy_firing_reversal r
                     JOIN proc.bp_policy_firing o ON o.firing_id = r.firing_id
                    WHERE o.firing_id = f.firing_id
                       OR (o.decision_id IS NOT NULL AND o.decision_id = f.decision_id)) AS reversed
      FROM proc.bp_policy_firing f
     WHERE f.policy_key = %s AND f.policy_version = ANY(%s)
       AND f.outcome = 'approve' AND f.result IN ('approved', 'rejected')
       AND f.decided_at IS NOT NULL AND f.decided_by IS NOT NULL
       AND f.decided_by NOT LIKE 'system:%%'
     ORDER BY COALESCE(f.decision_id, -f.firing_id), f.firing_id
     LIMIT %s
"""
```
Each row becomes `Decided(firing_id, values=matched_values (parsed), approved=result == 'approved', decided_by, asked_at=created_at, decided_at, reversed)`. When the limit is hit, log one warning naming the policy.

2. `targets(cur)`:

```python
_TARGETS_SQL = """
    SELECT p.policy_key, p.live_version, v.form_state, v.compiled
      FROM proc.bp_agent_policy p
      JOIN proc.bp_agent_policy_version v ON v.policy_key = p.policy_key AND v.version = p.live_version
      LEFT JOIN proc.bp_business_area a ON a.area_name = p.area_name
     WHERE p.status = 'live' AND v.form_state ->> 'outcome' = 'approve'
       AND COALESCE(a.never_suggest, FALSE) = FALSE
       AND COALESCE((v.compiled -> 'learning' ->> 'eligible')::boolean, FALSE)
"""
_SAME_CONDITION_SQL = """
    SELECT version FROM proc.bp_agent_policy_version
     WHERE policy_key = %s AND saved_as = 'live' AND form_state ->> 'outcome' = 'approve'
       AND form_state -> 'hidden' -> 'condition' = %s::jsonb
     ORDER BY version
"""
```

3. `scan` for each target, each in its own `approvals._tx(conn)` and its own `try/except` (one bad policy never stops the others; count it in `errors` and `logger.exception` it by key):
   - `cfg = settings.load_settings(conn)["learning"]` (read once per scan).
   - `decided = decisions(cur, key, versions)`; `proposals = learning_patterns.propose(condition, decided, cfg)`; add `plain`, `evidence`, `inputName` to each proposal with `learning_draft.plain/evidence` and the live form.
   - Lock the policy's rows: `SELECT suggestion_id, change_key, status, n_at_action, proposal FROM proc.bp_agent_policy_suggestion WHERE policy_key = %s FOR UPDATE`.
   - For each proposal, by `changeKey`:

     | Existing row | Action | Counter |
     |---|---|---|
     | none | INSERT `status 'open'`, `confident true` | `raised` |
     | `superseded` | UPDATE to `open`, `confident true`, new proposal/stats/version, `n_at_action NULL` | `raised` |
     | `open` | UPDATE proposal, stats, `policy_version`, `confident true`, `updated_at` | `updated` |
     | `not_yet` | if `stats.decisions >= n_at_action + cfg["not_yet_more"]`: UPDATE to `open` (as above); else nothing | `reopened` |
     | `dismissed` | if `stats.decisions >= n_at_action + cfg["dismiss_more"]`: UPDATE to `open`; else nothing | `reopened` |
     | `saved` | nothing, ever | |

   - For each existing row in `open`, `not_yet` or `dismissed` with no proposal this time:
     - if `not learning_patterns.targets(row.proposal, condition)`: UPDATE `status 'superseded'` (`superseded`). The policy's condition has moved on.
     - else if `status == 'open'`: UPDATE `stats = learning_stats.stats(learning_patterns.group_for(row.proposal, decided))`, `confident false` (`weakened`). This is "Not enough yet".
   - After all targets, and only when `only is None`: one statement that supersedes cards of policies that are no longer scanned. Real keys only, so a concurrent test's rows are never touched:

```sql
UPDATE proc.bp_agent_policy_suggestion s SET status = 'superseded', updated_at = %s
 WHERE s.status IN ('open','not_yet','dismissed') AND s.policy_key ~ '^[A-Z]{3}-[0-9]{4,}$'
   AND NOT (s.policy_key = ANY(%s))          -- the keys targets() returned
```

4. Scheduler job, modelled line for line on the conflict scan:

```python
    AGENT_POLICY_LEARNING_SCAN_JOB_NAME = "agent-policy-learning-scan"
    AGENT_POLICY_LEARNING_SCAN_LANE = "agent-policy-learning"

    def _register_agent_policy_learning_scan_job(self) -> None:
        """Look for approve policies people keep approving, every hour (first run 10 min after start).

        ON unless AGENT_POLICY_LEARNING_SCAN=off. It only ever writes suggestion rows for an owner
        to review; it never changes a policy. Idempotent: one row per policy and change.
        """
        import os
        if os.environ.get("AGENT_POLICY_LEARNING_SCAN", "on").strip().lower() == "off":
            logger.info("agent policy learning scan not registered (AGENT_POLICY_LEARNING_SCAN=off)")
            return
        if self.AGENT_POLICY_LEARNING_SCAN_JOB_NAME in self._jobs:
            return
        self.register_job(
            self.AGENT_POLICY_LEARNING_SCAN_JOB_NAME,
            self._run_agent_policy_learning_scan,
            interval=timedelta(hours=1),
            initial_delay=timedelta(minutes=10),
            lane=self.AGENT_POLICY_LEARNING_SCAN_LANE,
        )

    def _run_agent_policy_learning_scan(self) -> None:
        try:
            from datetime import datetime, timezone

            from services.agent_policy import learning
            from services.db import get_conn

            with get_conn() as conn:
                counts = learning.scan(conn, datetime.now(timezone.utc))
            if any(counts.get(k) for k in ("raised", "reopened", "weakened", "superseded", "errors")):
                logger.info("agent policy learning scan: %s", counts)
        except Exception:
            logger.exception("agent policy learning scan failed")
```

- [ ] **Step 1: Write the live tests** in `tests/agent_policy/test_learning_scan_live.py`. Use this helper and write one test per row below; each asserts on `scan`'s counts and on the stored row.

```python
"""The learning scan against the real firing log. Needs PROCWISE_TEST_LIVE_DB=1.

Targets are injected (keys TST-<hex>a, never real policies). Firing rows are append-only and
stay; suggestion rows are removed by the fixture.
"""
import copy
import json
import os
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from services.agent_policy import learning as L
from services.db import get_conn
from tests.agent_policy.fixtures import FORM_EXAMPLE

pytestmark = pytest.mark.skipif(os.getenv("PROCWISE_TEST_LIVE_DB") != "1", reason="live DB required")
NOW = datetime(2026, 10, 10, tzinfo=timezone.utc)


@pytest.fixture
def world():
    key = f"TST-{uuid.uuid4().hex[:6]}a"
    form = copy.deepcopy(FORM_EXAMPLE)
    with get_conn() as conn:
        yield conn, L.Target(key=key, live_version=1, form=form, versions=[1])
        conn.cursor().execute("DELETE FROM proc.bp_agent_policy_suggestion WHERE policy_key = %s", (key,))


def fire(conn, key, amounts, *, result="approved", people=3, days=40, seconds=60, version=1,
         decision_ids=None, by=None, start=NOW - timedelta(days=45)):
    """One settled approve firing per amount, backdated and spread over `days`."""
    ids, n = [], len(amounts)
    cur = conn.cursor()
    for i, amount in enumerate(amounts):
        asked = start + timedelta(days=days * i / max(n - 1, 1))
        cur.execute(
            "INSERT INTO proc.bp_policy_firing (policy_key, policy_version, checkpoint, action_name, outcome, "
            "result, matched_values, decision_id, decided_level, decided_by, decided_at, created_at) "
            "VALUES (%s,%s,'tool.call.before','tst','approve',%s,%s,%s,0,%s,%s,%s) RETURNING firing_id",
            (key, version, result, json.dumps({"tool.name": "refund.issue", "args.amount": amount}),
             decision_ids[i] if decision_ids else None, by or f"person-{i % people}",
             asked + timedelta(seconds=seconds), asked))
        ids.append(cur.fetchone()[0])
    return ids


def row(conn, key):
    cur = conn.cursor()
    cur.execute("SELECT status, confident, proposal, stats, n_at_action FROM proc.bp_agent_policy_suggestion "
                "WHERE policy_key = %s", (key,))
    return cur.fetchall()


def test_thirty_approvals_by_three_people_over_thirty_days_raise_one_suggestion(world):
    conn, t = world
    fire(conn, t.key, [510 + 10 * i for i in range(33)])
    counts = L.scan(conn, NOW, only=[t])
    assert counts["raised"] == 1 and counts["errors"] == 0
    [(status, confident, proposal, stats, _)] = row(conn, t.key)
    assert (status, confident, proposal["from"], proposal["to"]) == ("open", True, 500, 830)
    assert proposal["plain"] == "Raise the limit on Refund amount from 500 USD to 830 USD?"
    assert stats["decisions"] == 33
    assert L.scan(conn, NOW, only=[t])["updated"] == 1 and len(row(conn, t.key)) == 1, "a second scan adds nothing"
```

| Test name | Arrange | Assert |
|---|---|---|
| `test_one_decision_on_several_linked_rows_counts_once` | 6 distinct `decision_ids`, 5 firing rows each (30 rows), all approved | `row` is empty; `len(L.decisions(cur, key, [1])) == 6` |
| `test_timeouts_and_system_decisions_are_not_evidence` | 30 rows `result='timed_out'`, 30 approved rows `by='system:precedent'`, 30 `by='system:group'` | no suggestion; `L.decisions(...) == []` |
| `test_a_masked_value_is_not_evidence` | 40 approved rows whose `args.amount` is `"•••"` | no suggestion, `errors == 0` |
| `test_a_reversal_blocks_the_band_and_shows_not_enough_yet` | raise as in the first test; then `INSERT INTO proc.bp_policy_firing_reversal` for one firing at 600; scan again | `weakened == 1`; row `open`, `confident False`, `stats.reversed == 1`, `proposal.to == 830` unchanged |
| `test_a_reversal_on_a_linked_sibling_row_counts` | 30 decisions each with 2 linked rows; reverse the *second* row of one | after scan: no suggestion (never raised) |
| `test_not_yet_hides_until_thirty_more_decisions` | raise; `UPDATE … SET status='not_yet', n_at_action=33`; add 29 more approvals in range; scan; add 1 more; scan | first scan leaves `not_yet`; second: `reopened == 1`, `open` |
| `test_dismissed_returns_only_on_materially_stronger_evidence` | as above with `dismissed` | same shape |
| `test_a_saved_suggestion_is_never_raised_again` | raise; `UPDATE … SET status='saved'`; add 40 approvals; scan | row still `saved`; `raised == 0` |
| `test_a_changed_condition_supersedes_the_card` | raise; change `t.form`'s limit to 700 (new `Target` with the same key); scan | `superseded == 1`; a new key `threshold:args.amount:gt:700` is raised only if its own band passes |
| `test_evidence_from_an_older_version_with_the_same_condition_counts` | 20 firings at `version=1`, 13 at `version=2`; `Target(versions=[1, 2], live_version=2)` | raised, `stats.decisions == 33` |
| `test_one_broken_policy_does_not_stop_the_others` | two targets; the first has `form["hidden"]["condition"] = "nonsense"` and the second valid evidence | `errors == 0` or `1` for the first (nothing to propose is not an error; an exception is), and the second is raised |
| `test_a_non_numeric_setting_raises_no_suggestion` | monkeypatch `settings.load_settings` to return `learning.min_decisions = "thirty"`; valid evidence | no row; `errors == 1` |

- [ ] **Step 2: Run to see them fail.** `LIVE=1 bptest tests/agent_policy/test_learning_scan_live.py -q` → `ImportError: cannot import name 'learning'`.

- [ ] **Step 3: Write `learning.py`** to the behaviour above: `Target`, `decisions`, `targets`, `scan`, and a private `_store(cur, target, proposals, decided, cfg, now) -> Counter`. Module docstring: say that this module never writes a policy version.

- [ ] **Step 4: Run to see them pass.** `LIVE=1 bptest tests/agent_policy/test_learning_scan_live.py -q` → all pass.

- [ ] **Step 5: Prove the one-decision-per-case guard fails.** Remove `DISTINCT ON (…)` from `_DECIDED_SQL`. `test_one_decision_on_several_linked_rows_counts_once` must go red (a suggestion is raised from 6 decisions). Capture, restore, green.

- [ ] **Step 6: Write the job tests** in `tests/agent_policy/test_learning_job.py`, copying the three conflict-scan tests in `tests/agent_policy/test_approvals_job.py` (`test_conflict_scan_toggle`, `test_conflict_scan_has_its_own_lane`, `test_conflict_scan_calls_detect_all_and_never_raises`) with the learning names: the toggle (`AGENT_POLICY_LEARNING_SCAN` unset/on → registered; `off` → not), the lane (`agent-policy-learning`), and "calls `learning.scan` and never raises when it throws". Then add the job and its registration call. Run `bptest tests/agent_policy/test_learning_job.py tests/agent_policy/test_approvals_job.py -q`. (`test_conflict_scan_has_its_own_lane` fails before and after on this branch; it is a known pre-existing failure. Report it; do not fix it here.)

- [ ] **Step 7: Commit**

```bash
git add src/services/agent_policy/learning.py src/services/backend_scheduler.py tests/agent_policy/test_learning_scan_live.py tests/agent_policy/test_learning_job.py
git commit -m "feat(agent-policy): the hourly learning scan raises suggestions from people's decisions"
```

---

### Task 6: The card, the queue, and the three actions

**Files:**
- Modify: `src/services/agent_policy/learning.py`
- Modify: `src/repositories/agent_policy_repo.py` (`get_policy`, `save_version`, `create_draft` unchanged)
- Modify: `src/api/routers/agent_policies.py`
- Modify (separate commit, ruling Q1): `src/api/main.py`
- Test: `tests/agent_policy/test_learning_cards_live.py`, `tests/agent_policy/test_learning_endpoints_live.py`

**Interfaces:**
- Produces (in `learning.py`):
  - `class Refused(Exception)` with `.code` and `.message`
  - `card(row: dict) -> dict`:
    ```json
    {"id": 12, "policyKey": "FIN-0012", "policyName": "Refund or credit over $500", "status": "open",
     "kind": "threshold", "change": "Raise the limit on Refund amount from 500 USD to 830 USD?",
     "evidence": "33 of 33 requests between … over 40 days.",
     "stats": {"decisions": 33, "approved": 33, "approvers": 3, "days": 40, "medianSeconds": 60.0, "wilsonLower": 0.8957},
     "suggestionConfidence": "High", "canDraft": true, "updatedAt": "…"}
    ```
    `suggestionConfidence` is `"High"` when `confident`, else `"Not enough yet"`; `canDraft` is `confident and status == 'open'`. The field is never called `confidence`.
  - `list_cards(conn, *, status: str = "open") -> list[dict]` (`open` | `all`; newest first; `policyName` from the policy's latest version form, `""` if the policy row is gone)
  - `for_policy(cur, key: str) -> list[dict]` (status `open` only)
  - `act(conn, suggestion_id: int, verb: str, *, actor: str, now: datetime) -> dict` (`verb` in `not_yet`, `dismiss`; only from `open`; sets `status`, `n_at_action = stats.decisions`, `acted_by`, `acted_at`; otherwise `Refused("not_open", "This suggestion is no longer open.")`)
  - `draft(conn, suggestion_id: int) -> dict` → `{"policyKey", "baseVersion", "form", "changeNote", "checkWording", "suggestionId"}`. Built from the policy's **latest** version form with `learning_draft.build`. Refusals: `not_open`; `not_confident` ("There is not enough evidence yet."); `policy_changed` ("This policy has changed since the suggestion was made.") when `build` returns None or the policy is not live.
  - `note_save(cur, key: str, version: int, form: dict) -> int`: every `open` or `not_yet` suggestion of the policy whose proposal is `learning_patterns.applied(proposal, form.hidden.condition)` becomes `saved` with `saved_version`. Returns how many.
- Router, declared **above** `@router.get("/{key}")` (order matters: `/{key}` would swallow `/suggestions`):
  - `GET /agent-policies/suggestions?status=open|all` → `{"suggestions": [card]}`; `_require(p, "Viewer", "agent_policy.read", {})`
  - `GET /agent-policies/suggestions/{suggestion_id}/draft` → the draft; `_require(p, "Buyer", "agent_policy.read", {...})`; `Refused` → 409 `{"code", "message"}`
  - `POST /agent-policies/suggestions/{suggestion_id}/act` body `{"verb": "not_yet"|"dismiss"}` → 201 card; `_require(p, "Buyer", "agent_policy.suggestion_act", {"suggestion": id, "verb": verb})`; `Refused` → 409; unknown id → 404
- `repo.get_policy` adds `"suggestions": learning.for_policy(cur, policy_key)`.
- `repo.save_version` calls `learning.note_save(cur, policy_key, version, form)` just before `conn.commit()`, for both intents (a draft save is "saved" in the brief's sense: "Once a suggested change is saved, do not suggest it again").

- [ ] **Step 1: Write the failing tests.** `test_learning_cards_live.py` (reuse `world`, `fire`, `row` by importing them from `tests.agent_policy.test_learning_scan_live`):

```python
def test_the_card_names_suggestion_confidence_and_never_extraction_confidence(world):
    conn, t = world
    fire(conn, t.key, [510 + 10 * i for i in range(33)])
    L.scan(conn, NOW, only=[t])
    [c] = [c for c in L.list_cards(conn) if c["policyKey"] == t.key]
    assert c["suggestionConfidence"] == "High" and c["canDraft"] is True and "confidence" not in c
    assert c["change"].startswith("Raise the limit") and c["stats"]["approvers"] == 3


def test_not_yet_and_dismiss_record_who_when_and_how_many(world):
    conn, t = world
    fire(conn, t.key, [510 + 10 * i for i in range(33)])
    L.scan(conn, NOW, only=[t])
    sid = L.for_policy(conn.cursor(), t.key)[0]["id"]
    assert L.act(conn, sid, "not_yet", actor="u1", now=NOW)["status"] == "not_yet"
    assert L.for_policy(conn.cursor(), t.key) == [], "hidden from the form"
    assert row(conn, t.key)[0][4] == 33
    with pytest.raises(L.Refused) as exc:
        L.act(conn, sid, "dismiss", actor="u1", now=NOW)
    assert exc.value.code == "not_open"


def test_a_weak_card_cannot_be_drafted(world):
    conn, t = world
    fire(conn, t.key, [510 + 10 * i for i in range(33)])
    L.scan(conn, NOW, only=[t])
    conn.cursor().execute("UPDATE proc.bp_agent_policy_suggestion SET confident = false WHERE policy_key = %s", (t.key,))
    sid = L.for_policy(conn.cursor(), t.key)[0]["id"]
    assert L.for_policy(conn.cursor(), t.key)[0]["suggestionConfidence"] == "Not enough yet"
    with pytest.raises(L.Refused) as exc:
        L.draft(conn, sid)
    assert exc.value.code == "not_confident"
```

`test_learning_endpoints_live.py` drives the real app (`from api.main import app`, `fastapi.testclient.TestClient`), with the gateway headers the other endpoint tests use (copy the `HEADERS`/`_as(role)` helper from `tests/agent_policy/test_conflict_endpoints_live.py`). It needs a **real** policy, because `draft` and `note_save` read policy rows: create one with `repo.create_draft` + `repo.save_version(intent="activate")` in business area "Operations" using the helper that file already has for making a live test policy, name it "TST …", and retire it in the fixture's teardown. Tests:

| Test | Assert |
|---|---|
| `test_viewer_reads_the_queue_buyer_acts_viewer_cannot` | GET as Viewer 200; POST act as Viewer 403; as Buyer 201 |
| `test_draft_returns_a_form_and_writes_nothing` | GET draft 200; `form.hidden.condition` carries the new limit; `checked` is null; the policy's `latestVersion` is unchanged; the suggestion is still `open` |
| `test_saving_the_draft_marks_the_suggestion_saved_and_it_is_not_suggested_again` | POST `/{key}/versions` with the draft form, `intent "draft"` → 200/201; row `status 'saved'`, `saved_version` = new version; `GET /{key}` has `suggestions == []`; a new scan with more approvals raises nothing for that change key |
| `test_saving_an_unrelated_edit_does_not_mark_it_saved` | save a draft that only changes `owner` → suggestion stays `open` |
| `test_a_policy_that_moved_on_refuses_the_draft` | save a draft with the limit raised by hand to 700; GET draft → 409 `policy_changed` |
| `test_the_policy_page_carries_its_open_cards` | `GET /{key}` → `suggestions[0].id` equals the queue's |
| `test_nothing_in_this_stage_activates_a_policy` | after scan + draft + act: `SELECT count(*) FROM proc.bp_agent_policy_version WHERE policy_key = %s` is unchanged, and `live_version` is unchanged |
| `test_unknown_suggestion_is_404_and_a_bad_verb_is_422` | as named |

- [ ] **Step 2: Run to see them fail.** `LIVE=1 bptest tests/agent_policy/test_learning_cards_live.py tests/agent_policy/test_learning_endpoints_live.py -q`.

- [ ] **Step 3: Implement** `Refused`, `card`, `list_cards`, `for_policy`, `act`, `draft`, `note_save`; the three routes; the two repo lines. `act` and `draft` lock the suggestion row (`FOR UPDATE`) inside `approvals._tx`. In the router:

```python
class SuggestionActBody(BaseModel):
    verb: str = Field(pattern="^(not_yet|dismiss)$")


def _refused_learning(exc: learning.Refused) -> JSONResponse:
    return JSONResponse(status_code=409, content={"code": exc.code, "message": exc.message})
```

- [ ] **Step 4: Run to see them pass**, then the whole policy suite in both modes, compared with the same run on `origin/Development` (record the baseline first; three failures are known pre-existing):

```bash
bptest tests/agent_policy tests/engines/test_decide_live_conflict.py -q
LIVE=1 bptest tests/agent_policy tests/engines/test_decide_live_conflict.py -q
```
Expected: no test that passed on the baseline fails now.

- [ ] **Step 5: Let the two reads through the real app's output filter (user ruling Q1, 2026-10-10).** Add `tests/agent_policy/test_learning_endpoints_scrub.py` modelled on `tests/agent_policy/test_conflict_endpoints_scrub.py`: with a policy whose situation contains `01/04/2026`, read the queue and the draft through `api.main.app`. First run it red: the text comes back as `[withheld]`. Then add to `_AGENT_POLICY_SCREEN_EXEMPT` in `src/api/main.py`, with a comment naming ruling Q1 and its date:

```python
    ("GET", re.compile(r"^/agent-policies/suggestions$")),
    ("GET", re.compile(r"^/agent-policies/suggestions/[0-9]{1,18}/draft$")),
```
Run it green. Also assert that an error answer from the same paths (a 409 from the draft) and `POST …/act` are still scrubbed. Commit these two files on their own: `feat(agent-policy): the suggestion reads pass the customer's own policy wording through (ruling Q1)`.

- [ ] **Step 6: Commit**

```bash
git add src/services/agent_policy/learning.py src/repositories/agent_policy_repo.py src/api/routers/agent_policies.py tests/agent_policy/test_learning_cards_live.py tests/agent_policy/test_learning_endpoints_live.py tests/agent_policy/test_learning_endpoints_scrub.py
git commit -m "feat(agent-policy): the suggestion card, the queue, and Draft / Not yet / Dismiss"
```

---

### Task 7: Record that an approved action was reversed

**Files:**
- Modify: `src/services/agent_policy/learning.py` (`record_reversal`)
- Modify: `src/services/agent_policy/approval_views.py` (`policy_firings` and `_firing_view`)
- Modify: `src/api/routers/agent_policies.py`
- Test: `tests/agent_policy/test_learning_reversal_live.py`

**Interfaces:**
- Produces:
  - `learning.record_reversal(conn, firing_id: int, *, actor: str, reason: str, now: datetime) -> dict` → `{"firingId": <the case's first firing>, "reversedBy", "reversedAt", "reason"}`. The reversal is stored on the **earliest** firing row of the same approval case (`decision_id`), so one decision has at most one reversal whichever linked row was clicked. Refusals (`Refused`): `not_found`; `not_approved` ("Only an approved request can be marked as reversed.") unless `outcome = 'approve' AND result = 'approved'`; `already_reversed`; `reason_required` (blank reason).
  - `POST /agent-policies/firings/{firing_id}/reversal` body `{"reason": str (1–2000 chars)}` → 201; `_require(p, "Approver", "agent_policy.reversal", {"firing": firing_id})`; `not_found` → 404, others → 409.
  - `policy_firings` rows gain `"reversed": {"by", "at", "reason"} | None` (LEFT JOIN on the reversal table by the case's first firing; a linked sibling row shows the same reversal).

- [ ] **Step 1: Write the failing tests** (import `world`, `fire` from the scan tests):

| Test | Assert |
|---|---|
| `test_an_approver_marks_an_approved_firing_reversed` | 201-shaped dict; row in `bp_policy_firing_reversal` with actor and reason |
| `test_a_rejected_or_paused_or_notify_firing_cannot_be_reversed` | `Refused.code == "not_approved"` for each |
| `test_a_second_reversal_of_the_same_decision_is_refused_even_from_a_sibling_row` | two linked rows, reverse row 2, then row 1 → `already_reversed`; the stored `firing_id` is row 1's |
| `test_a_blank_reason_is_refused` | `reason_required` |
| `test_the_policy_activity_list_shows_the_reversal` | `approval_views.policy_firings(conn, key, 50)` → the row and its sibling both carry `reversed.reason` |
| `test_the_endpoint_needs_the_approver_role` | Buyer 403, Approver 201 (through `api.main.app`) |
| `test_a_reversal_reaches_the_next_scan` | raise a suggestion, reverse one firing in its band through the endpoint, scan → `confident False` |

- [ ] **Step 2: Run to see them fail.** `LIVE=1 bptest tests/agent_policy/test_learning_reversal_live.py -q`.

- [ ] **Step 3: Implement.** `record_reversal` in one `approvals._tx`:

```python
_FIRST_OF_CASE_SQL = """
    SELECT o.firing_id, o.outcome, o.result
      FROM proc.bp_policy_firing f
      JOIN proc.bp_policy_firing o
        ON o.firing_id = f.firing_id OR (f.decision_id IS NOT NULL AND o.decision_id = f.decision_id)
     WHERE f.firing_id = %s
     ORDER BY o.firing_id LIMIT 1
"""
```
then `INSERT INTO proc.bp_policy_firing_reversal (firing_id, reversed_by, reversed_at, reason) VALUES (%s,%s,%s,%s) ON CONFLICT (firing_id) DO NOTHING RETURNING firing_id`; no row returned → `already_reversed`.

- [ ] **Step 4: Run to see them pass**; re-run `LIVE=1 bptest tests/agent_policy/test_approval_endpoints_live.py -q` (it reads `/firings`) to confirm nothing that passed before fails.

- [ ] **Step 5: Commit**

```bash
git add src/services/agent_policy/learning.py src/services/agent_policy/approval_views.py src/api/routers/agent_policies.py tests/agent_policy/test_learning_reversal_live.py
git commit -m "feat(agent-policy): an approver records that an approved action was reversed"
```

---

### Task 8: The second reviewer

**Files:**
- Create: `src/services/agent_policy/second_review.py`
- Modify: `src/repositories/agent_policy_repo.py` (`save_version` activate branch; `get_policy`)
- Modify: `src/api/routers/agent_policies.py`
- Test: `tests/agent_policy/test_second_review.py` (pure), `tests/agent_policy/test_second_review_live.py`

**Interfaces:**
- Consumes: `learning_patterns.applied`; `approvals._same_person(principal, who)`; `approval_views.caller_names(principal, mapping)`; `deciders` mapping loaded as `readiness._load_deciders()` does.
- Produces (in `second_review.py`):
  - `MESSAGE = "A second reviewer must review this change before it goes Active. It loosens the policy as learning suggested."`
  - `loosening(suggestions: list[dict], live_condition, new_condition) -> Optional[int]`: the id of the first suggestion (any status except `superseded`) whose proposal is `applied` in `new_condition` and was not already applied in `live_condition`. Pure.
  - `needed(cur, policy_key: str, live_version: Optional[int], new_form: dict) -> Optional[int]`: None when there is no live version, when the form's business area (resolved exactly as a save resolves it, `repo._area`) has `second_reviewer` false, or when `loosening` is None.
  - `activation_problems(cur, policy_key, live_version, base_version, new_form) -> list[dict]`: `[]`, or `[{"field": "secondReview", "code": "second_review_required", "message": MESSAGE}]` when `needed` and there is no review row for `(policy_key, base_version)` **or** the base version's `hidden.condition` differs from `new_form`'s (the review was of something else).
  - `record(conn, policy_key, version, *, principal, note, now) -> dict`. Refusals (`Refused(code, message)`): `not_latest` (the version is not the policy's latest, or it is not a draft); `not_needed`; `own_change` ("You saved this change, so someone else must review it.") when `_same_person(principal, saved_by)`; `owner` ("The policy's owner cannot be its second reviewer.") when the form's owner name is in `caller_names(principal, mapping)`; `already_reviewed`.
  - `status(cur, policy_key, head: dict, versions: list[dict], principal=None) -> dict` → `{"required": bool, "version": int|None, "reviewedBy": str|None, "reviewedAt": iso|None}` for the latest version when it is a draft on top of a live version; `{"required": False, …None}` otherwise.
- `repo.save_version`, activate branch, directly after `problems = list(_activation_problems(...))`:
  ```python
  problems += second_review.activation_problems(cur, policy_key, row["live_version"], base_version, form)
  ```
- `repo.get_policy` adds `"secondReview": second_review.status(cur, policy_key, …)`.
- `POST /agent-policies/{key}/second-review` body `{"version": int, "note": Optional[str] (max 2000)}` → 201 `{"policyKey", "version", "reviewedBy", "reviewedAt"}`; `_require(p, "Approver", "agent_policy.second_review", {"policy": key, "version": version})`; `Refused` → 409 `{"code", "message"}`; unknown policy → 404.

- [ ] **Step 1: Write the pure tests** (`test_second_review.py`):

```python
from services.agent_policy import second_review as SR

LIVE = {"all": [{"field": "tool.name", "op": "in", "value": ["refund.issue"]},
                {"field": "args.amount", "op": "gt", "value": 500}]}


def at(limit):
    return {"all": [LIVE["all"][0], {"field": "args.amount", "op": "gt", "value": limit}]}


def sug(status="open", sid=7):
    return {"id": sid, "status": status,
            "proposal": {"kind": "threshold", "field": "args.amount", "op": "gt", "from": 500, "to": 830}}


def test_raising_the_limit_a_suggestion_pointed_at_needs_a_review_whatever_its_status():
    for status in ("open", "not_yet", "dismissed", "saved"):
        assert SR.loosening([sug(status)], LIVE, at(830)) == 7
    assert SR.loosening([sug()], LIVE, at(600)) == 7, "a smaller loosening in the same direction still counts"


def test_no_review_for_an_unchanged_or_tighter_condition_or_without_a_suggestion():
    assert SR.loosening([sug()], LIVE, LIVE) is None
    assert SR.loosening([sug()], LIVE, at(400)) is None
    assert SR.loosening([], LIVE, at(830)) is None
    assert SR.loosening([sug("superseded")], LIVE, at(830)) is None
```

- [ ] **Step 2: Write the live tests** (`test_second_review_live.py`), with two real "TST …" policies made as in Task 6's endpoint tests: one in **Finance** (second reviewer on by seed) and one in **Operations** (off). Give each a suggestion row by injecting firings and scanning with a `Target` built from the real policy (`L.targets(cur)` filtered to the key). Principals: `saver` (Approver), `reviewer` (Approver, different subject and email), `owner_person` (Approver whose email is mapped to the form's owner name in a test row of `bp_policy_decider_map` that the fixture inserts and removes).

| Test | Assert |
|---|---|
| `test_finance_activation_of_the_suggested_loosening_is_refused_without_a_review` | save the draft as `saver`; activate → `NotReady`, problems contain `second_review_required` and nothing else new; `live_version` unchanged |
| `test_the_person_who_saved_the_draft_cannot_review_it` | `record` as `saver` → `own_change` |
| `test_the_owner_cannot_review_it` | `record` as `owner_person` → `owner` |
| `test_after_another_approver_reviews_it_activation_succeeds` | `record` as `reviewer`; activate with `base_version` = the reviewed draft → live; review row present |
| `test_a_review_of_one_draft_does_not_cover_a_different_condition` | review draft at 830; activate a form with 900 on the same base → refused |
| `test_typing_the_loosening_by_hand_still_needs_the_review` | dismiss the suggestion; save a hand-edited draft at 800; activate → refused |
| `test_operations_needs_no_second_review` | same flow in Operations → activates |
| `test_tightening_or_an_unrelated_edit_needs_no_review_in_finance` | limit 400, or owner changed → activates |
| `test_a_review_is_refused_when_none_is_needed_or_the_version_is_not_the_latest_draft` | `not_needed`, `not_latest` |
| `test_the_policy_page_says_a_review_is_required_and_then_who_did_it` | `get_policy(...)["secondReview"]` before and after |
| `test_the_endpoint_needs_the_approver_role` | Buyer 403 through `api.main.app` |

- [ ] **Step 3: Run to see them fail**, implement, run to see them pass.

```bash
bptest tests/agent_policy/test_second_review.py -q
LIVE=1 bptest tests/agent_policy/test_second_review_live.py -q
```

- [ ] **Step 4: Prove the self-review guard fails.** Make `record` skip the `own_change` check. `test_the_person_who_saved_the_draft_cannot_review_it` must go red. Capture, restore, green.

- [ ] **Step 5: Confirm ordinary activation is untouched.** Run the whole policy suite in both modes (the commands in Task 6 Step 4) and compare with the baseline. A policy with no suggestion rows must activate exactly as before: `second_review.needed` returns None after one `SELECT` on the suggestion table.

- [ ] **Step 6: Commit**

```bash
git add src/services/agent_policy/second_review.py src/repositories/agent_policy_repo.py src/api/routers/agent_policies.py tests/agent_policy/test_second_review.py tests/agent_policy/test_second_review_live.py
git commit -m "feat(agent-policy): a suggested loosening in a second-reviewer area needs a second person before it goes Active"
```

---

### Task 9: The gateway forwards the five new calls

**Files (gateway worktree):**
- Modify: `src/modules/agent-policy/agent-policy.controller.ts`
- Modify: `src/modules/agent-policy/agent-policy.yml`
- Test: `src/modules/agent-policy/agent-policy.controller.spec.ts`

**Interfaces:**
- Consumes: the backend routes of Tasks 6–8; the controller's existing `need(req, role)`, `id(v, what)`, `KEY`, `this.svc.forward(user, method, path, body?, timeoutMs?)`.
- Produces (gateway paths the UI calls; each forwards to the same path on the backend):

| Method and path | Role floor | Body the gateway accepts (closed shape; unknown keys → 400) |
|---|---|---|
| `GET agent-policies/suggestions?status=open\|all` | Viewer | none; `status` defaults to `open`, anything else → 400 |
| `GET agent-policies/suggestions/:suggestionId/draft` | Buyer | none |
| `POST agent-policies/suggestions/:suggestionId/act` | Buyer | `{verb}` with `verb` in `not_yet`, `dismiss` |
| `POST agent-policies/firings/:firingId/reversal` | Approver | `{reason}` text, 1–2000 chars after trim |
| `POST agent-policies/:key/second-review` | Approver | `{version}` positive integer, optional `{note}` text ≤ 2000 |

The role floor here is the same as the backend's. The backend re-checks it; the gateway check keeps a refused call from ever leaving the gateway.

- [ ] **Step 1: Write the failing tests.** Add `describe('learning (stage 5)', …)` to the spec, using its existing `req`, `ctl`, `fetchMock` helpers. Complete code for the first; write the others in the same shape.

```ts
describe('learning (stage 5)', () => {
  const V = req(['PROCWISE_VIEWER']);
  const B = req(['PROCWISE_PROCUMENT_BUYER_ANALYST']);
  const A = req(['PROCWISE_ADMIN']);
  const url = (i = 0) => fetchMock.mock.calls[i][0];
  const body = (i = 0) => JSON.parse(fetchMock.mock.calls[i][1].body);

  it('a viewer reads the queue; status defaults to open and is checked', async () => {
    await ctl().listSuggestions(V, undefined);
    expect(url()).toBe('http://py/agent-policies/suggestions?status=open');
    await ctl().listSuggestions(V, 'all');
    expect(url(1)).toBe('http://py/agent-policies/suggestions?status=all');
    await expect(ctl().listSuggestions(V, 'closed')).rejects.toThrow('status must be open or all');
  });

  it('a viewer cannot draft or act; a buyer can', async () => {
    await expect(ctl().suggestionDraft(V, '7')).rejects.toThrow('needs the Buyer role or higher');
    await expect(ctl().suggestionAct(V, '7', { verb: 'dismiss' })).rejects.toThrow('needs the Buyer role or higher');
    await ctl().suggestionDraft(B, '7');
    expect(url()).toBe('http://py/agent-policies/suggestions/7/draft');
    await ctl().suggestionAct(B, '7', { verb: 'not_yet' });
    expect(url(1)).toBe('http://py/agent-policies/suggestions/7/act');
    expect(body(1)).toEqual({ verb: 'not_yet' });
    expect(fetchMock.mock.calls[1][1].headers['X-User-Sub']).toBe('u1');
  });
```

Further cases, one `it` each:
- `act` rejects `{verb: 'apply'}`, `{verb: 'dismiss', extra: 1}` and a non-numeric id with 400, and makes no backend call (`expect(fetchMock).not.toHaveBeenCalled()`).
- `reversal`: Buyer → 403 text `needs the Approver role or higher`; Admin forwards `{reason: 'clawed back'}` trimmed; blank reason, 2001 characters, an extra key → 400.
- `second-review`: Buyer → 403; Admin forwards `{version: 4}` and `{version: 4, note: 'checked the evidence'}`; `version: '4'`, `0`, `1.5`, an extra key and a key that fails `KEY` → 400.
- A backend 409 `{code, message}` passes through with status 409 and the same body (mock `{ status: 409, json: async () => ({ code: 'policy_changed', message: 'x' }) }` and assert on `getStatus()` and `getResponse()`).
- Declaration order: extend the existing order assertion (spec, "static segments before `:key`") so that `suggestions` and `firings/:firingId/reversal` are declared before `@Get(':key')` and `@Get(':key/firings')`.

- [ ] **Step 2: Run to see them fail.** From `$GW`: `npx jest src/modules/agent-policy/agent-policy.controller.spec.ts` → the new tests fail with `ctl(...).listSuggestions is not a function`.

- [ ] **Step 3: Add the routes** above the first `:key` route, with two closed-shape validators beside `conflictDecision`:

```ts
const SUGGESTION_STATUS = /^(open|all)$/;
const VERB = /^(not_yet|dismiss)$/;

function only(b: any, keys: string[]): void {
  if (!b || typeof b !== 'object' || Array.isArray(b)) throw new BadRequestException('a body is required');
  const extra = Object.keys(b).filter((k) => !keys.includes(k));
  if (extra.length) throw new BadRequestException(`unknown field: ${extra[0]}`);
}
function text(v: any, what: string, max: number, required: boolean): string | undefined {
  if (v === undefined || v === null) {
    if (required) throw new BadRequestException(`${what} is required`);
    return undefined;
  }
  if (typeof v !== 'string' || v.length > max) throw new BadRequestException(`${what} must be text of at most ${max} characters`);
  const t = v.trim();
  if (required && !t) throw new BadRequestException(`${what} is required`);
  return t;
}
```

```ts
  @Get('suggestions') async listSuggestions(@Req() req, @Query('status') status?: string) {
    need(req, 'Viewer');
    const s = status === undefined || status === '' ? 'open' : status;
    if (typeof s !== 'string' || !SUGGESTION_STATUS.test(s)) throw new BadRequestException('status must be open or all');
    return this.svc.forward(req.user, 'GET', `/agent-policies/suggestions?status=${s}`);
  }
  @Get('suggestions/:suggestionId/draft') async suggestionDraft(@Req() req, @Param('suggestionId') s: string) {
    need(req, 'Buyer');
    return this.svc.forward(req.user, 'GET', `/agent-policies/suggestions/${id(s, 'suggestion id')}/draft`);
  }
  @Post('suggestions/:suggestionId/act') async suggestionAct(@Req() req, @Param('suggestionId') s: string, @Body() body: any) {
    need(req, 'Buyer');
    only(body, ['verb']);
    if (typeof body.verb !== 'string' || !VERB.test(body.verb)) throw new BadRequestException('verb must be not_yet or dismiss');
    return this.svc.forward(req.user, 'POST', `/agent-policies/suggestions/${id(s, 'suggestion id')}/act`, { verb: body.verb });
  }
  @Post('firings/:firingId/reversal') async firingReversal(@Req() req, @Param('firingId') f: string, @Body() body: any) {
    need(req, 'Approver');
    only(body, ['reason']);
    return this.svc.forward(req.user, 'POST', `/agent-policies/firings/${id(f, 'firing id')}/reversal`,
      { reason: text(body.reason, 'reason', 2000, true) });
  }
  @Post(':key/second-review') async secondReview(@Req() req, @Param('key') k: string, @Body() body: any) {
    need(req, 'Approver');
    only(body, ['version', 'note']);
    if (typeof k !== 'string' || !KEY.test(k)) throw new BadRequestException('not a policy id');
    if (!Number.isInteger(body.version) || body.version < 1) throw new BadRequestException('version must be a whole number');
    const note = text(body.note, 'note', 2000, false);
    return this.svc.forward(req.user, 'POST', `/agent-policies/${k}/second-review`,
      note === undefined ? { version: body.version } : { version: body.version, note });
  }
```
(If the controller already has a helper that validates `:key`, use it instead of the inline `KEY.test`. `secondReview` may sit with the other `:key/...` routes.)

- [ ] **Step 4: Add five function blocks to `agent-policy.yml`**, each a copy of the `decideAgentPolicyConflict` block (lines ~526–549: same `handler`, `timeout`, `cors` origins and headers, `authorizer`) with only the name, description, `path` and `method` changed:

| Function name | path | method |
|---|---|---|
| `listAgentPolicySuggestions` | `agent-policies/suggestions` | get |
| `getAgentPolicySuggestionDraft` | `agent-policies/suggestions/{suggestionId}/draft` | get |
| `actAgentPolicySuggestion` | `agent-policies/suggestions/{suggestionId}/act` | post |
| `reverseAgentPolicyFiring` | `agent-policies/firings/{firingId}/reversal` | post |
| `secondReviewAgentPolicy` | `agent-policies/{key}/second-review` | post |

- [ ] **Step 5: Run to see them pass**, then the whole gateway suite and the build:

```bash
npx jest src/modules/agent-policy/agent-policy.controller.spec.ts
npm test
npm run build
```
Expected: the module's spec all green; `npm test` shows no failure that the baseline (`origin/spendiq-ui`, recorded before you start) does not show; the build succeeds.

- [ ] **Step 6: Commit**

```bash
git add src/modules/agent-policy/agent-policy.controller.ts src/modules/agent-policy/agent-policy.yml src/modules/agent-policy/agent-policy.controller.spec.ts
git commit -m "feat(agent-policy): gateway routes for learning suggestions, reversals and the second review"
```

---

### Task 10: The suggestion card and the Suggestions queue on screen

**Files (UI worktree):**
- Create: `src/modules/SpendIQ/agentPolicy/suggestions.js`, `src/modules/SpendIQ/agentPolicy/suggestions.test.js`
- Modify: `src/modules/SpendIQ/index.jsx` (import the module; add `suggestions: agentPolicySuggestions` to `window.__SPENDIQ_AP__`)
- Modify: `src/modules/SpendIQ/engine.js` (new `apLn*` functions in a new `/* ============ Agent policies (stage 5)` section placed **before** the `/* The small edit form` marker; one entry in `apPoliciesTab`'s `views`; one call in `apFormHTML`)
- Create: `src/modules/SpendIQ/agentPolicy/engineWiring.stage5.contract.test.js`

**Interfaces:**
- Consumes: gateway paths of Task 9; `apApi`, `apOpen(key)`, `apEdit(state, mutate, rerender)`, `apRender(state)`, `apCanEdit()`, `apConfirmTwoStep(title, text, run, opts)`, `apApprRerender()`, `apErrText(e)`, `escH`, `TB`, `TBA`, `toast`.
- Produces:
  - `suggestions.js` (pure): `confidenceText(card)`, `statLines(card)`, `canDraft(card, canEdit)`, `draftNotice(checkWording)`, `openCount(cards)`, `actErrorText(e)`, `secondReviewLine(sr)`, `canRecordReview(sr, canApprove)` (the last two are used by Task 11).
  - `engine.js`: `apLnLib()`, `apLnLoad()`, `apLnHTML()` (the queue), `apLnCardHTML(card, where)`, `apLnPolicySection(state)` (cards on the form), `apLnDraft(id)`, `apLnAct(id, verb)`, `apLnBind()`.

**What the person sees.** Wording is fixed here; use it exactly.

- A new segment **Suggestions** after **Conflicts**, with the count of open cards the way Conflicts shows its count.
- A card (existing card style, as `apPcCardHTML` uses):
  - title: the policy id and name, linking to the policy;
  - the change in plain words, e.g. **Raise the limit on Refund amount from 500 USD to 830 USD?**;
  - one evidence sentence from the server;
  - four statistics on one line: `33 decisions · 3 approvers · 40 days · typical decision time 1 min`;
  - **Suggestion confidence: High**, or **Suggestion confidence: Not enough yet**;
  - buttons **Draft this change** (disabled, with the title "There is not enough evidence yet", when `canDraft` is false; hidden for a Viewer), **Not yet**, **Dismiss**.
- The same card inside the policy form, in a section titled **Suggested change**, directly above the Conflicts section. Only open cards.
- **Draft this change** opens the policy form (if it is not already open), fills it from the server's draft, and shows: "Opened as a draft from a suggestion. Nothing is saved until you press Save draft." If the server lists wording it could not update, a second line: "Check the wording of: the situation, the message to the agent." (names from `draftNotice`). It never saves.
- **Not yet**: no confirm; toast "Hidden until 30 more decisions arrive." (the number is not hard-coded: say "Hidden until more decisions arrive." if the card carries no number).
- **Dismiss**: the existing two-step confirm, "Dismiss this suggestion?" / "It is not suggested again unless the evidence becomes much stronger."
- A 409 from draft or act shows the server's `message` in a toast and reloads the queue.

- [ ] **Step 1: Write the pure tests** (`suggestions.test.js`):

```js
import { describe, it, expect } from 'vitest';
import * as S from './suggestions';

const card = (over = {}) => ({
  id: 7, policyKey: 'FIN-0012', policyName: 'Refund or credit over $500', status: 'open', kind: 'threshold',
  change: 'Raise the limit on Refund amount from 500 USD to 830 USD?', evidence: '33 of 33 …',
  stats: { decisions: 33, approved: 33, approvers: 3, days: 40, medianSeconds: 60, wilsonLower: 0.8957 },
  suggestionConfidence: 'High', canDraft: true, ...over,
});

describe('suggestions: the card', () => {
  it('names suggestion confidence, never extraction confidence', () => {
    expect(S.confidenceText(card())).toBe('Suggestion confidence: High');
    expect(S.confidenceText(card({ suggestionConfidence: 'Not enough yet' }))).toBe('Suggestion confidence: Not enough yet');
    expect(S.confidenceText(card())).not.toMatch(/extraction/i);
  });
  it('shows the four statistics in plain words', () => {
    expect(S.statLines(card())).toBe('33 decisions · 3 approvers · 40 days · typical decision time 1 min');
    expect(S.statLines(card({ stats: { decisions: 1, approvers: 1, days: 1, medianSeconds: 45 } })))
      .toBe('1 decision · 1 approver · 1 day · typical decision time 45 s');
    expect(S.statLines(card({ stats: { decisions: 0, approvers: 0, days: 0, medianSeconds: null } })))
      .toBe('0 decisions · 0 approvers · 0 days');
  });
  it('drafting needs evidence and the right to edit', () => {
    expect(S.canDraft(card(), true)).toBe(true);
    expect(S.canDraft(card(), false)).toBe(false);
    expect(S.canDraft(card({ canDraft: false }), true)).toBe(false);
    expect(S.canDraft(card({ status: 'not_yet' }), true)).toBe(false);
  });
  it('says which wording a person must check', () => {
    expect(S.draftNotice([])).toBe('');
    expect(S.draftNotice(['situation'])).toBe('Check the wording of: the situation.');
    expect(S.draftNotice(['situation', 'messageForAgent', 'messageForPerson']))
      .toBe('Check the wording of: the situation, the message to the agent, the message to the person.');
    expect(S.draftNotice(['somethingNew'])).toBe('Check the wording of this policy.');
  });
  it('counts open cards only', () => {
    expect(S.openCount([card(), card({ status: 'not_yet' }), card({ status: 'saved' })])).toBe(1);
    expect(S.openCount(null)).toBe(0);
  });
  it('reads the server message from a refusal', () => {
    expect(S.actErrorText({ response: { status: 409, data: { code: 'policy_changed', message: 'This policy has changed since the suggestion was made.' } } }))
      .toBe('This policy has changed since the suggestion was made.');
    expect(S.actErrorText({ response: { status: 403, data: 'needs the Buyer role or higher' } })).toBe('You do not have permission to do this.');
    expect(S.actErrorText(new Error('Network Error'))).toBe('Could not reach the server. Try again.');
  });
});
```

- [ ] **Step 2: Run to see them fail** (`npx vitest run src/modules/SpendIQ/agentPolicy/suggestions.test.js`), then write `suggestions.js` to make them pass. Pure functions only: no DOM, no fetch, no `TB`.

- [ ] **Step 3: Write the wiring contract test** `engineWiring.stage5.contract.test.js`. Copy the harness from `engineWiring.stage4.contract.test.js` (`loadAp`, `fakeGateway`, the `window.__SPENDIQ_AP__` object with `suggestions` added, and the new `apLn*` names added to the returned list). Tests:

| Test | Assert |
|---|---|
| the three old functions are byte-identical | the same three SHA-256 assertions as stage 4 (`policyEdit`, `policyDelete`, `openFormModal`), unchanged hashes |
| the stage 5 block calls the gateway only | the slice from `/* ============ Agent policies (stage 5)` to `/* The small edit form` has no `axios`, `fetch(`, `AI_API`, `__SPENDIQ_AGENT_RUN__` |
| drafting never saves | `topLevel('apLnDraft')` contains neither `apSave` nor `'post'` |
| the Suggestions segment and its count | with a gateway answering one open card, `apPoliciesTab()` contains `Suggestions` and a `segct` with `1`; with `AP.view='suggestions'` the body contains the change sentence, `Suggestion confidence: High`, and the three buttons |
| a Viewer sees no Draft button | `rbac` Viewer → no `data-ap-ln-act="draft"`; Not yet and Dismiss are also absent |
| a weak card's Draft button is disabled | `canDraft:false` → `disabled` on the draft button, the other two present |
| Draft opens the form and fills it without saving | click → `GET /agent-policies/suggestions/7/draft` called; no call whose method is `post`; the open form state's `hidden.condition` carries the new limit, `checked` is null, `changeNote` starts "Suggested by learning:"; the notice text is rendered |
| Not yet posts and reloads | `POST …/7/act` body `{verb:'not_yet'}`, then `GET /agent-policies/suggestions` |
| Dismiss asks first | no call until the fake confirm says yes; then `{verb:'dismiss'}` |
| a refusal shows the server's words | gateway rejects with 409 `{message:'This policy has changed since the suggestion was made.'}` → that text reaches `toast` |
| the form shows its open cards above Conflicts | `apFormHTML(state)` with `policy.suggestions=[card]` contains `Suggested change` before `Conflicts`; with `[]` it contains neither the title nor a card |
| every `TB(`/`TBA(` argument in the stage 5 block is a string literal | regex over the slice: no `TB(` followed by anything but a quote |
| text from the server is escaped | a card whose `change` is `<img src=x onerror=1>` renders as `&lt;img` |

- [ ] **Step 4: Run to see them fail, then write the engine code.** Model each function on its stage 4 twin: `apLnLib` on `apPcLib` (returns `window.__SPENDIQ_AP__ && window.__SPENDIQ_AP__.suggestions`), `apLnLoad` on `apPcLoad` (state `AP.ln`, `AP.lnErr`, `AP.lnLoading`; path `/agent-policies/suggestions?status=open`), `apLnBind` on `apPcBind` (one delegated click handler on `data-ap-ln-act`), `apLnPolicySection` on `apPcPolicySection`. The draft:

```js
/* Never saves: the server's draft goes into the open form, and the owner saves it the usual way. */
async function apLnDraft(id){
  const S=apLnLib(); if(!S) return;
  let d;
  try{ d=await apApi('get','/agent-policies/suggestions/'+encodeURIComponent(id)+'/draft'); }
  catch(e){ toast(S.actErrorText(e)); apLnLoad(); return; }
  if(!AP.openState||AP.openState.key!==d.policyKey) await apOpen(d.policyKey);
  const state=AP.openState; if(!state) return;
  if(state.policy&&state.policy.latestVersion!==d.baseVersion){ toast('This policy has changed since the suggestion was made.'); apLnLoad(); return; }
  apEdit(state,()=>({...d.form,changeNote:d.changeNote}),false);
  state.form.checked=null;
  state.lnDraft={id:d.suggestionId,checkWording:d.checkWording||[]};
  apRender(state);
  toast('Opened as a draft from a suggestion. Nothing is saved until you press Save draft.');
}
```
In `apPoliciesTab`: add `apLnLib()?[['suggestions','Suggestions']]:[]` to `views` after the conflicts entry, the eager load line `if(apLnLib()&&AP.ln==null&&!AP.lnErr) apLnLoad();`, the `segct` for `v[0]==='suggestions'`, and the `body` branch `AP.view==='suggestions'&&apLnLib() ? apLnHTML() : …`. Write the segment label as its own literal (`TB('Suggestions')`), not through `TB(v[1])`. In `apFormHTML`: one call, `+apLnPolicySection(state)`, immediately before `apPcPolicySection(state)`.

- [ ] **Step 5: Run everything**

```bash
npx vitest run src/modules/SpendIQ/agentPolicy
npx vitest run src/lib/i18n/language.contract.test.js
npx vitest run
```
Expected: the `agentPolicy` folder all green including the four older `engineWiring.*` files. For the other two commands, record the baseline on `origin/spendiq-ui` first (four `atb`/`composedPages` failures and the i18n "unreviewed TB argument" failure are known); no new failure, and the i18n test's list of unreviewed call sites must not grow.

- [ ] **Step 6: Commit**

```bash
git add src/modules/SpendIQ/agentPolicy/suggestions.js src/modules/SpendIQ/agentPolicy/suggestions.test.js src/modules/SpendIQ/agentPolicy/engineWiring.stage5.contract.test.js src/modules/SpendIQ/index.jsx
git add -p src/modules/SpendIQ/engine.js     # your hunks only
git commit -m "feat(agent-policy): the suggestion card and the Suggestions queue"
```

---

### Task 11: Second review, recent decisions and "Mark as reversed" on the policy form

**Files (UI worktree):**
- Modify: `src/modules/SpendIQ/agentPolicy/suggestions.js` and its test (`secondReviewLine`, `canRecordReview`, `decisionLine`, `canReverse`)
- Modify: `src/modules/SpendIQ/agentPolicy/model.js` and its test (`problemMessage` knows `second_review_required`)
- Modify: `src/modules/SpendIQ/engine.js` (stage 5 section: `apLnReviewSection(state)`, `apLnRecordReview(state)`, `apLnDecisionsHTML(state)`, `apLnReverse(state, firingId)`; in `apLoadHealth`, keep the rows it already fetches: `state.firings=(r&&r.firings)||[]`; two calls in `apFormHTML`)
- Modify: `src/modules/SpendIQ/agentPolicy/engineWiring.stage5.contract.test.js`

**Interfaces:**
- Consumes: `policy.secondReview` (`{required, version, reviewedBy, reviewedAt}`) from `GET /agent-policies/{key}`; firing rows from `GET /agent-policies/{key}/firings` (already fetched for the health line) with the new `reversed` field; gateway `POST …/second-review` and `POST …/firings/{id}/reversal`; `apCanApprove()`, `apUserSub()`, `apFmtTime`.
- Produces: the four pure functions and four engine functions named above.

**What the person sees.**

- **Second review**, in the policy form's Governance area, only when `secondReview.required`:
  - not yet reviewed: "This change loosens the policy as learning suggested. A second reviewer must review it before it goes Active." and, for an Approver, a button **Record my review…** (two-step confirm: "Record your review of version N?" / "You are confirming you have read the change and the evidence. You must not be the policy's owner or the person who saved it.").
  - reviewed: "Reviewed by {name} on {date}."
  - a refusal (409) shows the server's message under the line, e.g. "You saved this change, so someone else must review it."
  - the activation problem `second_review_required` appears in the form's normal problem list, and focus goes to this section (`data-ap-field="secondReview"`).
- **Recent decisions**, a collapsed block under the health line on an Active policy ("Recent decisions (N)"), listing the fetched approve firings that a person approved or rejected, newest first, at most 20: `2026-10-02 · Approved by {decidedBy} · Refund amount 620` and, when reversed, ` · Reversed on 2026-10-09: {reason}`. Sensitive values arrive already masked and are shown as "Hidden".
  - For an Approver, each approved and not-yet-reversed row has **Mark as reversed…**: asks for a reason (required, the existing reason-input pattern from `apApprovalReasonInput`), then the two-step confirm "Mark this approval as reversed?" / "This cannot be undone. Learning will not suggest loosening the policy on the strength of it."
  - The health line's counts are unchanged.

- [ ] **Step 1: Pure tests first** (append to `suggestions.test.js`):

```js
describe('suggestions: second review and reversals', () => {
  it('says what is needed, then who did it', () => {
    expect(S.secondReviewLine({ required: false })).toBe('');
    expect(S.secondReviewLine({ required: true, version: 4, reviewedBy: null }))
      .toBe('This change loosens the policy as learning suggested. A second reviewer must review it before it goes Active.');
    expect(S.secondReviewLine({ required: true, version: 4, reviewedBy: 'sam@x.test', reviewedAt: '2026-10-10T09:00:00+00:00' }))
      .toBe('Reviewed by sam@x.test on 2026-10-10.');
  });
  it('only an approver may record a review, and only while one is missing', () => {
    expect(S.canRecordReview({ required: true, reviewedBy: null }, true)).toBe(true);
    expect(S.canRecordReview({ required: true, reviewedBy: null }, false)).toBe(false);
    expect(S.canRecordReview({ required: true, reviewedBy: 'x' }, true)).toBe(false);
    expect(S.canRecordReview({ required: false }, true)).toBe(false);
    expect(S.canRecordReview(null, true)).toBe(false);
  });
  it('a decision line, with its reversal', () => {
    const f = { id: 9, outcome: 'approve', result: 'approved', decidedBy: 'sam@x.test', decidedAt: '2026-10-02T10:00:00+00:00',
                matchedValues: { 'args.amount': 620, 'tool.name': 'refund.issue' }, reversed: null };
    expect(S.decisionLine(f)).toBe('2026-10-02 · Approved by sam@x.test · args.amount 620');
    expect(S.decisionLine({ ...f, result: 'rejected' })).toBe('2026-10-02 · Rejected by sam@x.test · args.amount 620');
    expect(S.decisionLine({ ...f, matchedValues: { 'args.amount': '•••' } })).toBe('2026-10-02 · Approved by sam@x.test · args.amount Hidden');
    expect(S.decisionLine({ ...f, reversed: { by: 'a', at: '2026-10-09T00:00:00+00:00', reason: 'clawed back' } }))
      .toBe('2026-10-02 · Approved by sam@x.test · args.amount 620 · Reversed on 2026-10-09: clawed back');
  });
  it('only an approved, unreversed decision by a person can be marked reversed, by an approver', () => {
    const f = { outcome: 'approve', result: 'approved', decidedBy: 'sam@x.test', reversed: null };
    expect(S.canReverse(f, true)).toBe(true);
    expect(S.canReverse(f, false)).toBe(false);
    expect(S.canReverse({ ...f, result: 'rejected' }, true)).toBe(false);
    expect(S.canReverse({ ...f, reversed: { by: 'a' } }, true)).toBe(false);
    expect(S.canReverse({ ...f, decidedBy: 'system:precedent' }, true)).toBe(false);
    expect(S.canReverse({ ...f, outcome: 'notify' }, true)).toBe(false);
  });
});
```
(`decisionLine` shows every matched value except `tool.name`, in key order. In `model.test.js` add: `problemMessage({field:'secondReview', code:'second_review_required', message:'A second reviewer must review this change before it goes Active. It loosens the policy as learning suggested.'})` returns that message.)

- [ ] **Step 2: Contract tests** (append to the stage 5 wiring test):

| Test | Assert |
|---|---|
| the review line and button | `policy.secondReview={required:true,version:4,reviewedBy:null}` + Approver → the sentence, a `data-ap-field="secondReview"` container and the button; Buyer → sentence, no button; `required:false` → nothing |
| recording a review asks first, posts the latest version, reloads | fake confirm yes → `POST /agent-policies/FIN-0012/second-review` body `{version:4}`; then the policy is re-read |
| a refusal shows the server's words | 409 `{message:'You saved this change, so someone else must review it.'}` → rendered in the section |
| recent decisions list only people's approve decisions | given firings of mixed outcome/result/decider → only the approved/rejected approve rows by a person, newest first, capped at 20 |
| Mark as reversed needs a reason and a confirm | no call with a blank reason; with a reason and yes → `POST /agent-policies/firings/9/reversal` body `{reason:'clawed back'}`; then firings re-read |
| a Viewer or Buyer sees no reversal button | as named |
| the health line is unchanged | `apHealthHTML` output for a fixed firing set equals what it was before this task (pin the string) |

- [ ] **Step 3: Implement, then run** `npx vitest run src/modules/SpendIQ/agentPolicy` (all green) and the two wider commands from Task 10 Step 5 against the baseline.

- [ ] **Step 4: Commit**

```bash
git add src/modules/SpendIQ/agentPolicy/suggestions.js src/modules/SpendIQ/agentPolicy/suggestions.test.js src/modules/SpendIQ/agentPolicy/model.js src/modules/SpendIQ/agentPolicy/model.test.js src/modules/SpendIQ/agentPolicy/engineWiring.stage5.contract.test.js
git add -p src/modules/SpendIQ/engine.js
git commit -m "feat(agent-policy): the second review, recent decisions and marking an approval as reversed"
```

---

### Task 12: An administrator can switch "Never suggest changes" and "Second reviewer" per business area

Today these two switches exist only in the database: no screen calls `PUT /agent-policies/taxonomy/{area}`, which the backend and gateway have had since stage 1. Learning's behaviour depends on both, so an administrator must be able to see and change them.

**Files (UI worktree):**
- Modify: `src/modules/SpendIQ/agentPolicy/suggestions.js` and its test (`areaBody(area, change)`, `areaLine(area)`)
- Modify: `src/modules/SpendIQ/engine.js` (stage 5 section: `apLnAreasHTML()`, `apLnAreaToggle(areaName, field)`; a segment **Business areas** shown only when `apIsAdmin()`, after **Deciders**)
- Modify: `src/modules/SpendIQ/agentPolicy/engineWiring.stage5.contract.test.js`

**Interfaces:**
- Consumes: `GET /agent-policies/taxonomy` → `{areas: [{areaName, idPrefix, subAreas, neverSuggest, secondReviewer, unassigned}]}`; `PUT /agent-policies/taxonomy/{area}` body `{subAreas, neverSuggest, secondReviewer}` (Admin). Read `src/api/routers/agent_policies.py:111-148` for the exact body before writing `areaBody`.
- Produces: `areaBody(area, change) -> {subAreas, neverSuggest, secondReviewer}` (the area's current values with one switch changed; sub-areas are sent back untouched); `areaLine(area) -> string`.

**What the administrator sees.** A table, one row per business area (the Deciders table style): name, sub-areas (read-only text), and two checkboxes: **Never suggest changes** and **Second reviewer for suggested changes**. Changing a checkbox uses the two-step confirm ("Change this setting for Finance?" / for turning a protection off: "Learning will be able to suggest loosening policies from this area." or "A suggested loosening in this area will no longer need a second person."), then PUTs and re-reads the taxonomy. A failure restores the checkbox and shows the server's message.

- [ ] **Step 1: Tests.** Pure: `areaBody({areaName:'Finance', subAreas:['General','Payments'], neverSuggest:false, secondReviewer:true}, {secondReviewer:false})` → `{subAreas:['General','Payments'], neverSuggest:false, secondReviewer:false}`; an unknown change key throws. Contract: the segment is absent for a non-admin and `AP.view` falls back to `list`; toggling asks first; the PUT path is URL-encoded (`Legal%20and%20compliance`) and the body is exactly `areaBody`'s; a rejected PUT leaves the shown value unchanged.

- [ ] **Step 2: Implement, run** `npx vitest run src/modules/SpendIQ/agentPolicy`, **commit:**

```bash
git add src/modules/SpendIQ/agentPolicy/suggestions.js src/modules/SpendIQ/agentPolicy/suggestions.test.js src/modules/SpendIQ/agentPolicy/engineWiring.stage5.contract.test.js
git add -p src/modules/SpendIQ/engine.js
git commit -m "feat(agent-policy): an administrator sets Never suggest changes and the second reviewer per business area"
```

---

### Task 13: Live demonstration and verification on a stack of its own

**Files:**
- Create: `specs/2026-10-10-agent-policy-governance-stage5-verification.md` (BP worktree)
- Scratch only (never committed): the demo driver script and screenshots, under the session scratchpad.

Run a private stack from the three worktrees, as stage 4 did (`specs/2026-10-09-agent-policy-governance-stage4-verification.md`, "How the stack was run": backend on :8010 with `AGENT_POLICY_LEARNING_SCAN=off`, `AGENT_POLICY_CONFLICT_SCAN=off`, `AGENT_POLICY_APPROVAL_SWEEP=off`; gateway on :3011 with `AUTH_BYPASS_GROUPS` set for the demo identities; UI on :3010; headless Chrome). Never the shared ports, never a restart of a shared service. Database: `bp_testdb`.

Because the guardrails need 30 decisions over 30 days by 3 people, the demo's evidence is **backdated firing rows inserted by the driver script** for demo policies named "DEMO stage 5 …". Say so plainly in the note. Real suggestions on real traffic cannot appear until 30 days of real decisions exist.

- [ ] **Step 1: Run the steps, recording each request, response and database row.**

| # | Step | Passes when |
|---|---|---|
| 1 | Create and activate two demo approve policies with the same shape (tool `demo_s5_<hex>`, `args.amount gt 500`): **F** in Finance, **O** in Operations; one block policy **B**; one approve policy **L** in Legal and compliance | all live; `learning.eligible` is true for F and O, false for B and L in the technical view |
| 2 | Insert 33 backdated approved firings for each of F, O, B, L (510…830, three deciders, 40 days), and 29 for a fifth policy **T** | rows present |
| 3 | Run `learning.scan(conn, now)` for real (no `only`) | cards for F and O only; none for B (block), L (never suggest), T (29 decisions) |
| 4 | Screens: Suggestions segment shows 2 with the count; the card shows the change, the evidence sentence, four statistics, "Suggestion confidence: High"; the same card on F's form above Conflicts | screenshots |
| 5 | **Draft this change** on O as a Buyer | form opens with limit 830, situation updated, examples 829/830/831 computed "Nothing happens / Nothing happens / A person decides", confirmation cleared, change note filled; `latestVersion` unchanged in the database; no POST in the gateway log |
| 6 | Save the draft, tick the confirmation, activate O as an Approver | live at the new limit; suggestion `saved`; card gone from queue and form; a re-scan with 40 more backdated approvals raises nothing for the old change |
| 7 | F: draft, save as person P1; activate | refused with the second-review problem, focus on the section |
| 8 | F: P1 tries **Record my review** | refused, "You saved this change…" |
| 9 | F: the mapped owner tries | refused, "The policy's owner cannot be its second reviewer." |
| 10 | F: P2 (Approver) records the review; P1 activates | live; review row; "Reviewed by P2 on …" was shown before activation |
| 11 | A sixth policy **H** in Finance with a card: dismiss it, then type the limit 800 by hand, save, activate | refused: second review still required |
| 12 | A seventh policy **N** with a card: **Not yet** | card hidden; 29 more firings + scan → still hidden; 1 more + scan → back |
| 13 | An eighth policy **R** with a card: open its form, **Recent decisions**, **Mark as reversed…** on one row as an Approver, scan | row shows "Reversed on …"; card now "Suggestion confidence: Not enough yet", Draft disabled; a second reversal of the same row is refused |
| 14 | An unlinked Viewer | reads the queue; sees no Draft / Not yet / Dismiss / Record my review / Mark as reversed; direct gateway POSTs answer 403 |
| 15 | An Admin opens **Business areas**, turns "Never suggest changes" on for Operations, scan | O-area cards are superseded and disappear; turn it back off |
| 16 | The scheduler job | with `AGENT_POLICY_LEARNING_SCAN` unset, start a backend on :8010 and confirm the log line that the job is registered on lane `agent-policy-learning`; do not wait an hour |
| 17 | Console | no console errors in any screen visited (brief §8 test 22) |
| 18 | Nothing became Active on its own | for every demo policy, the only versions are the ones a step above saved; list them |

- [ ] **Step 2: Clean up.** Retire every demo policy through the API. Delete the demo suggestion rows, second-review rows and decider-map rows you created. Firing and reversal rows stay (append-only): list their policy keys in the note. Stop only your own processes; confirm ports 8010, 3011, 3010 are closed and `procwise.service` is still `active`.

- [ ] **Step 3: Run the three suites one last time against their baselines** and put the numbers in the note:

```bash
bptest tests/agent_policy tests/engines/test_decide_live_conflict.py -q
LIVE=1 bptest tests/agent_policy tests/engines/test_decide_live_conflict.py -q
(cd "$GW" && npm test)
(cd "$UI" && npx vitest run)
```

- [ ] **Step 4: Write the verification note** in the stage 4 note's format: "In plain English" first (what works, what was not shown and why), then each step with its evidence, defects found (recorded, not patched), observations, the red/green captures from Tasks 2, 3, 5 and 8, the `engine.js` diff summary (ruling A: list every function added and the lines touched in existing functions), and brief §8 test 19 mapped clause by clause to the test that proves it.

- [ ] **Step 5: Commit the note**

```bash
git add specs/2026-10-10-agent-policy-governance-stage5-verification.md
git commit -m "docs(agent-policy): stage 5 live verification and diff summary"
```

---

## Self-review notes (for the executor)

**Brief §6 and test 19, clause by clause**

| Requirement | Where |
|---|---|
| §6.1 log: policy and version, time, matched values, level, who, decision, time taken | Already written by stage 3 (`bp_policy_firing`); read in Task 5 |
| §6.1 later reversal, linked to the firing | Task 7 (table from Task 1) |
| §6.1 timeouts excluded | Task 5 SQL (`result IN ('approved','rejected')`), tested |
| §6.2 group by the driving values; n, k, Wilson, approvers, days, edited, reversed, median | Tasks 2, 3 |
| §6.3 six guardrails, each a company setting | Task 2; settings read in Task 5 |
| §6.3 never for block / "Never suggest changes" | Task 5 `targets`; demo steps 1–3; switch on screen in Task 12 |
| §6.4 card in the form and a queue | Tasks 6, 10 |
| §6.4 new value inside the evidence | Task 3 (tests + broken-guard proof) |
| §6.4 suggestion confidence High / Not enough yet, draft button off | Tasks 5 (`weakened`), 6, 10 |
| §6.4 Draft: unsaved, condition + situation updated, examples recomputed, confirmation cleared, change note filled | Tasks 4, 6, 10 |
| §6.4 Not yet: hidden until 30 more | Tasks 5, 6 |
| §6.4 Dismiss: not again unless materially stronger | Tasks 5, 6 |
| §6.4 once saved, never suggested again | Task 6 `note_save`; Task 5 never touches `saved` |
| §6.4 only loosening, only approve | Task 3 (direction by operator), Task 5 `targets` |
| Design §2 second reviewer per business area, Finance on, not the owner | Tasks 8, 11, 12 |
| §2 nothing becomes Active automatically | Global Constraints; Task 6 test `test_nothing_in_this_stage_activates_a_policy`; demo step 18 |
| §3.6 the two confidences are never confused | Task 6 (`suggestionConfidence`), Task 10 test |

**Known limits, stated rather than hidden**
- A sensitive driving value is masked in the log, so no suggestion can ever be made on it.
- `not_in`, `eq`, `ne`, `exists` conditions and anything under `any`/`not` are never analysed.
- An approval whose action was later blocked at replay still counts as an approval (stage 3 ruling 10).
- A reversal cannot be undone.
- Evidence is capped at the 5,000 oldest decisions per policy; a warning is logged when the cap is hit.
- Stage 3's `bp_policy_firing.reversal_of` column stays unused.

**Names used across tasks (check before you finish a task):** `Decided`, `stats`, `failed_guardrails`, `eligible_leaves`, `propose`, `group_for`, `targets` (two of them: `learning_patterns.targets(proposal, condition)` and `learning.targets(cur)`; import the modules, not the names), `applied`, `learning_draft.plain/evidence/build`, `learning.scan/decisions/list_cards/for_policy/act/draft/note_save/record_reversal/Refused`, `second_review.loosening/needed/activation_problems/record/status`. Card field `suggestionConfidence`; problem code `second_review_required`; statuses `open, not_yet, dismissed, saved, superseded`.
