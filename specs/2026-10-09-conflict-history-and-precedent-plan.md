# Conflict History and Precedent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every decision on a clash between agent policies is kept in full, readable and exportable (masked per reader), and the decision engine decides a live clash itself when people have decided the exact same clash the same way N times, where N is a customer-editable governed policy (default 5).

**Architecture:**
- **N is a governed limit.** One new `proc.bp_policy` row (`agent_policy_conflicts.precedent_count`), read through `governed_limits.limit(..., fresh=True)`. A fresh read builds its own PolicyEngine, so an edit made in the policy admin applies on the next clash without a restart. The standing-rule proposal (`maybe_propose`) and precedent both read this one value. The old company setting `live_conflict_repeat` is removed.
- **The decision engine decides.** `engines.decision_engine.decide_live_conflict(cur, lc, ctx=, now=)` looks up the last N settled live cases of the same pair, at the same versions, decided by people. It returns the module's existing `Decision`: resolved (approve/reject, with the cited cases as `Evidence`) or escalated with a reason. The gate calls it only for a `human` clash on a fresh call. A resolved approve runs the tool now. A resolved reject refuses the call with `refused_on_precedent`. An escalation goes to people exactly as in stage 4, carrying the clash's history.
- **One history reader.** `services/agent_policy/conflict_history.py` reads (`raw`) and masks per viewer (`shown`/`read`). The policy page, the Conflicts detail, the paused clash's approval card and the CSV exports all use it. Every closing path writes `facts.decidedBy.kind`, so the reader never guesses.

**Tech Stack:** Python 3 / FastAPI / psycopg2 (BP_Backend), NestJS + serverless yml (gateway), vanilla `engine.js` + vitest pure modules (UI), PostgreSQL (`bp_testdb`, `bp_sqldb`).

**Spec:** `specs/2026-10-09-conflict-history-and-precedent-design.md`. Its user rulings R1–R5 bind. Stage 4 context: `specs/2026-10-09-agent-policy-governance-plan-4-conflicts.md` and `specs/2026-10-09-agent-policy-governance-stage4-verification.md`.

**Repositories (all on branch `agent-policy-stage4`):**
- **BP:** `/home/muthu/PycharmProjects/BP_Backend/.claude/worktrees/agent-policy-stage4`. Every BP path below is relative to this folder.
- **Gateway:** `/home/muthu/PycharmProjects/beyond-procwaise-Api-worktrees/agent-policy-stage4/beyond_procwaise_api`.
- **UI:** `/home/muthu/PycharmProjects/beyond_procwise_ui-worktrees/agent-policy-stage4`.

**Commands used throughout.** Paste this once per shell:

```bash
BP=/home/muthu/PycharmProjects/BP_Backend/.claude/worktrees/agent-policy-stage4
GW=/home/muthu/PycharmProjects/beyond-procwaise-Api-worktrees/agent-policy-stage4/beyond_procwaise_api
UI=/home/muthu/PycharmProjects/beyond_procwise_ui-worktrees/agent-policy-stage4
# BP pytest: .env (bp_testdb), no GPU, src on the path. LIVE=1 turns on the live-DB tests.
bptest() { (cd "$BP" && set -a && . ./.env && set +a && export PROCWISE_TEST_LIVE_DB="${LIVE:-}" \
  && PYTHONPATH=.:src CUDA_VISIBLE_DEVICES="" /home/muthu/PycharmProjects/BP_Backend/venv/bin/python -m pytest "$@"); }
# psql against one database: bppsql <dbname> -f <file>  or  bppsql <dbname> -Atc "<sql>"
bppsql() { local db="$1"; shift; (cd "$BP" && set -a && . ./.env && set +a \
  && PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "${DB_PORT:-5432}" -U "$DB_USER" -d "$db" -v ON_ERROR_STOP=1 "$@"); }
```

(`.env` prints `supplierconnect: command not found` from line 27. It is harmless. Never print secret values.)

## Global Constraints

From `.superpowers/sdd/2026-10-09-agent-policy-governance-plan-4-conflicts/common-context.md`, its `stage3/global-constraints.md`, the stage 4 user rulings Q1–Q5 in that folder's `progress.md`, and the spec. Every task's requirements include this section.

**Where you work**
- Run every BP command from the BP worktree. Never `cd` into `/home/muthu/PycharmProjects/BP_Backend` itself: it is a shared checkout used by other sessions.
- Python interpreter: `/home/muthu/PycharmProjects/BP_Backend/venv/bin/python` (the worktree has no venv of its own). Load `.env` first. `CUDA_VISIBLE_DEVICES=""`. `PYTHONPATH=.:src` (modules import as `services.…`, `api.…`, `repositories.…`, `engines.…`; `governed_limits` is always imported as `src.services.governed_limits`, the name every caller and `tests/conftest.py` use).
- DB tests need `PROCWISE_TEST_LIVE_DB=1`; without it pytest uses a fake DB. Live tests run on `bp_testdb` only (`DB_NAME=bp_testdb`, which `.env` sets).
- `get_conn()` is AUTOCOMMIT: a rollback is a no-op unless `conn.autocommit = False` is set first; every write runs in `approvals._tx`.
- `bp_testdb` and `bp_sqldb` are SHARED with other sessions (same RDS cluster; `bp_sqldb` host 10.100.10.180, same credentials). Only additive, idempotent DDL/DML from this plan. Never DROP/TRUNCATE/DELETE anything that is not yours.
- `bp_testdb` holds about 3,500 stale never-retired test drafts. `tests/agent_policy/conftest.py` makes save-time conflict detection a no-op unless a test is marked `conflict_detection`. Do not mark any test in this plan; drive detection explicitly with `among=`.
- Test data is scoped: tool names `tst_<hex>`, policy keys `TST-<hex><letter>`, names "TST …", workflow ids from `new_world`. Every row a test makes is removed in FK order (notification, conflict_rule, conflict, decision); firing rows are append-only BY DESIGN and stay.

**Commits**
- `git add` explicit paths only; never `git add -A` / `git add .`; never `git stash`. Stage only your own hunks.
- `.superpowers/` is NOT gitignored: never commit anything under it. `docs/` is gitignored: write specs under `specs/`.
- End every commit message with a blank line and then a `Co-Authored-By:` line naming the model that wrote the commit (ledger ruling), e.g. `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Do not push. Do not merge. Never dispatch subagents yourself.

**Rules that bind (stages 1–4)**
- `proc.bp_policy`: stage 4's rule "never written, altered or migrated" is amended by user ruling R4 for exactly one additive row: the `agent_policy_conflicts` governed limit, following the 2026-09-10 pattern (`policy_type 'limit'`, no `applies_to`, idempotent on `policy_identifier` + `policy_status = 1`). Nothing else in `proc.bp_policy` is touched. Agent policies still live only in their own tables.
- New tables use the `bp_` prefix, indexes `ix_bp_*` (this plan adds no table).
- Migrations live in `deploy/sql/` with a `_rollback.sql`, are additive and idempotent, and are applied to BOTH `bp_testdb` and `bp_sqldb`. A rollback is never run against shared data without a user ruling.
- **One evaluator:** `conditions.to_engine` + `policy_condition.evaluate`.
- **Fail closed:** any failure in the gate refuses the call with `policy_check_unavailable`, in one transaction, leaving nothing behind. A timeout rejects and never approves.
- **Precedence (Q1, design):** any matching `block` blocks immediately; a live clash involving a block is recorded, never put to a person, and precedent is never consulted. A standing rule never lets a policy beat a block, and still applies before precedent. Notifies always send. Notify policies never take part in conflicts (Q2).
- **Repeat-N (Q4):** the same set of policy ids; the last N person-decided live cases all approve or all reject; a different outcome resets; timeouts, block records, standing-rule autos and (new) precedent decisions never count.
- **Precedent (R1–R3):** only the identical pair at identical versions, decided by people (`by_person = true`), N of N agreeing, resolves; any doubt escalates to people with the history attached. Precedent may approve (the tool runs with no human approving; `system:precedent` is the approver of record) or reject.
- **N (R2, R4):** one number, the governed `agent_policy_conflicts.precedent_count`. At N the engine starts resolving AND the standing-rule proposal is raised. A missing or unreadable value means no precedent and no proposal, with a warning; never a default in code. 0 or null switches both off.
- **Masking:** a sensitive input is masked as `•••` (`enforcement.MASK`) in every response and log, except for an eligible reader. History reasons are masked, never dropped (this reverses the stage 4 final-review ruling that dropped live reasons). Masking reuses stage 3/4 helpers: `approval_views.mask_witness`, `approval_views.mask_text` (from `replay`), `approval_views.sensitive_for`.
- **CSV export:** formula-injection neutralisation is done server side, with the same rule as the UI's `inventory.csvCell`.
- **Every write endpoint** audits via `agent_actions.record_action_or_fail` before returning (this plan adds only read endpoints; reads use `_require(..., "agent_policy.read", ...)`, which is not audited, as for every read).
- **OutputSafety (Q5):** no new exemptions. The CSV routes answer `text/csv`, which the middleware passes through untouched (it scrubs JSON and SSE only); errors are still scrubbed.
- **UI:** `engine.js` changes are new `apPc*` functions plus minimal wiring only; `policyEdit`, `policyDelete` and `openFormModal` stay byte-identical; the stage 4 block calls the gateway only (`apApi`), never `fetch(` or `axios`.
- **The stage 4 byte-for-byte guard** `tests/agent_policy/test_conflict_live_gate.py::test_no_live_conflict_is_stage3_byte_for_byte` must still pass after every task. With no conflict classified, the gate's statements are exactly stage 3's.
- **Never call the real model in tests.** The live demonstration drives `run_tools` with a scripted chat stand-in.
- **Prove the guard fails:** where a step says so, break it on purpose, capture the red output, restore it, and capture green. Put both in your report.
- **Deployment is not part of this plan.** No task restarts `procwise.service`, `bp-gateway` or `bp-ui`, or touches the shared main checkouts, ports :8000/:3000/:3001 or Ollama. The migration goes to both databases before any deployment (spec §7); the controller deploys after review.

## Review Focus

1. **A customer changes the precedent count in the policy admin while the server runs.** The very next clash uses the new number; no restart is needed. Tests: `test_a_fresh_read_sees_an_edit_without_a_restart` (Task 1) and `test_an_edit_takes_effect_without_a_restart` (Task 7).
2. **A third approval policy that is not part of the clash also matches the action.** Precedent never approves on its behalf; the action goes to people. Tests: `test_an_approval_outside_the_clash_escalates` (Task 5) and `test_precedent_never_approves_for_a_policy_outside_the_clash` (Task 7).
3. **A "not allowed" policy whose condition cannot be read matches alongside a clash.** The call is blocked, and precedent is not consulted. Test: `test_an_unreadable_block_beats_precedent` (Task 7).
4. **A decider's reason quotes a sensitive amount, and a stranger reads it** (policy page, Conflicts detail, approval card, CSV). The amount reads `•••`; the reason is still there. Tests: `test_owner_decider_and_admin_read_the_full_reason_a_stranger_reads_it_masked` (Task 6), `test_the_paused_clash_card_carries_its_history_masked_per_approver` (Task 7), `test_the_policy_export_masks_per_caller` (Task 8).
5. **An exported reason starts with a formula, or contains a date like 01/04/2026.** The formula is neutralised with a leading apostrophe; the date is exported unchanged. The CSV is not JSON, so OutputSafety does not rewrite it. Tests: `test_a_formula_in_a_reason_is_neutralised` and `test_a_date_in_a_reason_survives` (Task 8).

---

## Decisions taken in planning (reviewers may challenge)

- **`fresh=True` builds a new PolicyEngine** (`governed_limits._fresh_engine` → `rbac._build_engine()`). Bypassing only `governed_limits._CACHE` would not be enough: `rbac.policy_engine()` caches the engine itself for 60 s. One `proc.bp_policy` read per `human` clash and per person settlement is cheap.
- **`settings.precedent_count()` is the one reader of N.** It lives in `services/agent_policy/settings.py`, imports `src.services.governed_limits` lazily and raises `LimitUnavailable`. `conflict_live.threshold()` turns that into `None` plus a warning. `decision_engine._precedent_count()` is the engine's seam over it.
- **Precedent applies only when every matched approve policy is part of the clash.** `lc.required` (all matched approve policies) must be a subset of `lc.involved`; otherwise the engine escalates. R1 says any doubt goes to people.
- **Precedent is consulted only when `verdict.result == "paused_for_approval"`.** An unreadable condition is a block (`verdict.blocks`) even when `classify` says `human`, and a block always wins.
- **The lookup runs in a savepoint** (`SAVEPOINT live_conflict_precedent`). A failed statement would otherwise abort the gate's whole transaction; with the savepoint, a failed lookup escalates (spec §4), and the escalated case is still recorded.
- **Precedent notifications are firing notifications** with link `agent-policy:<key>`, one per involved policy (owner plus every decider of that policy, de-duplicated). The `conflict:<id>` link opens only design-time cases, which return 404 for a live case. The `ck_bp_policy_notification_target` CHECK needs a firing id or a `conflict:` link.
- **`to_agent` on a precedent approve** (`{"result": "allowed", "conflictCaseId", "precedent": true}`) is set on `GateResult` and tested there. `tool_runtime` gives the model the tool's own result when a call is allowed and does not pass `to_agent` on (unchanged; outside this feature).
- **The reader's decision shape is the spec's:** `{option, scope, decidedBy {kind, name}, decidedAt, reason}`. The policy page's `conflicts[]` entries change shape. The stage 4 tests asserting the old shape are updated in Task 6, and the UI's `historyLine` reads both shapes.
- **Cases closed before this change** carry no `facts.decidedBy`. The reader reports `kind: null` for them; it never derives a kind from an actor string.
- **`facts.versionsAtDecision` on a live case** is the versions that clashed (the conflict row's `policy_versions`). A design-time case keeps its stage 4 meaning (the latest versions when the owner decided).
- **The escalated clash stores a raw history snapshot** (`facts.history`, newest 20 cases). It includes the private `_owners`/`_deciders`/`_args` used for masking. The snapshot is never served raw: `approval_views.conflict_block` passes it through `conflict_history.shown`.
- **The CSV has one row per case** (a case has at most one decision); an open case reads "Waiting for a decision". Both CSVs are capped at the newest 1,000 cases (`HISTORY_LIMIT`).
- **There is no settings screen to change.** No UI or gateway code exposes `live_conflict_repeat` (checked by grep). Only `settings.DEFAULTS` loses it. The stale key stays in `proc.bp_admin_config` (shared data, additive-only rule); nothing reads it any more.

---

## File map

**BP_Backend**
| File | Responsibility |
|---|---|
| `src/services/governed_limits.py` (modify) | `limit(..., fresh=True)`, `_new_engine`, `_fresh_engine` seam |
| `deploy/sql/2026-10-13_agent_policy_conflict_precedent.sql` (+ `_rollback.sql`) | the `agent_policy_conflicts` governed-limit row |
| `src/services/agent_policy/settings.py` (modify) | drop `live_conflict_repeat`; `precedent_count()` |
| `src/services/agent_policy/conflict_cases.py` (modify) | `DECIDED_BY_KINDS`, `decided_by()`; `decidedBy` in `decide_policy`/`_moot`; `history_for` through the reader |
| `src/services/agent_policy/conflict_live.py` (modify) | `PRECEDENT`; `insert_live` precedent records and extras; `decidedBy` and `versionsAtDecision` on settle; `threshold()` from the governed value |
| `src/engines/decision_engine.py` (modify) | `decide_live_conflict`, `PRECEDENT_SQL`, `_precedent_count` |
| `src/services/agent_policy/conflict_history.py` (new) | `Viewer`, `viewer`, `raw`, `shown`, `may_see`, `read`, `csv_cell`, `to_csv` |
| `src/services/agent_policy/gate.py` (modify) | `_consult_precedent`, `_record_precedent`, `_notify_precedent`, `_precedent_answer`, `_lock_key`; history snapshot on an escalated clash |
| `src/services/agent_policy/approval_views.py` (modify `conflict_block`) | masked `history` and `precedentNote` on a paused clash's card |
| `src/services/agent_policy/conflict_views.py` (modify) | `pairKey` on a case view; `conflictHistory` on the detail |
| `src/repositories/agent_policy_repo.py` (modify `get_policy`) | `viewer=` passed to `history_for` |
| `src/api/routers/agent_policies.py` (modify) | viewer for `get_one`/`get_conflict`; two CSV routes |
| `tests/conftest.py`, `tests/agent_policy/fixtures.py` (modify) | seed row; `_fresh_engine` patch; `precedent_n()` |
| tests (new) | `tests/migrations/test_2026_10_13_agent_policy_conflict_precedent.py`, `tests/agent_policy/test_precedent_count.py`, `tests/agent_policy/test_conflict_decided_by.py`, `tests/agent_policy/test_conflict_decided_by_live.py`, `tests/engines/test_decide_live_conflict.py`, `tests/agent_policy/test_conflict_history.py`, `tests/agent_policy/test_conflict_history_live.py`, `tests/agent_policy/test_conflict_precedent_live.py`, `tests/agent_policy/test_conflict_export_live.py` |

**Gateway:** `src/modules/agent-policy/agent-policy.service.ts`, `agent-policy.controller.ts`, `agent-policy.yml`, `agent-policy.controller.spec.ts`.

**UI:** `src/modules/SpendIQ/agentPolicy/conflicts.js` (+ `conflicts.test.js`), `src/modules/SpendIQ/engine.js` (`apPc*`), `src/modules/SpendIQ/agentPolicy/engineWiring.stage4.contract.test.js`.

---

### Task 1: A governed limit can be read fresh

**Files:**
- Modify: `src/services/governed_limits.py` (`_rules`, `limit`; add `_new_engine`, `_fresh_engine`)
- Modify: `tests/conftest.py` (`_governed_limits_available`: also patch `_fresh_engine`)
- Test: `tests/governance/test_governed_limits.py`

**Interfaces:**
- Produces:
  - `governed_limits.limit(policy: str, rule: str, *, env: Optional[str] = None, cast=float, fresh: bool = False) -> Any`. With `fresh=True`, the module cache is skipped and the row is read through `_fresh_engine()`. The value read is cached as usual afterwards.
  - `governed_limits._new_engine() -> Optional[Any]`: returns `rbac._build_engine()`, a new PolicyEngine.
  - `governed_limits._fresh_engine`: the seam a fresh read calls, `= _new_engine` by default. Tests replace it, as they replace `_engine`.

- [ ] **Step 1: Write the failing tests.** Append to `tests/governance/test_governed_limits.py`:

```python
# ---------------------------------------------------------------------------
# a fresh read: a customer's edit in the policy admin applies without a restart
# ---------------------------------------------------------------------------
def _precedent(n):
    return _Engine({"precedent_count": n}, "agent_policy_conflicts")


def test_a_fresh_read_sees_an_edit_without_a_restart(monkeypatch):
    monkeypatch.setattr(GL, "_engine", lambda: _precedent(5))
    monkeypatch.setattr(GL, "_fresh_engine", lambda: _precedent(5))
    assert GL.limit("agent_policy_conflicts", "precedent_count", cast=int) == 5
    # the gateway's PUT /policy/update/:id writes a new row; a new engine now reads 2
    monkeypatch.setattr(GL, "_fresh_engine", lambda: _precedent(2))
    assert GL.limit("agent_policy_conflicts", "precedent_count", cast=int) == 5, "a cached read is still cached"
    assert GL.limit("agent_policy_conflicts", "precedent_count", cast=int, fresh=True) == 2


def test_a_fresh_read_never_uses_the_shared_engine(monkeypatch):
    def shared():
        raise AssertionError("the shared, cached engine was used")
    monkeypatch.setattr(GL, "_engine", shared)
    monkeypatch.setattr(GL, "_fresh_engine", lambda: _precedent(3))
    assert GL.limit("agent_policy_conflicts", "precedent_count", cast=int, fresh=True) == 3


def test_a_fresh_read_of_a_missing_row_refuses(monkeypatch):
    monkeypatch.setattr(GL, "_fresh_engine", lambda: _Engine(None, "agent_policy_conflicts"))
    with pytest.raises(GL.LimitUnavailable):
        GL.limit("agent_policy_conflicts", "precedent_count", cast=int, fresh=True)


def test_a_fresh_read_with_no_engine_refuses(monkeypatch):
    monkeypatch.setattr(GL, "_fresh_engine", lambda: None)
    with pytest.raises(GL.LimitUnavailable):
        GL.limit("agent_policy_conflicts", "precedent_count", cast=int, fresh=True)


def test_the_real_fresh_seam_builds_a_new_engine_every_time(monkeypatch):
    from src.services import rbac
    built = []
    monkeypatch.setattr(rbac, "_build_engine", lambda: built.append(1) or _precedent(4))
    monkeypatch.setattr(GL, "_fresh_engine", GL._new_engine)
    assert GL.limit("agent_policy_conflicts", "precedent_count", cast=int, fresh=True) == 4
    assert GL.limit("agent_policy_conflicts", "precedent_count", cast=int, fresh=True) == 4
    assert built == [1, 1]
```

- [ ] **Step 2: Run them and confirm they fail.**
Run: `bptest tests/governance/test_governed_limits.py -v -k "fresh"`
Expected: FAIL, with `TypeError: limit() got an unexpected keyword argument 'fresh'` and `AttributeError: ... has no attribute '_new_engine'`.

- [ ] **Step 3: Implement.** In `src/services/governed_limits.py`, add after `_engine()`:

```python
def _new_engine() -> Optional[Any]:
    """A PolicyEngine built for one read. The shared engine (rbac.policy_engine) is cached for a
    minute and this module caches what it read; a value a customer edits in the policy admin
    (the gateway's PUT /policy/update/:id writes proc.bp_policy directly) must apply on the next
    read, without a restart."""

    from src.services import rbac

    return rbac._build_engine()


#: The seam a fresh read goes through. Tests replace this, as they replace _engine.
_fresh_engine = _new_engine
```

Replace `_rules` and the head of `limit`:

```python
def _rules(policy: str, *, fresh: bool = False) -> Dict[str, Any]:
    if not fresh:
        with _LOCK:
            cached = _CACHE.get(policy)
        if cached is not None:
            return cached

    try:
        engine = _fresh_engine() if fresh else _engine()
        row = engine.get_policy(policy) if engine else None
    except Exception as exc:  # noqa: BLE001 - unreadable is not permission
        raise LimitUnavailable(
            f"could not read the {policy} policy: {exc}") from exc

    if not isinstance(row, dict):
        raise LimitUnavailable(
            f"no active {policy} policy — its limits are unset, and an unset "
            f"limit is not an unlimited one")

    rules = (row.get("details") or {}).get("rules")
    if not isinstance(rules, dict):
        raise LimitUnavailable(f"the {policy} policy states no rules")

    with _LOCK:
        _CACHE[policy] = rules
    return rules
```

```python
def limit(policy: str, rule: str, *, env: Optional[str] = None,
          cast: Callable[[Any], Any] = float, fresh: bool = False) -> Any:
    """The governed value of one limit, or raise.

    ``policy`` is a ``policy_identifier`` (e.g. ``promotion_thresholds``),
    ``rule`` a key under its ``rules``. ``env`` names the environment variable
    that still overrides it during the deprecation window. ``fresh=True`` reads
    the row now (a new PolicyEngine, no cache): for a value customers edit while
    the server runs.

    Returns ``None`` only when the policy states ``null`` for the rule, which
    means "no limit" and is not the same as the rule being absent.
    """

    rules = _rules(policy, fresh=fresh)
```

(the rest of `limit` is unchanged). In `tests/conftest.py`, inside `_governed_limits_available`, add the second patch:

```python
    monkeypatch.setattr(governed_limits, "_engine", lambda: _SeededPolicyEngine())
    monkeypatch.setattr(governed_limits, "_fresh_engine", lambda: _SeededPolicyEngine())
```

- [ ] **Step 4: Run and confirm they pass.**
Run: `bptest tests/governance/test_governed_limits.py tests/governance/test_governed_limit_callers.py -v -k "not live_policy_set and not seed_matches"`
Expected: PASS. (The two tests that read live rows are dealt with in Task 2.)

- [ ] **Step 5: Prove the guard fails.** Temporarily change the first line of `_rules` to `if True:` (a fresh read is served from the cache). Run `bptest tests/governance/test_governed_limits.py -k fresh_read_sees -v`: it must go red on `== 2`. Restore the line, rerun, and capture green.

- [ ] **Step 6: Commit.**

```bash
cd "$BP" && git add src/services/governed_limits.py tests/conftest.py tests/governance/test_governed_limits.py
git commit -F - <<'EOF'
feat(governance): read a governed limit fresh, so a policy-admin edit applies without a restart

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

### Task 2: The precedent count is a governed policy row, in both databases

**Files:**
- Create: `deploy/sql/2026-10-13_agent_policy_conflict_precedent.sql`, `deploy/sql/2026-10-13_agent_policy_conflict_precedent_rollback.sql`
- Create: `tests/migrations/test_2026_10_13_agent_policy_conflict_precedent.py`
- Modify: `tests/conftest.py` (`GOVERNED_LIMIT_SEED`), `tests/governance/test_governed_limits.py` (`test_every_governed_limit_is_present_in_the_live_policy_set` counts)

**Interfaces:**
- Produces: one active `proc.bp_policy` row in each database. `policy_name 'AgentPolicyConflictPolicy'`, `policy_type 'limit'`, `policy_details = {"policy_identifier": "agent_policy_conflicts", "rules": {"precedent_count": <n>}}`, `created_by 'agent_policy_conflicts'`. `<n>` is copied from `proc.bp_admin_config['agent_policy_settings'].live_conflict_repeat`, else 5. Both databases hold 5 today.
- Produces: `GOVERNED_LIMIT_SEED["agent_policy_conflicts"] == {"precedent_count": 5}` (every test reads N=5 through the seeded engine unless it says otherwise).

- [ ] **Step 1: Record the baseline of the two live governed-limit tests.**
Run: `bptest tests/governance/test_governed_limits.py -k "live_policy_set or seed_matches" -v` once with `.env` as is (bp_testdb), and once with `DB_NAME=bp_sqldb` exported inside the subshell (`(cd "$BP" && set -a && . ./.env && set +a && DB_NAME=bp_sqldb PYTHONPATH=.:src CUDA_VISIBLE_DEVICES="" /home/muthu/PycharmProjects/BP_Backend/venv/bin/python -m pytest tests/governance/test_governed_limits.py -k "live_policy_set or seed_matches" -v)`).
Expected today, and NOT caused by this plan (record it in your report):
- on bp_sqldb, the count test fails `assert 64 == 62` (the 22 `triage_tolerances` values were added on 2026-09-24 without updating 62); the seed test passes;
- on bp_testdb, both fail on `supplier_info_request`, another session's row that is not in the seed.

- [ ] **Step 2: Write the failing migration test** `tests/migrations/test_2026_10_13_agent_policy_conflict_precedent.py`:

```python
"""The precedent count as a governed limit (design §3.3), in both live databases.

Needs PROCWISE_TEST_LIVE_DB=1. Read-only against the shared databases, except the tests that run
the migration files inside a transaction that is always rolled back.
"""
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("PROCWISE_TEST_LIVE_DB") != "1",
    reason="live database required (PROCWISE_TEST_LIVE_DB=1)",
)
DATABASES = ("bp_testdb", "bp_sqldb")
ROOT = Path(__file__).resolve().parents[2]
UP = ROOT / "deploy/sql/2026-10-13_agent_policy_conflict_precedent.sql"
DOWN = ROOT / "deploy/sql/2026-10-13_agent_policy_conflict_precedent_rollback.sql"
SLUG = "agent_policy_conflicts"


def _connect(dbname):
    import psycopg2
    return psycopg2.connect(
        host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT", 5432),
        user=os.getenv("DB_USER"), password=os.getenv("DB_PASSWORD"),
        dbname=dbname, connect_timeout=10,
    )


def _body(path):
    """The file's statements without its own BEGIN/COMMIT, to run inside a rolled-back transaction."""
    return "\n".join(line for line in path.read_text().splitlines()
                     if line.strip().upper() not in ("BEGIN;", "COMMIT;"))


def _rows(cur):
    cur.execute("SELECT policy_name, policy_type, policy_status, policy_details, created_by "
                "FROM proc.bp_policy WHERE policy_details->>'policy_identifier' = %s ORDER BY policy_id", (SLUG,))
    return cur.fetchall()


def _setting(cur):
    cur.execute("SELECT config_value->>'live_conflict_repeat' FROM proc.bp_admin_config "
                "WHERE config_key = 'agent_policy_settings'")
    row = cur.fetchone()
    return int(row[0]) if row and row[0] and row[0].isdigit() else 5


@pytest.mark.parametrize("db", DATABASES)
def test_the_row_exists_with_the_copied_value(db):
    conn = _connect(db)
    try:
        with conn.cursor() as cur:
            active = [r for r in _rows(cur) if r[2] == 1]
            assert len(active) == 1, f"{db}: expected one active {SLUG} row, found {len(active)}"
            name, ptype, _status, details, created_by = active[0]
            assert (name, ptype, created_by) == ("AgentPolicyConflictPolicy", "limit", "agent_policy_conflicts")
            assert details["policy_identifier"] == SLUG
            assert "applies_to" not in details, "configuration read by name, never an authority statement"
            assert details["rules"] == {"precedent_count": _setting(cur)}
    finally:
        conn.close()


@pytest.mark.parametrize("db", DATABASES)
def test_the_policy_engine_reads_it(db):
    from src.engines.policy_engine import PolicyEngine
    conn = _connect(db)
    try:
        engine = PolicyEngine(connection_factory=lambda: conn)
        row = engine.get_policy(SLUG)
        assert row is not None and row["details"]["rules"]["precedent_count"] >= 0
    finally:
        conn.close()


def test_applying_again_adds_nothing_and_the_rollback_removes_it():
    conn = _connect("bp_testdb")
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            assert len([r for r in _rows(cur) if r[2] == 1]) == 1
            cur.execute(_body(UP))
            cur.execute(_body(UP))
            assert len([r for r in _rows(cur) if r[2] == 1]) == 1, "idempotent"
            cur.execute(_body(DOWN))
            assert _rows(cur) == [], "the rollback removes every version of the row"
    finally:
        conn.rollback()
        conn.close()


def test_a_first_apply_copies_the_company_setting():
    conn = _connect("bp_testdb")
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            cur.execute(_body(DOWN))
            cur.execute("UPDATE proc.bp_admin_config SET config_value = jsonb_set(config_value, "
                        "'{live_conflict_repeat}', '7') WHERE config_key = 'agent_policy_settings'")
            cur.execute(_body(UP))
            [(_n, _t, _s, details, _c)] = _rows(cur)
            assert details["rules"] == {"precedent_count": 7}
            cur.execute(_body(DOWN))
            cur.execute("UPDATE proc.bp_admin_config SET config_value = config_value - 'live_conflict_repeat' "
                        "WHERE config_key = 'agent_policy_settings'")
            cur.execute(_body(UP))
            [(_n, _t, _s, details, _c)] = _rows(cur)
            assert details["rules"] == {"precedent_count": 5}, "no setting: the ruled default"
    finally:
        conn.rollback()
        conn.close()
```

- [ ] **Step 3: Run it and confirm it fails.**
Run: `LIVE=1 bptest tests/migrations/test_2026_10_13_agent_policy_conflict_precedent.py -v`
Expected: FAIL. `expected one active agent_policy_conflicts row, found 0` on both databases, and the file-based tests fail with `FileNotFoundError`.

- [ ] **Step 4: Write the migration** `deploy/sql/2026-10-13_agent_policy_conflict_precedent.sql`:

```sql
-- 2026-10-13  Conflict history and precedent (specs/2026-10-09-conflict-history-and-precedent-design.md §3.3).
--
-- How many times people must have decided the same clash between agent policies the same way
-- before the decision engine decides it on precedent becomes a governed limit, so a customer can
-- change it (user ruling R4). One row, the established pattern of the 2026-09-10 governed limits:
-- policy_type 'limit', DELIBERATELY NO applies_to (configuration read by name, never an
-- authority statement). The value is copied from the company setting it replaces
-- (bp_admin_config.agent_policy_settings.live_conflict_repeat), else the ruled default 5.
-- Additive and idempotent: nothing is inserted while an active row exists.
BEGIN;

INSERT INTO proc.bp_policy (
    policy_name, policy_type, policy_desc, policy_details,
    policy_linked_agents, policy_status, version,
    created_date, created_by, last_modified_date, last_modified_by
)
SELECT 'AgentPolicyConflictPolicy', 'limit',
       'How many times people must have decided the same clash between two agent policies the '
       'same way, with the same policy versions, before the decision engine decides that clash '
       'itself on precedent (approve or reject, citing those cases), and the policy owners are '
       'asked to make it a standing rule. Any disagreement, or fewer decisions, sends the clash '
       'to people. 0 switches precedent and the standing-rule proposal off.',
       jsonb_build_object('policy_identifier', 'agent_policy_conflicts',
                          'rules', jsonb_build_object('precedent_count', n.value)),
       '', 1, 1, now(), 'agent_policy_conflicts', now(), 'agent_policy_conflicts'
  FROM (SELECT COALESCE(
          (SELECT (c.config_value->>'live_conflict_repeat')::int
             FROM proc.bp_admin_config c
            WHERE c.config_key = 'agent_policy_settings'
              AND c.config_value->>'live_conflict_repeat' ~ '^[0-9]{1,6}$'),
          5) AS value) AS n
 WHERE NOT EXISTS (
    SELECT 1 FROM proc.bp_policy p
     WHERE p.policy_details->>'policy_identifier' = 'agent_policy_conflicts'
       AND p.policy_status = 1
 );

COMMIT;
```

and `deploy/sql/2026-10-13_agent_policy_conflict_precedent_rollback.sql`:

```sql
-- Rollback of 2026-10-13_agent_policy_conflict_precedent.sql. Removes every version of the
-- agent_policy_conflicts row, including versions a customer saved through the policy admin.
-- Afterwards precedent never applies and no standing-rule proposal is raised (both escalate /
-- skip with a warning). Never run against shared data without a user ruling.
BEGIN;
DELETE FROM proc.bp_policy WHERE policy_details->>'policy_identifier' = 'agent_policy_conflicts';
COMMIT;
```

- [ ] **Step 5: Apply to bp_testdb, then to bp_sqldb, twice each** (the second run must insert 0 rows):

```bash
bppsql bp_testdb -f "$BP/deploy/sql/2026-10-13_agent_policy_conflict_precedent.sql"   # INSERT 0 1
bppsql bp_testdb -f "$BP/deploy/sql/2026-10-13_agent_policy_conflict_precedent.sql"   # INSERT 0 0
bppsql bp_sqldb  -f "$BP/deploy/sql/2026-10-13_agent_policy_conflict_precedent.sql"   # INSERT 0 1
bppsql bp_sqldb  -f "$BP/deploy/sql/2026-10-13_agent_policy_conflict_precedent.sql"   # INSERT 0 0
for db in bp_testdb bp_sqldb; do bppsql $db -Atc "SELECT policy_status, policy_details FROM proc.bp_policy WHERE policy_details->>'policy_identifier'='agent_policy_conflicts'"; done
```

Expected: one row per database, `1|{"rules": {"precedent_count": 5}, "policy_identifier": "agent_policy_conflicts"}`.

- [ ] **Step 6: Bring the in-memory seed and the stale count up to date.** In `tests/conftest.py`, add to `GOVERNED_LIMIT_SEED`, after `"triage_tolerances"`:

```python
    # 2026-10-13: how many same-way decisions by people make a precedent (design §3.3, ruling R4)
    "agent_policy_conflicts": {"precedent_count": 5},
```

In `tests/governance/test_governed_limits.py::test_every_governed_limit_is_present_in_the_live_policy_set`, change the two counts and their comment:

```python
    assert len(live) == 11, f"expected eleven limit rows, found {sorted(live)}"
    # 57, plus the two contract parent-proposal limits (2026-10-04), plus the
    # three receipt tolerances the three-way match reads (2026-10-04:
    # over_delivery_pct, billed_over_received_qty, uom_conversion_required),
    # plus the 22 triage tolerances (2026-09-24, never counted here: 64 on
    # bp_sqldb), plus precedent_count (2026-10-13).
    assert sum(len(r) for r in live.values()) == 65, (
```

- [ ] **Step 7: Run and confirm they pass.**
Run: `LIVE=1 bptest tests/migrations/test_2026_10_13_agent_policy_conflict_precedent.py -v` → PASS (6 tests).
Rerun Step 1's two commands. Expected:
- on bp_sqldb both pass;
- on bp_testdb both still fail, only on `supplier_info_request` (pre-existing, another session's row; report it, do not change it).

- [ ] **Step 8: Prove the guard fails.** Temporarily remove `AND p.policy_status = 1` and the whole `WHERE NOT EXISTS (...)` clause from the migration. Rerun `test_applying_again_adds_nothing_and_the_rollback_removes_it`: it must go red on "idempotent". Restore the file and capture green. Do NOT apply the broken file to either database.

- [ ] **Step 9: Commit.**

```bash
cd "$BP" && git add deploy/sql/2026-10-13_agent_policy_conflict_precedent.sql deploy/sql/2026-10-13_agent_policy_conflict_precedent_rollback.sql tests/migrations/test_2026_10_13_agent_policy_conflict_precedent.py tests/conftest.py tests/governance/test_governed_limits.py
git commit -F - <<'EOF'
feat(agent-policy): the precedent count is a governed policy row a customer can change

Applied to bp_testdb and bp_sqldb (value copied from live_conflict_repeat: 5).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

### Task 3: The standing-rule proposal reads N from the governed value

**Files:**
- Modify: `src/services/agent_policy/settings.py` (drop `live_conflict_repeat` from `DEFAULTS`; add `PRECEDENT_POLICY`, `PRECEDENT_RULE`, `precedent_count()`)
- Modify: `src/services/agent_policy/conflict_live.py` (`threshold`, `_propose_safely`)
- Modify: `tests/agent_policy/fixtures.py` (add `precedent_n`)
- Modify: `tests/agent_policy/test_settings_registry.py` (line 8), `tests/agent_policy/test_conflict_live_live.py` (replace `test_threshold_comes_from_company_setting`)
- Test: `tests/agent_policy/test_precedent_count.py` (new)

**Interfaces:**
- Consumes: `governed_limits.limit(..., fresh=True)` (Task 1); the seeded row (Task 2).
- Produces:
  - `settings.PRECEDENT_POLICY = "agent_policy_conflicts"`, `settings.PRECEDENT_RULE = "precedent_count"`.
  - `settings.precedent_count() -> Optional[int]`: the governed value read fresh; `None` means null (off). It raises `src.services.governed_limits.LimitUnavailable` when the row or rule is missing, and `ValueError`/`TypeError` when the value is not a number.
  - `conflict_live.threshold() -> Optional[int]`: `precedent_count()`, or `None` plus a WARNING `"repeat proposal skipped: the precedent count cannot be read (<ExcType>)"`.
  - `tests/agent_policy/fixtures.precedent_n(monkeypatch, n, *, missing=False) -> None`: every fresh read of `agent_policy_conflicts` sees `n`, or no row when `missing`.

- [ ] **Step 1: Write the failing tests.** Add to `tests/agent_policy/fixtures.py` (end of file):

```python
def precedent_n(monkeypatch, n, *, missing=False):
    """Set the governed precedent count (agent_policy_conflicts.precedent_count) every fresh read
    sees. missing=True: the row does not exist, so the read raises LimitUnavailable."""
    from src.services import governed_limits as GL

    class _Engine:
        def get_policy(self, slug):
            if missing or slug != "agent_policy_conflicts":
                return None
            return {"policyName": "AgentPolicyConflictPolicy",
                    "details": {"policy_identifier": slug, "rules": {"precedent_count": n}}}

    monkeypatch.setattr(GL, "_fresh_engine", lambda: _Engine())
```

Create `tests/agent_policy/test_precedent_count.py`:

```python
"""N, the precedent count, is one governed value read fresh (design §3.3, rulings R2 and R4)."""
import inspect
from datetime import datetime, timezone

import pytest

from services.agent_policy import conflict_live as CL
from services.agent_policy import settings as S
from src.services import governed_limits as GL
from tests.agent_policy.fixtures import precedent_n


def test_precedent_count_is_the_governed_value_read_fresh(monkeypatch):
    def shared():
        raise AssertionError("the cached engine was used")
    monkeypatch.setattr(GL, "_engine", shared)
    precedent_n(monkeypatch, 3)
    assert S.precedent_count() == 3


def test_the_seed_holds_five():
    assert S.precedent_count() == 5


def test_a_missing_row_raises(monkeypatch):
    precedent_n(monkeypatch, None, missing=True)
    with pytest.raises(GL.LimitUnavailable):
        S.precedent_count()


def test_threshold_is_none_and_warns_when_the_row_is_missing(monkeypatch, caplog):
    precedent_n(monkeypatch, None, missing=True)
    with caplog.at_level("WARNING", logger=CL.__name__):
        assert CL.threshold() is None
    assert "repeat proposal skipped: the precedent count cannot be read (LimitUnavailable)" in caplog.text


def test_threshold_is_none_for_a_value_that_is_not_a_number(monkeypatch, caplog):
    precedent_n(monkeypatch, "five")
    with caplog.at_level("WARNING", logger=CL.__name__):
        assert CL.threshold() is None
    assert "(ValueError)" in caplog.text


class _NoSql:
    def execute(self, sql, params=None):
        raise AssertionError(f"nothing may run when the proposal is off: {sql}")


@pytest.mark.parametrize("n", [None, 0, -1])
def test_no_proposal_when_off_or_unreadable(monkeypatch, n):
    monkeypatch.setattr(CL, "threshold", lambda: n)
    CL._propose_safely(_NoSql(), 1, now=datetime.now(timezone.utc))


def test_live_conflict_repeat_is_no_longer_a_setting():
    assert "live_conflict_repeat" not in S.DEFAULTS
    assert "live_conflict_repeat" not in inspect.getsource(CL)
```

In `tests/agent_policy/test_settings_registry.py`, change line 8 from `assert d["live_conflict_repeat"] == 5` to:

```python
    assert "live_conflict_repeat" not in d, "N is the governed agent_policy_conflicts.precedent_count"
```

In `tests/agent_policy/test_conflict_live_live.py`, add `from tests.agent_policy import fixtures as F` to the imports, delete `from services.agent_policy import settings as S` and `test_threshold_comes_from_company_setting`, and add:

```python
def test_threshold_comes_from_the_governed_policy(conn, world, monkeypatch):
    F.precedent_n(monkeypatch, 2)
    once(conn, world, monkeypatch)
    assert repeat_cases(conn, world) == []
    once(conn, world, monkeypatch)
    [pc] = repeat_cases(conn, world)
    assert pc["facts"]["proposal"] == {"from": "repeat", "count": 2, "outcome": "approve"}


def test_a_missing_precedent_row_skips_the_proposal(conn, world, monkeypatch, caplog):
    F.precedent_n(monkeypatch, None, missing=True)
    with caplog.at_level("WARNING", logger=CL.__name__):
        for _ in range(5):
            once(conn, world, monkeypatch)
    assert repeat_cases(conn, world) == []
    assert "the precedent count cannot be read" in caplog.text


def test_zero_switches_the_proposal_off(conn, world, monkeypatch):
    F.precedent_n(monkeypatch, 0)
    for _ in range(3):
        once(conn, world, monkeypatch)
    assert repeat_cases(conn, world) == []
```

- [ ] **Step 2: Run them and confirm they fail.**
Run: `bptest tests/agent_policy/test_precedent_count.py tests/agent_policy/test_settings_registry.py -v`
Expected: FAIL (`AttributeError: module ... has no attribute 'precedent_count'`; `live_conflict_repeat` still in DEFAULTS).

- [ ] **Step 3: Implement.** In `src/services/agent_policy/settings.py`, delete the line `"live_conflict_repeat": 5,` from `DEFAULTS`, and add at the end:

```python
#: N, the precedent count (design §3.3): a governed limit in proc.bp_policy that a customer can
#: change, not a company setting. One number for both the precedent and the standing-rule
#: proposal (ruling R2).
PRECEDENT_POLICY = "agent_policy_conflicts"
PRECEDENT_RULE = "precedent_count"


def precedent_count() -> Optional[int]:
    """How many same-way decisions by people make a precedent. Read fresh on every call: the
    policy admin writes the row directly and its edit must apply without a restart. None is a
    stated null (off). Raises governed_limits.LimitUnavailable when the row or rule is missing,
    and ValueError/TypeError when the value is not a number: never a default in code."""
    from src.services import governed_limits

    return governed_limits.limit(PRECEDENT_POLICY, PRECEDENT_RULE, cast=int, fresh=True)
```

In `src/services/agent_policy/conflict_live.py`, replace `threshold` and `_propose_safely`:

```python
def threshold() -> Optional[int]:
    """The governed precedent count (settings.precedent_count): at N the standing-rule proposal is
    raised, as the engine starts deciding on precedent (ruling R2). None when it cannot be read:
    no proposal then, with a warning, never a number made up here."""
    try:
        return _settings.precedent_count()
    except Exception as exc:  # noqa: BLE001 - LimitUnavailable, or a value that is not a number
        logger.warning("repeat proposal skipped: the precedent count cannot be read (%s)", type(exc).__name__)
        return None


def _propose_safely(cur, live_id: int, *, now: datetime) -> None:
    """maybe_propose in a savepoint: a failed proposal is logged and undone on its own and never
    rolls back the person's approval or the settle (they stay in the caller's transaction).
    N missing, null or 0: no proposal, nothing runs."""
    n = threshold()
    if not n or n <= 0:
        return
    cur.execute("SAVEPOINT live_conflict_propose")
    try:
        maybe_propose(cur, live_id, now=now, threshold=n)
    except Exception as exc:  # noqa: BLE001 - type only: a driver message can quote stored values
        cur.execute("ROLLBACK TO SAVEPOINT live_conflict_propose")
        logger.error("repeat proposal failed for live case %s: %s", live_id, type(exc).__name__)
    cur.execute("RELEASE SAVEPOINT live_conflict_propose")
```

Also update the module docstring line "A person's decision counts towards repeat-N (maybe_propose)" to add: "N is the governed agent_policy_conflicts.precedent_count (settings.precedent_count)."

- [ ] **Step 4: Run and confirm they pass.**
Run: `bptest tests/agent_policy/test_precedent_count.py tests/agent_policy/test_settings_registry.py -v` → PASS.
Run: `LIVE=1 bptest tests/agent_policy/test_conflict_live_live.py tests/agent_policy/test_conflict_live_gate.py -v` → PASS (the repeat-5 tests read N=5 from the seed; `test_a_failed_repeat_proposal_never_rolls_back_the_persons_approval` still patches `CL.threshold`).
Run: `grep -rn "live_conflict_repeat" "$BP/src" "$UI/src" "$GW/src"` → no hits.

- [ ] **Step 5: Prove the guard fails.** In `threshold`, temporarily return `5` from the `except` branch (a default in code). `test_threshold_is_none_and_warns_when_the_row_is_missing` must go red. Restore it and capture green.

- [ ] **Step 6: Commit.**

```bash
cd "$BP" && git add src/services/agent_policy/settings.py src/services/agent_policy/conflict_live.py tests/agent_policy/fixtures.py tests/agent_policy/test_precedent_count.py tests/agent_policy/test_settings_registry.py tests/agent_policy/test_conflict_live_live.py
git commit -F - <<'EOF'
feat(agent-policy): the standing-rule proposal reads N from the governed precedent count

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

### Task 4: Every closing path records who decided; live records can be precedent decisions

**Files:**
- Modify: `src/services/agent_policy/conflict_cases.py` (`DECIDED_BY_KINDS`, `decided_by`; `decide_policy`, `_moot`)
- Modify: `src/services/agent_policy/conflict_live.py` (`PRECEDENT`, `_AUTO`, `_KIND`, `insert_live`, `settle_for_group`)
- Test: `tests/agent_policy/test_conflict_decided_by.py` (new, no DB), `tests/agent_policy/test_conflict_decided_by_live.py` (new, live)

**Interfaces:**
- Produces:
  - `conflict_cases.DECIDED_BY_KINDS = ("person", "standing_rule", "precedent", "timeout", "block", "retired")`.
  - `conflict_cases.decided_by(kind: str, name: Optional[str]) -> Dict[str, Any]`, returning `{"kind", "name"}`. It raises `ValueError` for an unknown kind.
  - `facts.decidedBy` on the closing row of every path:
    - `decide_policy` → person;
    - `_moot` → retired;
    - `settle_for_group` → person | timeout;
    - `insert_live` actioned → block | standing_rule | precedent.
  - `facts.versionsAtDecision` on `settle_for_group` action rows and actioned `insert_live` rows. It is the clash's `policy_versions` (`{key: int}`).
  - `conflict_live.PRECEDENT = "system:precedent"`.
  - `conflict_live.insert_live(cur, lc, *, ctx, action, now, default_response_time, status, decision=None, actor=None, reason=None, extra_facts: Optional[Dict] = None, extra_evidence: Optional[List[Dict]] = None) -> int`. `status='actioned'` accepts decision `block` | `standing_rule` | `approve` | `reject`; approve/reject only with actor `PRECEDENT` (or none). `extra_facts` is merged into facts; `extra_evidence` is appended after the overlap entry.

- [ ] **Step 1: Write the failing unit tests** `tests/agent_policy/test_conflict_decided_by.py`:

```python
"""facts.decidedBy is written where a conflict case closes, never parsed from an actor (design §3.1)."""
import json
from datetime import datetime, timezone

import pytest

from services.agent_policy import conflict_cases as CC
from services.agent_policy import conflict_engine as CE
from services.agent_policy import conflict_live as CL
from tests.agent_policy.test_conflict_live_gate import _Cur, make_doc

NOW = datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc)
ACTION = {"tool": "refund.issue", "args": {"amount": 900, "note": "x"}, "agent": "a", "workflowId": "wf",
          "userId": "u"}


def _lc():
    a = make_doc("TSB-0101", tool="refund.issue", outcome="approve", deciders=["FM"], source="Finance")
    b = make_doc("TSB-0102", tool="refund.issue", outcome="approve", deciders=["CFO"], source="Customer")
    hits = [{"id": d["id"], "version": d["version"], "outcome": "approve", "policy": d} for d in (a, b)]
    return CE.LiveConflict("human", hits, [("TSB-0101", "TSB-0102")], hits, {"TSB-0101", "TSB-0102"})


def _insert(**kw):
    log = []
    CL.insert_live(_Cur(log, iter(range(7, 9))), _lc(), ctx={"tool.name": "refund.issue", "args": ACTION["args"]},
                   action=ACTION, now=NOW, default_response_time="PT4H", **kw)
    sql, params = next((s, json.loads(p)) for s, p in log if s.startswith("INSERT INTO proc.bp_decision"))
    return params, json.loads(params[7]), json.loads(params[8])


def test_a_precedent_record_says_precedent_and_cites_its_cases():
    cited = [{"kind": "precedent", "caseId": "pc_3", "decision_id": 3, "outcome": "approve",
              "actioned_by": "sub-x", "actioned_at": "2026-10-09T10:00:00+00:00"}]
    params, facts, evidence = _insert(status="actioned", decision="approve", actor=CL.PRECEDENT,
                                      reason="Decided the same way 1 times before", extra_evidence=cited)
    assert facts["decidedBy"] == {"kind": "precedent", "name": "system:precedent"}
    assert facts["versionsAtDecision"] == {"TSB-0101": 1, "TSB-0102": 1}
    assert evidence[0]["kind"] == "overlap" and evidence[1:] == cited
    assert (params[2], params[5], params[15], params[16]) == ("approve", "actioned", "this_action", "system:precedent")


@pytest.mark.parametrize("decision,kind,name", [("block", "block", "system:not_allowed"),
                                                ("standing_rule", "standing_rule", "system:standing_rule")])
def test_block_and_standing_rule_records_say_so(decision, kind, name):
    _params, facts, _ev = _insert(status="actioned", decision=decision)
    assert facts["decidedBy"] == {"kind": kind, "name": name}
    assert facts["versionsAtDecision"] == {"TSB-0101": 1, "TSB-0102": 1}


def test_only_precedent_may_record_approve_or_reject():
    with pytest.raises(ValueError):
        _insert(status="actioned", decision="approve", actor="sub-someone")
    with pytest.raises(ValueError):
        _insert(status="actioned", decision="maybe")


def test_an_open_record_has_no_decided_by_and_keeps_its_extras():
    _params, facts, _ev = _insert(status="open", extra_facts={"history": [{"caseId": "pc_1"}],
                                                               "precedent": {"why": "decisions disagree"}})
    assert "decidedBy" not in facts and "versionsAtDecision" not in facts
    assert facts["history"] == [{"caseId": "pc_1"}] and facts["precedent"] == {"why": "decisions disagree"}


def test_decided_by_refuses_an_unknown_kind():
    assert CC.decided_by("person", "sub-x") == {"kind": "person", "name": "sub-x"}
    with pytest.raises(ValueError):
        CC.decided_by("guess", "system:timeout")
```

(Parameter positions in the `INSERT INTO proc.bp_decision` of `insert_live`: 2 decision, 5 status, 7 facts, 8 evidence, 15 decision_scope, 16 actioned_by.)

Write the failing live tests `tests/agent_policy/test_conflict_decided_by_live.py`:

```python
"""Each closing path writes facts.decidedBy (and live settlements facts.versionsAtDecision).

Needs PROCWISE_TEST_LIVE_DB=1 and DB_NAME=bp_testdb. Uses the conflict-endpoint world: DRAFTS with
tool tst_<tag>, retired at teardown; the gate gets the same documents injected.
"""
import os
from datetime import timedelta

import pytest

from services.agent_policy import approvals as A
from services.agent_policy import conflict_cases as CC
from services.agent_policy import conflict_live as CL
from tests.agent_policy import test_conflict_live_gate as LG
from tests.agent_policy.test_conflict_endpoints_live import (  # noqa: F401 - fixtures
    NOW, _a, _b, _c, _design_case, _gate, conn, world)

pytestmark = pytest.mark.skipif(os.getenv("PROCWISE_TEST_LIVE_DB") != "1", reason="live DB required")


def _closing_facts(conn, subject_type, case_id):
    [row] = LG.rows(conn, "SELECT facts FROM proc.bp_decision WHERE subject_type = %s AND status = 'actioned' "
                          "AND actioned_by IS NOT NULL AND facts->>'caseId' = %s "
                          "ORDER BY decision_id DESC LIMIT 1", (subject_type, f"pc_{case_id}"))
    return row["facts"]


def test_a_person_deciding_a_policy_case(conn, world):
    a, c, did = _design_case(conn, world)
    CC.decide_policy(conn, did, principal=LG.who(world, world.oa), option=f"retire:{c}", reason="Old rule",
                     limit_text=None, now=NOW)
    assert _closing_facts(conn, CC.SUBJECT_POLICY, did)["decidedBy"] == \
        {"kind": "person", "name": f"sub-{world.email_a}"}


def test_a_retired_policy_closing_its_case(conn, world):
    a, c, did = _design_case(conn, world)
    assert CC.close_moot(conn, c, now=NOW) == 1
    assert _closing_facts(conn, CC.SUBJECT_POLICY, did)["decidedBy"] == {"kind": "retired", "name": "system:retired"}


def test_people_settling_a_live_clash(conn, world, monkeypatch):
    (a, da), (b, db) = _a(conn, world), _b(conn, world)
    _gate(monkeypatch, world, [da, db])
    [lv] = LG.lives(conn, world)
    ma, mb = LG.members(conn, world)
    LG.act(conn, world, ma["decision_id"], world.la2)
    LG.act(conn, world, mb["decision_id"], world.lb)
    facts = _closing_facts(conn, CL.SUBJECT_LIVE, lv["decision_id"])
    assert facts["decidedBy"] == {"kind": "person", "name": f"sub-{world.people[world.lb]}"}
    assert facts["versionsAtDecision"] == {a: 1, b: 1}


def test_a_timeout_settling_a_live_clash(conn, world, monkeypatch):
    (a, da), (b, db) = _a(conn, world), _b(conn, world)
    _gate(monkeypatch, world, [da, db])
    [lv] = LG.lives(conn, world)
    A.sweep(conn, lv["respond_by"] + timedelta(seconds=1), decision_ids=[m["decision_id"] for m in LG.members(conn, world)])
    facts = _closing_facts(conn, CL.SUBJECT_LIVE, lv["decision_id"])
    assert facts["decidedBy"] == {"kind": "timeout", "name": "system:timeout"}
    assert facts["versionsAtDecision"] == {a: 1, b: 1}


def test_a_block_record(conn, world, monkeypatch):
    (a, da), (c, dc) = _a(conn, world), _c(conn, world)
    _gate(monkeypatch, world, [da, dc], amount=20000)
    [lv] = LG.lives(conn, world)
    assert lv["facts"]["decidedBy"] == {"kind": "block", "name": "system:not_allowed"}
    assert lv["facts"]["versionsAtDecision"] == {a: 1, c: 1}


def test_a_standing_rule_record(conn, world, monkeypatch):
    (a, da), (b, db) = _a(conn, world), _b(conn, world)
    da = {**da, "conflicts": [{"with": b, "prevails": a, "rule": f"{a} takes priority over {b}"}]}
    _gate(monkeypatch, world, [da, db])
    [lv] = LG.lives(conn, world)
    assert lv["facts"]["decidedBy"] == {"kind": "standing_rule", "name": "system:standing_rule"}
```

- [ ] **Step 2: Run them and confirm they fail.**
Run: `bptest tests/agent_policy/test_conflict_decided_by.py -v` → FAIL (`AttributeError: ... 'PRECEDENT'`, `'decided_by'`).
Run: `LIVE=1 bptest tests/agent_policy/test_conflict_decided_by_live.py -v` → FAIL (`KeyError: 'decidedBy'`).

- [ ] **Step 3: Implement `conflict_cases`.** Add after the `_LABELS` block:

```python
#: What can close a conflict case (facts.decidedBy.kind), set where it happens and never
#: derived from an actor string (design §3.1).
DECIDED_BY_KINDS = ("person", "standing_rule", "precedent", "timeout", "block", "retired")


def decided_by(kind: str, name: Optional[str]) -> Dict[str, Any]:
    """facts.decidedBy for the row that closes a conflict case."""
    if kind not in DECIDED_BY_KINDS:
        raise ValueError(f"unknown decidedBy kind {kind!r}")
    return {"kind": kind, "name": name}
```

In `decide_policy`, right after `facts["versionsAtDecision"] = _latest_versions(cur, keys)`, add:

```python
                facts["decidedBy"] = decided_by("person", actor)
```

In `_moot`, right after `facts["caseId"] = conflict_payload.case_id(did)`, add:

```python
    facts["decidedBy"] = decided_by("retired", RETIRED_ACTOR)
```

- [ ] **Step 4: Implement `conflict_live`.** Replace the constants under `OPTIONS`:

```python
NOT_ALLOWED = "system:not_allowed"
STANDING_RULE = "system:standing_rule"
PRECEDENT = "system:precedent"
_AGENT = "agent_policy_conflicts"
#: decision -> (actor, scope) for a live case recorded closed; approve/reject only on precedent
_AUTO = {"block": (NOT_ALLOWED, "this_action"), "standing_rule": (STANDING_RULE, "standing_rule"),
         "approve": (PRECEDENT, "this_action"), "reject": (PRECEDENT, "this_action")}
_KIND = {"block": "block", "standing_rule": "standing_rule", "approve": "precedent", "reject": "precedent"}
```

Replace `insert_live` with:

```python
def insert_live(cur, lc, *, ctx: Dict[str, Any], action: Dict[str, Any], now: datetime,
                default_response_time: str, status: str, decision: Optional[str] = None,
                actor: Optional[str] = None, reason: Optional[str] = None,
                extra_facts: Optional[Dict[str, Any]] = None,
                extra_evidence: Optional[List[Dict[str, Any]]] = None) -> int:
    """Record one live conflict on the caller's cursor. Returns its decision_id.

    status 'open' -> decision 'approve_or_reject', waiting for its member cases;
    status 'actioned' -> closed now by the system: 'block' | 'standing_rule', or 'approve' |
    'reject' decided on precedent (actor PRECEDENT only). A closed record carries
    facts.decidedBy and facts.versionsAtDecision. extra_facts are merged into the facts (an
    escalated clash's history); extra_evidence follows the overlap entry (cited precedents)."""
    if status not in ("open", "actioned"):
        raise ValueError(f"status must be 'open' or 'actioned', got {status!r}")
    if status == "actioned" and decision not in _AUTO:
        raise ValueError("an actioned live case needs decision 'block', 'standing_rule', 'approve' or "
                         f"'reject', got {decision!r}")
    if status == "actioned" and decision in ("approve", "reject") and (actor or PRECEDENT) != PRECEDENT:
        raise ValueError("only precedent records a live case as decided approve or reject")
    docs = sorted((h["policy"] for h in lc.involved), key=lambda d: str(d.get("id")))
    keys = [str(d.get("id")) for d in docs]
    key = conflict_detect.pair_key(*keys)
    versions = {str(d.get("id")): _version(d) for d in docs}
    args = dict((action or {}).get("args") or {})
    approve_docs = [d for d in docs if (d.get("enforcement") or {}).get("outcome") == "approve"]
    within = (max((_within(d, default_response_time) for d in approve_docs), key=durations.parse)
              if approve_docs else None)
    live_action = {"tool": action.get("tool"), "args": conflict_payload.condition_args(docs, args),
                   "agent": action.get("agent"), "workflowId": action.get("workflowId"),
                   "plain": _plain(docs, action.get("tool"))}
    standing = [{"policy": str(d.get("id")), **r} for d in docs for r in (d.get("conflicts") or [])
                if isinstance(r, dict)]
    payload = conflict_payload.build(
        "live", raised_at=now.isoformat(), policies=[conflict_payload.policy_entry(d) for d in docs],
        overlap_example=conflict_payload.condition_values(docs, _flat(ctx)), standing_rules=standing,
        prior=_prior(cur, key), options=list(OPTIONS), respond_within=within, on_timeout="reject",
        action=live_action)
    cols = conflict_payload.to_columns(payload)
    facts = dict(cols["facts"])
    facts["pairs"] = [list(p) for p in lc.pairs]
    facts["requestedBy"] = action.get("userId")
    facts.update(dict(extra_facts or {}))
    evidence = list(cols["evidence"]) + [dict(e) for e in extra_evidence or []]

    open_ = status == "open"
    if open_:
        decision, actor, scope = "approve_or_reject", None, None
        respond_by = now + timedelta(seconds=durations.parse(durations.resolve(within, default_response_time)))
    else:
        actor = actor or _AUTO[decision][0]
        scope = _AUTO[decision][1]
        respond_by = None
        facts["decidedBy"] = conflict_cases.decided_by(_KIND[decision], actor)
        facts["versionsAtDecision"] = dict(versions)
    cur.execute(
        """
        INSERT INTO proc.bp_decision (
            subject_type, subject_id, decision, resolution, rationale, status,
            policy_id, policy_name, facts, evidence, workflow_id, agent, created_by,
            options, respond_by, on_timeout, decision_scope, actioned_by, actioned_at, override_reason
        ) VALUES (%s,%s,%s,%s,%s,%s,NULL,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        RETURNING decision_id
        """,
        (cols["subject_type"], key, decision, "escalated" if open_ else "resolved",
         conflict_payload.why_line(docs), "open" if open_ else "actioned", key,
         json.dumps(facts, default=str), json.dumps(evidence, default=str), action.get("workflowId"),
         action.get("agent"), action.get("userId") or f"agent:{action.get('agent') or 'unknown'}",
         json.dumps(cols["options"]), respond_by, cols["on_timeout"], scope, actor,
         None if open_ else now, None if open_ else reason),
    )
    decision_id = int(cur.fetchone()[0])
    cur.execute("UPDATE proc.bp_decision SET facts = facts || %s::jsonb WHERE decision_id = %s",
                (json.dumps({"caseId": conflict_payload.case_id(decision_id)}), decision_id))
    cur.execute(
        "INSERT INTO proc.bp_agent_policy_conflict (decision_id, kind, pair_key, policy_keys, policy_versions, "
        "raised_by, is_open, outcome, decided_by, decided_at, by_person) "
        "VALUES (%s, 'live', %s, %s, %s, 'live', %s, %s, %s, %s, %s)",
        (decision_id, key, keys, json.dumps(versions), open_, None if open_ else decision,
         None if open_ else actor, None if open_ else now, None if open_ else False),
    )
    return decision_id
```

In `settle_for_group`, right after `facts["memberCases"] = list(member_ids)`, add:

```python
        cur.execute("SELECT policy_versions FROM proc.bp_agent_policy_conflict WHERE decision_id = %s", (live_id,))
        pv = cur.fetchone()
        facts["versionsAtDecision"] = dict(_j(pv[0], {}) or {}) if pv else {}
        # A member group is closed by a person or by the sweep (a timeout); system:group rows are
        # consequences of another member's decision and _credited skips them.
        facts["decidedBy"] = conflict_cases.decided_by("person" if by_person else "timeout", actor)
```

- [ ] **Step 5: Run and confirm they pass.**
Run: `bptest tests/agent_policy/test_conflict_decided_by.py -v` → PASS.
Run: `LIVE=1 bptest tests/agent_policy/test_conflict_decided_by_live.py tests/agent_policy/test_conflict_live_gate.py tests/agent_policy/test_conflict_decide_live.py tests/agent_policy/test_conflict_cases_live.py -v` → PASS.
If an existing test compares a closing row's whole `facts`, it may fail. Update that test only by the two new keys, and name it in your report.

- [ ] **Step 6: Prove the guard fails.** In `settle_for_group`, temporarily write `decided_by("person", actor)` regardless of `by_person`. `test_a_timeout_settling_a_live_clash` must go red. Restore it and capture green.

- [ ] **Step 7: Commit.**

```bash
cd "$BP" && git add src/services/agent_policy/conflict_cases.py src/services/agent_policy/conflict_live.py tests/agent_policy/test_conflict_decided_by.py tests/agent_policy/test_conflict_decided_by_live.py
git commit -F - <<'EOF'
feat(agent-policy): every conflict decision records what decided it and the versions it was about

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

### Task 5: The decision engine decides a live clash on precedent

**Files:**
- Modify: `src/engines/decision_engine.py` (append a "live policy conflicts" section)
- Test: `tests/engines/test_decide_live_conflict.py` (new, no DB)

**Interfaces:**
- Consumes: `settings.precedent_count()` (Task 3); `conflict_detect.pair_key`; `conflict_payload.case_id`; a `conflict_engine.LiveConflict`.
- Produces:
  - `decision_engine.LIVE_CONFLICT_SUBJECT = "live_conflict"`, `decision_engine.PRECEDENT_SOURCE = "proc.bp_agent_policy_conflict"`.
  - `decision_engine.PRECEDENT_SQL: str`, taking the parameters `(pair_key, versions_json, n)`.
  - `decision_engine._precedent_count() -> Optional[int]` (seam).
  - `decision_engine.decide_live_conflict(cur, lc, *, ctx: Dict[str, Any], now) -> Decision`. Never writes.
    - Resolved: `decision` is `"approve"` | `"reject"`, `resolution == RESOLVED`, and `evidence` is a list of `Evidence(fact="precedent", value={"decision_id", "outcome", "actioned_by", "actioned_at"}, source=PRECEDENT_SOURCE, reference="pc_<id>")`, newest first. `facts` = `{pairKey, versions, consultedAt, precedentCount, citedCases}`.
    - Escalated: `decision == "escalate"`, `resolution == ESCALATED`, and `rationale` is one of:
      - `"only a clash that needs people can be decided on precedent"`;
      - `"<KEY>[, <KEY>] also needs approval and is not part of this clash"`;
      - `"precedent limit unavailable"`;
      - `"precedent is switched off"`;
      - `"precedent lookup failed"`;
      - `"only <k> of <n> decisions by people on this exact clash"`;
      - `"decisions disagree"`.

- [ ] **Step 1: Write the failing tests** `tests/engines/test_decide_live_conflict.py`:

```python
"""The decision engine decides a live clash between agent policies on precedent, or escalates.

Rule 1: it decides only from what it looks up (the governed N and the clash's own settled cases).
Rule 2: anything short of N of N agreeing decisions by people, at identical versions, escalates.
"""
import json
from datetime import datetime, timezone

import pytest

from engines import decision_engine as DE
from services.agent_policy import conflict_engine as CE
from src.services import governed_limits as GL
from tests.agent_policy.fixtures import precedent_n

NOW = datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc)
AT = datetime(2026, 10, 9, 11, 0, tzinfo=timezone.utc)


def _hit(key, version, outcome="approve"):
    return {"id": key, "version": version, "outcome": outcome, "policy": {"id": key, "version": version}}


A, B, C = _hit("TST-0001", 3), _hit("TST-0002", 1), _hit("TST-0003", 1)


def _lc(kind="human", involved=(A, B), required=None):
    involved = list(involved)
    return CE.LiveConflict(kind, involved, [("TST-0001", "TST-0002")],
                           list(required if required is not None else involved), {h["id"] for h in involved})


class _Cur:
    def __init__(self, rows=(), fail=False):
        self.rows, self.fail, self.sql = list(rows), fail, []

    def execute(self, sql, params=None):
        self.sql.append((" ".join(sql.split()), params))
        if self.fail and "bp_agent_policy_conflict" in sql:
            raise RuntimeError("lookup down")

    def fetchall(self):
        return list(self.rows)


def _n(monkeypatch, n):
    monkeypatch.setattr(DE, "_precedent_count", lambda: n)


def test_resolves_when_the_last_n_person_decisions_agree(monkeypatch):
    _n(monkeypatch, 2)
    cur = _Cur([(11, "approve", "sub-b", AT), (10, "approve", "sub-a", AT)])
    d = DE.decide_live_conflict(cur, _lc(), ctx={}, now=NOW)
    assert (d.resolution, d.decision, d.subject_type, d.subject_id) == (DE.RESOLVED, "approve", "live_conflict",
                                                                       "TST-0001|TST-0002")
    assert [e.reference for e in d.evidence] == ["pc_11", "pc_10"]
    assert d.evidence[0].to_dict() == {"fact": "precedent", "source": "proc.bp_agent_policy_conflict",
                                       "reference": "pc_11",
                                       "value": {"decision_id": 11, "outcome": "approve", "actioned_by": "sub-b",
                                                 "actioned_at": AT.isoformat()}}
    assert d.facts["citedCases"] == ["pc_11", "pc_10"] and d.facts["precedentCount"] == 2
    statements = [s for s, _ in cur.sql]
    assert statements[0] == "SAVEPOINT live_conflict_precedent"
    assert statements[-1] == "RELEASE SAVEPOINT live_conflict_precedent"
    [(_sql, params)] = [x for x in cur.sql if "bp_agent_policy_conflict" in x[0]]
    assert params == ("TST-0001|TST-0002", json.dumps({"TST-0001": 3, "TST-0002": 1}, sort_keys=True), 2)


def test_the_lookup_counts_only_settled_person_decisions_at_identical_versions():
    sql = " ".join(DE.PRECEDENT_SQL.split())
    for clause in ("c.kind = 'live'", "c.pair_key = %s", "NOT c.is_open", "c.by_person",
                   "c.policy_versions = %s::jsonb", "ORDER BY c.decided_at DESC, c.decision_id DESC", "LIMIT %s"):
        assert clause in sql, clause


def test_rejects_on_precedent_too(monkeypatch):
    _n(monkeypatch, 2)
    d = DE.decide_live_conflict(_Cur([(11, "reject", "sub-b", AT), (10, "reject", "sub-a", AT)]), _lc(),
                                ctx={}, now=NOW)
    assert (d.resolution, d.decision) == (DE.RESOLVED, "reject")


def test_fewer_than_n_escalates(monkeypatch):
    _n(monkeypatch, 2)
    d = DE.decide_live_conflict(_Cur([(10, "approve", "sub-a", AT)]), _lc(), ctx={}, now=NOW)
    assert (d.resolution, d.rationale) == (DE.ESCALATED, "only 1 of 2 decisions by people on this exact clash")


def test_disagreement_escalates(monkeypatch):
    _n(monkeypatch, 2)
    d = DE.decide_live_conflict(_Cur([(11, "approve", "sub-b", AT), (10, "reject", "sub-a", AT)]), _lc(),
                                ctx={}, now=NOW)
    assert (d.resolution, d.rationale) == (DE.ESCALATED, "decisions disagree")


@pytest.mark.parametrize("n", [0, None, -1])
def test_zero_or_null_switches_it_off(monkeypatch, n):
    _n(monkeypatch, n)
    cur = _Cur()
    d = DE.decide_live_conflict(cur, _lc(), ctx={}, now=NOW)
    assert (d.resolution, d.rationale) == (DE.ESCALATED, "precedent is switched off") and cur.sql == []


def test_an_unreadable_limit_escalates_with_a_warning(monkeypatch, caplog):
    def missing():
        raise GL.LimitUnavailable("no active agent_policy_conflicts policy")
    monkeypatch.setattr(DE, "_precedent_count", missing)
    cur = _Cur()
    with caplog.at_level("WARNING", logger=DE.__name__):
        d = DE.decide_live_conflict(cur, _lc(), ctx={}, now=NOW)
    assert (d.resolution, d.rationale) == (DE.ESCALATED, "precedent limit unavailable") and cur.sql == []
    assert "precedent limit unavailable, the clash TST-0001|TST-0002 goes to people" in caplog.text


def test_a_failed_lookup_rolls_back_to_its_savepoint_and_escalates(monkeypatch):
    _n(monkeypatch, 2)
    cur = _Cur(fail=True)
    d = DE.decide_live_conflict(cur, _lc(), ctx={}, now=NOW)
    assert (d.resolution, d.rationale) == (DE.ESCALATED, "precedent lookup failed")
    assert [s for s, _ in cur.sql if "SAVEPOINT" in s] == [
        "SAVEPOINT live_conflict_precedent", "ROLLBACK TO SAVEPOINT live_conflict_precedent",
        "RELEASE SAVEPOINT live_conflict_precedent"]


def test_an_approval_outside_the_clash_escalates(monkeypatch):
    _n(monkeypatch, 2)
    cur = _Cur([(11, "approve", "sub-b", AT), (10, "approve", "sub-a", AT)])
    d = DE.decide_live_conflict(cur, _lc(required=[A, B, C]), ctx={}, now=NOW)
    assert (d.resolution, d.rationale) == (DE.ESCALATED, "TST-0003 also needs approval and is not part of this clash")
    assert cur.sql == []


def test_only_a_human_clash_is_considered(monkeypatch):
    _n(monkeypatch, 2)
    d = DE.decide_live_conflict(_Cur(), _lc(kind="auto"), ctx={}, now=NOW)
    assert (d.resolution, d.rationale) == (DE.ESCALATED, "only a clash that needs people can be decided on precedent")


def test_the_real_count_is_the_governed_fresh_value(monkeypatch):
    precedent_n(monkeypatch, 4)
    assert DE._precedent_count() == 4
    precedent_n(monkeypatch, None, missing=True)
    with pytest.raises(GL.LimitUnavailable):
        DE._precedent_count()
```

- [ ] **Step 2: Run them and confirm they fail.**
Run: `bptest tests/engines/test_decide_live_conflict.py -v`
Expected: FAIL (`AttributeError: module 'engines.decision_engine' has no attribute 'decide_live_conflict'`).

- [ ] **Step 3: Implement.** Append to `src/engines/decision_engine.py`:

```python
# ======================================================================================
# Live conflicts between agent policies: decided on precedent, or escalated
# (specs/2026-10-09-conflict-history-and-precedent-design.md §3.2, rulings R1-R3)
# ======================================================================================
LIVE_CONFLICT_SUBJECT = "live_conflict"
PRECEDENT_SOURCE = "proc.bp_agent_policy_conflict"

#: The last N settled live cases of this exact clash: the same pair key, the same version of every
#: involved policy, decided by a person. Precedent and every other automatic decision has
#: by_person false, so the engine can never reinforce itself.
PRECEDENT_SQL = """
    SELECT c.decision_id, c.outcome, c.decided_by, c.decided_at
      FROM proc.bp_agent_policy_conflict c
     WHERE c.kind = 'live' AND c.pair_key = %s AND NOT c.is_open AND c.by_person
       AND c.policy_versions = %s::jsonb
     ORDER BY c.decided_at DESC, c.decision_id DESC
     LIMIT %s
"""


def _precedent_count() -> Optional[int]:
    """The governed precedent count, read fresh. A seam; raises when it cannot be read."""
    from services.agent_policy import settings as agent_policy_settings

    return agent_policy_settings.precedent_count()


def _as_iso(value: Any) -> Any:
    return value.isoformat() if hasattr(value, "isoformat") else value


def _hit_version(hit: Dict[str, Any]) -> int:
    try:
        return int(hit.get("version") or 0)
    except (TypeError, ValueError):
        return 0


def decide_live_conflict(cur, lc: Any, *, ctx: Dict[str, Any], now: Any) -> Decision:
    """Decide a live clash between agent policies from its own history, or escalate it.

    Facts first (rule 1): the governed precedent count and this clash's settled cases, looked up
    here. Escalate rather than guess (rule 2): resolved only when the last N cases of this exact
    clash (same policies, same versions) were all decided the same way by people, and every
    approval the action needs is part of the clash. The action's values (ctx) play no part:
    precedent is about the clash, not the amount.

    `lc` is a conflict_engine.LiveConflict. Runs on the caller's cursor inside the caller's
    transaction and never writes. A failed lookup rolls back to its own savepoint and escalates,
    so it never poisons that transaction."""
    from services.agent_policy.conflict_detect import pair_key
    from services.agent_policy.conflict_payload import case_id
    from src.services.governed_limits import LimitUnavailable

    key = pair_key(*[str(h["id"]) for h in lc.involved])
    versions = {str(h["id"]): _hit_version(h) for h in lc.involved}
    base = {"pairKey": key, "versions": versions, "consultedAt": _as_iso(now)}

    def escalate(why: str, **more: Any) -> Decision:
        return Decision(subject_type=LIVE_CONFLICT_SUBJECT, subject_id=key, decision="escalate",
                        resolution=ESCALATED, rationale=why, facts={**base, **more})

    if lc.kind != "human":
        return escalate("only a clash that needs people can be decided on precedent")
    outside = [str(h["id"]) for h in lc.required if not any(h is x for x in lc.involved)]
    if outside:
        return escalate(f"{', '.join(outside)} also needs approval and is not part of this clash")
    try:
        n = _precedent_count()
    except (LimitUnavailable, TypeError, ValueError) as exc:
        logger.warning("precedent limit unavailable, the clash %s goes to people: %s", key, exc)
        return escalate("precedent limit unavailable")
    if not n or n <= 0:
        return escalate("precedent is switched off", precedentCount=n)

    cur.execute("SAVEPOINT live_conflict_precedent")
    try:
        cur.execute(PRECEDENT_SQL, (key, json.dumps(versions, sort_keys=True), int(n)))
        rows = cur.fetchall()
    except Exception as exc:  # noqa: BLE001 - a lookup that failed is doubt: people decide
        cur.execute("ROLLBACK TO SAVEPOINT live_conflict_precedent")
        cur.execute("RELEASE SAVEPOINT live_conflict_precedent")
        logger.warning("precedent lookup failed for %s: %s", key, type(exc).__name__)
        return escalate("precedent lookup failed", precedentCount=n)
    cur.execute("RELEASE SAVEPOINT live_conflict_precedent")

    if len(rows) < n:
        return escalate(f"only {len(rows)} of {n} decisions by people on this exact clash", precedentCount=n)
    outcomes = {r[1] for r in rows}
    if len(outcomes) != 1 or not outcomes <= {"approve", "reject"}:
        return escalate("decisions disagree", precedentCount=n)
    outcome = outcomes.pop()
    cited = [case_id(int(r[0])) for r in rows]
    evidence = [Evidence(fact="precedent",
                         value={"decision_id": int(r[0]), "outcome": r[1], "actioned_by": r[2],
                                "actioned_at": _as_iso(r[3])},
                         source=PRECEDENT_SOURCE, reference=case_id(int(r[0]))) for r in rows]
    return Decision(subject_type=LIVE_CONFLICT_SUBJECT, subject_id=key, decision=outcome, resolution=RESOLVED,
                    rationale=f"Decided the same way ({outcome}) {n} times before by people: {', '.join(cited)}.",
                    facts={**base, "precedentCount": n, "citedCases": cited}, evidence=evidence)
```

- [ ] **Step 4: Run and confirm they pass.**
Run: `bptest tests/engines -v` → PASS (the new file plus every existing engine test).

- [ ] **Step 5: Prove the guard fails.** Temporarily delete `AND c.by_person` from `PRECEDENT_SQL`. `test_the_lookup_counts_only_settled_person_decisions_at_identical_versions` must go red (the live self-reinforcement test in Task 7 also catches it). Restore it and capture green.

- [ ] **Step 6: Commit.**

```bash
cd "$BP" && git add src/engines/decision_engine.py tests/engines/test_decide_live_conflict.py
git commit -F - <<'EOF'
feat(decisions): the decision engine decides a live policy clash on clear precedent, else escalates

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

### Task 6: One history reader for the policy page, the Conflicts detail and the approval card

**Files:**
- Create: `src/services/agent_policy/conflict_history.py`
- Modify: `src/services/agent_policy/conflict_cases.py` (`history_for` through the reader; delete the now-unused `_action_dict`)
- Modify: `src/repositories/agent_policy_repo.py` (`get_policy(conn, policy_key, *, viewer=None)`)
- Modify: `src/services/agent_policy/conflict_views.py` (`view` adds `pairKey`; `get_conflict(conn, decision_id, principal, *, is_admin=False)` adds `conflictHistory`)
- Modify: `src/services/agent_policy/approval_views.py` (`conflict_block` adds `history`, `precedentNote`)
- Modify: `src/api/routers/agent_policies.py` (`get_one`, `get_conflict` pass the viewer)
- Modify (shape update): `tests/agent_policy/test_conflict_decide_live.py` (lines ~200–202 and ~381), `tests/agent_policy/test_conflict_endpoints_live.py` (lines ~347–352 and ~363)
- Test: `tests/agent_policy/test_conflict_history.py` (new, no DB), `tests/agent_policy/test_conflict_history_live.py` (new, live)

**Interfaces:**
- Consumes: `facts.decidedBy` (Task 4); `approval_views.overlap_example`, `mask_witness`, `mask_text`, `sensitive_for`; `deciders.eligible`, `deciders.load_map`.
- Produces (`conflict_history`):
  - `HISTORY_LIMIT = 1000`, `IN_CASE_LIMIT = 20`.
  - `Viewer(principal=None, is_admin=False, mapping={})` (frozen dataclass); `ANONYMOUS = Viewer()`.
  - `viewer(conn, principal, *, is_admin: bool) -> Viewer`.
  - `raw(cur, *, policy_key=None, pair_key=None, limit=HISTORY_LIMIT) -> List[Dict]`. Exactly one key must be given (`ValueError` otherwise). Entries are newest first: `{caseId, kind, isOpen, raisedAt, policies [{id, version}], example, decision {option, scope, decidedBy {kind, name}, decidedAt, reason} | None, citedCases [caseId], proposal, _owners, _deciders, _args}`.
  - `may_see(viewer: Viewer, entry) -> bool`: Admin, or linked to any owner or decider in the entry.
  - `shown(entries, *, sensitive: Set[str], unmasked_for: Callable[[Dict], bool]) -> List[Dict]`: strips the `_` keys; when masked, masks the example (`mask_witness`) and the reason (`mask_text` against `_args`).
  - `sensitive_of(cur, entries) -> Set[str]`.
  - `read(cur, *, policy_key=None, pair_key=None, viewer: Viewer, limit=HISTORY_LIMIT) -> List[Dict]`.
- Produces (others):
  - `conflict_cases.history_for(cur, policy_key, latest_version, status, *, viewer=None)`: `conflicts[]` = reader entries plus `otherPolicies`.
  - `repo.get_policy(conn, policy_key, *, viewer=None)`.
  - The conflict case view gains `pairKey`; `get_conflict` gains `conflictHistory`.
  - `approval_views.conflict_block(...)` output gains `history` (masked like the block) and `precedentNote`.

- [ ] **Step 1: Write the failing unit tests** `tests/agent_policy/test_conflict_history.py`:

```python
"""The history reader's masking and who may see what (no database)."""
from types import SimpleNamespace

import pytest

from services.agent_policy import conflict_history as CH
from services.agent_policy.enforcement import MASK

MAPPING = {"Owner A": {"groups": [], "emails": ["oa@x.test"]}, "Approver B": {"groups": ["G-B"], "emails": []}}


def _p(email="", groups=()):
    return SimpleNamespace(subject=f"sub-{email}", email=email, claims={"cognito:groups": list(groups)})


def _entry(reason="Paying the 900 refund is fine.", example=None):
    return {"caseId": "pc_9", "kind": "live", "isOpen": False, "raisedAt": "2026-10-09T10:00:00+00:00",
            "policies": [{"id": "TST-0001", "version": 1}, {"id": "TST-0002", "version": 1}],
            "example": example if example is not None else {"tool.name": "t", "args.amount": 900},
            "decision": {"option": "approve", "scope": "this_action", "decidedBy": {"kind": "person", "name": "sub-b"},
                         "decidedAt": "2026-10-09T11:00:00+00:00", "reason": reason},
            "citedCases": [], "proposal": None,
            "_owners": ["Owner A"], "_deciders": ["Approver B"], "_args": {"amount": 900}}


@pytest.mark.parametrize("principal,admin,expect", [
    (_p("oa@x.test"), False, True),                 # linked to an owner
    (_p("b@x.test", ["G-B"]), False, True),          # linked to a decider
    (_p("admin@x.test"), True, True),                # the Admin role
    (_p("stranger@x.test"), False, False),
    (None, False, False),
])
def test_who_may_see_full_values(principal, admin, expect):
    assert CH.may_see(CH.Viewer(principal, admin, MAPPING), _entry()) is expect


def test_a_stranger_reads_the_reason_masked_never_dropped():
    [e] = CH.shown([_entry()], sensitive={"args.amount"}, unmasked_for=lambda _e: False)
    assert e["decision"]["reason"] == f"Paying the {MASK} refund is fine."
    assert e["example"] == {"tool.name": "t", "args.amount": MASK}
    assert not any(k.startswith("_") for k in e)


def test_an_eligible_reader_sees_everything_and_the_input_is_never_changed():
    entry = _entry()
    [e] = CH.shown([entry], sensitive={"args.amount"}, unmasked_for=lambda _e: True)
    assert e["decision"]["reason"] == "Paying the 900 refund is fine." and e["example"]["args.amount"] == 900
    [masked] = CH.shown([entry], sensitive={"args.amount"}, unmasked_for=lambda _e: False)
    assert entry["decision"]["reason"] == "Paying the 900 refund is fine.", "shown() never edits its input"
    assert masked["decision"]["reason"] != entry["decision"]["reason"]


def test_masking_is_by_whole_token():
    entry = _entry(reason="Approved in 2900 cases; 900 is fine")
    [e] = CH.shown([entry], sensitive={"args.amount"}, unmasked_for=lambda _e: False)
    assert e["decision"]["reason"] == f"Approved in 2900 cases; {MASK} is fine"


def test_an_open_case_and_a_missing_reason_pass_through():
    entry = {**_entry(), "isOpen": True, "decision": None}
    [e] = CH.shown([entry], sensitive={"args.amount"}, unmasked_for=lambda _e: False)
    assert e["decision"] is None
    entry = _entry(reason=None)
    [e] = CH.shown([entry], sensitive={"args.amount"}, unmasked_for=lambda _e: False)
    assert e["decision"]["reason"] is None


def test_raw_needs_exactly_one_key():
    with pytest.raises(ValueError):
        CH.raw(object())
    with pytest.raises(ValueError):
        CH.raw(object(), policy_key="TST-0001", pair_key="TST-0001|TST-0002")
```

Write the failing live tests `tests/agent_policy/test_conflict_history_live.py`:

```python
"""One reader for every screen: complete, masked per reader (design §3.1). Needs PROCWISE_TEST_LIVE_DB=1."""
import os
from types import SimpleNamespace

import pytest

from services.agent_policy import conflict_cases as CC
from services.agent_policy import conflict_history as CH
from services.agent_policy.enforcement import MASK
from tests.agent_policy import test_conflict_live_gate as LG
from tests.agent_policy.test_conflict_endpoints_live import (  # noqa: F401 - fixtures
    NOW, STRANGER, _a, _as, _b, _c, _design_case, _gate, client, conn, world)

pytestmark = pytest.mark.skipif(os.getenv("PROCWISE_TEST_LIVE_DB") != "1", reason="live DB required")
REASON = "Paying the 900 refund is fine."


def _viewer(conn, w, name=None, *, admin=False):
    p = LG.who(w, name) if name else SimpleNamespace(subject="tst-stranger", email="stranger@example.test",
                                                     claims={"cognito:groups": []})
    return CH.viewer(conn, p, is_admin=admin)


def settled_live(conn, w, monkeypatch, reason=REASON):
    """A/B (approve, different sources and deciders, args.amount sensitive) paused on 900 and
    approved by both last-level deciders, B's with `reason`. Returns (a, b, live decision id)."""
    (a, da), (b, db) = _a(conn, w), _b(conn, w)
    _gate(monkeypatch, w, [da, db])
    [lv] = LG.lives(conn, w)
    ma, mb = LG.members(conn, w)
    LG.act(conn, w, ma["decision_id"], w.la2)
    LG.act(conn, w, mb["decision_id"], w.lb, reason=reason)
    return a, b, lv["decision_id"]


def _entry(conn, key, case, v):
    with conn.cursor() as cur:
        [e] = [e for e in CH.read(cur, policy_key=key, viewer=v) if e["caseId"] == f"pc_{case}"]
    return e


def test_owner_decider_and_admin_read_the_full_reason_a_stranger_reads_it_masked(conn, world, monkeypatch):
    a, b, live_id = settled_live(conn, world, monkeypatch)
    for v in (_viewer(conn, world, world.oa), _viewer(conn, world, world.lb), _viewer(conn, world, admin=True)):
        e = _entry(conn, a, live_id, v)
        assert e["decision"]["reason"] == REASON and e["example"]["args.amount"] == 900
    e = _entry(conn, a, live_id, _viewer(conn, world))
    assert e["decision"]["reason"] == f"Paying the {MASK} refund is fine."
    assert e["example"]["args.amount"] == MASK
    assert e["decision"]["decidedBy"] == {"kind": "person", "name": f"sub-{world.people[world.lb]}"}
    assert (e["kind"], e["isOpen"], e["decision"]["option"]) == ("live", False, "approve")
    assert e["policies"] == [{"id": k, "version": 1} for k in sorted([a, b])]
    assert not any(k.startswith("_") for k in e)


def test_the_policy_page_and_the_conflicts_detail_return_identical_entries(client, conn, world):
    a, c, did = _design_case(conn, world)
    CC.decide_policy(conn, did, principal=LG.who(world, world.oa), option=f"change:{a}", reason="Narrow it",
                     limit_text=None, now=NOW)
    for hdr in (_as(world, world.oa), STRANGER):
        page = client.get(f"/agent-policies/{a}", headers=hdr).json()["conflicts"]
        detail = client.get(f"/agent-policies/conflicts/{did}", headers=hdr).json()
        assert detail["pairKey"] == "|".join(sorted([a, c]))
        ids = {h["caseId"] for h in detail["conflictHistory"]}
        mine = [{k: v for k, v in e.items() if k != "otherPolicies"} for e in page if e["caseId"] in ids]
        assert mine and mine == detail["conflictHistory"]
        [one] = [e for e in page if e["caseId"] == f"pc_{did}"]
        assert one["otherPolicies"] == [c]
        assert one["decision"]["decidedBy"] == {"kind": "person", "name": f"sub-{world.email_a}"}


def test_the_reader_reports_the_kind_each_path_stored(conn, world):
    a, c, did = _design_case(conn, world)
    CC.close_moot(conn, c, now=NOW)
    with conn.cursor() as cur:
        [e] = [e for e in CH.raw(cur, policy_key=a) if e["caseId"] == f"pc_{did}"]
    assert e["decision"]["option"] == "moot"
    assert e["decision"]["decidedBy"] == {"kind": "retired", "name": "system:retired"}


def test_a_case_closed_before_the_reader_existed_reads_with_no_kind(conn, world):
    a, c, did = _design_case(conn, world)
    with conn.cursor() as cur:
        cur.execute("UPDATE proc.bp_agent_policy_conflict SET is_open = false, outcome = 'moot', "
                    "decided_by = 'legacy', decided_at = now() WHERE decision_id = %s", (did,))
        [e] = [e for e in CH.raw(cur, policy_key=a) if e["caseId"] == f"pc_{did}"]
    assert e["decision"]["option"] == "moot" and e["decision"]["decidedBy"] == {"kind": None, "name": "legacy"}


def test_history_is_newest_first_and_capped(conn, world, monkeypatch):
    a, c, did = _design_case(conn, world)
    _a2, b, live_id = settled_live(conn, world, monkeypatch)   # a second A/B pair; A is a new key
    with conn.cursor() as cur:
        both = CH.raw(cur, pair_key="|".join(sorted([_a2, b])))
        assert [e["caseId"] for e in both] == [f"pc_{live_id}"]
        one = CH.raw(cur, policy_key=c, limit=1)
    assert [e["caseId"] for e in one] == [f"pc_{did}"]
```

(`test_history_is_newest_first_and_capped`: `_a()` creates a fresh policy key on every call, so the second A is a different key from the design case's.)

- [ ] **Step 2: Run them and confirm they fail.**
Run: `bptest tests/agent_policy/test_conflict_history.py -v` → FAIL (`ModuleNotFoundError: services.agent_policy.conflict_history`).
Run: `LIVE=1 bptest tests/agent_policy/test_conflict_history_live.py -v` → FAIL.

- [ ] **Step 3: Create** `src/services/agent_policy/conflict_history.py`:

```python
"""Conflict history: every decision on a conflict between agent policies, complete and readable.

One reader for every screen and export -- the policy page, the Conflicts detail, the approval
card of a paused clash and the CSV -- so they can never disagree (the stage 4 D1 class of bug).
An entry is one conflict case, newest first:

    {caseId, kind (policy|live), isOpen, raisedAt, policies [{id, version}], example,
     decision {option, scope, decidedBy {kind, name}, decidedAt, reason} | None,
     citedCases [caseId], proposal}

Who sees what (design §3.1): full reasons and example values go to anyone linked (decider map)
to an owner or a decider of any of the case's policies, and to the Admin role. Everyone else
reads the same entry with sensitive values replaced by the mask, in the example and wherever the
reason quotes one of the action's values: the reason is masked, never dropped.

raw() reads (with the private _owners/_deciders/_args masking needs), shown() masks and strips
them, read() is both for one viewer. Nothing here writes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set

from services.agent_policy import conflict_payload, deciders

SUBJECT_POLICY = "policy_conflict"     # conflict_cases.SUBJECT_POLICY (not imported: it pulls in approvals)
SUBJECT_LIVE = "live_conflict"
HISTORY_LIMIT = 1000                   # the newest cases a page or an export reads
IN_CASE_LIMIT = 20                     # the newest cases copied into a paused clash for its approvers
_PRIVATE = ("_owners", "_deciders", "_args")


@dataclass(frozen=True)
class Viewer:
    principal: Any = None
    is_admin: bool = False
    mapping: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)


ANONYMOUS = Viewer()


def viewer(conn, principal, *, is_admin: bool) -> Viewer:
    return Viewer(principal=principal, is_admin=bool(is_admin), mapping=deciders.load_map(conn))


def _av():
    from services.agent_policy import approval_views   # lazily: approval_views reaches the gate
    return approval_views


def _j(v: Any, default=None):
    if v is None:
        return default
    if isinstance(v, (str, bytes)):
        return json.loads(v) if v else default
    return v


def _iso(v: Any) -> Any:
    return v.isoformat() if isinstance(v, datetime) else v


_CASES_SQL = """
    SELECT c.decision_id, c.kind, c.is_open, c.policy_versions, c.created_at, c.outcome, c.decided_by,
           c.decided_at, d.facts, d.evidence
      FROM proc.bp_agent_policy_conflict c
      JOIN proc.bp_decision d ON d.decision_id = c.decision_id
     WHERE {where}
     ORDER BY c.created_at DESC, c.decision_id DESC
     LIMIT %s
"""
# The row that closed each case: a person's or the system's action row, or a live record closed
# when it was written (block, standing rule, precedent). Open originals have no actioned_by.
_ACTIONS_SQL = """
    SELECT facts->>'caseId', decision, decision_scope, actioned_by, actioned_at, override_reason,
           facts->'decidedBy', evidence
      FROM proc.bp_decision
     WHERE subject_type IN (%s, %s) AND status = 'actioned' AND actioned_by IS NOT NULL
       AND facts->>'caseId' = ANY(%s)
     ORDER BY actioned_at, decision_id
"""


def raw(cur, *, policy_key: Optional[str] = None, pair_key: Optional[str] = None,
        limit: int = HISTORY_LIMIT) -> List[Dict[str, Any]]:
    """Every conflict case naming the policy, or of exactly this pair, newest first, unmasked."""
    if (policy_key is None) == (pair_key is None):
        raise ValueError("give exactly one of policy_key or pair_key")
    where, arg = (("c.policy_keys @> ARRAY[%s]::text[]", policy_key) if policy_key is not None
                  else ("c.pair_key = %s", pair_key))
    cur.execute(_CASES_SQL.format(where=where), (arg, int(limit)))
    cases = cur.fetchall()
    ids = [conflict_payload.case_id(r[0]) for r in cases]
    acts: Dict[str, tuple] = {}
    if ids:
        cur.execute(_ACTIONS_SQL, (SUBJECT_POLICY, SUBJECT_LIVE, ids))
        for row in cur.fetchall():
            acts[row[0]] = tuple(row[1:])          # oldest first: the latest closing row wins
    return [_entry(r, cid, acts.get(cid)) for r, cid in zip(cases, ids)]


def _entry(row, cid: str, act: Optional[tuple]) -> Dict[str, Any]:
    _did, kind, is_open, versions, created, outcome, by, at, facts, evidence = row
    facts = _j(facts, {}) or {}
    example = _av().overlap_example(evidence)
    pols = [p for p in facts.get("policies") or [] if isinstance(p, dict)]
    decision: Optional[Dict[str, Any]] = None
    cited: List[str] = []
    if act is not None:
        option, scope, actor, acted_at, reason, decided, act_evidence = act
        decided = _j(decided, {}) or {}
        decision = {"option": option, "scope": scope, "decidedBy": {"kind": decided.get("kind"), "name": actor},
                    "decidedAt": _iso(acted_at), "reason": reason}
        cited = [str(e["caseId"]) for e in _j(act_evidence, []) or []
                 if isinstance(e, dict) and e.get("kind") == "precedent" and e.get("caseId")]
    elif not is_open and outcome is not None:
        # closed with no closing row the reader knows (written before it existed): the index says
        decision = {"option": outcome, "scope": None, "decidedBy": {"kind": None, "name": by},
                    "decidedAt": _iso(at), "reason": None}
    if kind == "live":
        args = dict(((facts.get("action") or {}).get("args")) or {})
    else:
        args = {k[len("args."):]: v for k, v in example.items() if k.startswith("args.")}
    return {
        "caseId": cid, "kind": kind, "isOpen": bool(is_open), "raisedAt": _iso(created),
        "policies": [{"id": k, "version": int(v)} for k, v in sorted((_j(versions, {}) or {}).items())],
        "example": example, "decision": decision, "citedCases": cited, "proposal": facts.get("proposal"),
        "_owners": sorted({str(p.get("owner")).strip() for p in pols if str(p.get("owner") or "").strip()}),
        "_deciders": sorted({str(n).strip() for p in pols for n in p.get("deciders") or [] if str(n or "").strip()}),
        "_args": args,
    }


def may_see(v: Viewer, entry: Dict[str, Any]) -> bool:
    """Admin, or linked to an owner or a decider of any of the case's policies."""
    if v.is_admin:
        return True
    names = list(entry.get("_owners") or []) + list(entry.get("_deciders") or [])
    return any(deciders.eligible(v.principal, n, v.mapping) for n in names)


def shown(entries: List[Dict[str, Any]], *, sensitive: Set[str],
          unmasked_for: Callable[[Dict[str, Any]], bool]) -> List[Dict[str, Any]]:
    """The entries as a reader may see them: private keys gone, and for an entry the reader may
    not see in full, the example and the reason masked. Never edits its input."""
    av = _av()
    arg_names = {f[len("args."):] for f in sensitive if f.startswith("args.")}
    out = []
    for e in entries or []:
        if not isinstance(e, dict):
            continue
        view = {k: v for k, v in e.items() if k not in _PRIVATE}
        view["example"] = dict(e.get("example") or {})
        view["decision"] = dict(e["decision"]) if isinstance(e.get("decision"), dict) else None
        if not unmasked_for(e):
            view["example"] = av.mask_witness(view["example"], sensitive)
            d = view["decision"]
            if d and isinstance(d.get("reason"), str):
                d["reason"] = av.mask_text(d["reason"], dict(e.get("_args") or {}), arg_names)
        out.append(view)
    return out


def sensitive_of(cur, entries: List[Dict[str, Any]]) -> Set[str]:
    """Stage 3's union: every live policy's sensitive fields and those of the versions named."""
    pairs = [(p["id"], p["version"]) for e in entries for p in e.get("policies") or [] if p.get("id")]
    return _av().sensitive_for(cur, pairs)


def read(cur, *, policy_key: Optional[str] = None, pair_key: Optional[str] = None, viewer: Viewer,
         limit: int = HISTORY_LIMIT) -> List[Dict[str, Any]]:
    entries = raw(cur, policy_key=policy_key, pair_key=pair_key, limit=limit)
    return shown(entries, sensitive=sensitive_of(cur, entries), unmasked_for=lambda e: may_see(viewer, e))
```

- [ ] **Step 4: Route every screen through it.**

In `conflict_cases.py`, replace `history_for` and delete `_action_dict`:

```python
def history_for(cur, policy_key: str, latest_version: int, status: str, *, viewer=None) -> Dict[str, Any]:
    """A policy's conflicts through the one history reader, newest first, each also naming the
    other policies, and the change, limit or retire decision still waiting for the owner, if any.
    `viewer` (conflict_history.Viewer) decides what is masked; none reads as a stranger."""
    from services.agent_policy import conflict_history   # lazily: it reads approval_views

    entries = conflict_history.read(cur, policy_key=policy_key, viewer=viewer or conflict_history.ANONYMOUS)
    conflicts = [{**e, "otherPolicies": [p["id"] for p in e["policies"] if p["id"] != policy_key]}
                 for e in entries]
    return {"conflicts": conflicts, "pendingAction": _pending(cur, policy_key, latest_version, status)}
```

In `src/repositories/agent_policy_repo.py`:

```python
def get_policy(conn, policy_key: str, *, viewer=None) -> Dict[str, Any]:
```

and pass it on: `history = conflict_cases.history_for(cur, policy_key, head[2], head[0], viewer=viewer)`.

In `src/api/routers/agent_policies.py`, import `conflict_history` in the `services.agent_policy` import list. In `get_one`:

```python
    with _conn() as conn:
        try:
            got = repo.get_policy(conn, key, viewer=conflict_history.viewer(conn, p, is_admin=role == "Admin"))
```

In `get_conflict`:

```python
    role = _require(p, "Viewer", "agent_policy.read", {"conflict": decision_id})
    with _conn() as conn:
        got = conflict_views.get_conflict(conn, decision_id, p, is_admin=role == "Admin")
```

In `conflict_views.py`, import `conflict_history as CH`. In `view(...)`, add `"pairKey": case.get("subject_id"),` after `"decisionId"`. Replace `get_conflict`:

```python
def get_conflict(conn, decision_id: int, principal, *, is_admin: bool = False) -> Optional[Dict[str, Any]]:
    """One policy case with its history (every action row, oldest first) and the whole history of
    its pair through the one reader; None for a live case, an action row or an unknown id."""
    mapping = deciders.load_map(conn)
    with conn.cursor() as cur:
        cases = _select(cur, "AND d.decision_id = %s", (decision_id,), 1)
        if not cases:
            return None
        [out] = _views(cur, cases, principal, mapping)
        out["history"] = _actions(cur, cases).get(int(decision_id), [])
        out["conflictHistory"] = CH.read(cur, pair_key=str(cases[0]["subject_id"]),
                                         viewer=CH.Viewer(principal, bool(is_admin), mapping))
    return out
```

In `approval_views.conflict_block`, before `return`, add:

```python
    from services.agent_policy import conflict_history   # lazily: conflict_history reads this module
    history = conflict_history.shown(facts.get("history") or [], sensitive=sensitive,
                                     unmasked_for=lambda _e: unmasked)
```

and add to the returned dict: `"history": history, "precedentNote": (facts.get("precedent") or {}).get("why")`.

- [ ] **Step 5: Update the stage 4 tests that asserted the old entry shape** (the shape is now the spec's).

In `tests/agent_policy/test_conflict_decide_live.py::test_decision_stored_on_both_histories`, replace the three lines from `d = c["decision"]`:

```python
        d = c["decision"]
        assert d["option"] == f"keep_both:{b}" and d["scope"] == "standing_rule"
        assert d["decidedBy"] == {"kind": "person", "name": f"sub-{world.email_b}"}
        assert d["reason"] == "Customer terms win"
```

and, further down (the moot test), `c["decision"]["decision"] == "moot"` → `c["decision"]["option"] == "moot"`.

In `tests/agent_policy/test_conflict_endpoints_live.py::test_history_shows_a_settled_live_conflict_as_the_approvers_decision`, replace from `d = entry["decision"]` to the end of the test:

```python
    d = entry["decision"]
    assert (d["option"], d["decidedBy"]) == ("approve", {"kind": "person", "name": f"sub-{world.people[world.lb]}"})
    assert d["decidedAt"]
    # the decider's free text quotes the sensitive amount: masked for a stranger, never dropped (design §3.1)
    assert d["reason"] == f"Paying the {MASK} refund is fine."
    assert "Paying the 900" not in text
```

and in `test_history_shows_a_timed_out_live_conflict_as_a_system_reject`:

```python
    assert (entry["decision"]["option"], entry["decision"]["decidedBy"]) == \
        ("reject", {"kind": "timeout", "name": "system:timeout"})
```

- [ ] **Step 6: Run and confirm they pass.**
Run: `bptest tests/agent_policy/test_conflict_history.py -v` → PASS.
Run: `LIVE=1 bptest tests/agent_policy/test_conflict_history_live.py tests/agent_policy/test_conflict_decide_live.py tests/agent_policy/test_conflict_endpoints_live.py tests/agent_policy/test_conflict_cases_live.py tests/agent_policy/test_repo_live.py -v` → PASS.
Run: `bptest tests/agent_policy -q` → no new failures against the baseline taken before Task 1.

- [ ] **Step 7: Prove the guard fails.** In `shown`, temporarily change `if not unmasked_for(e):` to `if False:`. `test_owner_decider_and_admin_read_the_full_reason_a_stranger_reads_it_masked` and `test_a_stranger_reads_the_reason_masked_never_dropped` must go red. Restore it and capture green.

- [ ] **Step 8: Commit.**

```bash
cd "$BP" && git add src/services/agent_policy/conflict_history.py src/services/agent_policy/conflict_cases.py src/repositories/agent_policy_repo.py src/services/agent_policy/conflict_views.py src/services/agent_policy/approval_views.py src/api/routers/agent_policies.py tests/agent_policy/test_conflict_history.py tests/agent_policy/test_conflict_history_live.py tests/agent_policy/test_conflict_decide_live.py tests/agent_policy/test_conflict_endpoints_live.py
git commit -F - <<'EOF'
feat(agent-policy): one conflict history reader, masked per reader, for every screen

A decider's reason is now masked for readers who may not see the values, never dropped
(reverses the stage 4 final-review ruling, design §3.1).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

### Task 7: The gate consults precedent on a live clash

**Files:**
- Modify: `src/services/agent_policy/gate.py` (imports; `PRECEDENT_REFUSED`; `_lock_key`; `_consult_precedent`, `_record_precedent`, `_notify_precedent`, `_precedent_answer`; `_before_tool`; `_open_cases`)
- Modify: `tests/agent_policy/test_conflict_live_live.py` (`test_same_conflict_decided_same_way_5_times_raises_policy_case` tail; new proposal test)
- Modify: `tests/agent_policy/test_conflict_history_live.py` (append the approval-card history test)
- Test: `tests/agent_policy/test_conflict_precedent_live.py` (new, live)

**Interfaces:**
- Consumes: `decision_engine.decide_live_conflict`, `RESOLVED` (Task 5); `conflict_live.insert_live(..., extra_facts=, extra_evidence=)`, `conflict_live.PRECEDENT` (Task 4); `conflict_history.raw`, `IN_CASE_LIMIT` (Task 6); `conflict_detect.pair_key`.
- Produces:
  - `gate.PRECEDENT_REFUSED = "refused_on_precedent"`.
  - Precedent approve: `GateResult(allow=True, to_agent={"result": "allowed", "conflictCaseId": "pc_<id>", "precedent": True}, firing_ids=[...])`.
  - Precedent reject: `GateResult(allow=False, to_agent={"result": "blocked", "reasonCode": "refused_on_precedent", "reason": "<one sentence>", "conflictCaseId": "pc_<id>", "precedent": True}, firing_ids=[...])`.
  - On either, the live record is closed by `system:precedent`. Its firing rows record `allowed` | `blocked`, linked to the live case (`decision_id`). Notifications go to each involved policy's owner and deciders (link `agent-policy:<key>`), with no input values.
  - An escalated clash's open live record carries `facts.history` (raw, newest 20 cases of the pair) and `facts.precedent = {"why": <rationale>}`.

- [ ] **Step 1: Write the failing live tests** `tests/agent_policy/test_conflict_precedent_live.py`:

```python
"""A live clash decided on precedent at the gate (design §3.2). Needs PROCWISE_TEST_LIVE_DB=1.

Policies are injected into the gate (tool tst_<hex>, keys TST-<hex><letter>), so no live policy is
written to the shared database. N comes from the governed limit (fixtures.precedent_n). Every
case, conflict and notification row made here is removed by LG.cleanup; firing rows stay.
"""
import copy
import os

import pytest

from engines import decision_engine as DE
from services.agent_policy import approvals as A
from services.agent_policy import conflict_live as CL
from services.agent_policy import gate as G
from services.agent_policy import replay as R
from tests.agent_policy import test_conflict_live_gate as LG
from tests.agent_policy.fixtures import precedent_n
from tests.agent_policy.test_conflict_live_gate import conn, ran, world  # noqa: F401 - fixtures

pytestmark = pytest.mark.skipif(os.getenv("PROCWISE_TEST_LIVE_DB") != "1", reason="live DB required")
OWNER = "Chief Financial Officer"          # FORM_EXAMPLE's owner, on every test policy


def call(monkeypatch, w, docs, ran, *, wf=None):
    """One scripted agent call (amount 900), in its own workflow unless `wf` repeats one."""
    if wf is None:
        wf = f"{w.wf}-{len(w.wfs)}"
        w.wfs.append(wf)
    LG.use(monkeypatch, docs, w, ran)
    res, _ = LG.run(monkeypatch, LG.stub_tools(w, ran), [LG._round(w.tool), LG.FINAL], workflow_id=wf)
    return wf, res.calls[0].result


def members_of(conn, wf):
    return LG.rows(conn, "SELECT decision_id, policy_name FROM proc.bp_decision WHERE subject_type = %s "
                         "AND workflow_id = %s AND decision = 'approve_or_reject' ORDER BY decision_id",
                   (A.SUBJECT_TYPE, wf))


def live_of(conn, wf):
    return LG.rows(conn, "SELECT d.decision_id, d.decision, d.status, d.actioned_by, d.decision_scope, d.facts, "
                         "d.evidence, c.is_open, c.outcome, c.by_person FROM proc.bp_decision d "
                         "JOIN proc.bp_agent_policy_conflict c USING (decision_id) "
                         "WHERE d.subject_type = %s AND d.workflow_id = %s ORDER BY d.decision_id",
                   (CL.SUBJECT_LIVE, wf))


def firings_of(conn, wf):
    return LG.rows(conn, "SELECT firing_id, policy_key, result, decision_id, reason FROM proc.bp_policy_firing "
                         "WHERE workflow_id = %s ORDER BY firing_id", (wf,))


def decided(conn, monkeypatch, w, docs, ran, verb="approve", reason=None):
    """A clash paused for people and settled by them: approve = both members, reject = the first."""
    wf, out = call(monkeypatch, w, docs, ran)
    assert out["result"] == "paused_for_approval", out
    last = {w.key("A"): w.la2, w.key("B"): w.lb}
    ms = members_of(conn, wf)
    if verb == "approve":
        for m in ms:
            LG.act(conn, w, m["decision_id"], last[m["policy_name"]], reason=reason)
    else:
        LG.act(conn, w, ms[0]["decision_id"], last[ms[0]["policy_name"]], verb="reject", reason=reason or "No.")
    [lv] = live_of(conn, wf)
    return lv["decision_id"]


def _pair(w, **kw):
    return [LG.doc_a(w, **kw), LG.doc_b(w, **kw)]


def test_precedent_approves_after_n_person_approvals_and_the_tool_runs(conn, world, monkeypatch, ran):
    precedent_n(monkeypatch, 2)
    docs = _pair(world)
    first = decided(conn, monkeypatch, world, docs, ran)
    second = decided(conn, monkeypatch, world, docs, ran)
    assert ran == []
    wf, out = call(monkeypatch, world, docs, ran)
    assert ran == [LG.ARGS], "the tool runs now, with no person approving"
    assert out == {"refunded": 900}
    [lv] = live_of(conn, wf)
    assert (lv["status"], lv["decision"], lv["actioned_by"], lv["decision_scope"]) == \
           ("actioned", "approve", "system:precedent", "this_action")
    assert (lv["is_open"], lv["outcome"], lv["by_person"]) == (False, "approve", False)
    assert lv["facts"]["decidedBy"] == {"kind": "precedent", "name": "system:precedent"}
    cited = [e for e in lv["evidence"] if e.get("kind") == "precedent"]
    assert [e["caseId"] for e in cited] == [f"pc_{second}", f"pc_{first}"]
    assert {e["outcome"] for e in cited} == {"approve"} and all(e["actioned_by"].startswith("sub-") for e in cited)
    assert members_of(conn, wf) == []
    fs = firings_of(conn, wf)
    assert {f["result"] for f in fs} == {"allowed"} and {f["decision_id"] for f in fs} == {lv["decision_id"]}
    notes = LG.notes(conn, [f["firing_id"] for f in fs])
    assert {n["recipient"] for n in notes} == {OWNER, world.la1, world.la2, world.lb}
    assert all("ran on precedent" in n["message"] and "900" not in n["message"] for n in notes)


def test_the_agent_is_told_it_ran_on_precedent(conn, world, monkeypatch, ran):
    precedent_n(monkeypatch, 1)
    docs = _pair(world)
    decided(conn, monkeypatch, world, docs, ran)
    wf = f"{world.wf}-direct"
    world.wfs.append(wf)
    LG.use(monkeypatch, docs)
    res = G.before_tool(tool_name=world.tool, args=dict(LG.ARGS), agent="overcharge_hunter", reason=LG.REASON,
                        workflow_id=wf, user_id="req@example.test")
    [lv] = live_of(conn, wf)
    assert res.allow is True
    assert res.to_agent == {"result": "allowed", "conflictCaseId": f"pc_{lv['decision_id']}", "precedent": True}


def test_precedent_rejects_after_n_person_rejects(conn, world, monkeypatch, ran):
    precedent_n(monkeypatch, 2)
    docs = _pair(world)
    decided(conn, monkeypatch, world, docs, ran, verb="reject")
    decided(conn, monkeypatch, world, docs, ran, verb="reject")
    wf, out = call(monkeypatch, world, docs, ran)
    [lv] = live_of(conn, wf)
    assert ran == []
    assert out["result"] == "blocked" and out["reasonCode"] == "refused_on_precedent" and out["precedent"] is True
    assert out["conflictCaseId"] == f"pc_{lv['decision_id']}" and "refused on precedent" in out["reason"]
    assert (lv["decision"], lv["actioned_by"], lv["facts"]["decidedBy"]["kind"]) == ("reject", "system:precedent",
                                                                                      "precedent")
    fs = firings_of(conn, wf)
    assert {f["result"] for f in fs} == {"blocked"}
    assert all(f["reason"].startswith("refused_on_precedent") for f in fs)
    assert all("was refused on precedent" in n["message"] for n in LG.notes(conn, [f["firing_id"] for f in fs]))


def _escalated(conn, monkeypatch, w, docs, ran):
    wf, out = call(monkeypatch, w, docs, ran)
    assert out["result"] == "paused_for_approval" and ran == []
    [lv] = live_of(conn, wf)
    return lv


def test_below_n_goes_to_people(conn, world, monkeypatch, ran):
    precedent_n(monkeypatch, 3)
    docs = _pair(world)
    decided(conn, monkeypatch, world, docs, ran)
    decided(conn, monkeypatch, world, docs, ran)
    lv = _escalated(conn, monkeypatch, world, docs, ran)
    assert lv["facts"]["precedent"] == {"why": "only 2 of 3 decisions by people on this exact clash"}


def test_mixed_outcomes_go_to_people(conn, world, monkeypatch, ran):
    precedent_n(monkeypatch, 2)
    docs = _pair(world)
    decided(conn, monkeypatch, world, docs, ran)
    decided(conn, monkeypatch, world, docs, ran, verb="reject")
    lv = _escalated(conn, monkeypatch, world, docs, ran)
    assert lv["facts"]["precedent"] == {"why": "decisions disagree"}
    assert len(lv["facts"]["history"]) == 2, "the people deciding see what came before"


def test_a_new_policy_version_goes_to_people(conn, world, monkeypatch, ran):
    precedent_n(monkeypatch, 2)
    docs = _pair(world)
    decided(conn, monkeypatch, world, docs, ran)
    decided(conn, monkeypatch, world, docs, ran)
    newer = copy.deepcopy(docs)
    newer[0]["version"] = 2
    lv = _escalated(conn, monkeypatch, world, newer, ran)
    assert lv["facts"]["precedent"] == {"why": "only 0 of 2 decisions by people on this exact clash"}


def test_precedent_never_counts_toward_a_later_precedent(conn, world, monkeypatch, ran):
    precedent_n(monkeypatch, 2)
    docs = _pair(world)
    decided(conn, monkeypatch, world, docs, ran)
    decided(conn, monkeypatch, world, docs, ran)
    for _ in range(2):
        assert call(monkeypatch, world, docs, ran)[1] == {"refunded": 900}
    precedent_n(monkeypatch, 3)
    lv = _escalated(conn, monkeypatch, world, docs, ran)
    assert lv["facts"]["precedent"] == {"why": "only 2 of 3 decisions by people on this exact clash"}


@pytest.mark.parametrize("n,missing,why", [(0, False, "precedent is switched off"),
                                           (None, True, "precedent limit unavailable")])
def test_off_or_unreadable_goes_to_people(conn, world, monkeypatch, ran, n, missing, why):
    precedent_n(monkeypatch, n, missing=missing)
    lv = _escalated(conn, monkeypatch, world, _pair(world), ran)
    assert lv["facts"]["precedent"] == {"why": why}


def test_an_edit_takes_effect_without_a_restart(conn, world, monkeypatch, ran):
    precedent_n(monkeypatch, 3)
    docs = _pair(world)
    decided(conn, monkeypatch, world, docs, ran)
    decided(conn, monkeypatch, world, docs, ran)
    _escalated(conn, monkeypatch, world, docs, ran)
    precedent_n(monkeypatch, 2)                    # the row is edited; nothing restarts
    assert call(monkeypatch, world, docs, ran)[1] == {"refunded": 900}


def test_a_precedent_path_failure_refuses_and_leaves_nothing(conn, world, monkeypatch, ran):
    precedent_n(monkeypatch, 1)
    docs = _pair(world)
    decided(conn, monkeypatch, world, docs, ran)

    def boom(*a, **k):
        raise RuntimeError("notification store down")
    monkeypatch.setattr(G, "_notify_precedent", boom)
    wf, out = call(monkeypatch, world, docs, ran)
    assert out == G.UNAVAILABLE and ran == []
    assert live_of(conn, wf) == [] and members_of(conn, wf) == []
    assert [f["policy_key"] for f in firings_of(conn, wf)] == ["*"], "only the refusal is logged"


def test_a_failed_precedent_lookup_goes_to_people(conn, world, monkeypatch, ran):
    precedent_n(monkeypatch, 1)
    docs = _pair(world)
    decided(conn, monkeypatch, world, docs, ran)
    monkeypatch.setattr(DE, "PRECEDENT_SQL", "SELECT no_such_column FROM proc.bp_agent_policy_conflict "
                                             "WHERE %s IS NOT NULL AND %s IS NOT NULL LIMIT %s")
    lv = _escalated(conn, monkeypatch, world, docs, ran)
    assert lv["facts"]["precedent"] == {"why": "precedent lookup failed"}, "the gate's transaction survived"


def test_a_repeat_call_with_open_member_cases_does_not_consult_precedent(conn, world, monkeypatch, ran):
    precedent_n(monkeypatch, 3)
    docs = _pair(world)
    decided(conn, monkeypatch, world, docs, ran)
    decided(conn, monkeypatch, world, docs, ran)
    wf, first = call(monkeypatch, world, docs, ran)
    assert first["result"] == "paused_for_approval"
    precedent_n(monkeypatch, 2)                    # precedent would now decide a NEW call
    _wf, again = call(monkeypatch, world, docs, ran, wf=wf)
    assert again == first and ran == []
    assert len(live_of(conn, wf)) == 1, "the repeat reuses the paused clash"


def test_the_replay_recheck_never_consults_precedent(conn, world, monkeypatch, ran):
    docs = _pair(world)
    wf, out = call(monkeypatch, world, docs, ran)
    assert out["result"] == "paused_for_approval"
    monkeypatch.setattr(DE, "decide_live_conflict", lambda *a, **k: pytest.fail("the replay consulted precedent"))
    replay = lambda d: R.run(d, agent_nick=object())   # noqa: E731
    last = {world.key("A"): world.la2, world.key("B"): world.lb}
    for m in members_of(conn, wf):
        LG.act(conn, world, m["decision_id"], last[m["policy_name"]], replay=replay)
    assert ran == [LG.ARGS]


def test_a_block_still_wins_and_precedent_is_not_consulted(conn, world, monkeypatch, ran):
    monkeypatch.setattr(DE, "decide_live_conflict", lambda *a, **k: pytest.fail("precedent consulted on a block"))
    wf, out = call(monkeypatch, world, [LG.doc_a(world), LG.doc_b(world, outcome="block")], ran)
    assert out["result"] == "blocked" and ran == []
    [lv] = live_of(conn, wf)
    assert lv["facts"]["decidedBy"]["kind"] == "block"


def test_an_unreadable_block_beats_precedent(conn, world, monkeypatch, ran):
    precedent_n(monkeypatch, 1)
    docs = _pair(world)
    decided(conn, monkeypatch, world, docs, ran)
    unreadable = LG.make_doc(world.key("U"), tool=world.tool, outcome="block", source=f"TST Legal {world.tag}")
    unreadable["trigger"]["condition"] = {"all": [{"field": "args.amount", "op": "between", "value": "x"}]}
    monkeypatch.setattr(DE, "decide_live_conflict", lambda *a, **k: pytest.fail("precedent consulted on a block"))
    _wf, out = call(monkeypatch, world, docs + [unreadable], ran)
    assert out["result"] == "blocked" and ran == []


def test_precedent_never_approves_for_a_policy_outside_the_clash(conn, world, monkeypatch, ran):
    precedent_n(monkeypatch, 1)
    docs = _pair(world)
    decided(conn, monkeypatch, world, docs, ran)
    # same source as A (tiered, no clash with A) and the same decider as B (no clash with B)
    third = LG.make_doc(world.key("T"), tool=world.tool, outcome="approve", deciders=[world.lb],
                        source=f"TST Finance {world.tag}")
    lv = _escalated(conn, monkeypatch, world, docs + [third], ran)
    assert lv["facts"]["precedent"] == {"why": f"{world.key('T')} also needs approval and is not part of this clash"}
```

Append to `tests/agent_policy/test_conflict_history_live.py` (Task 6's file). It uses the endpoint world, because masking reads each policy version's sensitive inputs from the database, and the injected-only documents of `test_conflict_live_gate` are not there. Add `from services.agent_policy import approval_views as AV`, `from services.agent_policy import approvals as A` and `from tests.agent_policy.fixtures import precedent_n` to its imports:

```python
def test_the_paused_clash_card_carries_its_history_masked_per_approver(conn, world, monkeypatch):
    precedent_n(monkeypatch, 3)
    (a, da), (b, db) = _a(conn, world), _b(conn, world)          # args.amount is sensitive in both
    last = {a: world.la2, b: world.lb}

    def paused(i):
        wf = f"{world.wf}-h{i}"
        world.wfs.append(wf)
        LG.use(monkeypatch, [da, db])
        res, _ = LG.run(monkeypatch, LG.stub_tools(world, []), [LG._round(world.tool), LG.FINAL], workflow_id=wf)
        assert res.calls[0].result["result"] == "paused_for_approval"
        return LG.rows(conn, "SELECT decision_id, policy_name FROM proc.bp_decision WHERE subject_type = %s "
                             "AND workflow_id = %s AND decision = 'approve_or_reject' ORDER BY decision_id",
                       (A.SUBJECT_TYPE, wf))

    for i in range(2):
        for m in paused(i):
            LG.act(conn, world, m["decision_id"], last[m["policy_name"]], reason=REASON)
    [ma] = [m for m in paused(2) if m["policy_name"] == a]
    stranger = SimpleNamespace(subject="tst-stranger", email="s@example.test", claims={"cognito:groups": []})
    seen = AV.get_case(conn, ma["decision_id"], stranger, is_admin=True)["conflict"]
    assert [h["decision"]["reason"] for h in seen["history"]] == [f"Paying the {MASK} refund is fine."] * 2
    assert all(h["example"]["args.amount"] == MASK for h in seen["history"])
    assert seen["precedentNote"] == "only 2 of 3 decisions by people on this exact clash"
    mine = AV.get_case(conn, ma["decision_id"], LG.who(world, world.la2), is_admin=False)["conflict"]
    assert [h["decision"]["reason"] for h in mine["history"]] == [REASON] * 2
    assert not any(k.startswith("_") for h in mine["history"] for k in h)
```

(`settle_for_group` credits the last approver's reason (`_credited`); both approvers give `REASON` here.)

In `tests/agent_policy/test_conflict_live_live.py`, replace the last two lines of `test_same_conflict_decided_same_way_5_times_raises_policy_case` (from the second `once(conn, world, monkeypatch)` on) with:

```python
    world.n += 1
    wf = f"{world.wf}-{world.n}"
    world.wfs.append(wf)
    T.use(monkeypatch, [world.docs["A"], world.docs["B"]])
    res, _ = T.run(monkeypatch, T.stub_tools(world, []), [T._round(world.tool), T.FINAL], workflow_id=wf)
    assert res.calls[0].result == {"refunded": 900}, "the 6th identical clash runs on precedent (ruling R2)"
    assert len(repeat_cases(conn, world)) == 1, "a 6th raises no second case while one is open"
```

and add:

```python
def test_precedent_decisions_never_count_toward_the_proposal(conn, world, monkeypatch):
    F.precedent_n(monkeypatch, 2)
    once(conn, world, monkeypatch)
    once(conn, world, monkeypatch)
    assert len(repeat_cases(conn, world)) == 1
    proposed = []
    monkeypatch.setattr(CL, "maybe_propose", lambda *a, **k: proposed.append(a) or [])
    for _ in range(2):
        world.n += 1
        wf = f"{world.wf}-{world.n}"
        world.wfs.append(wf)
        T.use(monkeypatch, [world.docs["A"], world.docs["B"]])
        res, _ = T.run(monkeypatch, T.stub_tools(world, []), [T._round(world.tool), T.FINAL], workflow_id=wf)
        assert res.calls[0].result == {"refunded": 900}
    assert proposed == [] and len(repeat_cases(conn, world)) == 1
```


- [ ] **Step 2: Run them and confirm they fail.**
Run: `LIVE=1 bptest tests/agent_policy/test_conflict_precedent_live.py -v`
Expected: FAIL. The precedent tests still pause; `facts["precedent"]` raises `KeyError`; `G._notify_precedent` does not exist.

- [ ] **Step 3: Implement in `gate.py`.** Imports:

```python
from engines import decision_engine as DE
from services.agent_policy import (approvals, conflict_detect, conflict_engine, conflict_history, conflict_live,
                                   conflict_payload, deciders, enforcement, live_policies, settings)
```

Constants and helpers (after `UNAVAILABLE`):

```python
PRECEDENT_REFUSED = "refused_on_precedent"
```

```python
def _lock_key(tool_name: str, digest: str, workflow_id, user_id) -> str:
    """The per-call advisory lock: two identical calls (tool, args, workflow, requester) serialise."""
    return f"agent_policy_gate:{tool_name}:{digest}:{workflow_id}:{user_id}"
```

In `_open_cases`, replace the line `lock_key = f"agent_policy_gate:{tool_name}:{digest}:{workflow_id}:{user_id}"` with `lock_key = _lock_key(tool_name, digest, workflow_id, user_id)` (the same string, so the byte-for-byte guard holds). Add the parameter `note: Optional[str] = None` after `live=None`. In the `if lc.kind == "human":` branch, record the history:

```python
                if lc.kind == "human":
                    extra_live: Dict[str, Any] = {"history": conflict_history.raw(
                        cur, pair_key=conflict_detect.pair_key(*[str(h["id"]) for h in lc.involved]),
                        limit=conflict_history.IN_CASE_LIMIT)}
                    if note:
                        extra_live["precedent"] = {"why": note}
                    live_id = conflict_live.insert_live(cur, lc, ctx=ctx, action=action, now=now,
                                                        default_response_time=default_rt, status="open",
                                                        extra_facts=extra_live)
```

Add the precedent functions (after `_record_block`):

```python
def _consult_precedent(conn, lc, *, ctx, digest, tool_name, workflow_id, user_id, now):
    """The decision engine on a 'human' clash, under the call's own lock (inside the gate's
    transaction). None when this call repeats one whose member cases are still open: that call is
    already with people, and precedent is not consulted again."""
    with conn.cursor() as cur:
        cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (_lock_key(tool_name, digest, workflow_id, user_id),))
        if any(_open_case_for(cur, key=str(h["id"]), tool_name=tool_name, digest=digest,
                              workflow_id=workflow_id, requested_by=user_id) for h in lc.required):
            return None
        return DE.decide_live_conflict(cur, lc, ctx=ctx, now=now)


def _notify_precedent(cur, lc, firing_of, decision, case: str, tool_name: str) -> None:
    """Each involved policy's owner and deciders hear what precedent did. No input values."""
    verb = "ran on precedent" if decision.decision == "approve" else "was refused on precedent"
    n = len(decision.evidence)
    for hit in sorted(lc.involved, key=lambda h: str(h["id"])):
        policy = hit["policy"]
        names: List[str] = []
        for name in [str(policy.get("owner") or "").strip(), *_level_names(policy)]:
            if name and name not in names:
                names.append(name)
        _notify(cur, firing_of[id(hit)], names,
                f"{_what(policy, tool_name)} (policy {hit['id']}) {verb}: decided the same way {n} times "
                f"before ({case}).", f"agent-policy:{hit['id']}")


def _record_precedent(conn, lc, verdict, matched, decision, *, action, ctx, tool_name, agent, workflow_id,
                      user_id, duration_ms, now):
    """A clash the engine decided on precedent, in the gate's transaction: the closed live record
    (system:precedent, citing its cases), the firing rows linked to it, and the notifications.
    Returns (live_id, firing_ids)."""
    approve = decision.decision == "approve"
    cited = [{"kind": "precedent", "caseId": e.reference, "source": e.source, **dict(e.value or {})}
             for e in decision.evidence]
    with conn.cursor() as cur:
        live_id = conflict_live.insert_live(cur, lc, ctx=ctx, action=action, now=now,
                                            default_response_time=_default_response_time(), status="actioned",
                                            decision=decision.decision, actor=conflict_live.PRECEDENT,
                                            reason=decision.rationale, extra_evidence=cited)
        case = conflict_payload.case_id(live_id)
        firing_of = record_matches(cur, matched, verdict.notifies, result="allowed" if approve else "blocked",
                                   tool_name=tool_name, agent=agent, workflow_id=workflow_id, user_id=user_id,
                                   duration_ms=duration_ms,
                                   reason=(f"Decided on precedent ({case})" if approve
                                           else f"{PRECEDENT_REFUSED}: decided on precedent ({case})"))
        fids = [firing_of[id(h)] for h in matched]
        cur.execute("UPDATE proc.bp_policy_firing SET decision_id = %s WHERE firing_id = ANY(%s)", (live_id, fids))
        _notify_precedent(cur, lc, firing_of, decision, case, tool_name)
    return live_id, fids


def _precedent_answer(decision, live_id: int, firing_ids: List[int]) -> GateResult:
    case = conflict_payload.case_id(live_id)
    if decision.decision == "approve":
        return GateResult(allow=True, to_agent={"result": "allowed", "conflictCaseId": case, "precedent": True},
                          firing_ids=firing_ids)
    n = len(decision.evidence)
    return GateResult(allow=False, firing_ids=firing_ids, to_agent={
        "result": "blocked", "reasonCode": PRECEDENT_REFUSED,
        "reason": f"People refused this same action {n} times before, so it was refused on precedent ({case}).",
        "conflictCaseId": case, "precedent": True})
```

Replace the transaction block and the tail of `_before_tool` (everything from `live: Dict[str, Optional[int]] = {"id": None}` to the end):

```python
    live: Dict[str, Optional[int]] = {"id": None}
    precedent = None          # the engine's Decision when it resolved the clash on precedent
    note: Optional[str] = None  # why precedent did not decide a 'human' clash

    # One transaction for every row this call writes (firing rows, notifications, cases): on any
    # failure nothing is left behind -- never an open case for a call the agent was refused.
    with _connect() as conn:
        with approvals._tx(conn):
            if lc.kind == "human" and verdict.result == "paused_for_approval":
                consulted = _consult_precedent(conn, lc, ctx=ctx, digest=digest, tool_name=tool_name,
                                               workflow_id=workflow_id, user_id=user_id, now=now)
                if consulted is not None and consulted.resolution == DE.RESOLVED:
                    precedent = consulted
                elif consulted is not None:
                    note = consulted.rationale
            if precedent is not None:
                live["id"], firing_ids = _record_precedent(
                    conn, lc, verdict, matched, precedent, action=action, ctx=ctx, tool_name=tool_name,
                    agent=agent, workflow_id=workflow_id, user_id=user_id, duration_ms=duration_ms, now=now)
            else:
                with conn.cursor() as cur:
                    firing_of = record_matches(cur, matched, verdict.notifies, result=verdict.result,
                                               tool_name=tool_name, agent=agent, workflow_id=workflow_id,
                                               user_id=user_id, duration_ms=duration_ms)
                firing_ids.extend(firing_of[id(h)] for h in matched)

                if lc.kind == "block_record":
                    live["id"] = _record_block(conn, lc, action=action, ctx=ctx, now=now)

                if verdict.result == "paused_for_approval":
                    case_ids = _open_cases(conn, verdict, firing_of, action=action, ctx=ctx, digest=digest,
                                           tool_name=tool_name, workflow_id=workflow_id, user_id=user_id,
                                           now=now, **({"lc": lc, "live": live, "note": note} if lc.kind else {}))
                    # A notify row of a paused call is itself paused and linked to the group's first
                    # case; approvals._close_group settles it once the whole group is decided.
                    if case_ids:
                        with conn.cursor() as cur:
                            link_paused_notifies(cur, [firing_of[id(h)] for h in verdict.notifies], case_ids[0])

    if precedent is not None:
        return _precedent_answer(precedent, live["id"], firing_ids)
    if verdict.result == "allowed":
        return GateResult(allow=True, firing_ids=firing_ids)
    to_agent = dict(verdict.to_agent or {})
    if verdict.result == "paused_for_approval":
        to_agent["requestIds"] = list(case_ids)
    if live["id"] is not None:
        to_agent["conflictCaseId"] = conflict_payload.case_id(live["id"])
    return GateResult(allow=False, to_agent=to_agent, firing_ids=firing_ids, case_ids=case_ids)
```

Update the module docstring's "Live conflicts (stage 4)" list with a line for precedent: "human, on a fresh call: the decision engine (decision_engine.decide_live_conflict) may decide it on precedent: the action runs, or is refused with refused_on_precedent; otherwise it goes to people as below, with the clash's history attached."

- [ ] **Step 4: Run and confirm they pass.**
Run: `LIVE=1 bptest tests/agent_policy/test_conflict_precedent_live.py tests/agent_policy/test_conflict_history_live.py tests/agent_policy/test_conflict_live_live.py tests/agent_policy/test_conflict_live_gate.py tests/agent_policy/test_conflict_endpoints_live.py tests/agent_policy/test_conflict_decided_by_live.py -v` → PASS.
Run: `bptest "tests/agent_policy/test_conflict_live_gate.py::test_no_live_conflict_is_stage3_byte_for_byte" "tests/agent_policy/test_conflict_live_gate.py::test_insert_case_default_is_stage3_byte_for_byte" -v` → PASS (6 + 1).

- [ ] **Step 5: Prove the guards fail.**
- Temporarily delete `AND c.by_person` from `DE.PRECEDENT_SQL`. `test_precedent_never_counts_toward_a_later_precedent` must go red: the two precedent rows are counted, and the call runs. Restore.
- Temporarily drop `and verdict.result == "paused_for_approval"` from `_before_tool`. `test_an_unreadable_block_beats_precedent` must go red. Restore.
- Capture green after both restores.

- [ ] **Step 6: Commit.**

```bash
cd "$BP" && git add src/services/agent_policy/gate.py tests/agent_policy/test_conflict_precedent_live.py tests/agent_policy/test_conflict_live_live.py tests/agent_policy/test_conflict_history_live.py
git commit -F - <<'EOF'
feat(agent-policy): the gate lets the decision engine decide a live clash on precedent

N agreeing decisions by people on the identical clash (same policies, same versions) decide
it: approve runs the tool now, reject refuses it (refused_on_precedent). Anything less goes
to people with the clash's history attached. A block and a standing rule still come first.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

### Task 8: Export the conflict history as CSV

**Files:**
- Modify: `src/services/agent_policy/conflict_history.py` (`HEADER`, `KIND_WORDS`, `DECIDED_BY_WORDS`, `decision_words`, `csv_cell`, `to_csv`)
- Modify: `src/api/routers/agent_policies.py` (two routes, before `/conflicts/{decision_id}` and `/{key}/firings`)
- Test: `tests/agent_policy/test_conflict_history.py` (CSV unit tests), `tests/agent_policy/test_conflict_export_live.py` (new, live)

**Interfaces:**
- Consumes: `conflict_history.read`, `viewer` (Task 6); `conflict_detect.pair_key`; `conflict_cases.option_label`.
- Produces:
  - `conflict_history.HEADER = ("Case", "Kind", "Raised", "Policies", "Decided by", "Name", "Decision", "Scope", "Decided at", "Reason", "Cited cases")`.
  - `conflict_history.csv_cell(v) -> str`: quoted; neutralised like the UI's `csvCell`.
  - `conflict_history.to_csv(entries) -> str`: CRLF lines, header first, one row per case, trailing CRLF.
  - `GET /agent-policies/{key}/conflicts/history.csv` → 200 `text/csv; charset=utf-8`, `Content-Disposition: attachment; filename="conflict-history-<KEY>.csv"`. 404 for a non-key or an unknown policy.
  - `GET /agent-policies/conflicts/history.csv?pair=<K1|K2[|…]>` → the same, for the canonical (sorted, de-duplicated) pair; filename `conflict-history-<K1>_<K2>.csv`. 422 when `pair` is not two or more policy ids joined by `|`.
  - Both routes: Viewer floor (`_require(p, "Viewer", "agent_policy.read", ...)`); masking per caller as `read`.

- [ ] **Step 1: Write the failing tests.** Append to `tests/agent_policy/test_conflict_history.py`:

```python
# ---------------------------------------------------------------------------- CSV
@pytest.mark.parametrize("v", ["=1+1", "+1", "-1", "@SUM(A1)", "\t=1", "\r=1", "  =1+1", "\tfoo", "\rbar"])
def test_csv_cell_neutralises_formulas_like_the_ui(v):
    assert CH.csv_cell(v).strip('"').startswith("'")


def test_csv_cell_quotes_commas_and_quotes_and_blanks_none():
    assert CH.csv_cell('a,"b"') == '"a,""b"""'
    assert CH.csv_cell(None) == '""'


def test_to_csv_has_the_header_and_one_row_per_case():
    open_case = {**_entry(), "caseId": "pc_10", "kind": "policy", "isOpen": True, "decision": None}
    precedent = {**_entry(), "caseId": "pc_11", "citedCases": ["pc_9", "pc_8"],
                 "decision": {"option": "reject", "scope": "this_action",
                              "decidedBy": {"kind": "precedent", "name": "system:precedent"},
                              "decidedAt": "2026-10-09T12:00:00+00:00", "reason": "=cmd|' /C calc'!A0"}}
    text = CH.to_csv([precedent, open_case, _entry()])
    lines = text.split("\r\n")
    assert text.endswith("\r\n") and lines[-1] == ""
    assert lines[0] == '"Case","Kind","Raised","Policies","Decided by","Name","Decision","Scope","Decided at","Reason","Cited cases"'
    assert lines[1] == ('"pc_11","During an action","2026-10-09T10:00:00+00:00","TST-0001 v1; TST-0002 v1",'
                        '"Precedent","system:precedent","Reject","this_action","2026-10-09T12:00:00+00:00",'
                        '"\'=cmd|\' /C calc\'!A0","pc_9 pc_8"')
    assert lines[2].startswith('"pc_10","Between policies",') and '"Waiting for a decision"' in lines[2]
    assert '"Person","sub-b","Approve"' in lines[3]


def test_decision_words_use_the_screen_labels():
    assert CH.decision_words("keep_both:FIN-0001") == "Keep both: FIN-0001 takes priority"
    assert CH.decision_words("moot") == "Closed: policy retired"
    assert CH.decision_words(None) == ""
```

Write the failing live tests `tests/agent_policy/test_conflict_export_live.py`:

```python
"""The conflict history CSV through the whole app (design §3.1 Export). Needs PROCWISE_TEST_LIVE_DB=1."""
import os
from urllib.parse import quote

import pytest

from api.routers import agent_policies as R
from services.agent_policy import conflict_cases as CC
from services.agent_policy import conflict_history as CH
from services.agent_policy.enforcement import MASK
from tests.agent_policy import test_conflict_live_gate as LG
from tests.agent_policy.test_conflict_endpoints_live import (  # noqa: F401 - fixtures
    NOW, STRANGER, _as, _design_case, client, conn, world)
from tests.agent_policy.test_conflict_history_live import settled_live

pytestmark = pytest.mark.skipif(os.getenv("PROCWISE_TEST_LIVE_DB") != "1", reason="live DB required")
HEAD = ",".join(f'"{h}"' for h in CH.HEADER)


def _row(text, case):
    [row] = [line for line in text.split("\r\n") if line.startswith(f'"pc_{case}"')]
    return row


def test_the_policy_export_masks_per_caller(client, conn, world, monkeypatch):
    a, b, live_id = settled_live(conn, world, monkeypatch)
    r = client.get(f"/agent-policies/{a}/conflicts/history.csv", headers=STRANGER)
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/csv")
    assert r.headers["content-disposition"] == f'attachment; filename="conflict-history-{a}.csv"'
    assert r.text.split("\r\n")[0] == HEAD
    row = _row(r.text, live_id)
    assert f'"Paying the {MASK} refund is fine."' in row and "900" not in row
    assert '"During an action"' in row and '"Person"' in row and '"Approve"' in row
    owner = client.get(f"/agent-policies/{a}/conflicts/history.csv", headers=_as(world, world.oa)).text
    assert '"Paying the 900 refund is fine."' in _row(owner, live_id)


def test_the_pair_export_takes_the_pair_in_any_order(client, conn, world):
    a, c, did = _design_case(conn, world)
    pair = "|".join(sorted([a, c], reverse=True))
    r = client.get(f"/agent-policies/conflicts/history.csv?pair={quote(pair)}", headers=STRANGER)
    assert r.status_code == 200, r.text
    first, second = sorted([a, c])
    assert r.headers["content-disposition"] == f'attachment; filename="conflict-history-{first}_{second}.csv"'
    assert '"Waiting for a decision"' in _row(r.text, did)


def test_a_formula_in_a_reason_is_neutralised(client, conn, world):
    a, c, did = _design_case(conn, world)
    CC.decide_policy(conn, did, principal=LG.who(world, world.oa), option=f"change:{a}",
                     reason='=HYPERLINK("http://x.test","go")', limit_text=None, now=NOW)
    text = client.get(f"/agent-policies/{a}/conflicts/history.csv", headers=_as(world, world.oa)).text
    assert '"\'=HYPERLINK(""http://x.test"",""go"")"' in _row(text, did)


def test_a_date_in_a_reason_survives(client, conn, world):
    a, c, did = _design_case(conn, world)
    CC.decide_policy(conn, did, principal=LG.who(world, world.oa), option=f"change:{a}",
                     reason="Agreed with Legal on 01/04/2026 under DL/2024/001", limit_text=None, now=NOW)
    text = client.get(f"/agent-policies/{a}/conflicts/history.csv", headers=_as(world, world.oa)).text
    assert '"Agreed with Legal on 01/04/2026 under DL/2024/001"' in _row(text, did)
    assert "[withheld]" not in text


@pytest.mark.parametrize("pair", ["FIN-0001", "fin-0001|FIN-0002", "FIN-0001|FIN-0001", "FIN-0001,FIN-0002",
                                  "FIN-0001|FIN-0002|"])
def test_a_bad_pair_is_422(client, world, pair):
    r = client.get(f"/agent-policies/conflicts/history.csv?pair={quote(pair)}", headers=STRANGER)
    assert r.status_code == 422, r.text


def test_an_unknown_policy_is_404(client, world):
    assert client.get("/agent-policies/ZZQ-99999999/conflicts/history.csv", headers=STRANGER).status_code == 404
    assert client.get("/agent-policies/not-a-key/conflicts/history.csv", headers=STRANGER).status_code == 404


def test_below_viewer_is_refused(client, world, monkeypatch):
    monkeypatch.setattr(R, "_role_of", lambda p: "None")
    assert client.get("/agent-policies/conflicts/history.csv?pair=FIN-0001%7CFIN-0002",
                      headers=STRANGER).status_code == 403
    assert client.get("/agent-policies/FIN-0001/conflicts/history.csv", headers=STRANGER).status_code == 403
```

- [ ] **Step 2: Run them and confirm they fail.**
Run: `bptest tests/agent_policy/test_conflict_history.py -v -k "csv or decision_words"` → FAIL (`AttributeError: ... 'csv_cell'`).
Run: `LIVE=1 bptest tests/agent_policy/test_conflict_export_live.py -v` → FAIL. Expect 422 from `/conflicts/{decision_id}` for `history.csv`, and 404 on the policy route.

- [ ] **Step 3: Implement the CSV in `conflict_history.py`** (add `import re` at the top):

```python
# ---------------------------------------------------------------------------- export
HEADER = ("Case", "Kind", "Raised", "Policies", "Decided by", "Name", "Decision", "Scope", "Decided at",
          "Reason", "Cited cases")
KIND_WORDS = {"policy": "Between policies", "live": "During an action"}
DECIDED_BY_WORDS = {"person": "Person", "standing_rule": "Standing rule", "precedent": "Precedent",
                    "timeout": "Timeout", "block": "Not allowed (block)", "retired": "Policy retired"}
_DECISION_WORDS = {"approve": "Approve", "reject": "Reject", "moot": "Closed: policy retired",
                   "block": "Blocked", "standing_rule": "Decided by a standing rule"}
_FORMULA = re.compile(r"^\s*[=+\-@]")
_CONTROL = re.compile(r"^[\t\r]")


def csv_cell(v: Any) -> str:
    """One quoted CSV cell, neutralised against formula injection with the UI's inventory csvCell
    rule: a cell a spreadsheet would read as a formula (=, +, -, @, even after leading spaces), or
    one starting with a tab or CR, gets a leading apostrophe."""
    s = "" if v is None else str(v)
    if _FORMULA.match(s) or _CONTROL.match(s):
        s = "'" + s
    return '"' + s.replace('"', '""') + '"'


def decision_words(option: Optional[str]) -> str:
    if option is None:
        return ""
    if option in _DECISION_WORDS:
        return _DECISION_WORDS[option]
    from services.agent_policy.conflict_cases import option_label   # lazily: conflict_cases pulls in approvals
    return option_label(option)


def to_csv(entries: List[Dict[str, Any]]) -> str:
    """The (already masked) entries as CSV, one row per case, CRLF line ends."""
    lines = [",".join(csv_cell(h) for h in HEADER)]
    for e in entries or []:
        d = e.get("decision") or {}
        by = d.get("decidedBy") or {}
        waiting = bool(e.get("isOpen"))
        lines.append(",".join(csv_cell(v) for v in (
            e.get("caseId"), KIND_WORDS.get(e.get("kind"), e.get("kind")), e.get("raisedAt"),
            "; ".join(f"{p.get('id')} v{p.get('version')}" for p in e.get("policies") or []),
            "" if waiting else DECIDED_BY_WORDS.get(by.get("kind"), ""),
            None if waiting else by.get("name"),
            "Waiting for a decision" if waiting else decision_words(d.get("option")),
            d.get("scope"), d.get("decidedAt"), d.get("reason"),
            " ".join(e.get("citedCases") or []))))
    return "\r\n".join(lines) + "\r\n"
```

- [ ] **Step 4: Add the routes** to `src/api/routers/agent_policies.py`. Import `re`, `Response` (`from fastapi.responses import JSONResponse, Response`) and `conflict_detect`. Place both routes immediately before `@router.get("/conflicts/{decision_id}")`:

```python
_PAIR_RE = re.compile(r"[A-Z]{3}-[0-9]{4,}(\|[A-Z]{3}-[0-9]{4,}){1,19}")


def _csv(text: str, filename: str) -> Response:
    # text/csv: OutputSafety scrubs JSON and SSE only, so the customer's own words pass unchanged;
    # formulas are neutralised by conflict_history.csv_cell.
    return Response(content=text, media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@router.get("/conflicts/history.csv")
def export_pair_history(pair: str = Query(..., max_length=400), p: Principal = Depends(gateway_principal)):
    role = _require(p, "Viewer", "agent_policy.read", {"pair": pair})
    key = conflict_detect.pair_key(*pair.split("|")) if _PAIR_RE.fullmatch(pair) else ""
    if "|" not in key:
        raise HTTPException(status_code=422, detail="pair must be two or more policy ids joined by |")
    with _conn() as conn:
        who = conflict_history.viewer(conn, p, is_admin=role == "Admin")
        with conn.cursor() as cur:
            entries = conflict_history.read(cur, pair_key=key, viewer=who)
    return _csv(conflict_history.to_csv(entries), f"conflict-history-{key.replace('|', '_')}.csv")


@router.get("/{key}/conflicts/history.csv")
def export_policy_history(key: str, p: Principal = Depends(gateway_principal)):
    role = _require(p, "Viewer", "agent_policy.read", {"policy": key})
    if not approval_views.KEY_RE.match(key):
        raise HTTPException(status_code=404, detail="no such policy")
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM proc.bp_agent_policy WHERE policy_key = %s", (key,))
            if cur.fetchone() is None:
                raise HTTPException(status_code=404, detail="no such policy")
        who = conflict_history.viewer(conn, p, is_admin=role == "Admin")
        with conn.cursor() as cur:
            entries = conflict_history.read(cur, policy_key=key, viewer=who)
    return _csv(conflict_history.to_csv(entries), f"conflict-history-{key}.csv")
```

- [ ] **Step 5: Run and confirm they pass.**
Run: `bptest tests/agent_policy/test_conflict_history.py -v` → PASS.
Run: `LIVE=1 bptest tests/agent_policy/test_conflict_export_live.py tests/agent_policy/test_conflict_endpoints_live.py -v` → PASS.
Run: `bptest tests/agent_policy/test_conflict_endpoints_scrub.py tests/agent_policy/test_screens_through_app.py tests/agent_policy/test_router.py -v` → PASS (route order and the OutputSafety exemptions are unchanged).

- [ ] **Step 6: Prove the guard fails.** Temporarily make `csv_cell` skip the apostrophe. `test_a_formula_in_a_reason_is_neutralised` and `test_csv_cell_neutralises_formulas_like_the_ui` must go red. Restore it and capture green.

- [ ] **Step 7: Commit.**

```bash
cd "$BP" && git add src/services/agent_policy/conflict_history.py src/api/routers/agent_policies.py tests/agent_policy/test_conflict_history.py tests/agent_policy/test_conflict_export_live.py
git commit -F - <<'EOF'
feat(agent-policy): export a policy's or a pair's conflict history as CSV, masked per caller

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

### Task 9: The gateway forwards the two exports

**Files (gateway repo):**
- Modify: `src/modules/agent-policy/agent-policy.service.ts` (`call`, `failure`, `send` via them, new `forwardText`)
- Modify: `src/modules/agent-policy/agent-policy.controller.ts` (`PAIR`, `pair()`, `csvFile()`, two routes)
- Modify: `src/modules/agent-policy/agent-policy.yml` (two functions)
- Test: `src/modules/agent-policy/agent-policy.controller.spec.ts`

**Interfaces:**
- Consumes: the BP routes of Task 8.
- Produces:
  - `AgentPolicyService.forwardText(user, path, timeoutMs = FORWARD_TIMEOUT_MS): Promise<string>`. It returns the backend body verbatim; backend errors raise the same `HttpException` as `send`.
  - `GET agent-policies/conflicts/history.csv?pair=` and `GET agent-policies/:key/conflicts/history.csv`. Viewer floor; 400 on a bad pair or key; `Content-Type: text/csv; charset=utf-8` and `Content-Disposition: attachment; filename="conflict-history-<…>.csv"`. Both are declared before `conflicts/:decisionId` and `:key`.

- [ ] **Step 1: Write the failing tests.** Append inside the outer `describe('AgentPolicyController', ...)` in `agent-policy.controller.spec.ts`:

```ts
  describe('conflict history export', () => {
    const V = req(['PROCWISE_VIEWER']);
    const CSV = '"Case","Kind"\r\n"pc_1","During an action"\r\n';
    const resOf = () => ({ set: jest.fn() });
    beforeEach(() => fetchMock.mockResolvedValue({ status: 200, text: async () => CSV, json: async () => ({}) }));

    it('forwards a policy export with identity and key, and answers the CSV as a file', async () => {
      const res = resOf();
      await expect(ctl().policyConflictHistoryCsv(V, 'FIN-0001', res)).resolves.toBe(CSV);
      const [u, init] = fetchMock.mock.calls[0];
      expect(u).toBe('http://py/agent-policies/FIN-0001/conflicts/history.csv');
      expect(init.method).toBe('GET');
      expect(init.headers['X-Gateway-Key']).toBe('k1');
      expect(init.headers['X-User-Sub']).toBe('u1');
      expect(init.headers.Authorization).toBeUndefined();
      expect(res.set).toHaveBeenCalledWith({ 'Content-Type': 'text/csv; charset=utf-8',
        'Content-Disposition': 'attachment; filename="conflict-history-FIN-0001.csv"' });
    });

    it('forwards a pair export with the pair encoded', async () => {
      const res = resOf();
      await expect(ctl().conflictHistoryCsv(V, 'CUS-0004|FIN-0012', res)).resolves.toBe(CSV);
      expect(fetchMock.mock.calls[0][0]).toBe('http://py/agent-policies/conflicts/history.csv?pair=CUS-0004%7CFIN-0012');
      expect(res.set).toHaveBeenCalledWith(expect.objectContaining({
        'Content-Disposition': 'attachment; filename="conflict-history-CUS-0004_FIN-0012.csv"' }));
    });

    it.each(['FIN-0001', 'fin-0001|FIN-0002', 'FIN-0001|FIN-0002|', 'FIN-0001,FIN-0002', 'FIN-0001|../x', '', undefined])(
      'rejects pair %p before the backend', async (p) => {
        await expect(ctl().conflictHistoryCsv(V, p as any, resOf())).rejects.toMatchObject({ status: 400 });
        expect(fetchMock).not.toHaveBeenCalled();
      });

    it('rejects a key that is not a policy id', async () => {
      await expect(ctl().policyConflictHistoryCsv(V, '../system', resOf())).rejects.toMatchObject({ status: 400 });
      expect(fetchMock).not.toHaveBeenCalled();
    });

    it('passes a backend error through', async () => {
      fetchMock.mockResolvedValue({ status: 404, json: async () => ({ detail: 'no such policy' }), text: async () => '' });
      const err: any = await ctl().policyConflictHistoryCsv(V, 'FIN-0001', resOf()).catch((e) => e);
      expect(err.getStatus()).toBe(404);
      expect(err.getResponse()).toBe('no such policy');
    });

    it('an unsigned caller is refused with 401 before the backend', async () => {
      const bad = { user: { sub: 'u1', tokenVerified: false, 'cognito:groups': ['PROCWISE_ADMIN'] } };
      await expect(ctl().conflictHistoryCsv(bad, 'A-0001|B-0002', resOf())).rejects.toMatchObject({ status: 401 });
      await expect(ctl().policyConflictHistoryCsv(bad, 'FIN-0001', resOf())).rejects.toMatchObject({ status: 401 });
      expect(fetchMock).not.toHaveBeenCalled();
    });

    it('is declared before the id and key routes', () => {
      const names: string[] = Reflect.ownKeys(AgentPolicyController.prototype).map(String);
      expect(names.indexOf('conflictHistoryCsv')).toBeLessThan(names.indexOf('getConflict'));
      expect(names.indexOf('policyConflictHistoryCsv')).toBeLessThan(names.indexOf('getOne'));
    });
  });
```

(`'A-0001|B-0002'` in the 401 test is refused before validation, by the token check.)

- [ ] **Step 2: Run them and confirm they fail.**
Run: `cd "$GW" && npx jest src/modules/agent-policy`
Expected: FAIL (`ctl(...).policyConflictHistoryCsv is not a function`).

- [ ] **Step 3: Implement the service.** In `agent-policy.service.ts`, replace `send` with:

```ts
  /** As forward, for a backend answer that is text (the CSV exports): the body verbatim. */
  async forwardText(user: any, path: string, timeoutMs = FORWARD_TIMEOUT_MS): Promise<string> {
    const res = await this.call(user, 'GET', path, undefined, timeoutMs);
    if (res.status >= 400) throw await this.failure(res);
    return res.text();
  }

  /** As forward, but also says which success status the backend answered (e.g. 202 for a started run). */
  async send(user: any, method: string, path: string, body?: unknown, timeoutMs = FORWARD_TIMEOUT_MS): Promise<{ status: number; data: any }> {
    const res = await this.call(user, method, path, body, timeoutMs);
    if (res.status >= 400) throw await this.failure(res);
    const data = await res.json().catch(() => ({}));
    return { status: res.status, data };
  }

  /** The backend request: the gateway key and the verified identity, never the browser's token. */
  private async call(user: any, method: string, path: string, body: unknown, timeoutMs: number): Promise<any> {
    const base = process.env.BP_BACKEND_URL;
    const key = process.env.AGENT_POLICY_GATEWAY_KEY;
    if (!base || !key) throw new HttpException('agent policies are not configured', 503);
    const groups = user?.['cognito:groups'];
    try {
      return await fetch(`${base.replace(/\/$/, '')}${path}`, {
        method,
        // A backend that never answers must not hold the request until the Lambda times out.
        signal: AbortSignal.timeout(timeoutMs),
        headers: {
          'Content-Type': 'application/json',
          'X-Gateway-Key': key,
          'X-User-Sub': String(user?.sub ?? ''),
          'X-User-Email': String(user?.email ?? ''),
          'X-User-Groups': JSON.stringify(Array.isArray(groups) ? groups : groups ? [groups] : []),
        },
        body: body === undefined ? undefined : JSON.stringify(body),
      });
    } catch (err: any) {
      // No answer in time -> 504; could not reach it at all -> 502. Never the raw error text.
      if (err?.name === 'TimeoutError' || err?.name === 'AbortError') {
        throw new HttpException('the policy service did not answer', 504);
      }
      throw new HttpException('the policy service could not be reached', 502);
    }
  }

  /** A backend error, keeping its shape: the backend answers 422 with a top-level {"problems": [...]};
   *  other errors may use {"detail": ...}. The UI receives `problems` intact. */
  private async failure(res: any): Promise<HttpException> {
    const data = await res.json().catch(() => ({}));
    const payload = data && typeof data === 'object' && 'detail' in data && !('problems' in data) ? data.detail : data;
    return new HttpException(payload, res.status);
  }
```

- [ ] **Step 4: Implement the controller.** Add after `const DECIDE_KEYS`:

```ts
const PAIR = /^[A-Z]{3}-[0-9]{4,}(\|[A-Z]{3}-[0-9]{4,}){1,19}$/;
```

and after `version(...)`:

```ts
function pair(v?: string): string {
  if (typeof v !== 'string' || !PAIR.test(v)) throw new BadRequestException('pair must be policy ids joined by |');
  return v;
}
function csvFile(res: any, name: string): void {
  res.set({ 'Content-Type': 'text/csv; charset=utf-8', 'Content-Disposition': `attachment; filename="${name}"` });
}
```

Insert both routes immediately before `@Get('conflicts') async listConflicts`:

```ts
  // The exports come before 'conflicts/:decisionId' and ':key', so 'history.csv' is never read as an id or a key.
  @Get('conflicts/history.csv') async conflictHistoryCsv(@Req() req, @Query('pair') p: string, @Res({ passthrough: true }) res: any) {
    need(req, 'Viewer');
    const ok = pair(p);
    const text = await this.svc.forwardText(req.user, `/agent-policies/conflicts/history.csv?pair=${encodeURIComponent(ok)}`);
    csvFile(res, `conflict-history-${ok.split('|').join('_')}.csv`);
    return text;
  }
  @Get(':key/conflicts/history.csv') async policyConflictHistoryCsv(@Req() req, @Param('key') k: string, @Res({ passthrough: true }) res: any) {
    need(req, 'Viewer');
    const ok = key(k);
    const text = await this.svc.forwardText(req.user, `/agent-policies/${ok}/conflicts/history.csv`);
    csvFile(res, `conflict-history-${ok}.csv`);
    return text;
  }
```

- [ ] **Step 5: Add the two serverless functions** to `agent-policy.yml`, copying the `listAgentPolicyFirings` block exactly (cors origins, headers, authorizer):
- `getAgentPolicyConflictHistoryCsv`, `path: agent-policies/conflicts/history.csv`, `method: get`;
- `getAgentPolicyHistoryCsv`, `path: agent-policies/{key}/conflicts/history.csv`, `method: get`.

- [ ] **Step 6: Run and confirm they pass.**
Run: `cd "$GW" && npx jest src/modules/agent-policy && npm run build` → all agent-policy specs PASS (the existing `send` tests too) and the build succeeds.

- [ ] **Step 7: Prove the guard fails.** Temporarily change `PAIR` to `/.*/`. The `rejects pair` cases must go red. Restore it and capture green.

- [ ] **Step 8: Commit (gateway repo).**

```bash
cd "$GW" && git add src/modules/agent-policy/agent-policy.service.ts src/modules/agent-policy/agent-policy.controller.ts src/modules/agent-policy/agent-policy.yml src/modules/agent-policy/agent-policy.controller.spec.ts
git commit -F - <<'EOF'
feat(agent-policy): forward the conflict history CSV exports

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

### Task 10: The screens show who decided, the history of a clash, and export it

**Files (UI repo):**
- Modify: `src/modules/SpendIQ/agentPolicy/conflicts.js` (`decidedByText`, `historyLine` reads both shapes, `reasonText`, `pairOf`, `historyExportPath`, `historyFileName`)
- Modify: `src/modules/SpendIQ/engine.js` (stage 4 block only: `apPcHistoryList`, `apPcExportHistory`, `apPcDownload`; wiring in `apPcBind`, `apPcPolicySection`, `apPcCardHTML`, `apPcApprovalBlock`)
- Test: `src/modules/SpendIQ/agentPolicy/conflicts.test.js`, `src/modules/SpendIQ/agentPolicy/engineWiring.stage4.contract.test.js`

**Interfaces:**
- Consumes: the gateway routes (Task 9); reader entries (Task 6); the case view's `pairKey` and `conflictHistory`; the approval block's `history` and `precedentNote`.
- Produces (`conflicts.js`):
  - `decidedByText(decidedBy) -> string`: `'by a person' | 'by a standing rule' | 'on precedent' | 'timed out' | 'blocked: not allowed' | ''`.
  - `historyLine(h) -> string`.
  - `reasonText(r) -> string`: a mask reads "Hidden".
  - `pairOf(view) -> string | null`.
  - `historyExportPath({policyKey} | {pairKey}) -> string | null`.
  - `historyFileName({policyKey} | {pairKey}) -> string`.
- Produces (`engine.js`):
  - `apPcExportHistory(target) -> Promise<void>`.
  - `apPcDownload(text, name)`.
  - `apPcHistoryList(list) -> string`.

- [ ] **Step 1: Write the failing tests.** Append to `conflicts.test.js`:

```js
describe('conflicts: history', () => {
  it('reads the reader\'s decision shape and says who decided', () => {
    expect(C.historyLine({ caseId: 'pc_7', kind: 'live', isOpen: false, otherPolicies: ['FIN-0002'], citedCases: ['pc_3', 'pc_2'],
      decision: { option: 'approve', decidedBy: { kind: 'precedent', name: 'system:precedent' } } }))
      .toBe('pc_7 with FIN-0002 · During an action · Decided: Approve · on precedent (pc_3, pc_2)');
    expect(C.historyLine({ caseId: 'pc_8', kind: 'live', isOpen: false, decision: { option: 'reject', decidedBy: { kind: 'timeout', name: 'system:timeout' } } }))
      .toBe('pc_8 · During an action · Decided: Reject · timed out');
    expect(C.historyLine({ caseId: 'pc_9', kind: 'policy', isOpen: false, decision: { option: 'keep_both:FIN-0001', decidedBy: { kind: 'person', name: 'sub-x' } } }))
      .toBe('pc_9 · Between policies · Decided: Keep both: FIN-0001 takes priority · by a person');
  });
  it('still reads the stage 4 shape, and says nothing of an unknown decider', () => {
    expect(C.historyLine({ caseId: 'pc_12', kind: 'policy', isOpen: false, otherPolicies: ['FIN-0002'],
      decision: { caseId: 'pc_12', decision: 'limit:FIN-0001', decidedBy: 'fm' } }))
      .toBe('pc_12 with FIN-0002 · Between policies · Decided: Limit FIN-0001');
    expect(C.historyLine({ caseId: 'pc_13', kind: 'live', isOpen: false, decision: { option: 'moot', decidedBy: { kind: null, name: 'legacy' } } }))
      .toBe('pc_13 · During an action · Closed: policy retired');
    expect(C.historyLine({ caseId: 'pc_14', kind: 'live', isOpen: true, decision: null }))
      .toBe('pc_14 · During an action · Waiting for a decision');
  });
  it('a masked value in a reason reads "Hidden"', () => {
    expect(C.reasonText(`Paying the ${MASK} refund`)).toBe('Paying the Hidden refund');
    expect(C.reasonText(null)).toBe('');
  });
  it('export paths: a policy id, or two or more ids; anything else is null', () => {
    expect(C.historyExportPath({ policyKey: 'FIN-0001' })).toBe('/agent-policies/FIN-0001/conflicts/history.csv');
    expect(C.historyExportPath({ pairKey: 'FIN-0001|CUS-0004' })).toBe('/agent-policies/conflicts/history.csv?pair=FIN-0001%7CCUS-0004');
    for (const bad of [{ policyKey: '../x' }, { policyKey: 'fin-0001' }, { pairKey: 'FIN-0001' }, { pairKey: 'FIN-0001|x' }, {}, undefined]) {
      expect(C.historyExportPath(bad)).toBeNull();
    }
  });
  it('file names', () => {
    expect(C.historyFileName({ policyKey: 'FIN-0001' })).toBe('conflict-history-FIN-0001.csv');
    expect(C.historyFileName({ pairKey: 'CUS-0004|FIN-0012' })).toBe('conflict-history-CUS-0004_FIN-0012.csv');
  });
  it('the pair of a case: its pairKey, else its policies sorted', () => {
    expect(C.pairOf({ pairKey: 'A-0001' })).toBeNull();
    expect(C.pairOf({ pairKey: 'CUS-0004|FIN-0012' })).toBe('CUS-0004|FIN-0012');
    expect(C.pairOf({ policies: [{ id: 'FIN-0012' }, { id: 'CUS-0004' }] })).toBe('CUS-0004|FIN-0012');
    expect(C.pairOf({ policies: [{ id: 'FIN-0012' }] })).toBeNull();
  });
});
```

In `engineWiring.stage4.contract.test.js`, extend `loadAp`. Add `download` to its options, `'__download'` after `'__save'` in the `new Function` parameter list, `apPcDownload=(t,n)=>__download(t,n);\n` after `apSave=(...a)=>__save(a);\n`, `apPcExportHistory` to the returned names, and `download || (() => {})` as the last argument of `make(...)`:

```js
function loadAp(api, { rbac, confirm, document: doc, save, download } = {}) {
  const block = between('/* ============ Agent policies (stage 1)', '/* The small edit form');
  const toasts = [];
  const window = { __SPENDIQ_AP__: { model, inventory, upload, runView, approvals, conflicts }, __SPENDIQ_API_WRITE__: api, __SPENDIQ_RBAC__: rbac };
  const document = doc || { removeEventListener: () => {}, addEventListener: () => {}, querySelector: () => null };
  const saves = [];
  const make = new Function('window', 'escH', 'TB', 'TBA', 'svg', 'I', 'CA_POLICY_CATS', 'current', 'toast', 'wfRerender', 'document', '__confirm', '__save', '__download',
    block + '\napConfirmTwoStep=(t,b,yes,second)=>__confirm(t,b,yes,second);\napSave=(...a)=>__save(a);\napPcDownload=(t,n)=>__download(t,n);\n'
    + 'return {AP,apPoliciesTab,apListHTML,apInventoryHTML,apFormHTML,apApprovalCardHTML,apApprovalApprove,apApprovalLoadDetail,apNotificationsHTML,apNotificationOpen,'
    + 'apPcLoad,apPcHTML,apPcCardHTML,apPcSelect,apPcReason,apPcLimit,apPcDecide,apPcOpen,apPcTag,apPcPolicySection,apPcOpenDraft,apPcRetire,apPcApprovalBlock,apPcExportHistory};');
  const escH = (s) => String(s == null ? '' : s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const fns = make(window, escH, (t) => t, (t) => t, () => '', {}, [], 'policies',
    (m) => toasts.push(m), () => {}, document, confirm || ((t, b, yes) => yes()), save || ((a) => saves.push(a)),
    download || (() => {}));
  return { ...fns, toasts, saves };
}
```

Append:

```js
describe('conflict history: lists and export', () => {
  const hist = [{ caseId: 'pc_3', kind: 'live', isOpen: false, otherPolicies: ['FIN-0002'], citedCases: [],
    decision: { option: 'approve', decidedBy: { kind: 'person', name: 'sub-x' }, reason: `Paying the ${MASK} refund` } }];

  it('the policy form lists who decided, hides masked values and offers the export', async () => {
    const got = [];
    const { api, calls } = fakeGateway({ 'get /agent-policies/FIN-0001/conflicts/history.csv': () => Promise.resolve('"Case"\r\n') });
    const ap = loadAp(api, { download: (t, n) => got.push([t, n]) });
    const html = ap.apPcPolicySection({ key: 'FIN-0001', policy: { status: 'live', conflicts: hist, pendingConflictAction: null } });
    expect(html).toContain('pc_3 with FIN-0002 · During an action · Decided: Approve · by a person — Paying the Hidden refund');
    expect(html).not.toContain(MASK);
    expect(html).toContain('data-ap-pc-act="export" data-ap-pc-key="FIN-0001"');
    await ap.apPcExportHistory({ policyKey: 'FIN-0001' });
    expect(calls).toEqual([{ method: 'get', path: '/agent-policies/FIN-0001/conflicts/history.csv', body: undefined }]);
    expect(got).toEqual([['"Case"\r\n', 'conflict-history-FIN-0001.csv']]);
  });

  it('a Conflicts-screen case lists the history of its pair and exports the pair', async () => {
    const got = [];
    const { api, calls } = fakeGateway({ 'get /agent-policies/conflicts/history.csv': () => Promise.resolve('x') });
    const ap = loadAp(api, { download: (t, n) => got.push([t, n]) });
    const html = ap.apPcCardHTML(caseOf({ pairKey: 'FIN-0001|FIN-0002', conflictHistory: hist }));
    expect(html).toContain('History of this clash');
    expect(html).toContain('data-ap-pc-act="export" data-ap-pc-pair="FIN-0001|FIN-0002"');
    await ap.apPcExportHistory({ pairKey: 'FIN-0001|FIN-0002' });
    expect(calls[0].path).toBe('/agent-policies/conflicts/history.csv?pair=FIN-0001%7CFIN-0002');
    expect(got).toEqual([['x', 'conflict-history-FIN-0001_FIN-0002.csv']]);
  });

  it('something that is not a policy or a pair is never requested', async () => {
    const { api, calls } = fakeGateway({});
    const ap = loadAp(api);
    await ap.apPcExportHistory({ policyKey: '../x' });
    expect(calls).toEqual([]);
    expect(ap.toasts).toEqual(['This history cannot be exported.']);
  });

  it('a failed export is a message, never a download', async () => {
    const got = [];
    const { api } = fakeGateway({ 'get /agent-policies/FIN-0001/conflicts/history.csv': () => Promise.reject(new Error('down')) });
    const ap = loadAp(api, { download: (t, n) => got.push([t, n]) });
    await ap.apPcExportHistory({ policyKey: 'FIN-0001' });
    expect(got).toEqual([]);
    expect(ap.toasts[0]).toMatch(/^Could not export the conflict history/);
  });

  it('a paused clash\'s approval card lists the earlier decisions and why precedent did not decide', () => {
    const ap = loadAp(fakeGateway({}).api);
    const html = ap.apPcApprovalBlock({ conflict: { caseId: 'pc_9', policies: [], options: ['approve', 'reject'], example: {},
      history: hist, precedentNote: 'only 2 of 3 decisions by people on this exact clash' } });
    expect(html).toContain('Earlier decisions on this clash');
    expect(html).toContain('Decided: Approve · by a person — Paying the Hidden refund');
    expect(html).toContain('Not decided on precedent: only 2 of 3 decisions by people on this exact clash');
    expect(html).not.toContain(MASK);
  });
});
```

- [ ] **Step 2: Run them and confirm they fail.**
Run: `cd "$UI" && npx vitest run src/modules/SpendIQ/agentPolicy`
Expected: FAIL (`C.decidedByText is not a function`, `apPcExportHistory is not defined`).

- [ ] **Step 3: Implement `conflicts.js`.** Replace `historyLine` and add after it:

```js
const DECIDED_BY = { person: 'by a person', standing_rule: 'by a standing rule', precedent: 'on precedent', timeout: 'timed out', block: 'blocked: not allowed' };

/* Who or what decided a conflict, as words; '' when not known (a case closed before this was recorded). */
export function decidedByText(decidedBy) {
  const k = decidedBy && typeof decidedBy === 'object' ? decidedBy.kind : null;
  return DECIDED_BY[k] || '';
}

/* A reason as the screens show it: a masked value reads "Hidden", never the mask. */
export function reasonText(r) {
  return String(r == null ? '' : r).split(MASK).join('Hidden');
}

/* One line of a conflict history (the server's one reader). Reads the stage 4 shape too. */
export function historyLine(h) {
  if (!h) return '';
  const others = Array.isArray(h.otherPolicies) && h.otherPolicies.length ? ` with ${h.otherPolicies.join(', ')}` : '';
  const kind = h.kind === 'live' ? 'During an action' : 'Between policies';
  const d = h.decision && typeof h.decision === 'object' ? h.decision : null;
  const opt = d && (d.option || d.decision);
  const state = h.isOpen ? 'Waiting for a decision' : (opt ? statusText({ status: 'closed', decision: { decision: opt } }) : 'Closed');
  const by = !h.isOpen && d ? decidedByText(d.decidedBy) : '';
  const cited = by === 'on precedent' && Array.isArray(h.citedCases) && h.citedCases.length ? ` (${h.citedCases.join(', ')})` : '';
  return `${h.caseId || ''}${others} · ${kind} · ${state}${by ? ` · ${by}${cited}` : ''}`;
}

const POLICY_KEY = /^[A-Z]{3}-[0-9]{4,}$/;
const pairKeys = (pairKey) => String(pairKey == null ? '' : pairKey).split('|');
const isPair = (keys) => keys.length >= 2 && keys.every((k) => POLICY_KEY.test(k));

/* The pair a Conflicts-screen case is about: its pairKey, else its policies' ids, sorted. */
export function pairOf(view) {
  if (!view) return null;
  if (view.pairKey !== undefined && view.pairKey !== null) return isPair(pairKeys(view.pairKey)) ? String(view.pairKey) : null;
  const ids = (Array.isArray(view.policies) ? view.policies : []).map((p) => p && p.id).filter(Boolean).map(String).sort();
  return isPair(ids) ? ids.join('|') : null;
}

/* The gateway path of a history export, or null for anything that is not a policy id or a pair. */
export function historyExportPath({ policyKey, pairKey } = {}) {
  if (policyKey !== undefined) return POLICY_KEY.test(String(policyKey)) ? `/agent-policies/${policyKey}/conflicts/history.csv` : null;
  const keys = pairKeys(pairKey);
  return isPair(keys) ? `/agent-policies/conflicts/history.csv?pair=${encodeURIComponent(keys.join('|'))}` : null;
}

export function historyFileName({ policyKey, pairKey } = {}) {
  const tail = policyKey !== undefined ? String(policyKey) : pairKeys(pairKey).join('_');
  return `conflict-history-${tail.replace(/[^A-Za-z0-9_-]/g, '') || 'export'}.csv`;
}
```

- [ ] **Step 4: Wire `engine.js`** (stage 4 block only).

Add after `apPcLabel`:

```js
/* A conflict history as a list: who decided each case, its reason with masked values "Hidden". */
function apPcHistoryList(list){
  const C=apPcLib();
  return '<ul class="fm-help">'+(Array.isArray(list)?list:[]).filter(h=>h&&typeof h==='object').map(h=>'<li>'+escH(C.historyLine(h)+(h.decision&&h.decision.reason?' — '+C.reasonText(h.decision.reason):''))+'</li>').join('')+'</ul>';
}
/* The server writes the CSV (masked for this caller, formulas neutralised); this only saves it. */
async function apPcExportHistory(target){
  const C=apPcLib(); if(!C) return;
  const path=C.historyExportPath(target||{});
  if(!path){ toast('This history cannot be exported.'); return; }
  try{
    const text=await apApi('get',path);
    apPcDownload(typeof text==='string'?text:String(text==null?'':text),C.historyFileName(target));
  }catch(e){ toast('Could not export the conflict history: '+apErrText(e)); }
}
function apPcDownload(text,name){
  const url=URL.createObjectURL(new Blob([text],{type:'text/csv;charset=utf-8'}));
  const a=document.createElement('a'); a.href=url; a.download=name;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(()=>URL.revokeObjectURL(url),0);
}
```

In `apPcBind`'s click handler, after `else if(act==='retire'&&st) apPcRetire(st);` add:

```js
    else if(act==='export') apPcExportHistory(t.hasAttribute('data-ap-pc-key')?{policyKey:t.getAttribute('data-ap-pc-key')}:{pairKey:t.getAttribute('data-ap-pc-pair')});
```

In `apPcPolicySection`, replace its last line (the `hist.length` list) with:

```js
    +(hist.length?apPcHistoryList(hist)
      +'<div style="margin-top:6px"><button type="button" class="btn sm" data-ap-pc-act="export" data-ap-pc-key="'+escH(state.key||'')+'">'+TB('Export conflict history')+'</button></div>':'');
```

In `apPcCardHTML`, insert before the final `+'</div>';`:

```js
    +(Array.isArray(v.conflictHistory)&&v.conflictHistory.length?sec('History of this clash')+apPcHistoryList(v.conflictHistory):'')
    +(C.pairOf(v)?'<div style="margin-top:6px"><button type="button" class="btn sm" data-ap-pc-act="export" data-ap-pc-pair="'+escH(C.pairOf(v))+'">'+TB('Export conflict history')+'</button></div>':'')
```

In `apPcApprovalBlock`, insert before the final `+'</div>';`:

```js
    +(k.precedentNote?'<div class="fm-help">'+escH('Not decided on precedent: '+k.precedentNote)+'</div>':'')
    +(Array.isArray(k.history)&&k.history.length?sec('Earlier decisions on this clash')+apPcHistoryList(k.history):'')
```

- [ ] **Step 5: Run and confirm they pass.**
Run: `cd "$UI" && npx vitest run src/modules/SpendIQ/agentPolicy && node --check src/modules/SpendIQ/engine.js` → PASS. This includes the stage 4 contract tests: the byte-identical hashes, "the stage 4 block talks only to the gateway", and the existing `'pc_12 with FIN-0002 · Between policies · Decided: Limit FIN-0001 — They &lt;overlap&gt;'`.
Run: `cd "$UI" && npx vitest run src/modules/SpendIQ` → no new failures against the baseline.

- [ ] **Step 6: Prove the guard fails.** Temporarily make `apPcHistoryList` use `h.decision.reason` instead of `C.reasonText(...)`. The "never contains MASK" assertions must go red. Restore it and capture green.

- [ ] **Step 7: Commit (UI repo).**

```bash
cd "$UI" && git add src/modules/SpendIQ/agentPolicy/conflicts.js src/modules/SpendIQ/agentPolicy/conflicts.test.js src/modules/SpendIQ/engine.js src/modules/SpendIQ/agentPolicy/engineWiring.stage4.contract.test.js
git commit -F - <<'EOF'
feat(agent-policy): conflict history shows who decided, lists a clash's history, and exports it

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

### Task 11: Live demonstration and verification on a stack of its own

The running stack (procwise :8000, bp-gateway :3001, bp-ui :3000) runs from the shared main checkouts. It does not have this code until the controller deploys after review. Do not restart, rebuild or touch it, its checkouts, `procwise.service` or Ollama. Run the worktree code on free ports against `bp_testdb`, the way stage 4 Task 11 did (`specs/2026-10-09-agent-policy-governance-stage4-verification.md`, "How the stack was run").

**Files:**
- Create: `specs/2026-10-09-conflict-history-and-precedent-verification.md`
- Scratch only, never committed: driver scripts in the session scratchpad.

- [ ] **Step 1: Baselines and ports.** Record the full-suite results on the base commit `f2bab3ac` before Task 1 (or from the stage 4 verification spec, if identical). Run, per repo, against the final commit:
- `LIVE=1 bptest tests/agent_policy tests/engines tests/approvals tests/governance tests/migrations -q`;
- `cd "$GW" && npx jest src/modules/agent-policy`;
- `cd "$UI" && npx vitest run src/modules/SpendIQ`.

Then confirm ports 8010, 3011, 3010 and 9223 are free (`ss -ltn`).

- [ ] **Step 2: Start the stack** (each in its own process group; record the PIDs):
- **Backend :8010:** `/home/muthu/PycharmProjects/BP_Backend/.venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8010 --workers 1` from the BP worktree. Use `PYTHONPATH=$BP:$BP/src`, `.env` (bp_testdb), `CUDA_VISIBLE_DEVICES=""`, `AGENT_POLICY_ENFORCEMENT=on`, `AGENT_POLICY_APPROVAL_SWEEP=off`, `AGENT_POLICY_CONFLICT_SCAN=off`, `OLLAMA_BASE_URL=OLLAMA_HOST=http://127.0.0.1:9`, and `OLLAMA_CLOUD_*` unset.
- **Gateway :3011:** `npm run build`, then `node --experimental-global-webcrypto ./dist/main.js` with `PORT=3011 AUTH_BYPASS=true IS_OFFLINE=true NODE_ENV=development AUTH_BYPASS_GROUPS=PROCWISE_ADMIN,PROCWISE_FINANCE_REVIEWER_APPROVER BP_BACKEND_URL=http://127.0.0.1:8010`. Read the gateway key from the BP `.env`; never print it.
- **UI :3010:** Vite from the UI worktree with `VITE_API_URL=http://127.0.0.1:3011 VITE_DEV_AUTH_BYPASS=true`. If `mockPipelineDeals.js` is needed, copy it in for the run and delete it afterwards.
- **Headless Chrome :9223** for the screens.

- [ ] **Step 3: Keep the shared database safe.**
- Check `select current_database()` = `bp_testdb` before any write.
- Check the governed row: `bppsql bp_testdb -Atc "SELECT policy_details->'rules' FROM proc.bp_policy WHERE policy_details->>'policy_identifier'='agent_policy_conflicts' AND policy_status=1"` → `{"precedent_count": 5}`. Do not edit it: other sessions share it. The fresh read is proven by tests.
- Name the demo pair "DEMO precedent A/B <hex>". Scope it with `tool.name in ["get_policy"]` and `args.query eq "DEMO-PREC-<hex>"`. Use two decider names and two owner names, each prefixed "Demo Prec".
- Before each save, run a read-only dry run (`conflict_cases._pairs`) of the demo form against every non-retired policy, as stage 4 did. The only witnessed pair allowed is the intended demo pair.

- [ ] **Step 4: Demonstrate (driver: `run_tools` with a scripted chat stand-in, never the real model).**
1. Link the owners and deciders in the decider map. Create A (approve, doc "Demo Finance", deciders "Demo Prec Finance Approver") and B (approve, doc "Demo Customer", deciders "Demo Prec Customer Approver"), then activate both.
2. Five times: one scripted `get_policy` call (query `DEMO-PREC-<hex>`) is paused. Record the live case and its `facts.precedent.why` (`only k of 5 …`). Approve both member cases as the linked deciders; on approval 3, give a reason that quotes the query value.
3. After the fifth settlement, record the standing-rule proposal policy case (`raised_by 'repeat'`, `proposal.count 5`).
4. **The sixth identical call runs on precedent.** Record:
   - the tool's run;
   - the live record (`system:precedent`, `facts.decidedBy`, `evidence` citing the five cases);
   - the `allowed` firing rows linked to it;
   - the notifications (no input values).
5. Show `GET /agent-policies/<A>` `conflicts[]` through the gateway (:3011, `x-customer-id: 001`), once as a linked owner and once as an unlinked caller (masked reason).
6. Export `GET /agent-policies/<A>/conflicts/history.csv` and `GET /agent-policies/conflicts/history.csv?pair=<A>%7C<B>` through the gateway. Show the header, the six rows, the masking difference and the `Content-Disposition`.
7. **Screens (headless Chrome, 0 console errors and 0 uncaught exceptions):**
   - the policy form's Conflicts section, with "by a person" / "on precedent" lines and the Export button pressed;
   - a Conflicts-screen case with "History of this clash";
   - an approval card of a paused clash with "Earlier decisions on this clash" (repeat step 2's first call on a fresh pair C/D to get one).
8. `/decisions` shows neither `policy_conflict` nor `live_conflict`.

- [ ] **Step 5: Clean up.** Retire every demo policy (ids are never deleted). Delete your decider-map rows. Change no company setting and no governed row. Stop your processes by process group (`kill -TERM -- -<pgid>`, your PIDs only), then confirm ports 8010, 3011, 3010 and 9223 are closed and `procwise.service` is still `active` and untouched.

- [ ] **Step 6: Write** `specs/2026-10-09-conflict-history-and-precedent-verification.md`, in the stage 4 verification's layout:
- a plain-English summary first;
- how the stack was run;
- every request and response (keys redacted);
- the red/green captures from every task;
- the full-suite results per repo against the baseline;
- the diff summary per repo;
- the pre-existing failures (Task 2 Step 1), named as such.

- [ ] **Step 7: Commit (BP).**

```bash
cd "$BP" && git add specs/2026-10-09-conflict-history-and-precedent-verification.md
git commit -F - <<'EOF'
docs(agent-policy): conflict history and precedent live verification

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

## Self-review notes (for the executor)

**Spec coverage:**
- §3.1, stored additions:
  - `decidedBy` → Task 4;
  - `versionsAtDecision` → Task 4;
  - cited `evidence` → Tasks 5 and 7.
- §3.1, one reader → Task 6 (`raw`/`shown`/`read`; `history_for`, the Conflicts detail and the approval card use it).
- §3.1, who sees what → Task 6 (`may_see`; reasons masked, not dropped).
- §3.1, export → Task 8 (BP), Task 9 (gateway) and Task 10 (UI buttons).
- §3.2:
  - where → Task 5 (`decide_live_conflict` in `decision_engine.py`);
  - when → Task 7 (only `human`, a fresh call, `paused_for_approval`; never on a repeat or in the replay);
  - rules 1–6 → Task 5;
  - approve/reject records, notifications and `to_agent` → Task 7;
  - escalated history → Tasks 6 and 7;
  - fail-closed → Task 7;
  - the proposal reads the same N → Task 3.
- §3.3:
  - the row → Task 2;
  - the migration on both databases → Task 2;
  - `fresh=True` → Task 1;
  - one source, `DEFAULTS` removal → Task 3.
- §4 → the rows of the table are tested in Tasks 3, 5 and 7.
- §5 → every bullet has a named test. The demonstration is Task 11.
- §7 → the migration is applied in Task 2, before any deployment; deployment is out of scope.

**Names used across tasks:**
- Task 1: `governed_limits.limit(..., fresh=True)`, `_new_engine`, `_fresh_engine`.
- Task 3: `settings.precedent_count()`, `conflict_live.threshold()`, `fixtures.precedent_n(monkeypatch, n, *, missing=False)`.
- Task 4: `conflict_cases.decided_by(kind, name)`, `DECIDED_BY_KINDS`, `conflict_live.PRECEDENT`, `insert_live(..., extra_facts=, extra_evidence=)`.
- Task 5: `decision_engine.decide_live_conflict(cur, lc, *, ctx, now)`, `PRECEDENT_SQL`, `_precedent_count`, `RESOLVED`/`ESCALATED`.
- Task 6: `conflict_history.Viewer`/`ANONYMOUS`/`viewer`/`raw`/`shown`/`may_see`/`sensitive_of`/`read`/`IN_CASE_LIMIT`/`HISTORY_LIMIT`; `repo.get_policy(..., viewer=)`; `conflict_views.get_conflict(..., is_admin=)`.
- Task 7: `gate.PRECEDENT_REFUSED`, `_lock_key`, `_consult_precedent`, `_record_precedent`, `_notify_precedent`, `_precedent_answer`.
- Task 8: `conflict_history.csv_cell`/`to_csv`/`decision_words`/`HEADER`.
- Task 9: `forwardText`, `conflictHistoryCsv`, `policyConflictHistoryCsv`.
- Task 10: `decidedByText`, `reasonText`, `pairOf`, `historyExportPath`, `historyFileName`, `apPcHistoryList`, `apPcExportHistory`, `apPcDownload`.
