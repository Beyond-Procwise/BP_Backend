# Agent Policy Governance — Stage 4 (Conflicts) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Contradicting agent policies are found and sent to the decision engine, never resolved by this product:
- **Design time:** when a policy is saved (Draft or Active) or extracted, code looks for one example input that matches it and a policy from a *different source* with a *different outcome*. If it finds one, a `policy` conflict case is raised, with that input as the witness. The owners decide one of four things: keep both with a standing rule, change one, limit one, or retire one. The decision is applied only through the owner's own save.
- **Run time:** when one agent action matches several policies that disagree, a `live` conflict case is raised. A standing rule decides it automatically when one covers it. Otherwise it goes to people, and the action stays paused. A timeout or an outage rejects it.
- **Feedback:** after the same live conflict has been decided the same way 5 times (a company setting), a policy case is raised that proposes a standing rule.

**Architecture:**
- **Three pure modules** hold the logic, with no DB, model or clock:
  - `conflict_detect` finds witnesses;
  - `conflict_payload` builds the `policy-conflict/1` facts and the approver summary;
  - `conflict_engine` classifies a run-time verdict.
- **Two store modules** write `proc.bp_decision` rows, using the new subject types `policy_conflict` and `live_conflict` (design §5 mapping):
  - `conflict_cases` handles policy cases, decisions and standing rules;
  - `conflict_live` handles live records, settling them, and the repeat-5 check.
- **New tables:** one conflict-index table and one standing-rule table, which ruling B's "conflict case" table covers.
- **Approvers of live conflicts** reuse the stage 3 approval machinery unchanged in kind:
  - one `agent_policy_approval` member case per conflicting approve policy, routed to that policy's **last** escalation level (ruling fallback);
  - every member must approve; the sweeper rejects on timeout; replay runs the action.
- **Standing rules** are stored in their own table and overlaid onto each live policy's `conflicts[]` at read time, because version rows are immutable (see the contradictions section).
- **The repo** (the stage 1 policy store) gains one overlay hook.
- **The decision engine** gets new subject types, hidden from the generic `/decisions` endpoints exactly as stage 3's are.

**Tech Stack:** Python 3 / FastAPI / psycopg2 (BP_Backend), NestJS + serverless yml (gateway), vanilla `engine.js` + vitest pure modules (UI).

**Spec:**
- `specs/2026-10-08-agent-policy-governance-brief.md`:
  - §4 (keep the list, inventory and CSV; replace the overlap check);
  - §4.1–4.5;
  - §7 (the `conflicts[]` rule);
  - §8 tests 2 (no conflict part), 11, 13, 14, 15, 16, 17, 18.
- `specs/2026-10-08-agent-policy-governance-design.md`. Its rulings win over the brief:
  - §2: "Not allowed in a conflict: still blocks immediately";
  - §2: the live conflict approver is the engine's routing, falling back to the last escalation level of each matching approve policy, with all of them approving; if no approve policy matches, no approver is needed;
  - §2: "Five same-way live decisions raise a policy case", which is a company setting;
  - §3.8: no session resume;
  - §4 (stage 4 row);
  - §5 (field mapping).
- `.superpowers/sdd/2026-10-08-agent-policy-governance-plan-3-enforcement/progress.md`: every ruling binds.

---

## Open user decisions (answer before Task 4 starts; tasks 1–3 are safe either way)

1. **Can a standing rule let a policy beat a Not allowed (`block`) policy?**
   - **Recommend: no.** The design ruling says a block always blocks, so a rule that says otherwise would be a dead letter or a back door.
   - For a pair that includes a block, the only standing rule offered is "<block policy> takes priority", which records that the owners accept the block. Otherwise the owners change, limit or retire one of the policies.
2. **Do `notify` policies ever take part in a conflict?**
   - **Recommend: no.** The brief says a notify always sends, whatever is decided, so it never changes what happens to the action.
   - Conflicts are therefore computed only among `block` and `approve`. Approve + notify and block + notify raise nothing.
3. **Who decides a `policy` case?**
   - **Recommend:** anyone linked, through the stage 3 decider map, to *either* policy's `owner` name. One decision closes the case, and both owners are notified.
   - When an owner name is not linked, the case is "unroutable", and "Administrators" are notified (the stage 3 pattern).
   - An unlinked owner does NOT block Active. The brief allows Active while a conflict is open.
4. **What counts as "the same live conflict decided the same way 5 times"?**
   - **Recommend:**
     - the same set of conflicting policy ids, whatever their versions;
     - the last N person-decided live cases for that set all approved, or all rejected;
     - a different outcome resets the count;
     - timeouts, block records and standing-rule auto decisions never count;
     - no second proposal is raised while one is open, or while a standing rule covers the pair.
5. **OutputSafety exemption for the new screen reads.**
   - **Recommend:** extend the 2xx exemption to exactly `GET /agent-policies/conflicts` and `GET /agent-policies/conflicts/[0-9]{1,18}`. These are the same terms as ruling 23:1x: errors are still filtered.
   - Conflict cases quote policy excerpts, source file names and the action's own values, the same kinds of text the user already exempted for approvals.
   - Task 8 ships strict xfails until this ruling arrives.

## Decisions taken in planning (no ruling needed; reviewers may challenge)

- **Same source** means the same `source.document`, case-folded and trimmed, and not empty. A policy with no source document is never the same source as another. These policies are compared on `source.document` because the compiled JSON carries it, while `source_document_id` lives only on `bp_agent_policy`.
- **A witness must give every condition field a value.** A match that happens only because data is missing (`fail_closed`) is not a design-time conflict. At run time, `fail_closed` still applies, and the live path handles it.
- **The witness search is capped at `MAX_CANDIDATES = 5000` inputs.** Reaching the cap without a match means "no conflict found". The brief says that if no such input is found, there is no conflict.
- **Policy cases have no deadline** (`respond_by` NULL, `onTimeout` `"none"`). The brief allows Active meanwhile, and live conflicts fail safe. Live member cases use their policy's `respondWithin` with `onTimeout` `reject`.
- **Design-time detection compares outcomes only, as brief §4.1 says.** Two approve policies with different deciders are left to the live path and its repeat-5 proposal.
- **Policy case options** are `keep_both:<KEY>`, `change:<KEY>`, `limit:<KEY>` and `retire:<KEY>`, for each policy. Q1 trims `keep_both` for block pairs. The returned `scope` is `standing_rule` for `keep_both`, otherwise `this_action`.
- **Live options are `["approve","reject"]` only.** Ruling: no approve-with-changes, and approve or reject only. The brief's sample options `approve_by:CFO` and `reject_and_escalate_to_owners` are not used.
- **Subject types are exactly design §5's `policy_conflict` and `live_conflict`.** They are added to `decisions.HIDDEN_SUBJECT_TYPES`.
- **Witness keys are flat field names** (`{"tool.name": "refund.issue", "args.amount": 12400}`), the same shape as form examples, not the brief's illustrative `{"tool","amount"}`.

## Global Constraints

- Everything in stages 1–3 still binds, including `common-context.md` and `global-constraints.md` in the stage 3 SDD folder.
- `proc.bp_policy` is never written, altered or migrated. bp_rule detects and bp_policy authorizes; agent policies live only in their own tables.
- **New tables:** `bp_` prefix, indexes `ix_bp_*`. Migrations live in `deploy/sql/` with a `_rollback.sql`, are additive and idempotent, and are applied to **bp_testdb and bp_sqldb**.
- **`proc.bp_decision`:** no column changes in stage 4; stage 3 already added `options`, `respond_by`, `on_timeout` and `decision_scope`. Existing subject types must behave exactly as before. `tests/engines`, `tests/approvals` and the decisions router tests must stay green before and after.
- **One evaluator:** `conditions.to_engine` + `policy_condition.evaluate`. The witness search uses exactly this pair.
- **Precedence:** the design ruling stands. Any matching `block` blocks immediately. A live conflict involving a block is *recorded*, never put to a person. Notifies always send. There is no other precedence logic.
- **Fail closed:** any failure while recording a live conflict refuses the call with `policy_check_unavailable`, as the stage 3 gate does. One transaction, nothing left behind. A timeout rejects and never approves.
- **Never apply a decision silently.**
  - `change` and `limit` open a Draft in the UI and never write a version.
  - `retire` starts the existing two-step retire.
  - `keep_both` writes only a standing-rule row.
- **Live cases store only condition fields** in `facts.action.args`. Sensitive inputs are masked in every view, except for an approver eligible at the member case's current level.
- **Never call the real model in tests.** Live DB tests need `PROCWISE_TEST_LIVE_DB=1` and `DB_NAME=bp_testdb`.
- **No new OutputSafety exemptions without a user ruling** (open decision 5).
- **UI:** `engine.js` changes are new `apPc*` functions plus minimal wiring only. `policyEdit`, `policyDelete` and `openFormModal` stay byte-identical. The UI calls the gateway only.
- **Every write endpoint** audits via `agent_actions.record_action_or_fail` before returning.
- **Prove the guard fails:** where a step says so, break it on purpose, capture the red output, restore it, and capture green.
- **Concurrency:** another session may be finishing stage 3 in the same worktrees. Stage only your own hunks, `git add` explicit paths only, never stash. Where this plan names a stage 3 function, find it by name; line numbers may have moved.

## Review Focus

1. **Two saves racing for the same pair.** Exactly one open policy case per pair. The partial unique index makes the second insert a no-op that returns the existing case. Test: `test_concurrent_raise_one_case_per_pair` (Task 4).
2. **A sensitive input in a conflict case** (witness, live args, approver summary) shown to a non-approver. It is masked as `•••`. Tests: `test_conflict_view_masks_sensitive_witness` (Task 8) and `test_live_member_view_masks_for_non_approver` (Task 8).
3. **A model that loops on a call paused by a live conflict.** The repeat reuses every member case and writes no second live record. Test: `test_repeat_call_reuses_live_conflict` (Task 7).
4. **A policy retired while its conflict case is open.** The case closes as moot, and a later decide gets 409. Test: `test_retire_closes_open_case_as_moot` (Task 5).
5. **A huge condition** (long `in` lists, many fields). The witness search stops at the cap within 2 s, and "not found" is not an error. Test: `test_witness_search_is_bounded` (Task 2).

---

## File map

**BP_Backend** (`/home/muthu/PycharmProjects/BP_Backend/.claude/worktrees/agent-policy`)
| File | Responsibility |
|---|---|
| `deploy/sql/2026-10-12_bp_agent_policy_conflicts.sql` (+ `_rollback.sql`) | conflict index table, standing-rule table, notification `firing_id` nullable |
| `src/services/agent_policy/conflict_detect.py` (new, pure) | same-source, deciding outcomes, witness search, pair keys |
| `src/services/agent_policy/conflict_payload.py` (new, pure) | `policy-conflict/1` facts, condition-only args, why-line, options, §5 column mapping, returned decision |
| `src/services/agent_policy/conflict_engine.py` (new, pure) | classify a run-time `Verdict` into none / block_record / auto / human |
| `src/services/agent_policy/conflict_cases.py` (new) | raise policy cases (dedup), detect on save/scan, decide + apply, standing rules + overlay, moot close, history |
| `src/services/agent_policy/conflict_live.py` (new) | live record insert, settle on group close, repeat-5 proposal |
| `src/services/agent_policy/conflict_views.py` (new) | list/detail/readable for policy cases; conflict block for approval views |
| `src/services/agent_policy/gate.py` (modify `_before_tool`, `_open_cases`) | call `conflict_engine.classify`, record live conflicts |
| `src/services/agent_policy/approvals.py` (modify `_insert_case`, `_close_group`) | `last_level_only`; settle the live record when a group closes |
| `src/services/agent_policy/approval_views.py` (modify `case_view`/`get_case`, `_link_ref`) | `conflict` block on member cases; `conflict:<id>` links |
| `src/repositories/agent_policy_repo.py` (modify `live_documents`, `list_policies`, `get_policy`) | overlay `conflicts[]`; `openConflicts`; history + pending action |
| `src/services/agent_policy/extraction_run.py` (modify `_decide`) | detection after each saved draft |
| `src/api/routers/agent_policies.py` | conflict endpoints; detection after create/save; moot close after retire |
| `src/api/routers/decisions.py` (modify `HIDDEN_SUBJECT_TYPES`) | hide `policy_conflict`, `live_conflict` |
| `src/services/backend_scheduler.py` | `_register_agent_policy_conflict_scan_job` (hourly safety net) |
| `tests/migrations/test_2026_10_12_bp_agent_policy_conflicts.py`, `tests/agent_policy/test_conflict_*.py` | tests |

**Gateway** (`/home/muthu/PycharmProjects/beyond-procwaise-Api-worktrees/agent-policy/beyond_procwaise_api`): `src/modules/agent-policy/agent-policy.controller.ts`, `agent-policy.yml`, `agent-policy.controller.spec.ts`.

**UI** (`/home/muthu/PycharmProjects/beyond_procwise_ui-worktrees/agent-policy`): `src/modules/SpendIQ/agentPolicy/conflicts.js` (+ `conflicts.test.js`), `src/modules/SpendIQ/index.jsx` (register `conflicts` on `window.__SPENDIQ_AP__`), `src/modules/SpendIQ/engine.js` (`apPc*`), `src/modules/SpendIQ/agentPolicy/engineWiring.stage4.contract.test.js`.

---

### Task 1: Migration

**Files:**
- Create: `deploy/sql/2026-10-12_bp_agent_policy_conflicts.sql`, `deploy/sql/2026-10-12_bp_agent_policy_conflicts_rollback.sql`
- Test: `tests/migrations/test_2026_10_12_bp_agent_policy_conflicts.py` (live, both databases, same pattern as `tests/agent_policy/test_2026_10_10_bp_agent_policy_enforcement.py`)

**Interfaces:**
- Produces: `proc.bp_agent_policy_conflict`, `proc.bp_agent_policy_conflict_rule`, and a nullable `proc.bp_policy_notification.firing_id`.

```sql
-- 2026-10-12  Agent policy governance, stage 4: conflicts. Additive, idempotent.
-- Cases themselves are proc.bp_decision rows (subject_type policy_conflict | live_conflict);
-- this table indexes them by policy so a pair has at most one open policy case and a
-- policy's history can list its conflicts. proc.bp_policy is untouched (ruling B).
BEGIN;
CREATE TABLE IF NOT EXISTS proc.bp_agent_policy_conflict (
    decision_id     BIGINT PRIMARY KEY REFERENCES proc.bp_decision (decision_id),
    kind            TEXT NOT NULL CHECK (kind IN ('policy','live')),
    pair_key        TEXT NOT NULL,                 -- sorted policy keys joined by '|'
    policy_keys     TEXT[] NOT NULL CHECK (cardinality(policy_keys) >= 2),
    policy_versions JSONB NOT NULL,                -- {"FIN-0012": 3, "CUS-0004": 1}
    raised_by       TEXT NOT NULL CHECK (raised_by IN ('save','scan','live','repeat')),
    is_open         BOOLEAN NOT NULL DEFAULT TRUE,
    outcome         TEXT,                          -- the returned decision value
    decided_by      TEXT,
    decided_at      TIMESTAMPTZ,
    by_person       BOOLEAN,                       -- false for timeouts, block records, standing rules
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT ck_bp_agent_policy_conflict_settled
        CHECK (is_open OR (outcome IS NOT NULL AND decided_at IS NOT NULL))
);
CREATE UNIQUE INDEX IF NOT EXISTS ix_bp_agent_policy_conflict_open_pair
    ON proc.bp_agent_policy_conflict (pair_key) WHERE kind = 'policy' AND is_open;
CREATE INDEX IF NOT EXISTS ix_bp_agent_policy_conflict_keys
    ON proc.bp_agent_policy_conflict USING GIN (policy_keys);
CREATE INDEX IF NOT EXISTS ix_bp_agent_policy_conflict_pair_decided
    ON proc.bp_agent_policy_conflict (kind, pair_key, decided_at DESC);

CREATE TABLE IF NOT EXISTS proc.bp_agent_policy_conflict_rule (
    rule_id        BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pair_key       TEXT NOT NULL,
    prevails       TEXT NOT NULL,
    yields         TEXT NOT NULL,
    rule_text      TEXT NOT NULL,                  -- "FIN-0012 takes priority over CUS-0004"
    decision_id    BIGINT NOT NULL REFERENCES proc.bp_decision (decision_id),
    decided_by     TEXT NOT NULL,
    decided_at     TIMESTAMPTZ NOT NULL,
    superseded_at  TIMESTAMPTZ,
    superseded_by  BIGINT REFERENCES proc.bp_decision (decision_id),
    CONSTRAINT ck_bp_agent_policy_conflict_rule_two CHECK (prevails <> yields)
);
CREATE UNIQUE INDEX IF NOT EXISTS ix_bp_agent_policy_conflict_rule_in_force
    ON proc.bp_agent_policy_conflict_rule (pair_key) WHERE superseded_at IS NULL;

-- Policy-case notifications (to owners) have no firing row: the firing log is for actions.
ALTER TABLE proc.bp_policy_notification ALTER COLUMN firing_id DROP NOT NULL;
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'ck_bp_policy_notification_target') THEN
    ALTER TABLE proc.bp_policy_notification ADD CONSTRAINT ck_bp_policy_notification_target
      CHECK (firing_id IS NOT NULL OR link LIKE 'conflict:%');
  END IF;
END $$;
COMMIT;
```

The rollback runs inside a transaction:
1. drop the CHECK;
2. `DELETE FROM proc.bp_policy_notification WHERE firing_id IS NULL AND link LIKE 'conflict:%'` (stage 4's own rows);
3. `SET NOT NULL`;
4. drop both tables.

It is never run against shared data without a user ruling.

- [ ] **Step 1: Write the failing migration test.** It must check:
  - both tables, every column and every index exist in bp_testdb and bp_sqldb;
  - the partial unique index refuses a second open `policy` row for the same `pair_key`, but allows a second `live` row and a second closed row;
  - the in-force rule index refuses a second unsuperseded rule;
  - `ck_bp_agent_policy_conflict_settled` refuses `is_open=false` with no outcome;
  - the notification CHECK refuses a NULL `firing_id` with a `decision:` link and accepts a `conflict:` link;
  - `proc.bp_decision`'s column list equals the live list captured before applying (copy it first);
  - `proc.bp_policy` is unchanged (column list plus row count).
  
  Seed `bp_decision` rows with `subject_type='policy_conflict'`. Clean up in FK order: notification, conflict_rule, conflict, then decision.
- [ ] **Step 2: Run it and confirm it fails:** `PROCWISE_TEST_LIVE_DB=1 ... pytest tests/migrations/test_2026_10_12_bp_agent_policy_conflicts.py -v` → FAIL (relation does not exist).
- [ ] **Step 3: Write both SQL files. Apply to bp_testdb, then to bp_sqldb** (psql line in common-context). Apply twice to each to prove idempotence.
- [ ] **Step 4: Run the test against both databases** (`DB_NAME=bp_testdb` and `DB_NAME=bp_sqldb` for the read-only checks) → PASS. Prove a guard fails: temporarily drop the partial index in a transaction you roll back, and show the uniqueness test going red.
- [ ] **Step 5: Commit** `feat(agent-policy): conflict case and standing rule tables`.

---

### Task 2: Witness search (pure)

**Files:**
- Create: `src/services/agent_policy/conflict_detect.py`
- Test: `tests/agent_policy/test_conflict_detect.py`

**Interfaces:**
- Consumes: `conditions.to_engine`, `conditions.nest`, `conditions.condition_fields`, `policy_condition.evaluate` / `MissingField` / `ConditionError`.
- Produces:
  - `MAX_CANDIDATES: int = 5000`
  - `pair_key(*keys: str) -> str`: sorted, de-duplicated, `"|"`-joined.
  - `source_of(doc: dict) -> str | None`: the case-folded, stripped `source.document`, or None.
  - `same_source(a: dict, b: dict) -> bool`
  - `deciders_of(doc: dict) -> tuple[str, ...]`: the ordered `enforcement.intervention.escalateTo[].name`; `()` when not approve.
  - `deciding(doc: dict) -> bool`: outcome in `("block","approve")` (open decision 2).
  - `outcomes_differ(a, b) -> bool`, and `deciders_differ(a, b) -> bool`, which holds when both are approve and their decider tuples differ.
  - `design_time_pair(a: dict, b: dict) -> bool`: different keys, both deciding, not the same source, the same checkpoint, outcomes differ.
  - `witness(a: dict, b: dict, examples: Iterable[dict] = ()) -> dict | None`: a flat input matching both, or None. Raises `ConditionError` when a condition is unreadable; the caller skips that pair.

Core of the implementation (copy it, then make the tests pass):

```python
OTHER = "__none_of_these__"
PRESENT = "present"


def _tools(doc):
    return list(((doc.get("context") or {}).get("actions") or {}).get("tools") or [])


def _cond(doc):
    return conditions.to_engine((doc.get("trigger") or {}).get("condition"))


def _matches(doc, engine_cond, flat) -> bool:
    listed = _tools(doc)
    if listed and flat.get("tool.name") not in listed:
        return False          # stage 3 rule I1: a policy that lists tools applies to those only
    try:
        return bool(pc.evaluate(engine_cond, conditions.nest(flat)))
    except pc.MissingField:
        return False          # a design-time witness never relies on missing data


def _candidates(field, leaves, examples, tools):
    vals = []
    for leaf in leaves:
        if leaf.get("field") != field:
            continue
        op, v = leaf.get("op"), leaf.get("value")
        if op in ("gt", "gte", "lt", "lte", "eq", "ne") and isinstance(v, (int, float)) and not isinstance(v, bool):
            vals += [v, v + 1, v - 1, v + 0.01, v - 0.01]
        elif op in ("in", "not_in") and isinstance(v, list):
            vals += list(v) + [OTHER]
        elif op in ("eq", "ne"):
            vals += [v, OTHER]
        elif op == "exists":
            vals.append(PRESENT)
    vals += [ex[field] for ex in examples if field in ex]
    if field == "tool.name":
        vals += tools
    out, seen = [], set()
    for v in vals:
        k = json.dumps(v, sort_keys=True, default=str)
        if k not in seen:
            seen.add(k); out.append(v)
    return out


def witness(a, b, examples=()):
    ca, cb = _cond(a), _cond(b)            # ConditionError propagates
    examples = [dict(e) for e in examples if isinstance(e, dict)]
    tools = sorted(set(_tools(a)) | set(_tools(b)))
    fields = sorted(conditions.condition_fields((a.get("trigger") or {}).get("condition"))
                    | conditions.condition_fields((b.get("trigger") or {}).get("condition"))
                    | ({"tool.name"} if tools else set()))
    leaves = conditions._leaves((a.get("trigger") or {}).get("condition")) + \
             conditions._leaves((b.get("trigger") or {}).get("condition"))
    tried = 0
    for ex in examples:                    # 1. the examples as written, when complete
        if all(f in ex for f in fields):
            tried += 1
            flat = {f: ex[f] for f in fields}
            if _matches(a, ca, flat) and _matches(b, cb, flat):
                return flat
    pools = [_candidates(f, leaves, examples, tools) for f in fields]
    if any(not p for p in pools):
        return None
    for combo in itertools.product(*pools):  # 2. boundary values of both conditions
        tried += 1
        if tried > MAX_CANDIDATES:
            return None
        flat = dict(zip(fields, combo))
        if _matches(a, ca, flat) and _matches(b, cb, flat):
            return flat
    return None
```

(`conditions._leaves` is module-private. Promote it to a public `conditions.leaves`, keeping `_leaves` as an alias, so no existing caller breaks.)

- [ ] **Step 1: Write failing tests**, using `compile_policy` on `FORM_EXAMPLE` variants:
  - `test_finds_witness_between_approve_over_500_and_block_over_10000_from_other_source`: the witness has `args.amount > 10000` and `tool.name` in both tool lists, and both policies match it under the one evaluator. Assert that re-evaluation is True for both.
  - `test_no_witness_when_ranges_do_not_meet` (`gt 500` vs `lt 400`) → None.
  - `test_no_witness_when_tool_lists_are_disjoint` (refund.issue vs credit.issue only) → None.
  - `test_witness_never_relies_on_missing_field` (one condition uses `args.currency`, an `eq` with no example) → the found witness contains `args.currency`.
  - `test_examples_tried_first`: when a complete example matches both, it is returned verbatim.
  - `test_unreadable_condition_raises` → `ConditionError`.
  - `test_witness_search_is_bounded`: two conditions with `in` lists of 200 values on 3 fields and no overlap; returns None in < 2 s.
  - `design_time_pair` table:
    - same source → False (brief test 2: tiered);
    - same outcome → False;
    - notify involved → False;
    - different checkpoint → False;
    - different source + approve/block → True.
  - `pair_key("B-1","A-1") == pair_key("A-1","B-1") == "A-1|B-1"`.
- [ ] **Step 2: Run** `PYTHONPATH=.:src CUDA_VISIBLE_DEVICES="" /home/muthu/PycharmProjects/BP_Backend/venv/bin/python -m pytest tests/agent_policy/test_conflict_detect.py -v` → FAIL (module missing).
- [ ] **Step 3: Implement** as above.
- [ ] **Step 4: Run** → PASS. Also run `tests/agent_policy/test_conditions.py` (the alias). Prove the guard fails: remove the tool-list check in `_matches`, see the disjoint-tools test go red, then restore it.
- [ ] **Step 5: Commit** `feat(agent-policy): code-found witness for policy conflicts`.

---

### Task 3: `policy-conflict/1` payload and approver summary (pure)

**Files:**
- Create: `src/services/agent_policy/conflict_payload.py`
- Test: `tests/agent_policy/test_conflict_payload.py`

**Interfaces:**
- Consumes: `conflict_detect.deciders_of`, `conflict_detect.design_time_pair`, `conditions.condition_fields`.
- Produces:
  - `case_id(decision_id: int) -> str` → `"pc_<id>"`; `parse_case_id(s: str) -> int | None`. This also accepts bare digits; anything else gives None.
  - `policy_entry(doc: dict) -> dict` → `{id, version, outcome, situation, deciders, owner, businessArea, source:{document, reference, excerpt}}`. `deciders` is omitted when not approve, as in the brief's sample. `businessArea` is `"<primary> / <subArea>"`.
  - `condition_args(docs: list[dict], args: dict) -> dict`: only the `args.*` keys any listed condition uses (the brief's `args` rule).
  - `condition_values(docs, flat_ctx: dict) -> dict`: the flat values of every condition field present (used as the live witness).
  - `why_line(policies: list[dict]) -> str`, deterministic:
    - block + approve → `"One policy needs approval from <deciders of the approve policy, joined with ' then '>; the other does not allow this at all."`
    - two approves → `"The policies name different approvers: <A> and <B>."`
    - otherwise → `"The policies say different things about this action."`
  - `policy_options(a: dict, b: dict) -> list[str]`. In order: `keep_both:<A>`, `keep_both:<B>`, `change:<A>`, `change:<B>`, `limit:<A>`, `limit:<B>`, `retire:<A>`, `retire:<B>`. When either policy is a block, the only `keep_both` kept is the block's (open decision 1).
  - `scope_of(option: str) -> str` → `"standing_rule"` for `keep_both:*`, else `"this_action"`.
  - `build(kind: str, *, raised_at: str, policies: list[dict], overlap_example: dict, standing_rules: list[dict], prior: dict, options: list[str], respond_within: str | None, on_timeout: str, action: dict | None = None, case: str | None = None) -> dict`. Returns exactly the brief §4.4 keys: `schema`, `caseId`, `kind` (`"policy"`/`"live"`), `raisedAt`, `action` (omitted for policy), `policies`, `overlap:{example}`, `standingRules`, `priorDecisions:{sameConflict, lastOutcome}`, `options`, `respondWithin`, `onTimeout`. It also includes `summary`: `{actionPlain, why, policies:[{id, situation, outcome, owner, excerpt}], prior, options, respondWithin}`. The generated `why` sits beside the verbatim excerpts and never replaces them (§4.5).
  - `to_columns(payload: dict) -> dict`: the design §5 inbound mapping.
    - `subject_type`: `policy_conflict` | `live_conflict`;
    - `facts`: the `action`, `policies`, `standingRules`, `priorDecisions`, `summary` and `schema` keys;
    - `evidence`: `[{"kind": "overlap", "example": {...}}]`;
    - `options`: list;
    - `on_timeout`.
    
    `respond_by` is computed by the caller from `respondWithin`.
  - `returned_decision(row: dict) -> dict`: the design §5 return mapping. It reads an action row's `decision`, `decision_scope`, `actioned_by`, `actioned_at` and `override_reason`, and returns `{caseId, decision, scope, decidedBy, decidedAt, reason}`.

- [ ] **Step 1: Write failing tests:**
  - `test_build_matches_brief_shape`: the key set equals the brief §4.4 sample's key set for `live`. `policy` omits `action`.
  - `test_live_args_only_condition_fields`: args `{amount, currency, customer_email, note}` with conditions on `args.amount` give `{"amount": ...}` only (brief test 13).
  - `test_options_for_block_pair_offer_only_block_keep_both`;
  - `test_why_line_block_vs_approve` uses the brief's example wording;
  - `test_summary_keeps_excerpt_verbatim`: an excerpt with odd spacing and quotes is byte-equal;
  - `test_case_id_round_trip` and `parse_case_id("pc_x")` is None;
  - `test_to_columns_maps_design_section_5`;
  - `test_returned_decision_maps_design_section_5`.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(agent-policy): policy-conflict/1 payload and approver summary`.

---

### Task 4: Policy cases: raise, dedup, detect on save/extraction/scan

**Files:**
- Create: `src/services/agent_policy/conflict_cases.py`
- Modify: `src/api/routers/agent_policies.py` (`create`, `save`: call `conflict_cases.after_save` after the repo returns and after `live_policies.invalidate()`); `src/services/agent_policy/extraction_run.py` (`_decide`: after a successful `create_draft`/`save_version`, call `conflict_cases.after_save(conn, key)`); `src/services/backend_scheduler.py` (`_register_agent_policy_conflict_scan_job`, interval 1 h, initial delay 5 min, env `AGENT_POLICY_CONFLICT_SCAN=on|off`, default on, same shape as `_register_agent_policy_approvals_job`).
- Test: `tests/agent_policy/test_conflict_cases_live.py`, `tests/agent_policy/test_conflict_cases.py` (fake cursor / pure parts), plus an extension to `tests/agent_policy/test_approvals_job.py` for the job registration.

**Interfaces:**
- Consumes: Task 2 and Task 3 functions; `deciders.load_map`, `deciders.unmapped`; `approvals._tx`; `approvals.ADMIN_RECIPIENT`; `settings.load_settings`.
- Produces:
  - `SUBJECT_POLICY = "policy_conflict"`, `SUBJECT_LIVE = "live_conflict"`.
  - `raise_policy_case(cur, a: dict, b: dict, example: dict, *, raised_by: str, now: datetime, mapping, proposal: dict | None = None) -> int | None`. Runs inside the caller's transaction. Returns the new decision_id, or None when:
    - the pair already has an open policy case; or
    - an in-force standing rule covers the pair; or
    - a decided case exists for the pair whose `policy_versions` are both ≥ the current versions (nothing changed since it was decided).
    
    It writes:
    - the `bp_decision` row: `subject_type='policy_conflict'`, `subject_id=<pair_key>`, `decision='resolve_conflict'`, `resolution='escalated'`, `status='open'`, `policy_name=<pair_key>`, `rationale=why_line`, `created_by='system:conflict_detector'`, `agent='agent_policy_conflicts'`, `respond_by NULL`, `on_timeout 'none'`, `decision_scope NULL`, and the columns from `to_columns(build("policy", ...))`;
    - the `bp_agent_policy_conflict` row. Before any check, take `pg_advisory_xact_lock(hashtext('agent_policy_conflict:'||pair_key))`, then run the three dedup checks above, then insert. The partial unique index stays as the backstop: a unique violation there is a bug, so let it raise and roll back;
    - notifications: to both owners (`link='conflict:<id>'`, `firing_id NULL`, message `"Policies <A> and <B> conflict; a decision is needed."`, with no input values). For unlinked owners, `facts.unroutable=[names]` plus a notification to `approvals.ADMIN_RECIPIENT`.
  - `detect_for(conn, policy_key: str, *, now=None, among: list[str] | None = None, raised_by: str = "save") -> list[int]`.
    - It loads the saved policy's latest compiled doc plus its `form_state.examples` inputs.
    - Others are every non-retired policy (both its live and its latest version, de-duplicated), narrowed to `among` when given. Tests on the shared DB pass `among`.
    - For each pair where `design_time_pair` holds and `witness` returns an input, it calls `raise_policy_case`.
    - It skips a pair on `ConditionError`.
    - It runs in one `approvals._tx` transaction.
  - `detect_all(conn, *, now=None) -> dict`: `{"pairs": n, "raised": n, "errors": n}`. One transaction per policy; it never raises.
  - `after_save(conn, policy_key) -> None`: best effort. It calls `detect_for` and logs `logger.error("conflict detection failed for %s: %s", key, type(exc).__name__)` on failure. The save is never undone; the hourly scan catches what this missed.

- [ ] **Step 1: Write failing tests** (live; `world` fixture style; policies created through `repo.create_draft` with tools `tst_<tag>`, because ids can't be deleted; retire them at teardown):
  - `test_save_raises_policy_case_with_code_found_witness`. Save FIN-style approve > 500 (doc "TST Finance <tag>") and CUS-style block > 10000 (doc "TST Customer <tag>"). Then:
    - one open `policy_conflict` case;
    - `evidence[0].example` matches both (re-evaluate);
    - `facts.policies` has both, with verbatim excerpts;
    - options as in Task 3;
    - a conflict row with `kind='policy'` and `raised_by='save'` (brief test 11a).
  - `test_tiered_same_source_raises_nothing` (brief tests 2 and 11b).
  - `test_same_outcome_raises_nothing` (brief test 11b).
  - `test_resave_does_not_duplicate_open_case`.
  - `test_concurrent_raise_one_case_per_pair`: two threads call `detect_for` at once; exactly one open case.
  - `test_decided_pair_re_raised_only_after_a_new_version_still_overlaps`.
  - `test_owner_unlinked_marks_unroutable_and_notifies_administrators`.
  - `test_extraction_save_triggers_detection` (pure: patch `conflict_cases.after_save` and assert `_decide` calls it once per saved draft, never for `unchanged`).
  - `test_after_save_never_raises` (detect_for patched to raise).
  - The scan job registers, honours `AGENT_POLICY_CONFLICT_SCAN=off`, and calls `detect_all`.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement.** In the router, the `create`/`save` hooks go in the success path only (not in `finally`), after the connection closes, using a fresh `_conn()`.
- [ ] **Step 4: Run** → PASS. Also run `tests/agent_policy/test_router.py`, `test_extraction_run.py` and `test_repo_live.py`, which must stay green. Prove the guard fails: remove the advisory lock, run the concurrency test 20× and see a duplicate or a unique-violation error, then restore it.
- [ ] **Step 5: Commit** `feat(agent-policy): policy conflicts raised on save, extraction and hourly scan`.

---

### Task 5: Deciding a policy case; standing rules into `conflicts[]`; moot close; history

**Files:**
- Modify: `src/services/agent_policy/conflict_cases.py`; `src/repositories/agent_policy_repo.py` (`live_documents`, `list_policies`, `get_policy`); `src/api/routers/agent_policies.py` (`retire`: after success, `conflict_cases.close_moot`).
- Test: `tests/agent_policy/test_conflict_decide_live.py`, `tests/agent_policy/test_conflict_overlay.py` (pure overlay).

**Interfaces:**
- Consumes: Task 4; `deciders.eligible`; `live_policies.invalidate`; `conflict_payload.returned_decision`, `scope_of`, `case_id`.
- Produces:
  - `class ConflictRefused(Exception)` with `code`, `message`, `status` (the same shape as `approvals.ApprovalRefused`).
  - `decide_policy(conn, decision_id: int, *, principal, option: str, reason: str | None, limit_text: str | None, now: datetime, mapping=None) -> dict`. Refuses:
    - an unknown case → 404 `not_found`;
    - an option not in the case's `options` → 422 `unknown_option`;
    - a blank reason → 422 `reason_required`;
    - `limit:*` with blank `limit_text` → 422 `limit_required`;
    - a case that is not open → 409 `not_open`;
    - a principal not linked to either owner name → 403 `not_eligible` (open decision 3; Admin is not automatic).
    
    It locks the case `FOR UPDATE` inside `approvals._tx`, then writes:
    - the action row: a new `bp_decision` row, `subject_type='policy_conflict'`, same `subject_id`, `decision=<option>`, `decision_scope=scope_of(option)`, `status='actioned'`, `actioned_by`, `actioned_at=now`, `override_reason=reason`, and `facts` = the case facts plus `{"limitText": ..., "versionsAtDecision": {key: latest_version}}`;
    - the original row's `status='actioned'`;
    - the conflict row: `is_open=false`, `outcome=option`, `decided_by`, `decided_at`, `by_person=true`.
    
    For `keep_both:<P>`, it supersedes any in-force rule for the pair (`superseded_at=now`, `superseded_by=<action id>`) and inserts a rule (`prevails=P`, `yields=<other>`, `rule_text="<P> takes priority over <other>"`).
    
    It notifies both owners. After commit, it calls `live_policies.invalidate()`. It returns `returned_decision(action_row)` plus `{"applied": "standing_rule" | "draft_pending" | "retire_pending"}`.
  - `rules_for(cur, keys: list[str]) -> list[dict]`: the in-force rules touching any key.
  - `overlay(cur, docs: list[dict]) -> list[dict]`. Sets each doc's `conflicts` to `[{"with": other, "rule": rule_text, "caseId": case_id(decision_id), "decidedAt": iso, "prevails": prevails}]` (schema-compatible: the four required keys plus `prevails`). It does not mutate its input, because cached copies are shared.
  - `close_moot(conn, policy_key: str, *, now) -> int`. Closes every open policy case naming the key: an action row with `decision='moot'`, `actioned_by='system:retired'`, reason `"<KEY> was retired"`, `decision_scope='this_action'`; the conflict row is closed with `by_person=false`. It notifies the owners.
  - `history_for(cur, policy_key: str, latest_version: int, status: str) -> dict` → `{"conflicts": [{caseId, kind, isOpen, otherPolicies, raisedAt, decision: returned_decision | None}], "pendingAction": {caseId, action: "change"|"limit"|"retire", changeNote, limitText, decidedAt} | None}`.
    - `pendingAction` is the latest `change`/`limit`/`retire` decision on this key, while `latest_version == versionsAtDecision[key]`, or for `retire` until the status is `retired`.
    - `changeNote` = `"Conflict decision pc_<id>: <option label> — <reason>"`.
  - `open_cases_by_policy(cur) -> dict[str, list[str]]`: policy key → open policy `caseId`s.
  - **Repo:**
    - `live_documents` returns `overlay(cur, docs)`, which feeds both the orchestrator feed and `live_policies.load`;
    - `list_policies` rows gain `openConflicts: [caseId]`;
    - `get_policy` gains `conflicts` and `pendingConflictAction` from `history_for`, and Admin's live-version `compiled` is overlaid too.

- [ ] **Step 1: Write failing tests:**
  - `test_standing_rule_written_into_conflicts_of_both`: decide `keep_both:A`. `repo.live_documents` shows the entry on both A and B, `contract.validate` still passes, and the `/orchestrator/agent-policies/v2/live` feed (through the app, as in `test_feed_through_app.py`) carries it (brief test 16).
  - `test_decision_stored_on_both_histories` (brief test 15).
  - `test_change_decision_opens_pending_draft_and_writes_no_version`: the version count is unchanged and `pendingConflictAction.action == "change"`. After the owner saves a new version, `pendingConflictAction` is None (brief test 17).
  - `test_limit_decision_requires_limit_text_and_writes_no_version` (brief test 17).
  - `test_retire_decision_is_pending_until_two_step_retire`.
  - `test_not_eligible_without_owner_link`;
  - `test_reason_required`;
  - `test_unknown_option_refused`;
  - `test_second_keep_both_supersedes_first_rule`;
  - `test_retire_closes_open_case_as_moot` (also: decide afterwards → 409);
  - `test_overlay_does_not_mutate_cached_docs` (pure).
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** → PASS, plus `tests/agent_policy` fake and live in full. Prove the guard fails: make `overlay` skip the `yields` side, and see the both-policies test go red.
- [ ] **Step 5: Commit** `feat(agent-policy): owners decide policy conflicts; standing rules reach conflicts[]`.

---

### Task 6: Live-conflict classification (pure)

**Files:**
- Create: `src/services/agent_policy/conflict_engine.py`
- Test: `tests/agent_policy/test_conflict_engine.py`

**Interfaces:**
- Consumes: `enforcement.Verdict` (hits carry `id`, `outcome`, `unreadable`, `policy`); `conflict_detect.same_source`, `deciders_of`, `pair_key`.
- Produces:

```python
@dataclass
class LiveConflict:
    kind: Optional[str] = None        # None | "block_record" | "auto" | "human"
    involved: List[Dict[str, Any]] = field(default_factory=list)   # hits in any conflicting pair
    pairs: List[Tuple[str, str]] = field(default_factory=list)      # (key, key), sorted
    required: List[Dict[str, Any]] = field(default_factory=list)   # approve hits still needing approval
    last_level_only: Set[str] = field(default_factory=set)          # keys routed to their last level
    rules: List[Dict[str, Any]] = field(default_factory=list)       # standing rules applied (auto)


def conflicting(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    pa, pb = a["policy"], b["policy"]
    if a["id"] == b["id"] or same_source(pa, pb):
        return False                    # tiered policies from one source are not conflicts
    return a["outcome"] != b["outcome"] or deciders_of(pa) != deciders_of(pb)


def classify(verdict) -> LiveConflict:
    deciding = [h for h in verdict.blocks + verdict.approvals if not h.get("unreadable")]
    pairs, involved = [], []
    for i, a in enumerate(deciding):
        for b in deciding[i + 1:]:
            if conflicting(a, b):
                pairs.append(tuple(sorted((str(a["id"]), str(b["id"])))))
                for h in (a, b):
                    if not any(x is h for x in involved):
                        involved.append(h)
    if not pairs:
        return LiveConflict()
    if any(h["outcome"] == "block" for h in involved):
        # design ruling: Not allowed still blocks; the case is for the record and a policy fix
        return LiveConflict("block_record", involved, pairs)
    rules = {}
    for h in involved:
        for r in h["policy"].get("conflicts") or []:
            if isinstance(r, dict) and r.get("with") and r.get("prevails"):
                rules[pair_key(str(h["id"]), str(r["with"]))] = r
    if all(pair_key(*p) in rules for p in pairs):
        losers = {(set(p) - {rules[pair_key(*p)]["prevails"]}).pop() for p in pairs
                  if rules[pair_key(*p)]["prevails"] in p}
        required = [h for h in verdict.approvals if str(h["id"]) not in losers]
        if required and len(losers) == len({k for p in pairs for k in p}) - 1:
            return LiveConflict("auto", involved, pairs, required, set(),
                                [rules[pair_key(*p)] for p in pairs])
    return LiveConflict("human", involved, pairs, list(verdict.approvals), {str(h["id"]) for h in involved})
```

The `auto` branch applies only when the rules leave exactly one standing winner among the involved policies. Anything partial or circular goes to `human`.

- [ ] **Step 1: Write failing tests:**
  - no conflict when one policy matched;
  - no conflict between same-source tiered approves with different deciders;
  - no conflict between two approves with the same deciders from different sources (brief §4.1 "apply it normally");
  - `block_record` for block + approve from different sources;
  - unreadable block plus approve → None (it fails closed through stage 3, not as a conflict);
  - `human` for two approves with different deciders; `last_level_only` names both;
  - `auto` with rule A>B → `required == [A]`, and `rules` holds it;
  - a partial rule set of 3 policies → `human`;
  - a circular rule set → `human`;
  - notify hits never involved (open decision 2).
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(agent-policy): classify a multi-policy match as a live conflict`.

---

### Task 7: Live conflicts in the gate; settle on group close; repeat-5 proposal

**Files:**
- Create: `src/services/agent_policy/conflict_live.py`
- Modify:
  - `src/services/agent_policy/gate.py`: `_before_tool` calls `conflict_engine.classify(verdict)` right after `enforcement.check`, and `_open_cases` takes the classification.
  - `src/services/agent_policy/approvals.py`: `_insert_case` gains a keyword `last_level_only: bool = False` (when true, `levels = levels[-1:]`, so `on_timeout` becomes `reject`); `_close_group` calls `conflict_live.settle_for_group(...)` when `state["open"] == 0`, through a lazy import.
- Test: `tests/agent_policy/test_conflict_live_gate.py` (scripted chat stand-in through `run_tools`, as `test_gate.py` does) and `tests/agent_policy/test_conflict_live_live.py`.

**Interfaces:**
- Consumes: Tasks 3–6; stage 3 `gate.record_matches`, `gate._open_case_for`, `approvals._insert_case`, `approvals.group_state`; `settings.load_settings()["live_conflict_repeat"]`.
- Produces:
  - `conflict_live.insert_live(cur, lc: LiveConflict, *, ctx: dict, action: dict, now, default_response_time: str, status: str, decision: str | None = None, actor: str | None = None, reason: str | None = None) -> int`. It writes a `bp_decision` row, `subject_type='live_conflict'`, `subject_id=<pair_key of all involved>`, `decision='approve_or_reject'` (open) or the auto decision (`'block'` | `'standing_rule'`), with `facts`/`evidence`/`options` from `build("live", ...)`:
    - `action.args = condition_args(...)`;
    - `overlap.example = condition_values(...)`;
    - `standingRules` = every involved doc's `conflicts[]`;
    - `priorDecisions` from the conflict table: the count and last outcome of settled live cases for that pair_key;
    - `respond_by` = now + the longest member `respondWithin`; `on_timeout='reject'`.
    
    It also writes a conflict row with `kind='live'` and `raised_by='live'`. For `status='actioned'` (block_record or auto), the case is closed immediately with `actioned_by` = `'system:not_allowed'` or `'system:standing_rule'`, `decision_scope` = `'this_action'` (block) or `'standing_rule'` (auto), and the conflict row gets `by_person=false`.
  - `settle_for_group(cur, case: dict, state: dict, *, refused: str | None, actor: str, reason: str | None, now) -> int | None`. For the `facts.liveConflict` id of any group member: if that live case is open, it writes the action row (`decision` `'approve'` when every member approved, else `'reject'`; `decision_scope='this_action'`; `actioned_by=actor`; `override_reason=reason`), closes the original, and closes the conflict row with `by_person = not actor.startswith("system:")`. When `by_person`, it then calls `maybe_propose(cur, live_id, now)`. It returns the action row id.
  - `maybe_propose(cur, live_id: int, *, now, threshold: int) -> list[int]`. When the last `threshold` settled `by_person` live cases with the same `pair_key` all share one outcome (open decision 4), it raises, for each pair in the live case's `pairs`, `conflict_cases.raise_policy_case(..., raised_by='repeat', proposal={"from": "repeat", "count": threshold, "outcome": outcome})`. Dedup is the same as Task 4. The example is the live case's `overlap.example`.
  - **Gate behaviour by `lc.kind`**, all inside the existing single gate transaction:
    - `None` → stage 3 behaviour, byte-for-byte (assert it).
    - `"block_record"` → stage 3 block path, plus `insert_live(..., status='actioned', decision='block')`. For each pair containing a block, `raise_policy_case(..., raised_by='live')` with the live witness. `to_agent` gains `"conflictCaseId": "pc_<id>"`.
    - `"auto"` → open stage 3 cases only for `lc.required`, with normal levels. Record `insert_live(..., status='actioned', decision='standing_rule', reason=<rule texts>)`. `to_agent` gains `conflictCaseId`.
    - `"human"` → in `_open_cases`, after the reuse lookup:
      - if every approval was reused, the live id comes from the reused facts and no new live row is written;
      - otherwise `insert_live(..., status='open')` and open member cases with `last_level_only = key in lc.last_level_only` and `extra_facts={"liveConflict": live_id, ...}`.
      
      `to_agent` is the stage 3 `paused_for_approval` shape plus `conflictCaseId`.
  - Notifies are always written (stage 3, unchanged).

- [ ] **Step 1: Write failing tests** (scripted chat stand-in; no model):
  - `test_live_multi_match_pauses_and_sends_live_case_with_condition_fields_only`: two approve policies from different sources with different deciders. The tool is not run, one open `live_conflict` exists, `facts.action.args` keys equal the condition fields only, and two member cases each have one level, that policy's last decider (brief test 13).
  - `test_no_precedence_between_approve_policies`: neither member is dropped, and the tool runs only after both approve (via `approvals.act` + stub replay).
  - `test_block_in_conflict_still_blocks_and_is_recorded`: result `blocked`; a closed live case with `decision='block'`; a policy case raised for the pair.
  - `test_standing_rule_decides_automatically`: rule A>B in A's `conflicts[]`. Only A's normal case opens, and the live case is closed with `decision='standing_rule'`.
  - `test_live_conflict_timeout_rejects` (brief test 14): sweep past `respond_by`; the member case is rejected by `system:timeout`, its sibling is closed, the live case is settled `reject` with `by_person=false`, and the tool never runs.
  - `test_live_conflict_record_failure_refuses` (brief test 14, "unreachable"): patch `conflict_live.insert_live` to raise. The agent gets `policy_check_unavailable`, and no firing, case or notification rows are left (one transaction).
  - `test_repeat_call_reuses_live_conflict`;
  - `test_same_conflict_decided_same_way_5_times_raises_policy_case` (brief test 18): five live cases are approved by people; the 5th raises one `policy_conflict` with `raised_by='repeat'` and `facts.proposal.count == 5`. A 6th raises no second case.
  - `test_mixed_outcomes_reset_the_count`;
  - `test_timeouts_do_not_count`;
  - `test_threshold_comes_from_company_setting` (setting = 2);
  - `test_notify_still_sent_in_live_conflict`.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement.** Keep `_open_cases` changes additive. When `lc` is None, the code path must be the stage 3 one.
- [ ] **Step 4: Run** → PASS. Run the full `tests/agent_policy` suite (fake and live), plus `tests/engines` and `tests/approvals`. Prove the guard fails: make `classify` return None and see the live-case test go red; separately, remove `last_level_only` and see the one-level assertion go red.
- [ ] **Step 5: Commit** `feat(agent-policy): live conflicts pause the action and go to the decision engine`.

---

### Task 8: Endpoints, hidden subject types, approval-card conflict block

**Files:**
- Create: `src/services/agent_policy/conflict_views.py`
- Modify:
  - `src/api/routers/agent_policies.py`: the routes below, declared **before** `@router.get("/{key}")`;
  - `src/api/routers/decisions.py`: `HIDDEN_SUBJECT_TYPES` gains `"policy_conflict"` and `"live_conflict"`;
  - `src/services/agent_policy/approval_views.py`: `case_view` adds `conflict` when `facts.liveConflict` is set, and `_link_ref` maps `conflict:<id>` → `{"conflictId": id}`.
- Test: `tests/agent_policy/test_conflict_endpoints_live.py`, `tests/agent_policy/test_conflict_endpoints_scrub.py`, plus an extension to `tests/agent_policy/test_decisions_hide_approvals_live.py`.

**Interfaces:**
- Consumes: Tasks 3, 5 and 7.
- Produces:

| Method + path | Role | Returns |
|---|---|---|
| GET `/agent-policies/conflicts?status=open\|closed\|all` | Viewer | `{"conflicts": [view]}`. Policy cases only, newest first. `canDecide` is set per caller. |
| GET `/agent-policies/conflicts/{decision_id}` | Viewer | one view plus `history` (action rows as `returned_decision`). 404 for a live case or an unknown id. |
| POST `/agent-policies/conflicts/{decision_id}/decide` | Viewer (eligibility decides) | body `{option, reason, limitText?}` → `decide_policy`. 403/404/409/422 as raised. Audited as `agent_policy.conflict_decide` before and after, the way stage 3's `decide` is. |

- **View** = `{caseId, decisionId, status, raisedAt, raisedBy, policies:[policy_entry + {latestVersion, status}], example (sensitive masked unless canDecide), why, prior, options, optionLabels, respondWithin, unroutable, proposal, canDecide, decision}`.
- **Approval case `conflict` block** = `{caseId, why, policies, prior, options, respondWithin}`. It is masked with the same sensitive set as the case.
- **Every new GET** is checked through `scrub_payload` in `test_conflict_endpoints_scrub.py`. Any withheld field becomes a strict `xfail` naming open decision 5. **Do not** edit `_AGENT_POLICY_SCREEN_EXEMPT` until the user rules. When the ruling arrives, add exactly `("GET", r"^/agent-policies/conflicts$")` and `("GET", r"^/agent-policies/conflicts/[0-9]{1,18}$")` and turn the xfails into passes.

- [ ] **Step 1: Write failing tests:**
  - `/decisions` list and `/decisions/{id}` return neither subject type (seed both). Existing subject types are unchanged: run the decisions router tests before and after.
  - the three routes: role floors, the 404 for a live id, `canDecide`, and the audit rows written;
  - `test_conflict_view_masks_sensitive_witness`;
  - `test_live_member_view_masks_for_non_approver`;
  - a notification with a `conflict:` link returns `conflictId`;
  - `/conflicts` is not swallowed by `/{key}` (path order);
  - the scrub test (strict xfail where withheld).
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** → PASS, plus `tests/engines`, `tests/approvals`, the decisions router tests, `tests/api/test_p8_phase2_suppliers_support_rest.py::test_every_write_endpoint_resolves_the_caller` (the new POST uses `gateway_principal`) and `scripts/p8_endpoint_scan.py`.
- [ ] **Step 5: Commit** `feat(agent-policy): conflict endpoints; conflict cases hidden from /decisions`.

---

### Task 9: Gateway routes

**Files:** `src/modules/agent-policy/agent-policy.controller.ts`, `agent-policy.yml`, `agent-policy.controller.spec.ts` (gateway worktree).

**Interfaces:**
- Consumes: Task 8's paths and bodies.
- Produces:
  - `GET agent-policies/conflicts` (Viewer; `status` allow-list `open|closed|all`, default `open`);
  - `GET agent-policies/conflicts/:decisionId` (Viewer; the existing `id()` allow-list `^[0-9]{1,18}$`);
  - `POST agent-policies/conflicts/:decisionId/decide` (Viewer; body allow-list: `option` matches `^(keep_both|change|limit|retire):[A-Z]{3}-[0-9]{4,}$`, `reason` is a string ≤ 2000 chars, `limitText` is an optional string ≤ 500 chars; anything else gives 400).
  
  All three are declared **above** `@Get(':key')`, so `conflicts` is never read as a policy key. Matching `agent-policy.yml` http events are added.

- [ ] **Step 1: Write failing jest specs:**
  - each route forwards with the verified identity and `X-Gateway-Key`;
  - a bad id → 400;
  - a bad option → 400;
  - an unknown body key → 400;
  - `GET agent-policies/conflicts` does not hit `getOne`.
- [ ] **Step 2: Run** `npx jest src/modules/agent-policy` → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** → PASS, then the full jest suite against its baseline.
- [ ] **Step 5: Commit** `feat(agent-policy): conflict routes` (gateway).

---

### Task 10: Screens

**Files:**
- Create: `src/modules/SpendIQ/agentPolicy/conflicts.js`, `conflicts.test.js`, `engineWiring.stage4.contract.test.js`
- Modify: `index.jsx` (`conflicts: agentPolicyConflicts` on `window.__SPENDIQ_AP__`); `engine.js`, adding the new `apPc*` functions and minimal wiring:
  - `apPoliciesTab` gains a `['conflicts','Conflicts']` view with a badge counting open cases the caller can decide;
  - one tag cell in `apListHTML` and one in `apInventoryHTML`;
  - one section call in `apFormHTML`;
  - one block call in `apApprovalCardHTML`;
  - one branch in the notification target.

**Interfaces:**
- Consumes: Task 9 routes; `GET /agent-policies` `openConflicts`; `GET /agent-policies/{key}` `conflicts` and `pendingConflictAction`; the approval case `conflict` block.
- Produces:
  - **Pure `conflicts.js` exports:**
    - `CONFLICT_TAG = 'Conflict sent for decision'`;
    - `optionLabel(option)`:
      - `keep_both:X` → "Keep both: X takes priority";
      - `change:X` → "Change X";
      - `limit:X` → "Limit X";
      - `retire:X` → "Retire X";
    - `decideReady({option, reason, limitText})`;
    - `statusText(view)`: Waiting for a decision / Decided: <label> / Closed: policy retired;
    - `pendingActionText(p)`;
    - `priorLine(prior)`: "Decided N times before; last: approved", or "First time";
    - `openCountForCaller(list)`;
    - `notificationIsConflict(n)`.
  - **`engine.js`:**
    - `apPcLoad()`, `apPcHTML()`, `apPcCardHTML(c)`, `apPcSelect(id, option)`, `apPcReason(id, text)`, `apPcLimit(id, text)`, `apPcDecide(id)` (two-step confirm via `apConfirmTwoStep`), `apPcOpen(caseId)`;
    - `apPcTag(row)` for list and inventory: it links to the case;
    - `apPcPolicySection(state)`: the history list plus pending action buttons;
    - `apPcOpenDraft(state)`: it puts the form into edit mode with `changeNote` prefilled, and for limit sets `limit.on=true` with `limit.text` filled. It **never** calls `apSave`;
    - `apPcRetire(state)` → the existing `apRetire(state)`, which is the two-step retire;
    - `apPcApprovalBlock(c)`: the live conflict summary on the approval card.
  - **Each card shows** (brief §4.5, in this order):
    1. the example action in one sentence (`summary.actionPlain`, or the witness values; masked values show "Hidden");
    2. each policy: situation, outcome label, owner, and the source excerpt word for word with section and document;
    3. the why line, labelled "Why they conflict";
    4. previous decisions;
    5. the options and the response time ("No deadline" for policy cases).
    
    It also shows the proposal banner ("Decided the same way 5 times: make it a standing rule?") when `proposal` is set, and "Nobody can decide yet: link <names>" when unroutable.

- [ ] **Step 1: Write failing vitest tests** (pure plus the no-DOM harness, as stage 3's `engineWiring.stage3.contract.test.js` does):
  - Decide is disabled without an option and a reason, and for limit without limit text;
  - `apPcOpenDraft` sets `changeNote`/`limit` and makes **no** API write (spy on `apApi`) (brief test 17, UI side);
  - `apPcRetire` goes through `apConfirmTwoStep`;
  - masked witness values never render;
  - the excerpt renders verbatim (escaped, not rewritten);
  - list and inventory show `CONFLICT_TAG` for rows with `openConflicts`;
  - `policyEdit`, `policyDelete` and `openFormModal` are byte-identical (hash check as in stage 3);
  - the CSV export still neutralises formula injection (rerun `inventory.test.js`).
- [ ] **Step 2: Run** `npx vitest run src/modules/SpendIQ/agentPolicy` → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `npx vitest run src/modules/SpendIQ` → only the 4 known failures.
- [ ] **Step 5: Commit** `feat(agent-policy): conflicts tab, conflict tag, decision-to-draft` (UI).

---

### Task 11: Live demonstration and verification

On the stage 3 local stack:
- backend :8010 from the worktree;
- gateway :3011 with `AUTH_BYPASS_GROUPS=PROCWISE_ADMIN,PROCWISE_FINANCE_REVIEWER_APPROVER`;
- UI :3010;
- bp_testdb.

Use the tool the stage 3 demonstration used (`get_policy`, see `specs/2026-10-08-agent-policy-governance-stage3-verification.md`), driven through `run_tools` with a scripted chat stand-in. Never the real model.

1. **Owners:** link the two owner names and two decider names in the decider map.
2. **Design time:** create policy A (approve, doc "Demo Finance"), then policy B (block, doc "Demo Customer"), overlapping. Record:
   - the policy case;
   - its witness, re-evaluated;
   - the "Conflict sent for decision" tag on both policies and in the inventory;
   - the owner notifications.
3. **Tiered:** create a tiered pair from one document. No case.
4. **Standing rule:** decide `keep_both:B` as an owner. Show `conflicts[]` on both policies in `/orchestrator/agent-policies/v2/live`.
5. **Change:** on a second conflicting pair, decide `change:<key>`. Open the draft from the decision in the UI, cancel it, and show the version count is unchanged.
6. **Live conflict:** activate two approve policies from different sources with different deciders. The scripted call is paused; record the live case's args (condition fields only) and the two one-level member cases. Approve both, then record that the replay ran.
7. **Block record:** a call matching A plus a block from another source is blocked, and a closed live case records it.
8. **Timeout:** let a live member case time out, then run the sweeper. The action is rejected and the live case is settled `reject`.
9. **Repeat 5:** set `live_conflict_repeat` to its default 5. Drive 5 identical live conflicts and approve each; the 5th raises a proposal policy case. Restore any setting changed.
10. **`/decisions`:** neither new subject type appears.
11. **Screens:** headless Chrome on the Conflicts tab, the policy form section, and the approval card conflict block, with zero console errors (brief test 22).

Write `specs/2026-10-09-agent-policy-governance-stage4-verification.md` with:
- every request and response (keys redacted);
- the red/green captures;
- the full suite results per repo against baseline;
- the diff summary per repo (ruling A).

Stop only your own processes. Retire your demo policies, because ids can never be deleted.

---

## Self-review notes (for the executor)

**Coverage of brief §8:**
- tests 2 and 11 → Tasks 2 and 4;
- test 13 → Task 7;
- test 14 → Task 7 (timeout plus record failure);
- test 15 → Task 5;
- test 16 → Task 5;
- test 17 → Tasks 5 and 10;
- test 18 → Task 7;
- test 21 → Task 10 rerun;
- test 22 → Task 11.

**§4.5 summary** → Tasks 3, 8 and 10. **"Store every decision against the policies involved"** → Task 5 `history_for`.

**Names used across tasks:**
- `pair_key`;
- `case_id`;
- `raise_policy_case`;
- `detect_for` / `detect_all` / `after_save`;
- `decide_policy`;
- `overlay`;
- `close_moot`;
- `history_for`;
- `open_cases_by_policy`;
- `classify` / `LiveConflict`;
- `insert_live`;
- `settle_for_group`;
- `maybe_propose`;
- `last_level_only`;
- `SUBJECT_POLICY` / `SUBJECT_LIVE`.

**Brief vs design/code contradictions** (the design note wins where it carries a ruling):
1. **Standing rules in the JSON.** Brief §4.2 says to store standing rules "on both policies' JSON". But `bp_agent_policy_version` rows are immutable (stage 1 trigger), and a new version would be a silent change. The rules are therefore overlaid from `bp_agent_policy_conflict_rule` at read time.
2. **Precedence.** Brief §4.3 and test 13 say "remove block→approve→notify; the orchestrator holds no precedence". The design ruling says a block still blocks. A live conflict involving a block is therefore record-only, and with open decision 2, only approve-vs-approve live conflicts (different deciders, different sources) ever reach a person.
3. **"Limit one policy".** Stage 3 enforcement ignores `scope.limit` (free text, `limitIgnored`). A limit decision changes nothing that is enforced. The pair is re-detected after the owner's save unless the condition itself narrows.
4. **The overlap check to replace.** Brief §4 says to replace the existing "shares an event and a tool" overlap check. None exists (design §1), so there is nothing to remove.
5. **Live conflict options.** The brief's sample options (`approve_by:CFO`, `reject_and_escalate_to_owners`) conflict with the approve/reject-only ruling.
6. **Live conflict approver.** The brief recommends the "most senior decider"; the design ruling says the last level of each policy, and all must approve. The engine has no routing for these subject types, so the fallback always applies.
7. **Stage 3's group cases.** Stage 3's ruling was "multiple approve policies → one case each, with full levels". It is replaced, only for conflicting pairs, by one-level member cases (the stage 3 ledger anticipated this: "stage 4 replaces this path").
8. **Owner notifications.** `bp_policy_notification.firing_id` was NOT NULL, but policy-case owner notifications have no firing row. Task 1 relaxes it, guarded by a CHECK.
