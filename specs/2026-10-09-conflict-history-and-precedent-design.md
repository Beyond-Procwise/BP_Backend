# Conflict history and precedent: design

Date: 2026-10-09. Status: awaiting user review.
Builds on agent-policy stage 4 (conflicts), which is live: `specs/2026-10-09-agent-policy-governance-plan-4-conflicts.md`
and `specs/2026-10-09-agent-policy-governance-stage4-verification.md`.

## 1. What the user asked for

> "conflict history needs to be documented and will be part of the decision engine to resolve"

The user's rulings (2026-10-09, in conversation):

| # | Question | Ruling |
|---|---|---|
| R1 | How far may the decision engine go using history? | **Resolve on clear precedent.** If the same clash was decided the same way by people before and nothing contradicts it, the engine resolves it, citing those cases. Any doubt → escalate to people with the history attached. |
| R2 | How much history is a precedent? | **The same number as the existing "decided the same way N times" setting (5).** At N the engine starts resolving AND the standing-rule proposal is raised. |
| R3 | May precedent approve, or only reject? | **Approve, reject, and capture the decision.** The action may run with no human approving it; the precedent is the approver of record. |
| R4 | Where does N live? | **"Ensure the 5 times is captured as a policy so a customer could change it if they wanted to."** |
| R5 | Design approved in chat | "yes, write the spec" |

Understanding (assumptions marked *):
- "Documented" means every conflict decision is kept in full and is readable and exportable: who or what decided, the choice,
  the reason, the policy versions, the example, and what it relied on. *It does not mean a separate written manual.*
- "Part of the decision engine" means a live clash consults its own history before it is put to people, under the engine's
  existing rules: decide only from looked-up facts, record the evidence, escalate rather than guess.

## 2. What exists today (stage 4)

- Every conflict case is a `proc.bp_decision` row (`subject_type` `policy_conflict` | `live_conflict`). Each is indexed by
  `proc.bp_agent_policy_conflict`: `pair_key` (sorted policy ids), `policy_versions`, `is_open`, `outcome`, `decided_by`,
  `decided_at`, `by_person`.
- A live clash goes, in order:
  1. A block involved → blocked, recorded (`block_record`).
  2. A standing rule covers it → decided automatically (`auto`).
  3. Otherwise → people (`human`): one-level member cases, and all must approve.
- When a live case settles, `conflict_live.maybe_propose` counts the last N person-decided live cases for the same `pair_key`.
  If all N agree, it raises a policy case proposing a standing rule. N comes from the company setting
  `agent_policy_settings.live_conflict_repeat` (default 5).
- Each new case carries `priorDecisions: {sameConflict, lastOutcome}`.
- **Gaps this design closes:**
  - `history_for` drops a live clash's decider reason, because it cannot tell who is reading (stage 4 final review ruling).
  - Nothing exports the history.
  - Nothing uses the history to decide.
  - N is an internal setting, not a customer-editable policy.

## 3. Design

### 3.1 Conflict history: one complete, readable record

**Stored (no change to what is written, plus three additions).** Each conflict decision row in `proc.bp_decision` already
holds the option, scope, actor, time, reason, the case facts and the example. Three additions:

- **`facts.decidedBy.kind`** — one of `person`, `standing_rule`, `precedent`, `timeout`, `block`, `retired`. It is set
  wherever a conflict case is closed (`decide_policy`, `close_moot`, `insert_live` for block_record/auto, `settle_for_group`,
  and the new precedent path). It is derived from what happened, never guessed from the actor string.
- **`facts.versionsAtDecision`** on live settlements too (policy cases already carry it).
- **`evidence[]`** on a precedent decision lists each cited past case (`decision_id`, outcome, `actioned_by`, `actioned_at`).

**One reader: `conflict_history.read(cur, *, policy_key=None, pair_key=None, viewer)`.** It returns entries newest first:
`{caseId, kind (policy|live), raisedAt, policies [{id, version}], example, decision {option, scope, decidedBy {kind, name},
decidedAt, reason}, citedCases [caseId], proposal}`. `history_for` (policy page) and the Conflicts screen detail use it, so
the two can never disagree again (the stage 4 D1 class of bug).

**Who sees what.** The reader is given the caller (`viewer`). It shows full reasons and example values to:
- anyone linked (decider map) to either policy's owner;
- anyone linked to a decider of either policy;
- the Admin role.

For everyone else, sensitive values are replaced with `•••` (`approval_views.mask_text`/`mask_witness` against the stored
args and the sensitive set). The reason is still shown, masked rather than dropped. This reverses the stage 4 ruling that
dropped live reasons.

**Export.** `GET /agent-policies/{key}/conflicts/history.csv` and `GET /agent-policies/conflicts/history.csv?pair=<pair_key>`
(Viewer floor; same masking per caller) return one row per decision:
- case, kind, raised;
- policies and versions;
- decided by (kind, name), decision, scope, decided at, reason;
- cited cases.

CSV cells are neutralised against formula injection with the existing `csvCell` rule (server side here). The gateway
forwards both routes; the UI adds an "Export conflict history" button on the policy form's conflicts section and on a
Conflicts-screen case.

### 3.2 The decision engine resolves a live clash on precedent

**Where.** A new function in the decision engine module, `src/engines/decision_engine.py`:
`decide_live_conflict(cur, lc, *, ctx, now) -> Decision`. It needs no `agent_nick`, follows the engine's two rules (facts
first, escalate rather than guess), and returns the module's existing `Decision` (`resolution` resolved | escalated,
`facts`, `evidence`).

**When.** In the gate, after `conflict_engine.classify`, only for kind `human`:
- block_record and auto are unchanged, so a block still wins and a standing rule still applies first;
- a repeat call that reuses open member cases does not consult precedent again;
- the replay re-check after approval does not consult precedent.

**The rule (deterministic).** Let N = the governed precedent count (§3.3).

1. N is 0 or null → **escalated** ("precedent is switched off").
2. N cannot be read (`LimitUnavailable`) → **escalated** ("precedent limit unavailable"), logged as a warning.
3. Look up the last N **settled live** cases with the same `pair_key` whose `policy_versions` equal the current versions of
   every involved policy, and that were decided by a person (`by_person = true`). Precedent and other automatic decisions
   never count, so the engine cannot reinforce itself.
4. Fewer than N found → **escalated** ("only k of N decisions by people on this exact clash").
5. All N have the same outcome → **resolved** with that outcome. Otherwise → **escalated** ("decisions disagree").
6. The self-approval bar (stage 3) binds precedent too (final review C1): when the requester is the same person
   (`approvals._same_person`) as the credited decider of any cited case, or as any member approver of one →
   **escalated** ("the requester decided an earlier case of this clash"). R3 does not override it.
7. Any lookup error → **escalated**, never resolved.

**On resolved approve.** In the gate's single transaction:
- write the live case closed: `decision='approve'`, `actioned_by='system:precedent'`, `decision_scope='this_action'`,
  `facts.decidedBy.kind='precedent'`, `evidence` = the N cited cases; the conflict row has `by_person=false`;
- write firing rows (result `allowed`, linked to the case);
- **the tool runs now**;
- notify both policies' owners and every decider of each involved approve policy (`"<action> ran on precedent: decided the
  same way N times before (pc_…)"`, no input values);
- `to_agent` carries `conflictCaseId`, `precedent: true` and `precedentCount` (N); the model reads the tool's own result
  followed by one line: "Note: this action ran without a person approving it, on precedent (decided the same way N times
  before, case pc_…)." (final review I1). Every other allowed call reads exactly as before.
- the live record's `created_at` and its decision time are one clock (`now`), on both rows; so are a block record's and a
  standing rule's (final review M1).

**On resolved reject.** The same record with `decision='reject'`. The tool is refused (result `blocked`, reason code
`refused_on_precedent`), and the agent is told why in one sentence. The same notifications are sent.

**On escalated.** The stage 4 `human` path, unchanged, plus `facts.history` = the reader's entries for this `pair_key` (masked
per approver on view), so the people deciding see what came before.

**Fail-closed.** Any exception inside the precedent path refuses the call with `policy_check_unavailable` and leaves
nothing behind, the same as every other gate failure. Precedent can only ever turn a "human" clash into a decided one when
every check passes.

**The standing-rule proposal.** `maybe_propose` is unchanged in what it counts (person decisions only), and it now reads N from
the same governed value. When N is reached the owners get the proposal to make it permanent. Meanwhile the engine is
already resolving on precedent, as ruled in R2. Precedent decisions do not count toward it.

### 3.3 N is a customer-editable policy

- **The row.** A new governed-limit row in `proc.bp_policy` (`policy_type='limit'`, no `applies_to`, the established
  pattern of the 2026-09-10 governed limits). `policy_identifier = 'agent_policy_conflicts'`, name
  `AgentPolicyConflictPolicy`, rules `{"precedent_count": 5}`, with a description that says what the number does and that
  0 switches precedent off.
- **The migration.** `deploy/sql/2026-10-13_agent_policy_conflict_precedent.sql` (+ rollback) inserts the row on **bp_testdb
  and bp_sqldb**. If `bp_admin_config.agent_policy_settings.live_conflict_repeat` holds a value, the row takes that value;
  otherwise 5. Additive and idempotent.
- **Reading.** Through `governed_limits.limit("agent_policy_conflicts", "precedent_count", cast=int)`, which already refuses
  when the value is missing. A new keyword `fresh=True` bypasses its cache for this read. Customers edit the row through the
  existing policy admin (gateway `PUT /policy/update/:id`), which writes the database directly. A cached value would
  otherwise keep the old number until a restart.
- **One source.** `live_conflict_repeat` is removed from `services/agent_policy/settings.py` DEFAULTS and from the settings
  screen. The proposal and precedent both read the governed value. A missing value means no proposal and no precedent, with a
  warning; never a default in code.

## 4. Error handling summary

| Situation | Result |
|---|---|
| Limit row missing or unreadable | precedent escalates; the proposal is skipped; warning logged |
| N = 0 or null | precedent off; the proposal is skipped |
| History lookup fails | precedent escalates (people decide) |
| Precedent path raises inside the gate | call refused `policy_check_unavailable`, one transaction rolled back |
| Policy changed version since the precedents | not the same versions → escalates |
| A block is involved | blocked as today (precedent never consulted) |

## 5. Testing (live on bp_testdb, among=-scoped, as stage 4)

- Precedent approves after N person approvals, at identical versions: the tool runs; the live case has `system:precedent`,
  `decidedBy.kind=precedent` and evidence citing exactly those N cases. The owners and deciders are notified.
- Precedent rejects after N person rejects: the tool is refused with `refused_on_precedent`.
- No precedent below N; mixed outcomes escalate; a version change escalates.
- Precedent decisions never count toward a later precedent or toward the proposal (self-reinforcement test).
- N from the governed row: N=2 works; N=0 switches it off. A missing row escalates and skips the proposal. An edit
  takes effect without a restart (`fresh=True`).
- A precedent-path exception refuses the call and leaves nothing behind.
- The stage 4 byte-for-byte guard still holds when no conflict is classified.
- History reader:
  - the full reason for an owner, a decider and an Admin; masked values for a stranger;
  - the policy page and the Conflicts detail return identical entries;
  - `decidedBy.kind` is correct for every closing path.
- CSV export: columns, masking per caller, and formula-injection neutralised.
- Migration: the row exists on both databases with the copied value; idempotent; rollback.
- Prove each guard fails (break it on purpose, capture red, restore).
- One live demonstration on the running stack: 5 people-approved identical clashes, then the 6th runs on precedent; the
  export is shown.

## 6. Out of scope

- Precedent for design-time policy cases. Those are about policy wording; the stage 4 dedup already stops a decided pair
  re-raising until a policy changes.
- Precedent overriding a block or a standing rule.
- Fuzzy precedent ("similar" clashes). Only the identical pair at identical versions counts.
- A separate written manual. The history is documented as data, on screen and exportable.

## 7. Deployment note

The running stack has `AGENT_POLICY_CONFLICT_SCAN=off` (bp_testdb holds ~3,500 stale test drafts). Precedent is a live-call
path, unaffected by that switch. The migration must be applied to both databases before the code is deployed, because a
missing limit row means precedent never applies (it escalates), and that is safe but silent.
