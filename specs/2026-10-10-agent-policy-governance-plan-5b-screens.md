# Agent Policy Governance, Stage 5b (screen gaps) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. One implementer per lane at a time; test first; no push, no merge.

**Goal:** Close the screen gaps left after stage 5 (learning): names instead of sign-in ids, a settings screen for the learning thresholds, an owner's view of the Suggestions queue, a notification when a suggestion is raised, honest states and messages, the three small defects carried from stages 3-4, translations for the new phrases, and one browser run of everything.

**Builds on:** `specs/2026-10-10-agent-policy-governance-plan-5-learning.md` (read its Global Constraints: they all bind here), its ledger of rulings, and `specs/2026-10-10-agent-policy-governance-stage5-verification.md` (with its addendum). Same three worktrees, same branch `agent-policy-stage5`:
- BP `/home/muthu/PycharmProjects/BP_Backend/.claude/worktrees/agent-policy-stage5`
- Gateway `/home/muthu/PycharmProjects/beyond-procwaise-Api-worktrees/agent-policy-stage5/beyond_procwaise_api`
- UI `/home/muthu/PycharmProjects/beyond_procwise_ui-worktrees/agent-policy-stage5`

## Rules that bind (in addition to the stage 5 Global Constraints)
- Nothing here changes what learning suggests or when a second review is required. No change to `learning_stats`, `learning_patterns`, `learning_draft`, or the rule in `second_review._judge`.
- The learning scan stays OFF by default. Never run the unscoped scan on the shared test database.
- Every new write endpoint is role-checked at the gateway and again at the backend, audited with before/after values, and has a closed body shape.
- No new output-safety exemption. New reads carry no free text that the filter would scrub, or the screens must tolerate it.
- UI: new code goes in the stage 5 block of `engine.js` (`apLn*`), plus `agentPolicy/suggestions.js`. Edits inside functions that existed before stage 5 are allowed ONLY where a task names the function; each such edit is listed in the report and the contract tests' pins are updated. `policyEdit`, `policyDelete`, `openFormModal` stay byte-identical. Server text through `escH` only; `TB(`/`TBA(` take string literals only. Reuse existing card/table/confirm styles.
- Shared databases: only additive, idempotent DDL, applied to BOTH `bp_testdb` and `bp_sqldb`, with a rollback file that is never run. The controller announces DDL to other sessions before it is applied.

---

## Backend lane

### Task B1: People are shown by name
**Why:** "Reviewed by", "Reversed by" and "Approved by" show a Cognito sign-in id.
- Migration `deploy/sql/2026-10-16_bp_agent_policy_actor_names.sql` (+ rollback): `ALTER TABLE proc.bp_agent_policy_second_review ADD COLUMN IF NOT EXISTS reviewed_by_email TEXT;` and `ALTER TABLE proc.bp_policy_firing_reversal ADD COLUMN IF NOT EXISTS reversed_by_email TEXT;`. Nullable; nothing else. (The reversal table's append-only trigger guards UPDATE/DELETE on rows, not ADD COLUMN.)
- `second_review.record` stores the caller's email (trimmed, or NULL when the sign-in has none); `learning.record_reversal` the same.
- `secondReview` on `GET /agent-policies/{key}` gains `reviewedByName`: the stored email when present, else the id. `reviewedBy` (the id) stays. Each firing's `reversed` gains `byName` on the same rule.
- "Approved by" in Recent decisions: `policy_firings` rows gain `decidedByName`. Find where an approval decision already records the decider's email (look at how `approvals.act` records its action in `proc.bp_decision` / `agent_actions`, and at `approval_views`), and use it if it can be read with one join; do NOT alter `bp_policy_firing` or its guard trigger. If no email is recorded anywhere for a decision, `decidedByName` equals `decidedBy` and the report says so.
- A `system:*` decider is shown with the words the conflict history already uses (reuse `conflict_history.DECIDED_BY_WORDS` or its helper).
- Privacy: an email is shown only to callers who may already see the id in that answer (no wider than today).
- Tests: stored email round-trips; missing email falls back to the id; old rows (NULL email) still read; the names never appear where the id did not.

### Task B2: The learning thresholds are settings an administrator can change
- `GET /agent-policies/settings/learning` (Admin): `{"learning": {min_decisions, min_days, min_approvers, wilson_lower, median_seconds_floor, not_yet_more, dismiss_more}, "defaults": {…the ruled defaults…}, "scan": {"enabled": <bool: whether AGENT_POLICY_LEARNING_SCAN is on in this process>}}`.
- `PUT /agent-policies/settings/learning` (Admin), body exactly the seven keys, all required. Validation (refuse with 422 `{problems:[{field, message}]}` in plain words): `min_decisions` whole number 10-1000; `min_days` whole number 1-365; `min_approvers` whole number 2-50; `wilson_lower` number 0.5-0.99; `median_seconds_floor` whole number 0-3600; `not_yet_more`, `dismiss_more` whole numbers 1-1000. No booleans, no strings.
- Stored by merging into the existing `proc.bp_admin_config['agent_policy_settings']` row under `learning` (read-modify-write in one transaction with the row locked; other keys in that row untouched; create the row if it does not exist). Audited in the same transaction with before and after values (the taxonomy PUT is the model).
- Declared above `/{key}` routes. Tests through `api.main.app`: role floor; each validation; the other settings keys survive; audit before/after; the next `settings.load_settings` reads the new values.

### Task B3: The queue knows whose cards are whose
- Each card gains `mine: bool` (the caller is linked to the policy's owner name in the decider map, `approval_views.caller_names`) and `ownerName`. The list accepts `mine=1` to return only those. If the decider map cannot be read: `mine` is false for every card and the answer carries `mineUnavailable: true` (never an error).
- **Not yet / Dismiss** now need: the caller is linked to the owner, OR has the Approver role or higher. A Buyer who is not the owner gets 403 "Only the policy's owner or an approver can hide a suggestion." **Draft this change** stays at Buyer (it writes nothing).
- Each card gains `canAct: bool` on that rule so the screen can hide the two buttons.
- Tests: owner-linked Buyer may act; unlinked Buyer may not; Approver may; `mine=1` filters; map unreadable.

### Task B4: A notification when a suggestion is raised
- When the scan inserts a card or moves one back to `open` (raised or reopened; not on a plain refresh), it writes one in-product notification per owner name, in the same per-policy transaction, using the stage 3 notification table and helper the conflict cases use (`proc.bp_policy_notification`; `firing_id` is nullable). Message: `Learning suggests a change to {policyKey}: {plain change sentence}`. Link: follow the existing link convention (`approval_views._link_ref` parses it) with a new kind for suggestions, e.g. `suggestion:{id}`; extend `_link_ref` so `GET /agent-policies/notifications` returns a `target` the screen can open (`{kind: "suggestion", id, policyKey}`).
- No notification when the owner name is blank; one per card per (re)raise; a failure to write the notification must not lose the card (savepoint, logged).
- Tests (scoped scan with injected targets, as stage 5): raised → one row; refreshed → none; reopened → one; unreadable notification write → card still there.

### Task B5: Honest card states and messages
- A card whose change cannot be drafted right now says so before anyone clicks: in `card()`, when the policy's latest version is a draft whose condition no longer has the state the proposal starts from (the same test `draft` applies), `canDraft` is false and the card carries `blocked: "The latest draft of this policy already changes this. Activate or discard that draft first."`; when the policy is no longer live: `blocked: "This policy is no longer Active."`. `draft` keeps refusing as it does.
- `second_review`: a permanent fault (the stored form of the live or draft version cannot be read) returns the problem code `second_review_unreadable` with the message "This policy's saved form could not be read, so a second review cannot be checked. An administrator needs to look at it."; "Try again shortly" stays only for failed database reads.
- `secondReview` on the policy page gains `whoMayReview: "An approver who is not this policy's owner and has not saved this draft."` when a review is required and not yet recorded (a fixed sentence from the server so the rule is stated in one place).
- Tests for each.

### Task B6: The repeat proposal's "Previous decisions" is right (carried from stage 4, defect D3 there)
- A policy conflict case raised by the repeat rule (`facts.proposal.from == "repeat"`) reports its prior decisions as the live decisions it was raised from (count and last outcome), not "first time". Find where `priorDecisions` is computed for a policy case (`conflict_payload` / `conflict_cases`) and include the live cases of the same pair for a repeat proposal only. No other case's prior count changes: pin that with the existing tests plus one new one.

## Gateway lane

### Task G1: Forward the settings calls and the queue filter
- `GET agent-policies/settings/learning` (Admin) and `PUT agent-policies/settings/learning` (Admin; closed body of exactly the seven keys; each a finite number, not a boolean or string; ranges are the backend's job, the gateway checks only shape and type).
- `GET agent-policies/suggestions` accepts `mine=1` (anything else for `mine` → 400) and forwards it.
- `POST agent-policies/suggestions/:id/act`: the gateway floor stays Buyer (the backend applies the owner-or-Approver rule and its 403 passes through with its message).
- Static routes before `:key`; two new serverless function blocks copied from an existing one; spec cases for each, including that a refused call never reaches the backend. Mutation-check the role floors.

## Screens lane

### Task U1: Names, honest states, hints, and the small defects
Reads `reviewedByName`, `reversed.byName`, `decidedByName`, `blocked`, `whoMayReview`, the `second_review_unreadable` problem.
- "Reviewed by {name} on {date}", "Approved by {name}", "Reversed on {date} by {name}: {reason}" use the name fields, falling back to the id fields when absent.
- A card with `blocked` shows that sentence under the evidence and its Draft button is disabled with that sentence as its title.
- Under the second-review sentence, when present, show `whoMayReview`.
- "Recent decisions" stays open across redraws once a person has opened it, until the form is closed.
- The two refresh gaps: the "Business areas" tab requires the library (edit allowed in `apPoliciesTab`: one token), and a save that finishes during the very first queue read still causes a re-read (edit allowed: the `apReload` line may pass a flag).
- Carried from stages 3-4 (edits allowed in the named stage 3/4 functions; keep each to the minimum and list them):
  - The approval card, just after a decision with no replay row yet: when the case is alone in its group say "Approved. The action will run shortly."; only when other cases in the group are still open say "Waiting for the other approvals before the action runs." (the detail answer's `group` tells which; when `group` is null treat the case as alone).
  - A closed case never shows the raw word "actioned": show what happened from its history ("Approved by {who}", "Rejected by {who}", "Rejected: nobody decided in time").
  - The repeat proposal card shows the prior decisions the server now sends (B6) instead of "First time".
- Tests for each; break the guards for `blocked` and the names fallback on purpose.

### Task U2: Learning settings, my suggestions, and opening a suggestion from a notification
- **Learning settings** (Admin only): a section at the top of the Business areas screen titled "Learning thresholds": seven labelled number inputs in plain words — "Decisions needed", "Days the decisions must span", "Different approvers needed", "Confidence needed (0.85 = 85%)", "Shortest typical decision time (seconds)", "More decisions before a 'Not yet' card returns", "More decisions before a dismissed card returns" — each with its ruled default shown beside it ("default 30"); a line "The hourly scan is on / off on this server." from `scan.enabled`; a Save button (two-step confirm "Change the learning thresholds?" / "This changes when learning suggests loosening a policy, for every policy."); the server's 422 problems shown beside their fields; a "Reset to defaults" button that fills the inputs (does not save).
- **My suggestions**: when any card has `mine` true, the queue shows a two-button switch "Mine" / "All", defaulting to Mine; the tab's count is the count of open cards that are mine when any are, else all. When `mineUnavailable` is true the switch is hidden and one line says "Could not tell which policies are yours; showing all." Not yet / Dismiss are shown only when the card's `canAct` is true.
- **Notification → suggestion**: a notification whose target kind is `suggestion` opens the Suggestions view with that card scrolled into view and outlined, or, when it is no longer open, a toast "That suggestion is no longer open." (edit allowed in the stage 3 `apNotificationOpen` and its pure helper `notificationTarget`: one new branch).
- Tests for each, including Viewer/Buyer/Approver/Admin visibility.

### Task U3: The new phrases are in the phrase dictionary
- Read `src/lib/i18n/body.js`, `body.more.js`, `src/locales/`, and `src/lib/i18n/language.contract.test.js` first, and find how the other languages' entries are produced (a script, a pipeline, or by hand).
- Add every on-screen string introduced by stages 5 and 5b to the phrase inventory and `en.json`, following the file's own rules (every listed phrase must appear verbatim in `engine.js`).
- Other languages: if the repo has an offline, deterministic way to produce them, run it. If producing them needs a model, a GPU or a network call, do NOT run it: leave the other languages falling back to English and say exactly what command a person runs later.
- The contract test must be no worse than baseline (3 known failures, same lists). Do not "fix" the baseline failures.

## Task V: One browser run of everything, and the note
Private stack exactly as stage 5's Task 13 (ports 8010 / 3011 / 3010, headless Chrome, `bp_testdb`, scan switches off, scoped scans only, demo policies "DEMO stage 5b …", clean up, list what stays). Re-run the stage 5 walkthrough at the final commits (so the stage 5 fix wave is seen in a browser: queue refresh after save, plain names in Recent decisions, Business areas re-reading before a save, the card above a Conflicts section with one conflicting demo pair), then every 5b item. Append the results to `specs/2026-10-10-agent-policy-governance-stage5-verification.md` as "## Stage 5b browser run", plain English first, defects recorded not patched. Final suite numbers for all three repos.
