# Email assurance: what is NOT yet verified, and what must pass before bp_sqldb or production

Written 2026-10-08. Nothing in section 1 may be treated as working until its live check passes.
**Nothing is applied to bp_sqldb or production until sections 1 and 2 are cleared.**

The test environment has no GPU driver, so every model-dependent stage was proven against a FAKE
model that returns good output and also bad output (malformed JSON, an unknown family, a made-up PO
number, a reasoned field with no basis, a figure in no fact, out-of-range scores, a refusal as a
"repair"). That proves the plumbing and the validators. It says nothing about model quality.

## 1. Pending live verification (needs a real model)

| # | Item | What a real run must show | Gold data needed |
|---|---|---|---|
| 1 | Classifier accuracy (from_prompt) | >= an agreed accuracy on a labelled set of real requests; low-confidence cases ask rather than guess; no invented lookup keys survive | A labelled set of requests -> family. **Does not exist.** |
| 2 | Classifier output format | The real model honours `format="json"` (it is documented inert on the `messages=` path, see email_intent.py); prose-wrapped JSON still parses | none |
| 3 | Planner quality | Briefs follow the facts, put a figure nowhere that is not in a fact or the request, and name a real basis; `{"missing": [...]}` is returned when it should be | A set of requests with known facts |
| 4 | Planner steering | `from_prompt` actually writes better/closer drafts from the brief than without it | A/B on the same requests |
| 5 | Judge calibration | Scores agree with human scores on a sample (report agreement, not just that it returns 1-5); a deliberately bad draft scores low | Human-scored drafts. **Does not exist.** |
| 6 | Governed prompt text | The three prompts (`email_family_classify`, `email_brief_plan`, `email_draft_judge`) behave as intended on the live model; wording reviewed | none |
| 7 | Repair pass | The LLM repair removes the flagged problem and keeps the email; rejection rate measured | none |
| 8 | Latency | The judge and planner add acceptable seconds per draft (the model was 3.4s/turn when resident) | none |

## 2. Pending review / decision (not model-dependent)

| # | Item | Owner |
|---|---|---|
| 9 | `EmailToneRules` row: the eight variables (six original + `warmth`, `directness`), the country-to-formality map, seniority keyword lists, escalation thresholds (1/3/5 prior contacts), the `on_gap` rule per variable (`assume` for escalation_level and recipient_seniority, `default` for the rest), the 13 instruction override groups, and the `tone_cues` word list that decides whether an instruction is a tone request at all. **Written, not applied.** | You |
| 9a | `tone_cues` is a judgement list. Too short and an unmapped tone word slips through silently; too long and plain tasks raise false 'unmapped' questions. Review against real instructions once there are some. | You |
| 10 | The three governed prompts. **Written, not applied.** Until applied, classify/plan/judge report `unavailable`. | You |
| 11 | Tone variables are RECORDED but do not yet steer the drafter. The drafter's own tone logic (`_calculate_tone_guidance`, round and price gap) is unchanged. | Build decision |
| 12 | Mailbox -> user mapping: `bp_mailbox_binding` has 1 inactive test row, and mail is sent from a shared SES sender, so it cannot identify the sender. `reviewed_by` comes from `bp_approval.actioned_by`, `sent_by` from the signed-in principal. | Confirm |
| 13 | Style profiles still resolve from the sender mailbox at draft time (`resolve_user_ref`); only the NEW learning job will key them on `sent_by`. | Stage 6 |
| 14 | `shadow` -> `enforce` for any family: needs shadow evidence reviewed first. | You |
| 15 | Authority check at draft time uses the drafting AGENT's limit (`resolve_authority`); the human's authority is enforced at approval. A per-user spend limit does not exist (bands live in `bp_admin_config.authority_bands`; `bp_role_assignment` is empty). | Decision |
| 16 | A counter in a non-GBP currency has authority `unresolved` (no exchange rate is invented). | Decision |
| 17 | Conflicts (payload vs Postgres) are shown but do not gate `ready`. | Decision |
| 18 | Escalation level is derived from the count of `workflow_email_tracking` rows for the thread. Zero rows is treated as ZERO prior contacts (level 1, source `postgres`), because the data cannot tell "never contacted" from "contact not recorded". Only an unknown workflow or supplier id is a gap. | Decision |
| 19 | The eval database omits constraints on purpose (so cases can seed duplicates). The "multiple matching records" case therefore removes a fact's ordering rule rather than relying on duplicate primary keys, which production forbids. | Note |

### Stage 6 (learning job) - decisions and limits

| # | Item | Status |
|---|---|---|
| 20 | **Blocked by "no sent text is stored":** (a) style rules about WHICH words a person prefers ("opens with Hi, not Dear"), (b) detecting "the same wording correction from 3 users". The job instead proposes rules from LENGTH and REWRITE-INTENSITY signals only, and opens a review item for "3+ reviewers heavily rewrote this family" that says in its own text it is NOT the same-correction signal. Closing the gap needs a decision: keep a minimal set of non-prose edit features (greeting/sign-off changed, paragraph count), or retain edit text under a stated retention rule. | **Decision needed** |
| 21 | Deviation from the spec's four exemplar criteria: a draft whose reasoned value (price, deadline) a person overruled is NOT an exemplar candidate, however few words changed. | Flag |
| 22 | The spec allows promotion by "a second independent rubric pass". Not built: needs a model. Today promotion is a person other than the author. | Pending model |
| 23 | An edit's changed figures are paired in the order they appear to decide what replaced what. A figure that merely disappears (sentence cut) is an OMISSION, not a correction. The heuristic is wrong when several figures change in one send in a different order. | Known limit |
| 24 | Approved style rules and approved exemplars are NOT yet consumed by the drafter; the job only prepares them. | Not built |
| 25 | No endpoint or screen for people to act on the queues. Functions exist for deciding a style rule (owner only) and approving an exemplar (not the author); data-quality items and review items have no resolve function yet. | Not built (UI scope) |
| 26 | The job is OFF unless `EMAIL_LEARNING_ENABLED=1`. The migration is applied to bp_testdb only; thresholds are the `EmailLearningRules` policy row (missing/invalid -> the job refuses to run). | As designed |
| 27 | A reviewer's replacement for a Postgres-backed figure is kept ONLY in the data-quality queue (labelled unverified) and is never written to an eval candidate, exemplar, style rule or classifier example. Tested by scanning every learning table for the value. | Tested |

### Read-only role and the bp_sqldb pack

| # | Item | Status |
|---|---|---|
| 28 | `email_agent_reader` / `email_agent_writer` exist ONLY in migration files and in throwaway containers. A Postgres role is cluster-wide, so creating one on the RDS cluster that hosts bp_testdb would create it for bp_sqldb, uicanvas, ses and the rest; that is why it was NOT created on bp_testdb. | **Decision: apply?** |
| 29 | Until the login roles exist and `EMAIL_AGENT_RO_*` / `EMAIL_AGENT_RW_*` are set, reads run under the INTERIM control (a read-only session on the existing login). It is a guardrail, not a boundary: a session can switch it off and the existing login can still write. Every draft records `read_control` (`dedicated_role`, `interim_readonly_session` or `unenforced`). | Interim |
| 30 | The reader may read only `supplier_response`, `workflow_email_tracking` and nine `bp_supplier` columns (no bank, tax or registration columns). A family fact source naming anything else fails closed ("permission denied" -> unresolved). Adding a source table is a reviewed GRANT in a migration. | As designed |
| 31 | Only the assurance layer's reads and the capture/learning writes use the new doors. The drafter's own writes (`proc.draft_rfq_emails`), the send guard and the `/drafts/*` router still use the application login. | Not migrated |
| 32 | The bp_sqldb rehearsal restored the STRUCTURE of eight tables plus the policy and prompt rows; business rows, other tables, and lock behaviour on a busy database are not exercised. | Known limit |
| 33 | The tone rules and the three governed prompts are inside the pack (files 6 and 7). Applying the pack applies them. Say if they should be held back and applied separately after your review. | **Decision** |

## 3. Out of scope for now (explicit)

- **Supplier-reply outcomes** (`supplier_replied`, `reply_latency_s`, `issue_resolved`): deferred until inbound email
  integration. The columns are marked `OUT OF SCOPE` in the database; NULL means "not captured".
- **Team membership**: no user -> team mapping exists anywhere in the schema. `team_id` stays nullable; exemplar
  fallback is user -> organisation (the whole deployment; there is no tenant dimension).
- **UI**: the UI repo is not edited. See `2026-10-08-email-assurance-ui-patch-proposal.md`.
- **Abandoned-draft sweep** (server side): BUILT 2026-10-08, see the section at the end of this file.

## 4. Not started (next, in the agreed order)

1. Read-only database role (new, dedicated, non-prod first) and `default_transaction_read_only` as an interim control.
3. bp_sqldb: exact DDL, grant changes, rollback script, and results from a non-prod copy.
4. Metrics (edit distance and fact-conflict rate by family over time).

DONE since this was written: read-only roles + connection doors, and the bp_sqldb DDL pack with a passing rehearsal (`2026-10-09-bp-sqldb-email-assurance-ddl-pack.md`); Stage 6 learning job (`src/services/draft_assurance/learning.py`, migration `2026-10-09_email_agent_learning.sql`, off by default); evals and CI (`evals/email`, `.github/workflows/email-goldens.yml`, 56 golden cases). Making the eval job a REQUIRED check is a branch-protection setting and has NOT been done.

## 5. What IS applied (bp_testdb only)

| What | File |
|---|---|
| `email_agent` schema, capture + outcome tables | `deploy/sql/2026-10-07_email_agent_capture.sql`, `2026-10-08_email_agent_capture_v2.sql` |
| Learning queues + `EmailLearningRules` policy row | `deploy/sql/2026-10-09_email_agent_learning.sql` |
| Family rows `EmailFamily_negotiation_counter`, `EmailFamily_free_prompt` (+ v2 labels/rubric/authority agent) | `2026-10-07_email_family_*.sql`, `2026-10-08_email_family_v2.sql` |

Written, NOT applied anywhere: `2026-10-08_email_tone_rules.sql`, `2026-10-08_email_assurance_prompts.sql`.
Each has a `_rollback.sql`.

## RFQ batch wrap (2026-10-08, shadow)

The RFQ batch (`_render_supplier_draft`) now carries a per-supplier assurance record (family `rfq_batch`, config row in
`deploy/sql/2026-10-08_email_family_rfq_batch.sql`, NOT applied anywhere). It records and never changes the body, makes no
model call (the body is templated; the family has no rubric so the judge reports unavailable), and a fault on one supplier
leaves that draft `unassured` without losing the batch. What it does NOT do, and why it matters for criteria 1-2:

| # | Limit |
|---|---|
| R1 | An RFQ's deadline, items and quantities come from the request and upstream supplier profiles, not a Postgres row. They are CARRIED: numeric figures are listed under `unverified_figures`; nothing confirms them. |
| R2 | **Shared gap (all families):** a carried DATE is accepted but NOT listed under `unverified_figures` (only numbers are). So "flagged user_asserted" (criterion 1) is not met for dates anywhere. Needs a decision on whether to list dates too. |
| R3 | Quantities/items for an RFQ may exist in Postgres (requisition or PO lines); the family reads only the supplier contact. Adding a fact source is config, but the right table needs your ruling. |
| R4 | No repair pass and no judge for RFQs (shadow, templated). Model-composed RFQ bodies (`_render_dynamic_body`) are checked but not repaired. |
| R5 | One assurance read per supplier, in the batch's worker threads: N suppliers = N small reads. Not load-tested. |

## Human-written emails: reply panel, report panel, manual passthrough (2026-10-08, shadow)

Family `human_written` (`deploy/sql/2026-10-08_email_family_human_written.sql`, NOT applied anywhere). Intents recorded for
capture: `REPLY_PANEL`, `REPORT_PANEL` (`POST /workflows/email/prepare`) and `MANUAL_PASSTHROUGH` (the drafting agent's manual
body). No model writes these, so nothing is repaired or judged, and the person's text is stored exactly as before.

| # | Limit |
|---|---|
| H1 | The person is the author, so EVERY figure they typed is carried and therefore allowed: this family cannot fail a typed figure. It lists the ones Postgres does not hold under `unverified_figures` and cites the supplier's latest offer and currency next to them. Whether a typed price contradicts the offer is for the reviewer; no rule decides it. |
| H2 | It CAN fail: bank details, a liability admission, a waiver, an award commitment, a leaked internal limit, and a recipient not on the supplier master. |
| H3 | The manual passthrough has no supplier, so every recipient is reported `recipient_not_on_master` and no fact resolves. Correct, and noisy: such drafts will always show `needs_review`. Needs a ruling on whether the passthrough should resolve a supplier from the recipient address. |
| H4 | The manual passthrough's author is `data.requested_by` only; if the caller does not send it the draft is attributed to the agent, not a person. |
| H5 | Report-panel emails with no deal thread have no workflow or supplier, so only the text checks run. |
| H6 | Test-suite note: a test file that imports `src.api.routers.workflows` at module level breaks `tests/test_email_dispatch_service.py` (three tests) in the same session; `tests/api/test_email_prepare_endpoint.py` already does, independent of this work. The new tests import the router lazily. |

## The one path left unwrapped: the negotiation agent's own draft stub (decision 2026-10-08)

`NegotiationAgent._build_email_draft_stub` returns a dict with the agent's own composed text. It is deliberately NOT wrapped,
because it cannot reach a supplier by itself:

1. The negotiation agent never stores it (no write to `proc.draft_rfq_emails` anywhere in `negotiation_agent.py`), and
   `EmailDispatchService.send_draft` raises `No stored draft found` for any identifier that is not a stored draft.
   Pinned by `tests/services/test_negotiation_stub_not_sendable.py`; mutation-checked (removing the refusal turns it red).
2. The email a negotiation round actually sends is written and stored by `EmailDraftingAgent`, whose counter path IS assured.
3. A body handed to the dispatcher over that stored draft (what a stub would try) changes the transmitted content, and the
   approval's content hash refuses it (`tests/approvals/test_content_binding_enforced.py`). Live policy, read-only check on
   bp_sqldb and bp_testdb 2026-10-08: `EmailDispatchApprovalPolicy.on_content_mismatch = deny`, `approval_required = true`;
   `EmailReplyAutonomyPolicy.auto_reply_intents = []` (no unattended send).

**Residual risk, stated:** the hash protection holds while `on_content_mismatch` stays `deny` and `auto_reply_intents` stays
empty. Setting either to `warn` / a real intent would let an un-assured body reach a supplier, and this decision would have
to be reopened. Also, the stub text is still shown to anyone who reads the negotiation agent's `drafts` output; it carries
no assurance record and should not be presented to a reviewer as a checked draft.

## Sent text and diffs (decision 3, 2026-10-08) - built, not applied

`2026-10-08_email_agent_sent_text.sql` + `draft_assurance/retention.py`. Item 20 above (the learning job's blocked rules) is
now **unblocked in storage, not in use**: the text is kept but `learning.py` does not read it yet.

| # | What is true |
|---|---|
| T1 | On a successful send, with a retention period, the plain text that went out and a word-level diff from the model's draft are stored in `email_agent.bp_draft_sent_text` (one row per send). The diff is complete: `capture.apply_diff(draft, diff)` rebuilds the sent text exactly (tested). |
| T2 | **Access:** only `email_agent_writer` (SELECT/INSERT/DELETE on that table alone). The reader and PUBLIC have nothing (tested as real logins). No API returns it: `to_view` carries neither the sent text nor the draft (tested). Until the dedicated logins exist the app's own login is used, so this restriction is NOT in force in a running system yet. |
| T3 | **Retention:** `EmailTextRetention.raw_text_days` (policy row, default 90). Unreadable, missing, zero or negative = **no raw text is stored** and the purge refuses. A job (`email-text-retention`, ON unless `EMAIL_TEXT_RETENTION_ENABLED=0`, every 360 min) deletes sent text older than the period and blanks the model's draft text, keeping the row, hash, facts, scores and classes. |
| T4 | **Derived features are kept, with no expiry.** No derived-retention period is implemented, because deleting capture/outcome rows would break the learning tables that reference them. If you want one it needs its own ruling. |
| T5 | **Bank details are masked before anything is stored** (IBAN, UK sort code, "account number ..."), in the sent text, the diff, the model's draft and the changed-figure columns. Found and fixed while testing: the changed-figure columns had been storing an account number as a "figure". This is pattern-based: a bank detail in an unusual format would not be caught. |
| T6 | **Not redacted:** names, phone numbers and email addresses in signatures, and any quoted thread history the body carried. They are stored as sent, under T2/T3 only. A real PII pass is not built; `style/redaction.py` over-redacts figures and would destroy the diff, so it is not used here. |
| T7 | A draft sent after its own text expired records NO edit distance (NULL, not 1.0) and stores the sent text with no diff. |
| T8 | `request_text` / `user_instruction` (the person's own words) are still stored unmasked in `bp_draft_capture`; a bank detail typed there is not masked. |
| T9 | The retention job needs the DELETE grant, so it only works under the writer role or the app login; it was never run against a shared database. |

## Steering: tone, the author's approved style rules, approved exemplars (2026-10-08) - built, quality PENDING a live model

`draft_assurance/steering.py`, `deploy/sql/2026-10-08_email_agent_steering.sql`, and a `directives` block in the tone rules.
Appended as delimited data to the user message of the three model-writing prompts (counter, `from_decision`, `from_prompt`). The
RFQ batch, human-written and stub paths are not model-written here and are not steered.

| # | What is true |
|---|---|
| S1 | Tone steers only when a variable came from Postgres or the person's own words. A variable that fell back to its default has no data behind it and steers NOTHING (tested). |
| S2 | Style rules are the draft author's own, status `approved` or `edited` (the edit wins). Nobody's rules are applied to somebody else's email. The author is `requested_by`; an agent-initiated draft has none, so it gets organisation exemplars and no personal rules. |
| S3 | Exemplars: approved, in date (`review_after`), this family; the author's first, then the organisation's. Cut at a word, and a delimiter inside stored text is stripped so an exemplar cannot close its own block (tested). |
| S4 | **Config:** `EmailSteeringRules` (`enabled`, three limits). Missing, unreadable, malformed or `enabled: false` = no steering and a prompt identical to before (tested by comparing prompts). The migration's row ships `enabled: true`; it belongs to pack (b) with the tone rules, so it is held back with them. |
| S5 | Nothing steers a CHECK. A figure copied from an exemplar into a new email is caught by the ordinary grounding check (tested end to end). |
| S6 | What steered a draft is recorded by id (`bp_draft_capture.steering`, plus a `steering` stage in `stage_status`), never as text. |
| S7 | A steering failure (store down) is recorded `unavailable` and the draft is written exactly as without steering. |
| S8 | **PENDING LIVE VERIFICATION:** whether steered drafts are better, whether the model obeys "imitate register, never copy figures", whether 2 exemplars + 5 rules + tone lines is the right amount, and the effect on length/latency. Nothing here has met a real model. |
| S9 | Reads of the style rules and exemplars use the writer door (the reader role has no access to `email_agent`, and `bp_exemplar_candidate.draft_text` is raw text). |
| S10 | Existing mailbox-derived style (`services/style`, the system prompt's `_with_style`) is untouched and still applies. The two sources can both be present in one prompt; their interaction is unverified. |
| S11 | Tone directive wording (19 sentences in the tone rules row) is a proposal for your review, like the rest of that row. |

## Labelling set for the classifier and judge (2026-10-08) - built; nothing filled in yet

Items 1, 2, 5 and 8 of section 1 needed gold data that did not exist. `evals/email/labelling/` now holds blank sheets for your team
(55 requests; 15 + 14 draft emails to score) and a separate key. See its `README.md`. **No result exists until people fill the sheets
and a real model is run.** `python -m evals.email.labelling.live ... --live` refuses to run without `--live` so a stand-in cannot
produce a number that looks like one.

| # | What is true |
|---|---|
| L1 | The "intended" family on each request is MY hypothesis, kept in `key/`, not printed on the sheet, so your team's judgement is independent. Where the team and I disagree, the family definitions are ambiguous (reported as `intent_vs_team`). |
| L2 | The flawed drafts are deliberate controls, also keyed. The report shows whether the **team** found them (are the controls fair?) and whether the **model** did (does the judge separate good from bad?). A judge that gives everything a 4 is shown as not separating even if it never disagrees much (tested). |
| L3 | All requests and drafts are invented, in one house style. 55 + 29 items expose gross failure, not a ranking: treat accuracy as roughly +/- 10 points. Real requests should be added when there are some. |
| L4 | **Found while building it, and fixed (tested):** the classifier was offered every `email_family` row, so a free-text request could be classified as `rfq_batch` or `human_written`. Rows now carry `classifiable` (false for those two), and the classifier is shown a `request_description` (when to choose the family) instead of the config text ("Guardrails for ..."), and a short `request_label` in its question ("Is this a counter-offer, or an ordinary supplier message?"). Whether those descriptions classify well is exactly what the sheets will measure. |
| L5 | Only two families can be classified into. A third family is config plus new rows in the sheets. |
| L6 | The criteria definitions the team reads are my wording of the rubric names. Please check they say what you meant by them. |

## Endpoints for the learning queues and the metrics (2026-10-08) - built; no screen

`src/api/routers/email_learning.py` (registered with the authenticated routers), `draft_assurance/queues.py`, and decision functions in
`learning.py`. This closes item 25 (no way to act on the queues) at the API level only: **no UI exists**. The route list is in the
router's docstring and in `specs/2026-10-08-email-assurance-ui-patch-proposal.md`.

| # | What is true |
|---|---|
| Q1 | Three new closed-vocabulary actions: `email.learning.read` (read), `email.learning.decide` (write), `exemplar.approve` (configure). The class decides who may by default; a `configure` action is Admin-only unless a policy row says otherwise. If a team lead who is not an Admin should approve exemplars, that is a policy-row change, not code. |
| Q2 | Approving an exemplar is `configure` because it changes what steers EVERY user's drafts in that family. Its author can never approve it (tested through the API). Seeing an exemplar's TEXT is a separate call under the same gate; the listing carries no text. |
| Q3 | A person sees and decides only their OWN proposed style rules (tested: someone else's rule is a 404 and is never listed). |
| Q4 | Who is recorded is `principal.subject` and nothing else; a body naming someone is ignored; a blank person is a 401 before anything is asked (all tested). |
| Q5 | Data-quality and review items have no "owner" concept: anyone holding `email.learning.decide` (a `write`-class action, Buyer and above by default) may resolve or dismiss. If only the data owner should, that needs a role for it. |
| Q6 | **Rejecting an exemplar is not recorded on its row** (the table has no `decided_by`); only the authorisation audit says who. Resolving a data-quality item, deciding a review item and approving an exemplar ARE recorded on the row. |
| Q7 | Responses carry labels and row ids, never an internal table or column name, never an eval candidate's draft text, and never an exemplar's text outside the detail call (all tested by scanning the JSON). A reviewer's replacement value for a fact is returned marked `verified: false`. |
| Q8 | Metrics (`GET /email-learning/metrics`) are family aggregates (drafts, sent, abandoned, mean edit distance, fact-edit rate, fact-conflict rate, needs-review rate) by day, week or month. There is no per-person view. |
| Q9 | All reads and writes go through the writer door (the reader role has no access to `email_agent`). Until the dedicated logins exist this is the app's own login, so the role restriction is not yet in force. |
| Q10 | Until someone uses these endpoints there is nothing approved for the drafter to be steered by. With no screen, that means an API client or a person calling it by hand. |
| Q11 | **CLOSED 2026-10-08.** `GET /drafts/{id}/assurance` and `POST /drafts/{id}/preflight` asked no authorisation question beyond being logged in, so any logged-in user could read any draft's facts (with row ids), brief, violations and which facts had changed. Both now ask a governed read gate, `email.draft.read` (class `read`), with the draft named, and refuse a blank person first (all tested; five breakages caught). Being a `read`-class action, access is as open by default as before; the difference is an audit record and that a policy can now narrow it. Who should be allowed to read a draft's assurance is still your call. |

## The abandoned-draft sweep (2026-10-08) - built, off nowhere, applied nowhere

`draft_assurance/sweep.py`, `deploy/sql/2026-10-08_email_draft_sweep.sql`, job `email-draft-sweep` (ON unless
`EMAIL_DRAFT_SWEEP_ENABLED=0`, hourly). Closes the drafts nobody sent and nobody abandoned, so "how many drafts were never used" can
be answered.

| # | What is true |
|---|---|
| W1 | After `EmailDraftSweepRules.abandon_after_days` (14, proposed by me: **please confirm the number**) with no outcome, the LATEST capture of a draft gets an `abandoned` outcome by `system:draft-sweep`, reason "no send or decision within 14 days". A missing or non-positive value makes the sweep refuse. |
| W2 | **It abandons only what the product CONFIRMS was not sent.** Recording a send is best-effort, so a draft with no outcome may well have gone out. Sent per `draft_rfq_emails` (`sent`, or any `sent_on`) or per a `workflow_email_tracking` row = left alone. No product record at all = left alone (nothing confirmed). If the check cannot be made, nothing is abandoned. All tested, including each of those mutations. |
| W3 | A regeneration replaces its predecessor; only the newest capture is ever swept, so a replaced draft is never called abandoned. |
| W4 | A draft sent AFTER it was swept is counted as sent, not abandoned, in the metrics (tested). It will carry both outcomes. |
| W5 | **Needs a grant:** the reader role gets `SELECT (unique_id, sent, sent_on)` on `proc.draft_rfq_emails` and nothing else of it (tested as a real login). The DBA change request is updated; this widens what the reader can reach by one table's three columns. Until the dedicated reader exists the app's own login is used. |
| W6 | A draft the product has no record of (stored elsewhere, or the draft table write failed) is never swept; it stays "no outcome" forever. Counted in the report as `skipped_unverifiable`. |
| W7 | Batch size (500) bounds one run; the rest waits for the next. The job logs its report each run; nothing else surfaces it yet. |
| W8 | Not built: a draft that is waiting on a human approval for longer than the period is swept like any other, and then still can be sent. Whether a pending approval should postpone the sweep is a decision; today it does not. |

## Pack (a) alone, and the deploy order (2026-10-08)

Found while rewriting the DDL pack document: the `steering` column was in pack (b) (held back) while the capture code writes to it, so
applying (a) alone would have made every capture fail, quietly. Fixed: the column is its own pack (a) file, and
`tests/email_evals/test_pack_split.py` now applies (a) alone to a fresh database and runs capture, send, sweep, retention, learning and metrics
on it (and reproduces the bug if the column is put back into (b)). The rehearsal on a copy of bp_sqldb's structure was re-run with all fifteen files:
**PASS**, 364 eval tests against the copy, schema fingerprint identical after rollback.

**Deploy order matters, and failure is quiet.** The application code on `origin/Development` writes to the (a) tables and columns. Where the
migrations are not applied, capture, send-outcome recording, the sweep and retention each log an error and record nothing, while drafting and
sending carry on. `bp_testdb` has only the first capture files applied, so any code that reaches it from `Development` records nothing for the
newer columns. Apply pack (a) before the code reaches an environment you want captured, or accept that it will not be.
