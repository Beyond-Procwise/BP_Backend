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
- **Abandoned-draft sweep** (server side): not built; the abandon event is best-effort.

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
