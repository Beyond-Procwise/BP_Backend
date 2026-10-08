# bp_sqldb: email assurance DDL pack

Rewritten 2026-10-08 from the real files (it had grown a trail of addenda); refreshed after the pack-split fix below. **NOT APPLIED to bp_sqldb, to its cluster peers, or to anywhere
shared.** Nothing here is applied until live verification has passed (your ruling of 2026-10-08) and you say so.

## What it is

Seventeen files in four groups, per your rulings:

* **Pack (a), schema, capture, validators in shadow:** twelve files. Creates the `email_agent` schema and its tables, and inserts the family
  rows (all `shadow`: they record and never block) and the retention, learning and sweep settings. Nothing in (a) changes what the model is
  asked to write. **One exception to "records and never blocks":** once `bp_inbound_flag` exists, a reply flagged as a suspected request to change
  payment details DOES stop an agent drafting, and the send guard sending, on that supplier's thread until an approver clears it. That is its purpose;
  it also means a false alarm holds a thread until someone looks.
* **Pack (b), prompts, tone rules, steering:** three files, applied only after the live checks pass, because they change what the model is
  asked to do.
* **Product-table change:** one file, `2026-10-08_supplier_response_provenance.sql`, on `proc.supplier_response`. Not part of either pack; it needs its own change request (section below) and is applied before the rest.
* **Roles:** one file, a production change of its own (cluster-wide), handled through change control with its own document.

Each file is idempotent (re-running changes nothing), is its own transaction, and has a rollback. The hash is the first 16 hex of the
file's sha256, so you can check that the file applied is the file rehearsed.

| # | group | apply | sha256 (16) | rollback | what it does |
|---|---|---|---|---|---|
| 1 | product | `2026-10-08_supplier_response_provenance.sql` | `b7560c6f4d169f3e` | `2026-10-08_supplier_response_provenance_rollback.sql` | **A PRODUCT-table change, its own change request, applied FIRST.** Adds eight nullable columns to `proc.supplier_response` recording where an extracted price/lead time came from and whether a person confirmed it (see below). |
| 2 | a | `2026-10-07_email_agent_capture.sql` | `7bf01e433b523aac` | `2026-10-07_email_agent_capture_rollback.sql` | Creates schema `email_agent`; tables `bp_draft_capture`, `bp_draft_outcome` (+ indexes, unique-once index). |
| 3 | a | `2026-10-08_email_agent_capture_v2.sql` | `5b9625ec63b4b3c6` | `2026-10-08_email_agent_capture_v2_rollback.sql` | ALTER TABLE on those two tables: stage outputs, accountability, readiness; drops the unused `user_id`. |
| 4 | a | `2026-10-08_email_agent_sent_text.sql` | `5a60e134140e55ca` | `2026-10-08_email_agent_sent_text_rollback.sql` | Creates `bp_draft_sent_text` (raw sent text + diff; no access for PUBLIC); adds `bp_draft_capture.text_expired_at`; INSERT `EmailTextRetention` (`raw_text_days: 90`). |
| 5 | a | `2026-10-08_email_agent_steering_column.sql` | `9431b2298b96ccf9` | `2026-10-08_email_agent_steering_column_rollback.sql` | Adds `bp_draft_capture.steering`. **In (a), not (b): the capture code writes to this column, so without it every capture would fail.** |
| 6 | a | `2026-10-08_email_inbound_flag.sql` | `b01d67b674f7ac4e` | `2026-10-08_email_inbound_flag_rollback.sql` | Creates `bp_inbound_flag`: inbound replies a person must review (suspected request to change payment details). Stores ids, signals and keywords, never email text. |
| 7 | a | `2026-10-09_email_agent_learning.sql` | `0b11e29ea46484c0` | `2026-10-09_email_agent_learning_rollback.sql` | Six queue/candidate tables in `email_agent`; two columns on `bp_draft_outcome`; one index; INSERT `EmailLearningRules`. The learning job stays OFF. |
| 8 | a | `2026-10-07_email_family_negotiation_counter.sql` | `4e36029c3c49028a` | `2026-10-07_email_family_negotiation_counter_rollback.sql` | INSERT `EmailFamily_negotiation_counter` (mode `shadow`; offer and lead time are claims unless confirmed). |
| 9 | a | `2026-10-07_email_family_free_prompt.sql` | `728758becdd81f6d` | `2026-10-07_email_family_free_prompt_rollback.sql` | INSERT `EmailFamily_free_prompt` (mode `shadow`). |
| 10 | a | `2026-10-08_email_family_rfq_batch.sql` | `6b376258fd96ef41` | `2026-10-08_email_family_rfq_batch_rollback.sql` | INSERT `EmailFamily_rfq_batch` (mode `shadow`, not classifiable). |
| 11 | a | `2026-10-08_email_family_human_written.sql` | `115d90b1afc77537` | `2026-10-08_email_family_human_written_rollback.sql` | INSERT `EmailFamily_human_written` (mode `shadow`, not classifiable). |
| 12 | a | `2026-10-08_email_family_v2.sql` | `5a039ce0b4bf1d48` | `2026-10-08_email_family_v2_rollback.sql` | UPDATE only the two original family rows (created_by = `email_assurance_migration`): fact labels, rubric, authority agent. |
| 13 | a | `2026-10-08_email_draft_sweep.sql` | `61fc89e9772e4027` | `2026-10-08_email_draft_sweep_rollback.sql` | INSERT `EmailDraftSweepRules` (`abandon_after_days: 14`, `batch_size: 500`). |
| 14 | b | `2026-10-08_email_agent_steering.sql` | `7f21b66f8a5f762b` | `2026-10-08_email_agent_steering_rollback.sql` | INSERT `EmailSteeringRules`. **Held back with the tone rules and prompts.** |
| 15 | b | `2026-10-08_email_tone_rules.sql` | `436d01b122b0807b` | `2026-10-08_email_tone_rules_rollback.sql` | INSERT `EmailToneRules` (with tone `directives`). **Awaiting your review; held back.** |
| 16 | b | `2026-10-08_email_assurance_prompts.sql` | `bc18e33f3140386c` | `2026-10-08_email_assurance_prompts_rollback.sql` | INSERT three rows into `proc.bp_prompt`: classify, plan, judge. **Awaiting your review; held back.** |
| 17 | roles | `2026-10-09_email_agent_roles.sql` | `969b5c86b8bff0cf` | `2026-10-09_email_agent_roles_rollback.sql` | CREATE two NOLOGIN roles (cluster-wide) and the grants listed below. **Its own change request; see `2026-10-08-email-agent-roles-change-request.md`.** |

Not in any pack, on purpose: login roles and passwords (created by an operator, out of band), the `EMAIL_AGENT_*` environment variables,
flipping any family from `shadow` to `enforce`, `EMAIL_LEARNING_ENABLED` (the learning job stays off).
**Two jobs start on their own once the tables exist** (they are ON unless switched off, and do nothing where the schema or setting is absent):
`email-text-retention` (deletes raw text older than the period) and `email-draft-sweep` (closes quiet, confirmed-unsent drafts). Set
`EMAIL_TEXT_RETENTION_ENABLED=0` / `EMAIL_DRAFT_SWEEP_ENABLED=0` to hold either.

## Deploy the schema BEFORE the code that uses it

The application code on `origin/Development` writes to the (a) tables and columns. Where the migrations are not applied, capture,
send-outcome recording, the sweep and retention each log an error and record NOTHING (drafting and sending carry on, by design), so
the failure is quiet. Apply pack (a) first, or leave `email_agent` absent knowingly. `bp_testdb` has only the original capture
files applied (capture, capture_v2, family v2), so it is already in that state for the newer columns. A test now applies pack (a) alone
to a fresh database and runs capture, send, sweep, retention, learning and metrics on it
(`tests/email_evals/test_pack_split.py`). Nothing had ever applied (a) alone before that test, which is how the `steering` column came to
be in pack (b) while the capture code wrote to it; putting the column back into (b) makes the test fail.

## Exactly what changes in the database

**New objects (pack a):** schema `email_agent`; ten tables: `bp_draft_capture`, `bp_draft_outcome`, `bp_draft_sent_text`, `bp_inbound_flag`, `bp_dq_item`,
`bp_eval_candidate`, `bp_review_item`, `bp_style_rule`, `bp_classifier_example`, `bp_exemplar_candidate` (and their sequences and indexes).
**New rows:** 9 in `proc.bp_policy` (`EmailFamily_negotiation_counter`, `EmailFamily_free_prompt`, `EmailFamily_rfq_batch`,
`EmailFamily_human_written`, `EmailToneRules`, `EmailLearningRules`, `EmailTextRetention`, `EmailSteeringRules`, `EmailDraftSweepRules`; of
these `EmailToneRules` and `EmailSteeringRules` are pack b), and 3 in `proc.bp_prompt` (pack b).
**Existing objects altered: none.** No existing table, column, index, trigger, policy row, prompt row, role or grant is modified; the one UPDATE
(`family_v2`) touches only rows this pack created. A test connects to a database the roles file has never touched, records what every
pre-existing role and PUBLIC can effectively do, applies the file, and requires an identical answer.

## Grant changes (the only privileges the pack adds; all in the roles file)

| role | object | privilege |
|---|---|---|
| `email_agent_reader` | schema `proc` | USAGE |
| `email_agent_reader` | `proc.supplier_response`, `proc.workflow_email_tracking` | SELECT |
| `email_agent_reader` | `proc.bp_supplier` columns `supplier_id, supplier_name, contact_name_1, contact_email_1, contact_name_2, contact_email_2, contact_role_1, is_preferred_supplier, country` | SELECT (column-level; **no `bank_*`, tax or registration columns**) |
| `email_agent_reader` | `proc.draft_rfq_emails` columns `unique_id, sent, sent_on` | SELECT (column-level; **no body, subject, recipients, payload or attachments**) |
| `email_agent_writer` | schema `email_agent` | USAGE |
| `email_agent_writer` | the eight original `email_agent` tables | SELECT, INSERT, UPDATE (no DELETE, no TRUNCATE) |
| `email_agent_writer` | `email_agent.bp_draft_sent_text` | SELECT, INSERT, **DELETE** (the retention purge; the only table it may delete from) |
| `email_agent_writer` | `email_agent.bp_inbound_flag` | SELECT, INSERT, UPDATE (a flag is recorded and decided, never deleted) |
| `email_agent_writer` | all `email_agent` sequences | USAGE, SELECT |

The reader has no access to `email_agent` at all and PUBLIC has none to the raw-text table. Role attributes: `NOLOGIN NOSUPERUSER NOCREATEDB
NOCREATEROLE NOREPLICATION NOBYPASSRLS`. The reader also carries `default_transaction_read_only = on`, a guardrail only (a session may turn
it off); the missing write privileges are the boundary, and the role tests connect AS each role and attempt writes, DDL, GRANTs, `SET ROLE`,
and reads of ungranted tables, columns and bank fields. The deny tests run with that default switched OFF, so only the missing privilege can
stop them.

**A role is cluster-wide.** `bp_sqldb` shares the RDS cluster `procwisemvpdb01` with `bp_testdb`, `uicanvas`, `ses` and others. `CREATE ROLE`
creates the role for all of them, although the GRANTS apply only in the database they are run in (see the roles change request, which
also lists what PUBLIC can reach).

## The product-table change (its own change request)

`2026-10-08_supplier_response_provenance.sql` adds eight columns to **`proc.supplier_response`**: `extraction_status`
(`extracted_unverified` | `confirmed` | `rejected`), `extraction_method`, `extraction_model`, `extraction_prompt_version`,
`extraction_confidence`, `extracted_at`, `confirmed_by`, `confirmed_at`.

* **Why:** the analyser takes the first number in an inbound email as the price (a model may override it) and stores it with no record of
  that. The email layer reads that column as a Postgres fact. These columns let it tell a CLAIM from a value a person vouched for.
* **Risk:** all eight are nullable with no default, so it is a metadata-only change; on `bp_sqldb` the table holds 1 row. Existing rows are
  NOT backfilled (their origin cannot be known; NULL means "origin not recorded"). No existing column, index, constraint or reader is altered.
  The rollback drops the eight columns and nothing else.
* **What happens before it is applied:** the analyser's stamp is tolerant (a missing column costs nothing, nothing is stored); the email
  layer treats a missing column the same as "origin not recorded", so every offer reads as a claim.
* **What happens after:** new inbound replies are stamped `extracted_unverified`; old ones stay NULL. Either way the offer is a claim until a
  person confirms it; a way to confirm the ROW is not built, so confirming is per draft.
* **Rehearsed** as the first file of the 17 on a restored copy of the real table (with its real constraints): applies, rolls back to an
  identical schema, re-applies.

## Pre-flight (read-only, run on bp_sqldb before anything)

```sql
-- the tables the pack reads or alters must exist
SELECT table_name, count(*) FROM information_schema.columns WHERE table_schema = 'proc'
  AND table_name IN ('supplier_response','bp_supplier','workflow_email_tracking','bp_policy','bp_prompt','bp_approval','bp_mailbox_binding','bp_agent_actions','draft_rfq_emails')
GROUP BY 1;                                   -- all nine present
SELECT count(*) FROM pg_namespace WHERE nspname = 'email_agent';                       -- expect 0
SELECT count(*) FROM pg_roles WHERE rolname LIKE 'email_agent%';                        -- expect 0 (or 2 if the cluster peers already got them)
SELECT count(*) FROM proc.bp_policy WHERE policy_name LIKE 'Email%Rules' OR policy_name LIKE 'EmailFamily_%' OR policy_name LIKE 'Email%Retention';   -- expect 0
SELECT has_schema_privilege('public','proc','USAGE');                                   -- expect false (PUBLIC starts with nothing)
```

## Apply (as an operator with rights on the schemas)

```bash
cd deploy/sql
# PRODUCT-TABLE change first, through its own change request:
#   psql -v ON_ERROR_STOP=1 -h "$HOST" -U "$USER" -d bp_sqldb -f 2026-10-08_supplier_response_provenance.sql
# PACK (a) now:
for f in 2026-10-07_email_agent_capture 2026-10-08_email_agent_capture_v2 2026-10-08_email_agent_sent_text 2026-10-08_email_agent_steering_column 2026-10-08_email_inbound_flag 2026-10-09_email_agent_learning 2026-10-07_email_family_negotiation_counter 2026-10-07_email_family_free_prompt 2026-10-08_email_family_rfq_batch 2026-10-08_email_family_human_written 2026-10-08_email_family_v2 2026-10-08_email_draft_sweep; do
  psql -v ON_ERROR_STOP=1 -h "$HOST" -U "$USER" -d bp_sqldb -f "$f.sql" || break      # stop at the first failure
done
# PACK (b) only after live verification passes:
for f in 2026-10-08_email_agent_steering 2026-10-08_email_tone_rules 2026-10-08_email_assurance_prompts; do
  psql -v ON_ERROR_STOP=1 -h "$HOST" -U "$USER" -d bp_sqldb -f "$f.sql" || break
done
```
Do not use `psql -1`: the files are self-transactional and nesting only produces warnings. The roles file is applied through its own change
request, AFTER the files it grants on (capture and sent_text) and not before. Login roles and passwords are created separately, out of band
(see the header of `2026-10-09_email_agent_roles.sql`); until they exist the application uses its own login and records `read_control` on
every draft.

## Post-apply checks (read-only)

```sql
SELECT count(*) FROM information_schema.tables WHERE table_schema = 'email_agent';      -- 10
SELECT count(*) FROM proc.bp_policy WHERE created_by = 'email_assurance_migration';     -- 9 after (a) and (b); 7 after (a) alone
SELECT prompt_name FROM proc.bp_prompt WHERE prompt_name IN ('email_family_classify','email_brief_plan','email_draft_judge');   -- 3 after (b), 0 after (a) alone
-- after the roles change:
SELECT has_table_privilege('email_agent_reader','proc.supplier_response','INSERT');      -- false
SELECT has_column_privilege('email_agent_reader','proc.bp_supplier','bank_iban','SELECT');   -- false
SELECT has_column_privilege('email_agent_reader','proc.draft_rfq_emails','body','SELECT');   -- false
SELECT has_table_privilege('email_agent_writer','proc.supplier_response','SELECT');      -- false
SELECT has_table_privilege('email_agent_reader','email_agent.bp_draft_sent_text','SELECT');   -- false
```

## Rollback (newest first; each is idempotent)

The `*_rollback.sql` files in reverse order. Rollback removes the rows by `created_by = 'email_assurance_migration'`, drops the
`email_agent` schema, and drops the roles after `DROP OWNED BY`. **It destroys captured drafts, sent text and learning rows**: take a
`pg_dump -n email_agent` first if any exist.

## Rehearsal results (2026-10-08; a restored COPY of bp_sqldb's structure in a disposable Postgres 16 container)

Run with `python -m evals.email.rehearsal` (re-runnable before the real apply). Full log: `evals/email/rehearsal-log.md`.

- **The copy:** schema-only `pg_dump` of nine real tables (`supplier_response` with its sequence, `bp_supplier`, `workflow_email_tracking`,
  `bp_policy`, `bp_prompt`, `bp_approval`, `bp_mailbox_binding`, `bp_agent_actions`, `draft_rfq_emails`) with their real constraints and
  indexes, plus the rows of `bp_policy` (51) and `bp_prompt` (16). The only bp_sqldb access was that read-only dump.
- **Apply:** seventeen of seventeen files ok, each in about a tenth of a second, in the order the test harness uses (not the (a)/(b) split below).
- **Evals against the copy:** all golden cases pass (30 negotiation_counter, 26 free_prompt); the whole eval test suite passes against the copy, 406 of 406.
- **Rollback:** every rollback ok. Left behind: 0 roles, 0 `email_agent` schema, 0 policy rows, 0 prompt rows.
- **Schema fingerprint:** before the pack `74e91df8f5922063`, after rollback `74e91df8f5922063` (**identical**); the first apply gives
  `a86537b428a9a337` and re-applying after the rollback reproduces it exactly.

## What this rehearsal does NOT prove

- **The (a)/(b) split on the real structure.** The rehearsal applies all seventeen in one order. The split IS proven on a generated schema
  (pack (a) alone, then (b) on top, then (b) rolled back leaving (a) intact: `tests/email_evals/test_pack_split.py`) but was not run as two separate
  sessions against the restored bp_sqldb copy.
- **Data volume and live contents.** The copy holds the policy and prompt rows, not the business rows. Lock times on large tables are not
  exercised (the pack alters no existing table, only creates new ones and inserts 9 policy rows and 3 prompts).
- **Everything else on the bp_sqldb cluster.** Only nine tables were copied. Three constraints pointing at tables that are not part of the copy
  could not be restored (they concern `bp_agent_actions`, `bp_style_intent`, `bp_style_profile`; none is touched by the pack).
- **A real model.** Classifier accuracy, planner quality, judge calibration and the effect of steering remain unverified (checklist, section 1).
- **Your approvals.** The tone rules, the prompts and the steering settings are in pack (b) and await your review.
