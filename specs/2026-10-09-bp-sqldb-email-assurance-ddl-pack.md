# bp_sqldb: email assurance DDL pack

Prepared 2026-10-09. **NOT APPLIED to bp_sqldb, bp_testdb's cluster peers, or production.** Nothing here is applied until
(1) sections 1 and 2 of `2026-10-08-email-assurance-pending-live-verification.md` are cleared and (2) you say so.

## What it is

Nine files, run in this order. Each is idempotent (re-running changes nothing), is its own transaction, and has a rollback.
The hash is the first 16 hex of the file's sha256, so you can check the file applied is the file rehearsed.

| # | apply | sha256 (16) | rollback | what it does |
|---|---|---|---|---|
| 1 | `2026-10-07_email_agent_capture.sql` | `7bf01e433b523aac` | `2026-10-07_email_agent_capture_rollback.sql` | Creates schema `email_agent`; tables `bp_draft_capture`, `bp_draft_outcome` (+ index, unique-once index). |
| 2 | `2026-10-08_email_agent_capture_v2.sql` | `5b9625ec63b4b3c6` | `2026-10-08_email_agent_capture_v2_rollback.sql` | ALTER TABLE on the two new tables: stage outputs, accountability, readiness; drops the unused `user_id`; column comments marking supplier-reply outcomes OUT OF SCOPE. |
| 3 | `2026-10-07_email_family_negotiation_counter.sql` | `1415f2ac4ca14648` | `2026-10-07_email_family_negotiation_counter_rollback.sql` | INSERT one row into `proc.bp_policy`: `EmailFamily_negotiation_counter` (mode `shadow`). |
| 4 | `2026-10-07_email_family_free_prompt.sql` | `706151b0e4c0ebdb` | `2026-10-07_email_family_free_prompt_rollback.sql` | INSERT one row into `proc.bp_policy`: `EmailFamily_free_prompt` (mode `shadow`). |
| 5 | `2026-10-08_email_family_v2.sql` | `5a039ce0b4bf1d48` | `2026-10-08_email_family_v2_rollback.sql` | UPDATE only those two rows (created_by = `email_assurance_migration`): fact labels, rubric, authority agent. |
| 6 | `2026-10-08_email_tone_rules.sql` | `acfa9975bdc08efe` | `2026-10-08_email_tone_rules_rollback.sql` | INSERT one row into `proc.bp_policy`: `EmailToneRules`. **Awaiting your review.** |
| 7 | `2026-10-08_email_assurance_prompts.sql` | `bc18e33f3140386c` | `2026-10-08_email_assurance_prompts_rollback.sql` | INSERT three rows into `proc.bp_prompt`: classify, plan, judge. **Awaiting your review.** |
| 8 | `2026-10-09_email_agent_learning.sql` | `0b11e29ea46484c0` | `2026-10-09_email_agent_learning_rollback.sql` | Six queue/candidate tables in `email_agent`; two columns on `bp_draft_outcome`; one index; INSERT one row into `proc.bp_policy`: `EmailLearningRules`. |
| 9 | `2026-10-09_email_agent_roles.sql` | `e8440acf4ee1241e` | `2026-10-09_email_agent_roles_rollback.sql` | CREATE two NOLOGIN roles (cluster-wide) and the grants listed below. |

Not in the pack, on purpose: login roles and passwords (created by an operator, out of band), the `EMAIL_AGENT_*` environment
variables, flipping any family from `shadow` to `enforce`, and `EMAIL_LEARNING_ENABLED` (the learning job stays off).

## Exactly what changes in the database

**New objects:** schema `email_agent`; tables `bp_draft_capture`, `bp_draft_outcome`, `bp_dq_item`, `bp_eval_candidate`,
`bp_review_item`, `bp_style_rule`, `bp_classifier_example`, `bp_exemplar_candidate` (and their sequences/indexes); two roles.
**New rows:** 4 in `proc.bp_policy` (`EmailFamily_negotiation_counter`, `EmailFamily_free_prompt`, `EmailToneRules`,
`EmailLearningRules`), 3 in `proc.bp_prompt`. **Existing objects altered: none.** No existing table, column, index, trigger,
policy row, prompt row, role or grant is modified. A test connects to a database the roles file has never touched, records
what every pre-existing role and PUBLIC can effectively do (over 300 privilege checks across `proc` and `email_agent`, including column-level ones on `bp_supplier`),
applies the file, and requires an identical answer.

## Grant changes (the only privileges the pack adds)

| role | schema | object | privilege |
|---|---|---|---|
| `email_agent_reader` | `proc` | schema | USAGE |
| `email_agent_reader` | `proc` | `supplier_response`, `workflow_email_tracking` | SELECT |
| `email_agent_reader` | `proc` | `bp_supplier` columns `supplier_id, supplier_name, contact_name_1, contact_email_1, contact_name_2, contact_email_2, contact_role_1, is_preferred_supplier, country` | SELECT (column-level; **no `bank_*`, tax or registration columns**) |
| `email_agent_writer` | `email_agent` | schema | USAGE |
| `email_agent_writer` | `email_agent` | the eight tables above | SELECT, INSERT, UPDATE (no DELETE, no TRUNCATE) |
| `email_agent_writer` | `email_agent` | all sequences | USAGE, SELECT |

Role attributes: `NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS`. The reader also carries
`default_transaction_read_only = on`, a guardrail only (a session may turn it off); the missing write privileges are the boundary,
and 57 tests connect AS each role and attempt writes, DDL, GRANTs, `SET ROLE` and reads of ungranted tables and bank columns.
The deny tests run with that default switched OFF, so only the missing privilege can stop them.

**A role is cluster-wide.** `bp_sqldb` shares the RDS cluster `procwisemvpdb01` with `bp_testdb`, `uicanvas`, `ses` and others.
`CREATE ROLE` creates the role for all of them, although the GRANTS apply only in the database they are run in. For that reason
the roles were NOT created on `bp_testdb` as a "non-prod copy": there is no non-prod cluster. Rolling back needs the roles'
privileges revoked in every database that received them before the role can be dropped (the rollback fails loudly otherwise).

## Pre-flight (read-only, run on bp_sqldb before anything)

```sql
-- the tables and columns the pack reads or alters must exist (all were present when this was prepared 2026-10-09)
SELECT table_name, count(*) FROM information_schema.columns WHERE table_schema = 'proc'
  AND table_name IN ('supplier_response','bp_supplier','workflow_email_tracking','bp_policy','bp_prompt','bp_approval','bp_mailbox_binding','bp_agent_actions')
GROUP BY 1;                                   -- expect 36, 52, 15, 12, 11, 19, (>=13), and bp_agent_actions present
SELECT count(*) FROM pg_namespace WHERE nspname = 'email_agent';                       -- expect 0
SELECT count(*) FROM pg_roles WHERE rolname LIKE 'email_agent%';                        -- expect 0 (or 2 if bp_testdb's cluster already got them)
SELECT count(*) FROM proc.bp_policy WHERE policy_name LIKE 'Email%Rules' OR policy_name LIKE 'EmailFamily_%';   -- expect 0
SELECT policy_name, policy_status FROM proc.bp_policy WHERE policy_details->>'policy_identifier' IN ('email_reply_autonomy','approval_threshold');
                                              -- bp_sqldb holds 2 INACTIVE (status 0) + 1 active EmailReplyAutonomyPolicy; the active rules equal bp_testdb's
SELECT has_schema_privilege('public','proc','USAGE');                                   -- expect false (PUBLIC starts with nothing)
```

## Apply (as an operator with CREATEROLE and rights on the schemas)

```bash
cd deploy/sql
for f in 2026-10-07_email_agent_capture 2026-10-08_email_agent_capture_v2 2026-10-07_email_family_negotiation_counter \
         2026-10-07_email_family_free_prompt 2026-10-08_email_family_v2 2026-10-08_email_tone_rules \
         2026-10-08_email_assurance_prompts 2026-10-09_email_agent_learning 2026-10-09_email_agent_roles; do
  psql -v ON_ERROR_STOP=1 -h "$HOST" -U "$USER" -d bp_sqldb -f "$f.sql" || break      # stop at the first failure
done
```
Do not use `psql -1`: the files are self-transactional and nesting only produces warnings.

Then, separately and with the password supplied out of band (see the header of `2026-10-09_email_agent_roles.sql`):
`CREATE ROLE email_agent_ro_svc LOGIN PASSWORD :'pw' IN ROLE email_agent_reader; ALTER ROLE email_agent_ro_svc SET default_transaction_read_only = on;`
and the writer equivalent; then set `EMAIL_AGENT_RO_USER/PASSWORD` and `EMAIL_AGENT_RW_USER/PASSWORD`. Until they are set the application runs
the interim control and records `read_control` on every draft.

## Post-apply checks (read-only)

```sql
SELECT count(*) FROM information_schema.tables WHERE table_schema = 'email_agent';      -- 8
SELECT policy_name FROM proc.bp_policy WHERE created_by = 'email_assurance_migration' ORDER BY 1;   -- 4 rows
SELECT prompt_name FROM proc.bp_prompt WHERE prompt_name IN ('email_family_classify','email_brief_plan','email_draft_judge');   -- 3 rows
SELECT has_table_privilege('email_agent_reader','proc.supplier_response','INSERT');      -- false
SELECT has_column_privilege('email_agent_reader','proc.bp_supplier','bank_iban','SELECT');   -- false
SELECT has_table_privilege('email_agent_writer','proc.supplier_response','SELECT');      -- false
```

## Rollback (newest first; each is idempotent)

The nine `*_rollback.sql` files in reverse order. Rollback removes the rows by `created_by = 'email_assurance_migration'`
(and the prompts by their `created_by`), drops the `email_agent` schema, and drops the roles after `DROP OWNED BY`.
**It destroys captured drafts and learning rows**: take a `pg_dump -n email_agent` first if any exist.

## Rehearsal results (a restored COPY of bp_sqldb's structure in a disposable Postgres 16 container)

Run with `python -m evals.email.rehearsal` (re-runnable before the real apply). Full log: `evals/email/rehearsal-log.md`.

- **The copy:** schema-only `pg_dump` of eight real tables (`supplier_response` with its sequence, `bp_supplier`, `workflow_email_tracking`,
  `bp_policy`, `bp_prompt`, `bp_approval`, `bp_mailbox_binding`, `bp_agent_actions`) with their real constraints and indexes, plus the
  rows of `bp_policy` (51) and `bp_prompt` (16). The only bp_sqldb access was that read-only dump.
- **Apply:** nine of nine files ok, each in under a second.
- **Evals against the copy:** 56 of 56 golden cases pass (30 negotiation_counter, 26 free_prompt); the whole eval test suite passes, 207 of 207.
- **Rollback:** nine of nine ok. Left behind: 0 roles, 0 `email_agent` schema, 0 policy rows, 0 prompt rows.
- **Schema fingerprint** (a hash of the `proc` and `email_agent` schema, no data): before the pack `7c0678d42ef0dae3`, after rollback `7c0678d42ef0dae3` (**identical**); the first apply gives
  `a56c47013e69b98f` and re-applying after the rollback reproduces it exactly.

What the rehearsal found (all fixed): production's `supplier_response` and `workflow_email_tracking` have unique keys my test seeds violated;
`policy_id` is `GENERATED ALWAYS` (the eval harness could not re-insert rows); `supplier_response.id` uses a standalone sequence that
`RESTART IDENTITY` does not reset; the real `email_volume` policy makes the send guard read `proc.bp_agent_actions`, which it refused to send without.

## What this rehearsal does NOT prove

- **Data volume and live contents.** The copy holds the policy and prompt rows, not the business rows. Lock times on large tables are not exercised (the pack
  alters no existing table, only creates new ones and inserts 7 rows, so the exposure is small).
- **Everything else on the bp_sqldb cluster.** Only the eight tables above were copied. Interactions with triggers/functions elsewhere were not
  exercised (one function, `proc.bp_agent_actions_append_only`, was not part of the copy; the guard does not call it).
- **A real model.** Classifier accuracy, planner quality and judge calibration remain unverified (checklist section 1).
- **Your approvals.** The tone rules and the three prompts are applied by files 6 and 7 but are still awaiting your review.

## Addendum 2026-10-08: two more files, not yet rehearsed

`2026-10-08_email_family_rfq_batch.sql` (+ `_rollback.sql`) adds the `rfq_batch` family row (mode `shadow`, same shape as
`free_prompt`; INSERT one row into `proc.bp_policy`). It was added AFTER the rehearsal above and has run only in the
throwaway eval database (apply, roll back, re-apply: green), NOT against the restored bp_sqldb copy. Under the 2026-10-08
ruling it belongs in pack (a) (schema, capture, validators in shadow). Re-run `python -m evals.email.rehearsal` before
this pack is used; the counts above become eleven of eleven.

`2026-10-08_email_family_human_written.sql` (+ `_rollback.sql`) is the same shape for the `human_written` family: an
email a person typed (reply panel, report panel, manual passthrough). Same status: eval database only, pack (a).

`2026-10-08_email_agent_sent_text.sql` (+ `_rollback.sql`) adds `email_agent.bp_draft_sent_text` (raw text, writer-only),
`bp_draft_capture.text_expired_at`, and the `EmailTextRetention` policy row (`raw_text_days: 90`). It sits after
`capture_v2` and before the roles migration. **`2026-10-09_email_agent_roles.sql` was edited** to grant the writer
SELECT/INSERT/DELETE on that one table (guarded by `to_regclass`, so it is a no-op where the table is absent); its
checksum above is therefore stale. Twelve files now; rehearsal still to be re-run, and none of this touches bp_sqldb.

`2026-10-08_email_agent_steering.sql` (+ rollback): `bp_draft_capture.steering` and the `EmailSteeringRules` policy row. Belongs to
**pack (b)** (tone rules, prompts, steering: held back until live verification). `2026-10-08_email_tone_rules.sql` was also
edited (a `directives` block). Thirteen files now; checksums above stale for the tone rules; rehearsal to be re-run.

Also edited after the rehearsal: the four family rows gained config keys (`classifiable` on `rfq_batch` and `human_written`;
`request_description` and `request_label` on `negotiation_counter` and `free_prompt`). Insert-only rows, nothing applied; the
checksums of those four files above are stale and the rehearsal must be re-run.
