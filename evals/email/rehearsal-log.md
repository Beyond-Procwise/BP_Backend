# Rehearsal of the email-assurance DDL pack on a copy of `bp_sqldb`'s structure

- run at 2026-10-08 16:01:53 UTC; container `postgres:16-alpine`

## 1. The copy (read-only dump of the real structure)

- restored schema: 796 lines; errors: 3
  - `ERROR:  function proc.bp_agent_actions_append_only() does not exist`  (a constraint pointing at a table that is not part of the copy)
  - `ERROR:  relation "proc.bp_style_intent" does not exist`  (a constraint pointing at a table that is not part of the copy)
  - `ERROR:  relation "proc.bp_style_profile" does not exist`  (a constraint pointing at a table that is not part of the copy)
- restored policy+prompt data: 123 lines; errors: 0
- copy holds: policies / prompts / proc tables = `51|16|9`

## 2. Schema fingerprint BEFORE the pack: `74e91df8f5922063`

## 3. Apply the pack, one file at a time (`psql -v ON_ERROR_STOP=1 -f`; each file is its own transaction)

| file | result |
|---|---|
| `2026-10-07_email_agent_capture.sql` | ok (0.1s) |
| `2026-10-08_email_agent_capture_v2.sql` | ok (0.1s) |
| `2026-10-08_email_agent_sent_text.sql` | ok (0.1s) |
| `2026-10-08_email_agent_steering_column.sql` | ok (0.0s) |
| `2026-10-08_email_agent_steering.sql` | ok (0.1s) |
| `2026-10-08_email_draft_sweep.sql` | ok (0.0s) |
| `2026-10-07_email_family_negotiation_counter.sql` | ok (0.0s) |
| `2026-10-07_email_family_free_prompt.sql` | ok (0.1s) |
| `2026-10-08_email_family_rfq_batch.sql` | ok (0.0s) |
| `2026-10-08_email_family_human_written.sql` | ok (0.0s) |
| `2026-10-08_email_family_v2.sql` | ok (0.0s) |
| `2026-10-08_email_tone_rules.sql` | ok (0.0s) |
| `2026-10-08_email_assurance_prompts.sql` | ok (0.1s) |
| `2026-10-09_email_agent_learning.sql` | ok (0.1s) |
| `2026-10-09_email_agent_roles.sql` | ok (0.0s) |

- fingerprint after apply: `7ef79057b8574e67` (differs from before: True)

## 4. The eval suite, run against the copy

```
family                    cases  passed   rate
free_prompt                  26      26   100%
negotiation_counter          30      30   100%
```
- (the runner re-applies the migrations itself, which also proves they are idempotent on the copy)
- `pytest tests/email_evals` against the copy: **364 passed in 40.49s**

## 5. Roll the pack back, newest first

| file | result |
|---|---|
| `2026-10-09_email_agent_roles_rollback.sql` | ok |
| `2026-10-09_email_agent_learning_rollback.sql` | ok |
| `2026-10-08_email_assurance_prompts_rollback.sql` | ok |
| `2026-10-08_email_tone_rules_rollback.sql` | ok |
| `2026-10-08_email_family_v2_rollback.sql` | ok |
| `2026-10-08_email_family_human_written_rollback.sql` | ok |
| `2026-10-08_email_family_rfq_batch_rollback.sql` | ok |
| `2026-10-07_email_family_free_prompt_rollback.sql` | ok |
| `2026-10-07_email_family_negotiation_counter_rollback.sql` | ok |
| `2026-10-08_email_draft_sweep_rollback.sql` | ok |
| `2026-10-08_email_agent_steering_rollback.sql` | ok |
| `2026-10-08_email_agent_steering_column_rollback.sql` | ok |
| `2026-10-08_email_agent_sent_text_rollback.sql` | ok |
| `2026-10-08_email_agent_capture_v2_rollback.sql` | ok |
| `2026-10-07_email_agent_capture_rollback.sql` | ok |

- fingerprint after rollback: `74e91df8f5922063`  **equals the one before: True**
- left behind (roles / email_agent schema / policy rows / prompt rows): `0|0|0|0`

## 6. Re-applied after rollback: fingerprint `7ef79057b8574e67` (same as the first apply: True)

**Overall: PASS**
