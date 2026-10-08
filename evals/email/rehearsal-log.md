# Rehearsal of the email-assurance DDL pack on a copy of `bp_sqldb`'s structure

- run at 2026-10-07 21:35:45 UTC; container `postgres:16-alpine`

## 1. The copy (read-only dump of the real structure)

- restored schema: 670 lines; errors: 1
  - `ERROR:  function proc.bp_agent_actions_append_only() does not exist`  (a constraint pointing at a table that is not part of the copy)
- restored policy+prompt data: 123 lines; errors: 0
- copy holds: policies / prompts / proc tables = `51|16|8`

## 2. Schema fingerprint BEFORE the pack: `7c0678d42ef0dae3`

## 3. Apply the pack, one file at a time (`psql -v ON_ERROR_STOP=1 -f`; each file is its own transaction)

| file | result |
|---|---|
| `2026-10-07_email_agent_capture.sql` | ok (0.1s) |
| `2026-10-08_email_agent_capture_v2.sql` | ok (0.1s) |
| `2026-10-07_email_family_negotiation_counter.sql` | ok (0.1s) |
| `2026-10-07_email_family_free_prompt.sql` | ok (0.1s) |
| `2026-10-08_email_family_v2.sql` | ok (0.1s) |
| `2026-10-08_email_tone_rules.sql` | ok (0.1s) |
| `2026-10-08_email_assurance_prompts.sql` | ok (0.1s) |
| `2026-10-09_email_agent_learning.sql` | ok (0.2s) |
| `2026-10-09_email_agent_roles.sql` | ok (0.1s) |

- fingerprint after apply: `a56c47013e69b98f` (differs from before: True)

## 4. The eval suite, run against the copy

```
family                    cases  passed   rate
free_prompt                  26      26   100%
negotiation_counter          30      30   100%
```
- (the runner re-applies the migrations itself, which also proves they are idempotent on the copy)
- `pytest tests/email_evals` against the copy: **207 passed in 31.81s**

## 5. Roll the pack back, newest first

| file | result |
|---|---|
| `2026-10-09_email_agent_roles_rollback.sql` | ok |
| `2026-10-09_email_agent_learning_rollback.sql` | ok |
| `2026-10-08_email_assurance_prompts_rollback.sql` | ok |
| `2026-10-08_email_tone_rules_rollback.sql` | ok |
| `2026-10-08_email_family_v2_rollback.sql` | ok |
| `2026-10-07_email_family_free_prompt_rollback.sql` | ok |
| `2026-10-07_email_family_negotiation_counter_rollback.sql` | ok |
| `2026-10-08_email_agent_capture_v2_rollback.sql` | ok |
| `2026-10-07_email_agent_capture_rollback.sql` | ok |

- fingerprint after rollback: `7c0678d42ef0dae3`  **equals the one before: True**
- left behind (roles / email_agent schema / policy rows / prompt rows): `0|0|0|0`

## 6. Re-applied after rollback: fingerprint `a56c47013e69b98f` (same as the first apply: True)

**Overall: PASS**
