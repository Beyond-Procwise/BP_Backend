# Email golden evals

Per-family golden cases for the email drafting assurance layer, run against a **seeded, throwaway Postgres**.

    python -m evals.email.runner                       # all families; exits non-zero on any regression
    python -m evals.email.runner --family free_prompt  # one family (no baseline check)
    pytest tests/email_evals                           # the same cases, one pytest test each

A Postgres comes from `EMAIL_EVAL_DSN` (CI) or, locally, a `postgres:16-alpine` container started and
removed for you. It is never one of the shared databases.

## What is real and what is scripted

| Real | Scripted |
|---|---|
| Table definitions (generated from the live columns: `python -m evals.email.gen_schema`) | The model: each case lists the output of each stage, good or bad |
| The migration files in `deploy/sql` (so a migration that stops loading fails here) | The two authority policies' *values* (copied verbatim from the live rows) |
| The drafting paths, validators, capture, send guard, policy engine | Recipient/sensitivity/role checks in the send guard (not under test; stubbed) |

A stage a case does not script behaves like a dead model, so "the model is never asked" is checkable.

## A case

`cases/<family>/NNN-name.json`: `run` (path + payload), `model` (compose/classify/plan/judge/repair),
`seed` (rows, merged onto `base_seed.json` unless `"base": false`), optional `policy_patch`, `policy_delete`,
`prompts: false`, `reviewer` (confirmations), `before_send` (move a fact), `send` (guard simulation), and
`expect`: dotted paths into `rec` (the assurance record), `view`, `row` (the stored capture row), `draft`,
`send`, `sent`, `confirm`, `calls`. A bare value means equals; `{"has"|"lacks"|"contains"|"not_contains"|"len"|"null"|"in"|"gte": x}`.
`"path #2"` repeats a key.

Write the expectation from the INTENDED behaviour. If a case fails, decide whether the code or the expectation
is wrong; do not copy what the system happened to produce.

## Blocking regressions

`baseline.json` gives each family a `min_cases` and a `min_pass_rate`. A lower pass rate fails; so do fewer cases
(deleting a case to get a green), and a family with cases but no baseline. Raising the baseline is a deliberate edit.

## Not covered here

Model quality: classifier accuracy, planner quality, judge calibration. See
`specs/2026-10-08-email-assurance-pending-live-verification.md`.

## Rehearsing against the real structure

    python -m evals.email.rehearsal        # needs read access to bp_sqldb, Docker, psql, pg_dump

Restores a schema-only copy of eight real tables (and the policy/prompt rows) into a disposable container, applies the whole DDL pack the way an
operator would, runs these goldens and the whole eval test suite against the copy (`EMAIL_EVAL_EXISTING_SCHEMA=1`), rolls the pack back, and
requires the schema fingerprint to equal the one from before. The last log is `rehearsal-log.md`. Re-run it before the real apply.
