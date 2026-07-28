-- Resync the identity sequences behind the governance tables.
--
-- proc.bp_prompt.prompt_id and proc.bp_policy.policy_id are GENERATED ALWAYS AS
-- IDENTITY, but both sequences were left behind the rows already in the tables
-- (bp_prompt held ids up to 100 while its sequence was still handing out 5).
-- Every application INSERT therefore died on the primary key:
--
--   duplicate key value violates unique constraint "bp_prompt_pkey"
--   DETAIL: Key (prompt_id)=(4) already exists.
--
-- which took out POST /agents (create an agent from the workspace), the
-- extraction-feedback hint writer (services/extraction_feedback/apply.py) and
-- any policy insert — all of them reported to the user as a sanitised
-- "I couldn't retrieve that", because the gate strips the constraint name.
--
-- Idempotent: safe to run again, and a no-op once the sequences are ahead.
-- COALESCE covers an empty table (setval to 0 with is_called false, so the next
-- value is 1 rather than an error).

SELECT setval(
    pg_get_serial_sequence('proc.bp_prompt', 'prompt_id'),
    COALESCE((SELECT MAX(prompt_id) FROM proc.bp_prompt), 0) + 1,
    false
);

SELECT setval(
    pg_get_serial_sequence('proc.bp_policy', 'policy_id'),
    COALESCE((SELECT MAX(policy_id) FROM proc.bp_policy), 0) + 1,
    false
);
