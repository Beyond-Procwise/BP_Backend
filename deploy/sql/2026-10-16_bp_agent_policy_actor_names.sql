-- Agent policy governance, stage 5b: people are shown by name.
-- The sign-in email of whoever recorded a second review or a reversal, kept beside the sign-in id
-- so the screens can show who by name. Nullable: rows written before this, and sign-ins that carry
-- no email, have none (the id is shown). Additive and idempotent; applied to bp_testdb and bp_sqldb.
-- (The reversal table's append-only trigger guards UPDATE/DELETE on rows, not ADD COLUMN.)
BEGIN;
ALTER TABLE proc.bp_agent_policy_second_review ADD COLUMN IF NOT EXISTS reviewed_by_email TEXT;
ALTER TABLE proc.bp_policy_firing_reversal ADD COLUMN IF NOT EXISTS reversed_by_email TEXT;
COMMIT;
