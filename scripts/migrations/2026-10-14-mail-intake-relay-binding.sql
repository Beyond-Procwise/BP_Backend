-- scripts/migrations/2026-10-14-mail-intake-relay-binding.sql
--
-- A relay binding holds no credential. Mail is pushed to our own MTA (SES) and
-- written to our own bucket; nothing logs in anywhere. But credential_ref was
-- NOT NULL and had to look like a Secrets Manager ARN, so a relay binding could
-- only be stored with an ARN pointing at a secret that does not exist — a fake
-- reference the customer document ("no credential is issued") would contradict.
--
-- Now: a relay binding MUST have no credential, every other provider MUST have
-- an ARN. Spelled as CASE rather than OR because a CHECK passes on NULL, and
-- `credential_ref LIKE ...` is NULL for a NULL credential — the OR form would
-- quietly admit a graph or IMAP binding with no credential at all.
--
-- Idempotent. Every row valid before is valid after (no relay row exists yet).

ALTER TABLE proc.bp_mailbox_binding
    ALTER COLUMN credential_ref DROP NOT NULL;

ALTER TABLE proc.bp_mailbox_binding
    DROP CONSTRAINT IF EXISTS ck_bp_mailbox_binding_credential_is_arn;

ALTER TABLE proc.bp_mailbox_binding
    ADD CONSTRAINT ck_bp_mailbox_binding_credential_is_arn
    CHECK (CASE WHEN provider = 'relay' THEN credential_ref IS NULL
                ELSE credential_ref IS NOT NULL
                     AND credential_ref LIKE 'arn:aws:secretsmanager:%' END);
