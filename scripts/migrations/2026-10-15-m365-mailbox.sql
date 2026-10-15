-- scripts/migrations/2026-10-15-m365-mailbox.sql
-- Shared company mailboxes on Microsoft 365 (spec 2026-10-15-m365-shared-mailbox-design §4.2).
-- Runs AFTER 2026-08-07-mail-intake.sql and 2026-10-14-mail-intake-relay-binding.sql.
-- Idempotent. Every row valid before is valid after (no send_receive row exists yet).

ALTER TABLE proc.bp_mailbox_binding
    DROP CONSTRAINT IF EXISTS bp_mailbox_binding_role_check;
ALTER TABLE proc.bp_mailbox_binding
    ADD CONSTRAINT bp_mailbox_binding_role_check
    CHECK (role = ANY (ARRAY['exemplar_source', 'draft_target', 'both', 'intake', 'send_receive']));

ALTER TABLE proc.bp_mailbox_binding
    ADD COLUMN IF NOT EXISTS is_default        BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS last_test_send_at TIMESTAMPTZ;

-- A mailbox may be the default only once a person has watched a test email leave it:
-- sending cannot be probed without sending (spec §2.3 step 3).
ALTER TABLE proc.bp_mailbox_binding
    DROP CONSTRAINT IF EXISTS ck_bp_mailbox_binding_default_needs_test_send;
ALTER TABLE proc.bp_mailbox_binding
    ADD CONSTRAINT ck_bp_mailbox_binding_default_needs_test_send
    CHECK (NOT is_default OR (last_test_send_at IS NOT NULL AND role = 'send_receive'));

-- One default among active bindings.
CREATE UNIQUE INDEX IF NOT EXISTS ix_bp_mailbox_binding_default
    ON proc.bp_mailbox_binding ((1)) WHERE is_default AND is_active;

-- One active org binding per address (Review Focus 1).
CREATE UNIQUE INDEX IF NOT EXISTS ix_bp_mailbox_binding_org_address
    ON proc.bp_mailbox_binding (lower(mailbox_address))
    WHERE user_ref = 'org' AND is_active;
