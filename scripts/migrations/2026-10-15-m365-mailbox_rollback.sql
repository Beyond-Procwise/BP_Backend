-- Rollback of 2026-10-15-m365-mailbox.sql.
-- WARNING: the DELETE below removes every company (send_receive) mailbox binding, and with
-- them last_test_send_at - the "this company has sent from its own mailbox" signal that
-- keeps supplier email HELD (never sent via SES) after a disconnect. After this rollback a
-- customer that had been sending from Microsoft 365 goes back to the Procwise SES sender.
BEGIN;
DROP INDEX IF EXISTS proc.ix_bp_mailbox_binding_org_address;
DROP INDEX IF EXISTS proc.ix_bp_mailbox_binding_default;
ALTER TABLE proc.bp_mailbox_binding DROP CONSTRAINT IF EXISTS ck_bp_mailbox_binding_default_needs_test_send;
DELETE FROM proc.bp_mailbox_binding WHERE role = 'send_receive';
ALTER TABLE proc.bp_mailbox_binding DROP COLUMN IF EXISTS last_test_send_at;
ALTER TABLE proc.bp_mailbox_binding DROP COLUMN IF EXISTS is_default;
ALTER TABLE proc.bp_mailbox_binding DROP CONSTRAINT IF EXISTS bp_mailbox_binding_role_check;
ALTER TABLE proc.bp_mailbox_binding ADD CONSTRAINT bp_mailbox_binding_role_check
    CHECK (role = ANY (ARRAY['exemplar_source', 'draft_target', 'both', 'intake']));
COMMIT;
