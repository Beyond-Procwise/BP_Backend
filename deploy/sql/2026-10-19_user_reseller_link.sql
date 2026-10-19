-- Which reseller accounts a user may see through the data gateway (ruled 2026-10-19: a user -> reseller link table).
-- NOT YET APPLIED to any database: a schema change, held for sign-off. Additive and idempotent.
--
-- Same shape and rules as proc.bp_user_buyer_scope: appended and revoked, never edited; only an Admin grants (enforced in the
-- API that writes it, not built yet); an Admin sees every reseller and needs no row; no row means NO reseller access.
--
-- reseller_account_id is free text with no foreign key ON PURPOSE: no reseller entity exists in the estate yet (bp_account holds
-- sales customers). When the reseller model is defined this column gets its reference and a backfill, in its own migration.
CREATE TABLE IF NOT EXISTS proc.bp_user_reseller_link (
    link_id             bigserial PRIMARY KEY,
    subject             text        NOT NULL,
    reseller_account_id text        NOT NULL,
    granted_by          text        NOT NULL,
    granted_at          timestamptz NOT NULL DEFAULT now(),
    revoked_at          timestamptz,
    revoked_by          text
);
CREATE UNIQUE INDEX IF NOT EXISTS ix_bp_user_reseller_link_active
    ON proc.bp_user_reseller_link (subject, reseller_account_id) WHERE revoked_at IS NULL;
CREATE INDEX IF NOT EXISTS ix_bp_user_reseller_link_subject ON proc.bp_user_reseller_link (subject);
