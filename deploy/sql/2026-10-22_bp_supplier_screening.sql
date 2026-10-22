-- The compliance screening register: what a person or a provider found when they checked a supplier.
-- FOR REVIEW. NOT APPLIED to any database. Idempotent; one transaction. Rollback: the _rollback.sql beside it.
--
-- Why a register and not a lookup: the platform has no sanctions, credit, ESG, forced-labour or
-- conflict-minerals provider, and the agent library's skills (award, selection, negotiation, renewal)
-- need "the result and its date from compliance" before they may shortlist or award. Compliance records
-- each check here (who, which provider, what result, when, valid until); the agent tools
-- screen_sanctions / financial_health_score / esg_rating_lookup / forced_labour_screen /
-- conflict_minerals_check / monitor_risk_events READ the latest record and perform no check themselves.
-- No record is reported as "not performed", never as clear. A provider integration, when one exists,
-- writes into this same table.
--
-- Append-only: a later check supersedes an earlier one by date; nothing is edited or deleted, so what
-- an award rested on at the time can always be shown. tenant_id follows the tenancy convention
-- (deploy/sql/tenancy): 'default' until the stamping trigger (part 6) replaces the default.
BEGIN;

CREATE TABLE IF NOT EXISTS proc.bp_supplier_screening (
    screening_id   BIGSERIAL PRIMARY KEY,
    supplier_id    TEXT NOT NULL,
    check_type     TEXT NOT NULL
                   CONSTRAINT bp_supplier_screening_type_check
                   CHECK (check_type IN ('sanctions','financial_health','esg_rating','forced_labour',
                                         'conflict_minerals','risk_event')),
    subject        TEXT NOT NULL DEFAULT 'supplier',   -- the legal entity, owner or director screened
    result         TEXT NOT NULL
                   CONSTRAINT bp_supplier_screening_result_check
                   CHECK (result IN ('clear','flagged','rated','not_performed')),
    rating         TEXT,                               -- a provider's grade, e.g. 'BBB', 'Silver', 'high'
    score          NUMERIC,                            -- a provider's number, e.g. probability of default
    details        JSONB NOT NULL DEFAULT '{}'::jsonb, -- anything else the provider stated, as stated
    provider       TEXT NOT NULL,                      -- who performed the check: a provider or a person
    reference      TEXT,                               -- the provider's report or case reference
    checked_on     DATE NOT NULL,
    valid_until    DATE,
    recorded_by    TEXT NOT NULL,
    created_date   TIMESTAMPTZ NOT NULL DEFAULT now(),
    tenant_id      TEXT NOT NULL DEFAULT 'default',
    CONSTRAINT bp_supplier_screening_validity_check CHECK (valid_until IS NULL OR valid_until >= checked_on)
);

CREATE INDEX IF NOT EXISTS ix_bp_supplier_screening_latest
    ON proc.bp_supplier_screening (supplier_id, check_type, checked_on DESC);

CREATE OR REPLACE FUNCTION proc.bp_supplier_screening_append_only() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'proc.bp_supplier_screening is append-only: record a newer check instead of changing %',
        COALESCE(OLD.screening_id::text, '?');
END $$;

DROP TRIGGER IF EXISTS tr_bp_supplier_screening_append_only ON proc.bp_supplier_screening;
CREATE TRIGGER tr_bp_supplier_screening_append_only
    BEFORE UPDATE OR DELETE ON proc.bp_supplier_screening
    FOR EACH ROW EXECUTE FUNCTION proc.bp_supplier_screening_append_only();

COMMIT;
