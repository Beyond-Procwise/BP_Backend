BEGIN;
-- deploy/sql/2026-09-26_bp_value_outcome.sql
--
-- The value ledger: money a finding or opportunity actually produced, recorded by a
-- person, never overwritten. Spec: docs/superpowers/specs/2026-09-25-value-ledger-design.md
--
-- Replaces the overwrite-in-place columns bp_extraction_discrepancy.resolution_outcome /
-- recovered_amount and bp_opportunity.realised_savings_gbp as the place recovered value
-- is read from. Those columns are left in place and no longer read.
--
-- Append-only by TRIGGER, not REVOKE: the application role owns this table and an owner
-- may re-grant itself anything (see 2026-09-16_bp_agent_actions_immutable.sql). A mistake
-- is corrected by a new row whose supersedes_id names the row it replaces.

CREATE TABLE IF NOT EXISTS proc.bp_value_outcome (
    outcome_id     bigserial PRIMARY KEY,
    -- B2 decision: tenant_id on every new table, defaulted to one constant.
    tenant_id      text NOT NULL DEFAULT 'default',
    source_type    text NOT NULL CHECK (source_type IN ('finding', 'opportunity')),
    -- discrepancy_id (bigint) or opportunity_id (varchar), both stored as text.
    source_id      text NOT NULL,
    outcome_type   text NOT NULL CHECK (outcome_type IN (
                       'avoided', 'claimed', 'recovered', 'claim_dropped',
                       'realised_saving', 'terms_improved', 'cycle_time')),
    amount         numeric(18,2),
    currency       char(3),
    amount_gbp     numeric(18,2),
    fx_rate        numeric,
    fx_as_of       timestamptz,
    evidence_ref   text,
    note           text,
    supersedes_id  bigint REFERENCES proc.bp_value_outcome (outcome_id),
    recorded_by    text NOT NULL,
    valid_from     date NOT NULL DEFAULT current_date,
    recorded_at    timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT bp_value_outcome_amount_ck CHECK (
        CASE WHEN outcome_type = 'claim_dropped' THEN amount IS NULL
             ELSE amount IS NOT NULL AND amount > 0 END),
    CONSTRAINT bp_value_outcome_currency_ck CHECK (
        outcome_type IN ('claim_dropped', 'cycle_time') OR currency IS NOT NULL),
    CONSTRAINT bp_value_outcome_evidence_ck CHECK (
        outcome_type <> 'recovered' OR coalesce(btrim(evidence_ref), '') <> '')
);

CREATE INDEX IF NOT EXISTS ix_bp_value_outcome_source
    ON proc.bp_value_outcome (source_type, source_id);
CREATE INDEX IF NOT EXISTS ix_bp_value_outcome_type_valid
    ON proc.bp_value_outcome (outcome_type, valid_from);
CREATE INDEX IF NOT EXISTS ix_bp_value_outcome_supersedes
    ON proc.bp_value_outcome (supersedes_id);

CREATE OR REPLACE FUNCTION proc.bp_value_outcome_immutable() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'proc.bp_value_outcome is append-only: % refused. Record a correcting row (supersedes_id) instead.', TG_OP;
END;
$$;

DROP TRIGGER IF EXISTS tr_bp_value_outcome_immutable ON proc.bp_value_outcome;
CREATE TRIGGER tr_bp_value_outcome_immutable
    BEFORE UPDATE OR DELETE ON proc.bp_value_outcome
    FOR EACH ROW EXECUTE FUNCTION proc.bp_value_outcome_immutable();

DROP TRIGGER IF EXISTS tr_bp_value_outcome_no_truncate ON proc.bp_value_outcome;
CREATE TRIGGER tr_bp_value_outcome_no_truncate
    BEFORE TRUNCATE ON proc.bp_value_outcome
    FOR EACH STATEMENT EXECUTE FUNCTION proc.bp_value_outcome_immutable();

COMMIT;
