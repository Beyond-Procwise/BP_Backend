-- 2026-09-27  Rule book: detection rules move out of proc.bp_policy.
--
-- A rule reads facts and asserts something is the case, producing a finding.
-- A policy reads a proposed action and answers allowed / needs approval /
-- forbidden. They are separate concerns and from here they are separate
-- tables. The five policy_type='opportunity' rows were detector configuration
-- wearing a policy badge; they are deactivated below and re-expressed as rules.
--
-- Additive and idempotent. Safe to re-run.

BEGIN;

CREATE TABLE IF NOT EXISTS proc.bp_rule (
    rule_id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    rule_name          TEXT        NOT NULL,
    detector_slug      TEXT        NOT NULL,
    finding_type       TEXT        NOT NULL DEFAULT 'opportunity',
    scope              TEXT,
    required_fields    JSONB       NOT NULL DEFAULT '[]',
    -- Thresholds. An ABSENT key means this detector has no default and takes
    -- its threshold from the caller at run time. That is deliberate and it is
    -- not the same as zero: on a minimum-value filter 0.0 means "fire on
    -- everything". Ten of the twelve detectors below genuinely have no default
    -- anywhere in the codebase, so their conditions are '{}'. Do not helpfully
    -- fill them in with zeroes -- put a real, considered number or leave it out.
    conditions         JSONB       NOT NULL DEFAULT '{}',
    severity           TEXT,
    rule_status        SMALLINT    NOT NULL DEFAULT 1,
    version            INTEGER     NOT NULL DEFAULT 1,
    created_date       TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by         TEXT        NOT NULL DEFAULT 'system',
    last_modified_date TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_modified_by   TEXT        NOT NULL DEFAULT 'system'
);

COMMENT ON TABLE proc.bp_rule IS
    'Detection rules. What the system looks for. Authorization lives in proc.bp_policy.';
COMMENT ON COLUMN proc.bp_rule.conditions IS
    'Default thresholds. An absent key means no default: the caller supplies it. Absent is not zero.';

-- One rule per detector. Without this the seed''s ON CONFLICT has nothing to
-- conflict on and every re-run duplicates all twelve rows.
CREATE UNIQUE INDEX IF NOT EXISTS ux_bp_rule_detector_slug
    ON proc.bp_rule (detector_slug);
CREATE INDEX IF NOT EXISTS ix_bp_rule_status
    ON proc.bp_rule (rule_status);

COMMIT;

BEGIN;

-- The twelve detectors that have a handler in
-- opportunity_miner_agent._detector_handlers(). rule_name is the display name
-- the UI already shows, so moving these changes no label.
INSERT INTO proc.bp_rule
    (rule_name, detector_slug, finding_type, scope, required_fields, conditions, severity)
VALUES
 ('Price Benchmark Variance','price_variance_check','opportunity','po_lines',
  '["supplier_id","item_id","actual_price","benchmark_price"]','{}','medium'),
 ('Volume Consolidation','volume_consolidation_check','opportunity','po_lines',
  '["minimum_volume_gbp"]','{}','low'),
 ('Contract Expiry Opportunity','contract_expiry_check','opportunity','contracts',
  '["negotiation_window_days"]','{"negotiation_window_days":90}','medium'),
 ('Supplier Risk Alert','supplier_risk_check','non_conformance','supplier_master',
  '["risk_threshold"]','{}','high'),
 ('Maverick Spend Detection','maverick_spend_check','non_conformance','purchase_orders',
  '["minimum_value_gbp"]','{}','high'),
 ('Duplicate Supplier','duplicate_supplier_check','opportunity','po_lines',
  '["minimum_overlap_gbp"]','{}','low'),
 ('Category Overspend','category_overspend_check','non_conformance','invoice_lines',
  '["category_budgets"]','{}','medium'),
 ('Inflation Pass-Through','inflation_passthrough_check','anomaly','invoice_lines',
  '["market_inflation_pct"]','{}','medium'),
 ('Unused Contract Value','unused_contract_value_check','opportunity','contracts',
  '["minimum_unused_value_gbp"]','{}','low'),
 ('Supplier Performance Deviation','supplier_performance_check','anomaly','invoices',
  '["performance_records"]','{}','medium'),
 ('ESG Opportunity','esg_opportunity_check','opportunity',NULL,
  '["esg_scores"]','{}','low'),
 ('Invoice Overbilling','invoice_po_variance_check','anomaly','invoice_lines',
  '["variance_threshold_pct"]','{"variance_threshold_pct":10.0}','high')
ON CONFLICT (detector_slug) DO NOTHING;

-- Retire the five detector-configuration rows from the policy table.
-- Deactivated rather than deleted: PolicyEngine filters on
-- COALESCE(policy_status,1)=1, so status 0 removes them from every read path
-- while leaving the rows legible for audit and the rollback a one-line flip.
UPDATE proc.bp_policy
   SET policy_status      = 0,
       last_modified_date = now(),
       last_modified_by   = 'bp_rule_migration_2026_09_27'
 WHERE policy_type = 'opportunity'
   AND policy_status = 1;

COMMIT;
