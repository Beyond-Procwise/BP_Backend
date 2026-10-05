-- 2026-10-05  Contract expiry buckets: a detection rule plus the alert it records.
--
-- The rule lives in proc.bp_rule like every other detection rule, so the bucket
-- edges (3/6/9/12/18 months) are data an administrator can change, not numbers
-- in code. The alert table holds ONE row per contract per bucket per end date,
-- which is what makes "alert once per bucket, not repeatedly" a property of the
-- table (a unique index) rather than a promise in the code.
--
-- Additive and idempotent. Safe to re-run.

BEGIN;

CREATE TABLE IF NOT EXISTS proc.bp_contract_expiry_alert (
    alert_id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    contract_id        TEXT        NOT NULL,
    -- 'EXPIRED' | '0-3' | '3-6' | '6-9' | '9-12' | '12-18' | 'NO_END_DATE'
    -- (the labels follow the rule's bucket_months edges)
    bucket             TEXT        NOT NULL,
    -- NULL only for NO_END_DATE. Part of the key so an amendment that moves the
    -- end date opens a fresh alert instead of silently reusing the old one.
    end_date           DATE,
    -- open       = a person should look at it
    -- suppressed = an ACTIVE demand item covers the contract (suppressed_by names it)
    -- cleared    = no longer true: renewed, amended into another bucket, or inactive
    status             TEXT        NOT NULL DEFAULT 'open'
                       CHECK (status IN ('open', 'suppressed', 'cleared')),
    suppressed_by      TEXT,
    days_to_end        INTEGER,
    rule_id            BIGINT,
    rule_version       INTEGER,
    first_fired_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_evaluated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    cleared_at         TIMESTAMPTZ
);

COMMENT ON TABLE proc.bp_contract_expiry_alert IS
    'Alerts raised by the contract_expiry_bucket_check rule. One row per contract, bucket and end date.';

-- COALESCE so the NULL end date of a NO_END_DATE alert still collides with
-- itself; a plain unique index treats every NULL as distinct.
CREATE UNIQUE INDEX IF NOT EXISTS ux_bp_contract_expiry_alert_key
    ON proc.bp_contract_expiry_alert (contract_id, bucket, COALESCE(end_date, DATE '0001-01-01'));
CREATE INDEX IF NOT EXISTS ix_bp_contract_expiry_alert_status
    ON proc.bp_contract_expiry_alert (status, bucket);

COMMIT;

BEGIN;

INSERT INTO proc.bp_rule
    (rule_name, detector_slug, finding_type, scope, required_fields, conditions, severity)
VALUES
 ('Contract Expiry Buckets','contract_expiry_bucket_check','non_conformance','contracts',
  '["bucket_months"]',
  '{"bucket_months":[3,6,9,12,18],
    "alert_expired":true,
    "flag_missing_end_date":true,
    "lifecycle_status":"active",
    "demand_contract_key":"contract_id",
    "inactive_demand_statuses":["draft","closed","cancelled","rejected","completed","won","lost"]}',
  'high')
ON CONFLICT (detector_slug) DO NOTHING;

COMMIT;
