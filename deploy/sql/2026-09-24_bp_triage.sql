-- Discrepancy triage, procure-to-pay slice.
-- Spec: docs/superpowers/specs/2026-09-24-discrepancy-triage-p2p-design.md (§5.3, §8.2).
--
-- Findings that need action go to the EXISTING proc.bp_detection_finding (the Action
-- Centre reads it). These four tables hold everything else: one row per run, one row
-- per comparison (including matches -- "suppress from view, never from record"), and a
-- map from each finding's stable fingerprint to its bp_detection_finding row, kept here
-- so no column is added to a table the gateway maps field by field; and each deal's
-- content hash at its last triage, which the scheduled re-triage compares against.
--
-- The tolerance row: every value that decides whether a difference is a discrepancy.
-- Starting values are the triage spec's §14 example; tune them after the first backfill.
-- Additive and idempotent.
BEGIN;

CREATE TABLE IF NOT EXISTS proc.bp_triage_run (
    run_id              uuid PRIMARY KEY,
    mode                varchar NOT NULL CHECK (mode IN ('backfill', 'scheduled', 'single')),
    started_at          timestamptz NOT NULL DEFAULT now(),
    finished_at         timestamptz,
    rolled_back_at      timestamptz,
    config_fingerprint  varchar NOT NULL,
    config_values       jsonb NOT NULL,
    deal_count          integer,
    failed_deals        jsonb,
    report              jsonb
);

CREATE TABLE IF NOT EXISTS proc.bp_triage_result (
    result_id     bigserial PRIMARY KEY,
    run_id        uuid NOT NULL REFERENCES proc.bp_triage_run (run_id) ON DELETE CASCADE,
    deal_id       varchar NOT NULL,
    rule_id       varchar NOT NULL,
    claim_doc     varchar,
    claim_line    varchar,
    auth_doc      varchar,
    auth_line     varchar,
    field_name    varchar,
    claim_value   varchar,
    auth_value    varchar,
    outcome       varchar NOT NULL,
    severity      varchar NOT NULL,
    exposure_gbp  numeric,
    score         numeric,
    score_inputs  jsonb,
    tolerance     jsonb,
    fingerprint   varchar NOT NULL,
    finding_id    bigint
);
CREATE INDEX IF NOT EXISTS ix_bp_triage_result_run  ON proc.bp_triage_result (run_id);
CREATE INDEX IF NOT EXISTS ix_bp_triage_result_deal ON proc.bp_triage_result (deal_id);

CREATE TABLE IF NOT EXISTS proc.bp_triage_finding (
    fingerprint    varchar PRIMARY KEY,
    finding_id     bigint NOT NULL,
    deal_id        varchar NOT NULL,
    first_run_id   uuid NOT NULL,
    last_run_id    uuid NOT NULL,
    last_severity  varchar NOT NULL
);
ALTER TABLE proc.bp_triage_finding ADD COLUMN IF NOT EXISTS replaced_finding_id bigint;
ALTER TABLE proc.bp_triage_finding ADD COLUMN IF NOT EXISTS replaced_severity varchar;
-- The finding's mirror row in proc.bp_extraction_discrepancy (the table the SpendIQ
-- Action Centre reads), and the one a reopen replaced, so a rollback can hand it back.
ALTER TABLE proc.bp_triage_finding ADD COLUMN IF NOT EXISTS mirror_id bigint;
ALTER TABLE proc.bp_triage_finding ADD COLUMN IF NOT EXISTS replaced_mirror_id bigint;
CREATE INDEX IF NOT EXISTS ix_bp_triage_finding_deal      ON proc.bp_triage_finding (deal_id);
CREATE INDEX IF NOT EXISTS ix_bp_triage_finding_first_run ON proc.bp_triage_finding (first_run_id);

-- What each deal's documents hashed to (and which tolerances judged them) at its last
-- successful triage. The scheduled job re-triages a deal whose row is missing or
-- differs; a deal that failed keeps its old row (or none) and is retried.
CREATE TABLE IF NOT EXISTS proc.bp_triage_deal_state (
    deal_id             varchar PRIMARY KEY,
    content_hash        varchar NOT NULL,
    config_fingerprint  varchar NOT NULL,
    last_run_id         uuid NOT NULL,
    triaged_at          timestamptz NOT NULL DEFAULT now()
);

INSERT INTO proc.bp_policy (
    policy_name, policy_type, policy_desc, policy_details,
    policy_linked_agents, policy_status, version,
    created_date, created_by, last_modified_date, last_modified_by
)
SELECT 'TriageTolerancePolicy', 'limit',
       'What counts as a discrepancy between a deal''s quote, PO and invoices, and how '
       'serious it is. Widen these and findings stop reaching the Action Centre.',
       jsonb_build_object('policy_identifier', 'triage_tolerances', 'rules', jsonb_build_object(
           'unit_price_over_pct',        1.0,
           'unit_price_over_abs',        5.0,
           'unit_price_combine',         'min',
           'unit_price_under_pct',       5.0,
           'quantity_over_pct',          5.0,
           'rounding_per_line',          0.01,
           'cumulative_total_pct',       0.5,
           'cumulative_total_abs',       50.0,
           'cumulative_total_combine',   'min',
           'allowed_tax_rates',          jsonb_build_array(0, 5, 20),
           'min_link_confidence',        0.8,
           'unlinked_below',             0.5,
           'min_extraction_confidence',  0.7,
           'description_min_similarity', 0.4,
           'materiality_pct_of_total',   0.5,
           'materiality_floor',          25,
           'materiality_ceiling',        5000,
           'band_s1',                    70,
           'band_s2',                    40,
           'uplift_min_lines',           3,
           'uplift_same_pct_within',     0.1,
           'batch_size',                 200
       )),
       '', 1, 1, now(), 'triage', now(), 'triage'
 WHERE NOT EXISTS (
       SELECT 1 FROM proc.bp_policy
        WHERE policy_type = 'limit'
          AND policy_details->>'policy_identifier' = 'triage_tolerances');

COMMIT;
