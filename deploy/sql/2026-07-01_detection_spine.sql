-- Detection spine: proc.bp_detection_finding
--
-- The single unified output of the detection engine. Referenced by
-- beyond-procwaise-Api's DetectionFinding entity (src/modules/Detection), which
-- documents this exact path in its header comment -- but the file was never
-- committed, so GET /discrepancies and /discrepancies/metrics returned 500 and
-- the SpendIQ Action Centre queue had nothing to read.
--
-- Additive and idempotent: creates a new, empty table. No existing data is touched.
-- Column types mirror the TypeORM entity so the API can read and write it without
-- a schema sync (the datasources run with synchronize: false).

CREATE TABLE IF NOT EXISTS proc.bp_detection_finding (
    finding_id          bigserial PRIMARY KEY,

    engine_run_id       uuid,
    rule_id             varchar,
    category            varchar     NOT NULL,
    severity            varchar     NOT NULL,   -- critical | warning | info

    doc_type            varchar,
    doc_pk              varchar,
    deal_id             varchar,

    field_name          varchar,
    observed_value      varchar,
    expected_value      varchar,
    delta               varchar,

    blocks_promotion    boolean     NOT NULL DEFAULT false,
    confidence          numeric,
    notes               varchar,

    -- extraction-pipeline resolution semantics
    status              varchar     NOT NULL DEFAULT 'open',   -- open | resolved | ignored | superseded

    -- compliance lifecycle; the stage gate reads this one. DetectionService.resolveFinding()
    -- and PipelineService.patchIssue() must move `status` and `lifecycle_status` together.
    lifecycle_status    varchar     NOT NULL DEFAULT 'open',   -- open | remediating | resolved | accepted_risk

    pipeline_record_id  varchar,                                -- == deal_id
    stage_id            bigint,
    owner               varchar,
    due_date            date,
    regulation          varchar,

    resolution_action   varchar,
    resolved_value      varchar,
    resolved_by         varchar,
    resolved_at         timestamptz,

    detected_at         timestamptz NOT NULL DEFAULT now()
);

-- GET /discrepancies sorts by detected_at and filters on status/severity/category.
CREATE INDEX IF NOT EXISTS ix_bp_detection_finding_status_detected
    ON proc.bp_detection_finding (status, detected_at DESC);

CREATE INDEX IF NOT EXISTS ix_bp_detection_finding_severity
    ON proc.bp_detection_finding (severity);

CREATE INDEX IF NOT EXISTS ix_bp_detection_finding_category
    ON proc.bp_detection_finding (category);

-- Stage-gate lookups (blocking issues for a record's stage).
CREATE INDEX IF NOT EXISTS ix_bp_detection_finding_record_stage
    ON proc.bp_detection_finding (pipeline_record_id, stage_id)
    WHERE pipeline_record_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS ix_bp_detection_finding_lifecycle
    ON proc.bp_detection_finding (lifecycle_status);

CREATE INDEX IF NOT EXISTS ix_bp_detection_finding_deal
    ON proc.bp_detection_finding (deal_id)
    WHERE deal_id IS NOT NULL;
