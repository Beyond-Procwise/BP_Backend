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

-- A decision made on either side reaches the other straight away: on the finding (the
-- gateway's resolve and pipeline PATCH) it moves the Action Centre mirror row; on the
-- mirror (SpendIQ resolve / ignore / flag) it moves the finding. A superseded row is
-- never touched.
--
-- Echo guard: while one trigger writes the other table it sets the transaction-local
-- setting bp.triage_sync to 'on', and the other trigger returns at once when it sees it,
-- so a sync never bounces back (and cannot reopen the side that started it). It is set
-- back to 'off' straight after, so a later decision in the same transaction still syncs.
-- If the UPDATE raises, the whole transaction aborts and set_config(..., true) is undone
-- with it, so a stale 'on' can never survive into a commit.
--
-- A finding can move straight from one closed state to the other (the gateway's
-- POST /discrepancies/resolve writes 'resolved' or 'ignored' whatever the current state),
-- but the mirror's lifecycle guard only allows closed -> open -> closed, so the mirror is
-- taken there in those two legal steps.
CREATE OR REPLACE FUNCTION proc.bp_triage_finding_decision_to_mirror() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE
    r record;
BEGIN
    IF current_setting('bp.triage_sync', true) = 'on' THEN
        RETURN NULL;
    END IF;
    FOR r IN
        SELECT d.discrepancy_id, d.status
          FROM proc.bp_triage_finding m
          JOIN proc.bp_extraction_discrepancy d ON d.discrepancy_id = m.mirror_id
         WHERE m.finding_id = NEW.finding_id
           AND d.status IN ('open', 'resolved', 'ignored')    -- never a superseded mirror
           AND d.status <> NEW.status
    LOOP
        PERFORM set_config('bp.triage_sync', 'on', true);
        IF NEW.status IN ('resolved', 'ignored') AND r.status IN ('resolved', 'ignored') THEN
            -- closed -> closed: reopen first (resolved_by kept), then close the other way.
            UPDATE proc.bp_extraction_discrepancy
               SET status = 'open', resolved_at = NULL
             WHERE discrepancy_id = r.discrepancy_id;
        END IF;
        UPDATE proc.bp_extraction_discrepancy d
           SET status = NEW.status,
               resolved_by = CASE WHEN NEW.status = 'open' THEN d.resolved_by
                                  ELSE coalesce(NEW.resolved_by, 'detection-finding') END,
               resolved_at = CASE WHEN NEW.status = 'open' THEN NULL
                                  ELSE coalesce(NEW.resolved_at, now()) END
         WHERE d.discrepancy_id = r.discrepancy_id;
        PERFORM set_config('bp.triage_sync', 'off', true);
    END LOOP;
    RETURN NULL;
END;
$$;
DROP TRIGGER IF EXISTS tr_bp_triage_finding_decision_to_mirror ON proc.bp_detection_finding;
CREATE TRIGGER tr_bp_triage_finding_decision_to_mirror
    AFTER UPDATE OF status ON proc.bp_detection_finding
    FOR EACH ROW
    WHEN (OLD.status IS DISTINCT FROM NEW.status
          AND NEW.status IN ('resolved', 'ignored', 'open'))
    EXECUTE FUNCTION proc.bp_triage_finding_decision_to_mirror();

CREATE OR REPLACE FUNCTION proc.bp_triage_mirror_decision_to_finding() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF current_setting('bp.triage_sync', true) = 'on' THEN
        RETURN NULL;
    END IF;
    PERFORM set_config('bp.triage_sync', 'on', true);
    UPDATE proc.bp_detection_finding f
       SET status = NEW.status,
           lifecycle_status = CASE NEW.status WHEN 'resolved' THEN 'resolved'
                                              WHEN 'ignored' THEN 'accepted_risk'
                                              ELSE 'open' END,
           resolved_by = CASE WHEN NEW.status = 'open' THEN f.resolved_by
                              ELSE coalesce(NEW.resolved_by, 'action-centre') END,
           resolved_at = CASE WHEN NEW.status = 'open' THEN NULL
                              ELSE coalesce(NEW.resolved_at, now()) END
      FROM proc.bp_triage_finding m
     WHERE m.mirror_id = NEW.discrepancy_id AND f.finding_id = m.finding_id
       AND ((NEW.status IN ('resolved', 'ignored') AND f.status = 'open')
            OR (NEW.status = 'open' AND f.status IN ('resolved', 'ignored')));
    PERFORM set_config('bp.triage_sync', 'off', true);
    RETURN NULL;
END;
$$;
DROP TRIGGER IF EXISTS tr_bp_triage_mirror_decision_to_finding ON proc.bp_extraction_discrepancy;
CREATE TRIGGER tr_bp_triage_mirror_decision_to_finding
    AFTER UPDATE OF status ON proc.bp_extraction_discrepancy
    FOR EACH ROW
    WHEN (OLD.status IS DISTINCT FROM NEW.status AND NEW.source_file LIKE 'triage:%'
          AND NEW.status IN ('resolved', 'ignored', 'open'))
    EXECUTE FUNCTION proc.bp_triage_mirror_decision_to_finding();

-- One-time catch-up for decisions made before the triggers existed, or before they
-- handled a closed -> closed switch (idempotent: a second apply finds every mapped pair
-- already in step). The echo guard is on throughout so none of these writes bounces.
DO $$
DECLARE
    mirrors_switched integer;
    mirrors_closed   integer;
    findings_closed  integer;
BEGIN
    PERFORM set_config('bp.triage_sync', 'on', true);
    -- Finding closed one way, mirror closed the other: reopen the mirror, then close it
    -- to match (the two moves its lifecycle guard allows).
    CREATE TEMP TABLE bp_triage_switch ON COMMIT DROP AS
    SELECT d.discrepancy_id, f.status, f.resolved_by, f.resolved_at
      FROM proc.bp_triage_finding m
      JOIN proc.bp_detection_finding f ON f.finding_id = m.finding_id
      JOIN proc.bp_extraction_discrepancy d ON d.discrepancy_id = m.mirror_id
     WHERE f.status IN ('resolved', 'ignored') AND d.status IN ('resolved', 'ignored')
       AND d.status <> f.status;
    UPDATE proc.bp_extraction_discrepancy d
       SET status = 'open', resolved_at = NULL
      FROM bp_triage_switch s
     WHERE d.discrepancy_id = s.discrepancy_id;
    UPDATE proc.bp_extraction_discrepancy d
       SET status = s.status,
           resolved_by = coalesce(s.resolved_by, 'detection-finding'),
           resolved_at = coalesce(s.resolved_at, now())
      FROM bp_triage_switch s
     WHERE d.discrepancy_id = s.discrepancy_id;
    GET DIAGNOSTICS mirrors_switched = ROW_COUNT;
    DROP TABLE bp_triage_switch;
    UPDATE proc.bp_extraction_discrepancy d
       SET status = f.status,
           resolved_by = coalesce(f.resolved_by, 'detection-finding'),
           resolved_at = coalesce(f.resolved_at, now())
      FROM proc.bp_triage_finding m
      JOIN proc.bp_detection_finding f ON f.finding_id = m.finding_id
     WHERE d.discrepancy_id = m.mirror_id
       AND d.status = 'open' AND f.status IN ('resolved', 'ignored');
    GET DIAGNOSTICS mirrors_closed = ROW_COUNT;
    UPDATE proc.bp_detection_finding f
       SET status = d.status,
           lifecycle_status = CASE d.status WHEN 'resolved' THEN 'resolved'
                                            ELSE 'accepted_risk' END,
           resolved_by = coalesce(d.resolved_by, 'action-centre'),
           resolved_at = coalesce(d.resolved_at, now())
      FROM proc.bp_triage_finding m
      JOIN proc.bp_extraction_discrepancy d ON d.discrepancy_id = m.mirror_id
     WHERE f.finding_id = m.finding_id
       AND f.status = 'open' AND d.status IN ('resolved', 'ignored');
    GET DIAGNOSTICS findings_closed = ROW_COUNT;
    PERFORM set_config('bp.triage_sync', 'off', true);
    RAISE NOTICE 'triage decision sync: % mirror row(s) switched, % mirror row(s) closed, % finding(s) closed',
        mirrors_switched, mirrors_closed, findings_closed;
END;
$$;

-- The two decision-sync triggers look up the map by finding_id and by mirror_id on every
-- decision; without these each lookup scans the whole map.
CREATE INDEX IF NOT EXISTS ix_bp_triage_finding_finding ON proc.bp_triage_finding (finding_id);
CREATE INDEX IF NOT EXISTS ix_bp_triage_finding_mirror  ON proc.bp_triage_finding (mirror_id);

COMMIT;
