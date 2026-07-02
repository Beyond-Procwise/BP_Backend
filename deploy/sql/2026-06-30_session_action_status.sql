-- 2026-06-30  Session action-status tracking
-- ---------------------------------------------------------------------------
-- Adds session_id + action_status to process_monitor, introduces the
-- session_document_outcome table (one row per document per session), and
-- wires five PostgreSQL triggers so that every document's final outcome is
-- automatically recorded and — once every document in a session has an
-- outcome — pg_notify fires on the 'session_status' channel.
--
-- Outcome taxonomy (session_document_outcome.outcome):
--   'target'       document reached a _trgt table (success)
--   'discrepancy'  document has a blocking discrepancy; never reached _trgt
--   'failed'       extraction failed; no row produced anywhere
--
-- action_status values (process_monitor.action_status):
--   'completed'           all docs → target
--   'partially_completed' mix of outcomes (at least one target)
--   'failed'              zero docs reached target
--   NULL                  session in-progress / session_id not set
--
-- Triggers installed:
--   1. proc.bp_invoice_trgt          AFTER INSERT → fn_record_outcome(target)
--   2. proc.bp_quote_trgt            AFTER INSERT → fn_record_outcome(target)
--   3. proc.bp_purchase_order_trgt   AFTER INSERT → fn_record_outcome(target)
--   4. proc.bp_extraction_discrepancy AFTER INSERT (blocks_promotion=TRUE only)
--                                                 → fn_record_outcome(discrepancy)
--   5. proc.process_monitor          AFTER UPDATE (Extraction_Failed transition)
--                                                 → fn_record_outcome(failed)
--
-- Idempotent: uses IF NOT EXISTS, CREATE OR REPLACE, DROP … IF EXISTS.
-- Run against: bp_sqldb.
-- ---------------------------------------------------------------------------

BEGIN;

-- ============================================================
-- 1. Extend proc.process_monitor
-- ============================================================
ALTER TABLE proc.process_monitor
    ADD COLUMN IF NOT EXISTS session_id    TEXT,
    ADD COLUMN IF NOT EXISTS action_status TEXT
        CHECK (action_status IN ('completed', 'partially_completed', 'failed'));

-- Fast lookup: "all docs for session X"
CREATE INDEX IF NOT EXISTS idx_pm_session_id
    ON proc.process_monitor (session_id)
    WHERE session_id IS NOT NULL;

-- ============================================================
-- 2. proc.session_document_outcome — one row per document per session
-- ============================================================
CREATE TABLE IF NOT EXISTS proc.session_document_outcome (
    id            BIGSERIAL    PRIMARY KEY,
    session_id    TEXT         NOT NULL,
    file_path     TEXT         NOT NULL,
    document_type TEXT,
    outcome       TEXT         NOT NULL
        CHECK (outcome IN ('target', 'discrepancy', 'failed')),
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    UNIQUE (session_id, file_path)
);

CREATE INDEX IF NOT EXISTS idx_sdo_session_id
    ON proc.session_document_outcome (session_id);

COMMENT ON TABLE proc.session_document_outcome IS
    'Records the final processing outcome for each document in a session. '
    'Exactly one row per (session_id, file_path) guaranteed by UNIQUE constraint. '
    'Written exclusively by triggers; never modified by application code.';

-- ============================================================
-- 3. proc.fn_try_resolve_session
--    Called after every outcome INSERT.  Checks whether every document in
--    the session now has an outcome; if so, computes action_status, updates
--    process_monitor atomically, and fires pg_notify.
-- ============================================================
CREATE OR REPLACE FUNCTION proc.fn_try_resolve_session(p_session_id TEXT)
RETURNS VOID
LANGUAGE plpgsql
AS $$
DECLARE
    v_total      INT;
    v_resolved   INT;
    v_target     INT;
    v_discrep    INT;
    v_failed     INT;
    v_status     TEXT;
    v_rows_upd   INT;
BEGIN
    -- Guard: unknown or empty session
    IF p_session_id IS NULL OR p_session_id = '' THEN
        RETURN;
    END IF;

    -- Total documents uploaded for this session
    SELECT COUNT(*)
    INTO   v_total
    FROM   proc.process_monitor
    WHERE  session_id = p_session_id;

    IF v_total = 0 THEN RETURN; END IF;

    -- Documents that have reached a terminal outcome
    SELECT COUNT(*)
    INTO   v_resolved
    FROM   proc.session_document_outcome
    WHERE  session_id = p_session_id;

    -- Session still in-progress — at least one document not yet resolved
    IF v_resolved < v_total THEN RETURN; END IF;

    -- All documents resolved — compute outcome breakdown
    SELECT
        COUNT(*) FILTER (WHERE outcome = 'target'),
        COUNT(*) FILTER (WHERE outcome = 'discrepancy'),
        COUNT(*) FILTER (WHERE outcome = 'failed')
    INTO v_target, v_discrep, v_failed
    FROM proc.session_document_outcome
    WHERE session_id = p_session_id;

    -- Determine action_status
    IF v_target = v_total THEN
        v_status := 'completed';
    ELSIF v_target > 0 THEN
        v_status := 'partially_completed';
    ELSE
        v_status := 'failed';
    END IF;

    -- Atomic write — WHERE action_status IS NULL means only the first
    -- concurrent trigger that reaches this point wins; every other
    -- concurrent caller sees ROW_COUNT = 0 and exits cleanly.
    UPDATE proc.process_monitor
    SET    action_status = v_status
    WHERE  session_id    = p_session_id
      AND  action_status IS NULL;

    GET DIAGNOSTICS v_rows_upd = ROW_COUNT;

    -- Another trigger already resolved this session concurrently
    IF v_rows_upd = 0 THEN RETURN; END IF;

    -- Emit notification for the FastAPI SessionNotifyListener thread
    PERFORM pg_notify(
        'session_status',
        json_build_object(
            'session_id',    p_session_id,
            'action_status', v_status,
            'total',         v_total,
            'target',        v_target,
            'discrepancy',   v_discrep,
            'failed',        v_failed,
            'resolved_at',   to_char(NOW() AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"')
        )::text
    );
END;
$$;

-- ============================================================
-- 4. proc.fn_record_outcome
--    Shared helper used by all five triggers.
--    Looks up session_id from process_monitor via file_path,
--    inserts one outcome row (UNIQUE guard prevents double-recording),
--    then calls fn_try_resolve_session.
-- ============================================================
CREATE OR REPLACE FUNCTION proc.fn_record_outcome(
    p_file_path     TEXT,
    p_document_type TEXT,
    p_outcome       TEXT
)
RETURNS VOID
LANGUAGE plpgsql
AS $$
DECLARE
    v_session_id TEXT;
BEGIN
    IF p_file_path IS NULL THEN RETURN; END IF;

    -- Resolve session_id through process_monitor.file_path
    SELECT session_id
    INTO   v_session_id
    FROM   proc.process_monitor
    WHERE  file_path   = p_file_path
      AND  session_id IS NOT NULL
    LIMIT  1;

    -- Document does not belong to any session — skip silently
    IF v_session_id IS NULL THEN RETURN; END IF;

    -- Record outcome.  ON CONFLICT DO NOTHING: UNIQUE(session_id, file_path)
    -- ensures exactly one outcome row per document.  Because we only fire the
    -- discrepancy trigger when blocks_promotion = TRUE, a document that has
    -- non-blocking discrepancies AND reaches _trgt will only ever see the
    -- 'target' outcome recorded (discrepancy trigger stays silent for it).
    INSERT INTO proc.session_document_outcome
        (session_id, file_path, document_type, outcome)
    VALUES
        (v_session_id, p_file_path, p_document_type, p_outcome)
    ON CONFLICT (session_id, file_path) DO NOTHING;

    -- Attempt session resolution (exits early if not all docs are resolved)
    PERFORM proc.fn_try_resolve_session(v_session_id);
END;
$$;

-- ============================================================
-- 5a. Trigger: bp_invoice_trgt AFTER INSERT → outcome = 'target'
--     Resolves file_path via bp_invoice_raw.doc_pk_candidate
-- ============================================================
CREATE OR REPLACE FUNCTION proc.trg_fn_invoice_trgt_outcome()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    v_file_path TEXT;
BEGIN
    SELECT source_file
    INTO   v_file_path
    FROM   proc.bp_invoice_raw
    WHERE  doc_pk_candidate = NEW.invoice_id
    ORDER  BY extracted_at DESC
    LIMIT  1;

    IF v_file_path IS NOT NULL THEN
        PERFORM proc.fn_record_outcome(v_file_path, 'invoice', 'target');
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_invoice_trgt_outcome ON proc.bp_invoice_trgt;
CREATE TRIGGER trg_invoice_trgt_outcome
    AFTER INSERT ON proc.bp_invoice_trgt
    FOR EACH ROW
    EXECUTE FUNCTION proc.trg_fn_invoice_trgt_outcome();

-- ============================================================
-- 5b. Trigger: bp_quote_trgt AFTER INSERT → outcome = 'target'
-- ============================================================
CREATE OR REPLACE FUNCTION proc.trg_fn_quote_trgt_outcome()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    v_file_path TEXT;
BEGIN
    SELECT source_file
    INTO   v_file_path
    FROM   proc.bp_quote_raw
    WHERE  doc_pk_candidate = NEW.quote_id
    ORDER  BY extracted_at DESC
    LIMIT  1;

    IF v_file_path IS NOT NULL THEN
        PERFORM proc.fn_record_outcome(v_file_path, 'quote', 'target');
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_quote_trgt_outcome ON proc.bp_quote_trgt;
CREATE TRIGGER trg_quote_trgt_outcome
    AFTER INSERT ON proc.bp_quote_trgt
    FOR EACH ROW
    EXECUTE FUNCTION proc.trg_fn_quote_trgt_outcome();

-- ============================================================
-- 5c. Trigger: bp_purchase_order_trgt AFTER INSERT → outcome = 'target'
-- ============================================================
CREATE OR REPLACE FUNCTION proc.trg_fn_po_trgt_outcome()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    v_file_path TEXT;
BEGIN
    SELECT source_file
    INTO   v_file_path
    FROM   proc.bp_purchase_order_raw
    WHERE  doc_pk_candidate = NEW.po_id
    ORDER  BY extracted_at DESC
    LIMIT  1;

    IF v_file_path IS NOT NULL THEN
        PERFORM proc.fn_record_outcome(v_file_path, 'purchase_order', 'target');
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_po_trgt_outcome ON proc.bp_purchase_order_trgt;
CREATE TRIGGER trg_po_trgt_outcome
    AFTER INSERT ON proc.bp_purchase_order_trgt
    FOR EACH ROW
    EXECUTE FUNCTION proc.trg_fn_po_trgt_outcome();

-- ============================================================
-- 5d. Trigger: bp_extraction_discrepancy AFTER INSERT
--     Only fires when blocks_promotion = TRUE (blocking discrepancy).
--     Non-blocking discrepancies do not interrupt the _trgt path, so
--     the _trgt trigger records 'target' for those documents instead.
-- ============================================================
CREATE OR REPLACE FUNCTION proc.trg_fn_discrepancy_outcome()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    -- Ignore non-blocking discrepancies: those documents still proceed to
    -- _trgt, where the _trgt INSERT trigger records outcome = 'target'.
    IF NOT NEW.blocks_promotion THEN
        RETURN NEW;
    END IF;

    PERFORM proc.fn_record_outcome(NEW.source_file, NEW.doc_type, 'discrepancy');
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_discrepancy_outcome ON proc.bp_extraction_discrepancy;
CREATE TRIGGER trg_discrepancy_outcome
    AFTER INSERT ON proc.bp_extraction_discrepancy
    FOR EACH ROW
    EXECUTE FUNCTION proc.trg_fn_discrepancy_outcome();

-- ============================================================
-- 5e. Trigger: process_monitor AFTER UPDATE → outcome = 'failed'
--     Fires only on the transition INTO Extraction_Failed status.
--     Handles documents that never produced a _raw row (complete
--     extraction crash before any data was written).
-- ============================================================
CREATE OR REPLACE FUNCTION proc.trg_fn_pm_failed_outcome()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    -- Only on status transition into Extraction_Failed
    IF NEW.status = 'Extraction_Failed'
       AND (OLD.status IS DISTINCT FROM 'Extraction_Failed')
       AND NEW.session_id IS NOT NULL
       AND NEW.file_path  IS NOT NULL
    THEN
        -- Insert directly (we already have session_id and file_path on the row)
        INSERT INTO proc.session_document_outcome
            (session_id, file_path, document_type, outcome)
        VALUES
            (NEW.session_id, NEW.file_path, NEW.document_type, 'failed')
        ON CONFLICT (session_id, file_path) DO NOTHING;

        PERFORM proc.fn_try_resolve_session(NEW.session_id);
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_pm_failed_outcome ON proc.process_monitor;
CREATE TRIGGER trg_pm_failed_outcome
    AFTER UPDATE ON proc.process_monitor
    FOR EACH ROW
    EXECUTE FUNCTION proc.trg_fn_pm_failed_outcome();

-- ============================================================
-- 6. Indexes for efficient trigger joins
-- ============================================================
CREATE INDEX IF NOT EXISTS idx_invoice_raw_doc_pk_cand
    ON proc.bp_invoice_raw (doc_pk_candidate)
    WHERE doc_pk_candidate IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_quote_raw_doc_pk_cand
    ON proc.bp_quote_raw (doc_pk_candidate)
    WHERE doc_pk_candidate IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_po_raw_doc_pk_cand
    ON proc.bp_purchase_order_raw (doc_pk_candidate)
    WHERE doc_pk_candidate IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_pm_file_path
    ON proc.process_monitor (file_path)
    WHERE file_path IS NOT NULL;

COMMIT;
