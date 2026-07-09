-- 2026-07-09  Per-document quality actions + content-hash dedup
-- Additive & idempotent. Run against bp_sqldb.
BEGIN;

ALTER TABLE proc.process_monitor
    ADD COLUMN IF NOT EXISTS content_hash TEXT,
    ADD COLUMN IF NOT EXISTS doc_action   TEXT;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'process_monitor_doc_action_check'
    ) THEN
        ALTER TABLE proc.process_monitor
            ADD CONSTRAINT process_monitor_doc_action_check
            CHECK (doc_action IS NULL OR doc_action IN
                   ('duplicate', 'updated', 'needs_review', 'unsupported'));
    END IF;
END;
$$;

CREATE INDEX IF NOT EXISTS ix_bp_process_monitor_content_hash
    ON proc.process_monitor (content_hash)
    WHERE content_hash IS NOT NULL;

CREATE INDEX IF NOT EXISTS ix_bp_process_monitor_doc_action
    ON proc.process_monitor (session_id, doc_action)
    WHERE doc_action IS NOT NULL;

-- Enriched session resolver: adds per-document doc_action breakdown to the
-- pg_notify payload. Session-rollup logic (action_status) unchanged.
CREATE OR REPLACE FUNCTION proc.fn_try_resolve_session(p_session_id TEXT)
RETURNS VOID
LANGUAGE plpgsql
AS $$
DECLARE
    v_total    INT;
    v_resolved INT;
    v_target   INT;
    v_discrep  INT;
    v_failed   INT;
    v_status   TEXT;
    v_rows_upd INT;
    v_dup      INT;
    v_upd      INT;
    v_review   INT;
    v_unsup    INT;
    v_documents JSON;
BEGIN
    IF p_session_id IS NULL OR p_session_id = '' THEN RETURN; END IF;

    SELECT COUNT(*) INTO v_total
    FROM proc.process_monitor WHERE session_id = p_session_id;
    IF v_total = 0 THEN RETURN; END IF;

    SELECT COUNT(*) INTO v_resolved
    FROM proc.session_document_outcome WHERE session_id = p_session_id;
    IF v_resolved < v_total THEN RETURN; END IF;

    SELECT
        COUNT(*) FILTER (WHERE outcome = 'target'),
        COUNT(*) FILTER (WHERE outcome = 'discrepancy'),
        COUNT(*) FILTER (WHERE outcome = 'failed')
    INTO v_target, v_discrep, v_failed
    FROM proc.session_document_outcome
    WHERE session_id = p_session_id;

    IF v_target = v_total THEN
        v_status := 'completed';
    ELSIF v_target > 0 THEN
        v_status := 'partially_completed';
    ELSE
        v_status := 'failed';
    END IF;

    UPDATE proc.process_monitor
    SET    action_status = v_status
    WHERE  session_id = p_session_id AND action_status IS NULL;

    GET DIAGNOSTICS v_rows_upd = ROW_COUNT;
    IF v_rows_upd = 0 THEN RETURN; END IF;

    -- Per-document quality-action breakdown (new)
    SELECT
        COUNT(*) FILTER (WHERE doc_action = 'duplicate'),
        COUNT(*) FILTER (WHERE doc_action = 'updated'),
        COUNT(*) FILTER (WHERE doc_action = 'needs_review'),
        COUNT(*) FILTER (WHERE doc_action = 'unsupported')
    INTO v_dup, v_upd, v_review, v_unsup
    FROM proc.process_monitor
    WHERE session_id = p_session_id;

    SELECT COALESCE(
        json_agg(json_build_object('file_path', file_path, 'doc_action', doc_action))
            FILTER (WHERE doc_action IS NOT NULL),
        '[]'::json)
    INTO v_documents
    FROM proc.process_monitor
    WHERE session_id = p_session_id;

    PERFORM pg_notify(
        'session_status',
        json_build_object(
            'session_id',    p_session_id,
            'action_status', v_status,
            'total',         v_total,
            'target',        v_target,
            'discrepancy',   v_discrep,
            'failed',        v_failed,
            'duplicate',     COALESCE(v_dup, 0),
            'updated',       COALESCE(v_upd, 0),
            'needs_review',  COALESCE(v_review, 0),
            'unsupported',   COALESCE(v_unsup, 0),
            'documents',     v_documents,
            'resolved_at',   to_char(NOW() AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"')
        )::text
    );
END;
$$;

COMMIT;
