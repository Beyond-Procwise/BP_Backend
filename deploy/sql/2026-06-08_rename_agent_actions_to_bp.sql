-- Rename proc.agent_actions -> proc.bp_agent_actions to align with the bp_ table
-- naming convention used by the rest of the schema (bp_invoice_trgt, etc.).
-- Idempotent and order-independent: a no-op on fresh deploys that already
-- created proc.bp_agent_actions directly, and on DBs where it has already run.
DO $$
BEGIN
    IF to_regclass('proc.agent_actions') IS NOT NULL
       AND to_regclass('proc.bp_agent_actions') IS NULL THEN
        ALTER TABLE proc.agent_actions RENAME TO bp_agent_actions;
        -- Postgres does not auto-rename indexes when a table is renamed.
        ALTER INDEX IF EXISTS proc.ix_agent_actions_deal     RENAME TO ix_bp_agent_actions_deal;
        ALTER INDEX IF EXISTS proc.ix_agent_actions_document RENAME TO ix_bp_agent_actions_document;
        ALTER INDEX IF EXISTS proc.ix_agent_actions_doc_pk   RENAME TO ix_bp_agent_actions_doc_pk;
        ALTER INDEX IF EXISTS proc.ix_agent_actions_trace    RENAME TO ix_bp_agent_actions_trace;
        ALTER INDEX IF EXISTS proc.ix_agent_actions_created  RENAME TO ix_bp_agent_actions_created;
        ALTER INDEX IF EXISTS proc.ix_agent_actions_phase    RENAME TO ix_bp_agent_actions_phase;
    END IF;
END $$;
