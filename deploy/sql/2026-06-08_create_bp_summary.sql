-- Persona summary cache/history. summary_id is an opaque UUID. Each row stores
-- the generated summary plus the exact data snapshot it was built from (so an
-- as_of request can regenerate over a past data state). Idempotent.

CREATE TABLE IF NOT EXISTS proc.bp_summary (
    summary_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    persona         TEXT        NOT NULL,
    persona_source  TEXT        NOT NULL,
    scope           TEXT        NOT NULL,
    deal_id         VARCHAR(25),
    summary         TEXT        NOT NULL,
    data_snapshot   JSONB       NOT NULL,
    sources         JSONB,
    model           TEXT,
    is_current      BOOLEAN     NOT NULL DEFAULT true,
    generated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by      TEXT        NOT NULL DEFAULT 'system'
);

CREATE INDEX IF NOT EXISTS ix_bp_summary_current
    ON proc.bp_summary (persona, deal_id) WHERE is_current;
CREATE INDEX IF NOT EXISTS ix_bp_summary_lookup
    ON proc.bp_summary (persona, deal_id, generated_at DESC);
CREATE INDEX IF NOT EXISTS ix_bp_summary_deal
    ON proc.bp_summary (deal_id);

-- Seed persona framings into the governance prompt table. Guarded on
-- prompt_name so re-runs are no-ops.
INSERT INTO proc.bp_prompt (prompt_name, prompt_type, prompt_linked_agents, prompts_desc)
SELECT 'analysis', 'summary_persona', 'summary_agent',
       '{"prompt_template": "You are a procurement data analyst. Emphasize spend totals, price and volume trends, supplier concentration, and quantitative anomalies."}'::jsonb
WHERE NOT EXISTS (SELECT 1 FROM proc.bp_prompt WHERE prompt_name = 'analysis' AND prompt_type = 'summary_persona');

INSERT INTO proc.bp_prompt (prompt_name, prompt_type, prompt_linked_agents, prompts_desc)
SELECT 'negotiation', 'summary_persona', 'summary_agent',
       '{"prompt_template": "You are a procurement negotiation strategist. Emphasize leverage points, price gaps between quotes, purchase orders and invoices, contract and renewal timing, and concession opportunities."}'::jsonb
WHERE NOT EXISTS (SELECT 1 FROM proc.bp_prompt WHERE prompt_name = 'negotiation' AND prompt_type = 'summary_persona');

INSERT INTO proc.bp_prompt (prompt_name, prompt_type, prompt_linked_agents, prompts_desc)
SELECT 'compliance', 'summary_persona', 'summary_agent',
       '{"prompt_template": "You are a procurement compliance auditor. Emphasize policy adherence, discrepancies, missing approvals, tax and currency correctness, and audit flags."}'::jsonb
WHERE NOT EXISTS (SELECT 1 FROM proc.bp_prompt WHERE prompt_name = 'compliance' AND prompt_type = 'summary_persona');
