-- Create proc.bp_prompt and proc.bp_policy (bp_ naming convention) and seed
-- them from the defaults agents currently hardcode. Idempotent: safe to re-run.

-- ============================ bp_prompt ============================
CREATE TABLE IF NOT EXISTS proc.bp_prompt (
    prompt_id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    prompt_name          TEXT        NOT NULL,
    prompt_type          TEXT,
    prompt_linked_agents TEXT,
    prompts_desc         JSONB,
    prompts_status       SMALLINT    NOT NULL DEFAULT 1,
    version              INTEGER     NOT NULL DEFAULT 1,
    created_date         TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by           TEXT        NOT NULL DEFAULT 'system',
    last_modified_date   TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_modified_by     TEXT        NOT NULL DEFAULT 'system'
);
CREATE INDEX IF NOT EXISTS ix_bp_prompt_status
    ON proc.bp_prompt (prompts_status) WHERE prompts_status = 1;
CREATE INDEX IF NOT EXISTS ix_bp_prompt_name
    ON proc.bp_prompt (prompt_name);

-- ============================ bp_policy ============================
CREATE TABLE IF NOT EXISTS proc.bp_policy (
    policy_id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    policy_name          TEXT        NOT NULL,
    policy_type          TEXT,
    policy_desc          TEXT,
    policy_details       JSONB,
    policy_linked_agents TEXT,
    policy_status        SMALLINT    NOT NULL DEFAULT 1,
    version              INTEGER     NOT NULL DEFAULT 1,
    created_date         TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by           TEXT        NOT NULL DEFAULT 'system',
    last_modified_date   TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_modified_by     TEXT        NOT NULL DEFAULT 'system'
);
CREATE INDEX IF NOT EXISTS ix_bp_policy_status
    ON proc.bp_policy (policy_status) WHERE policy_status = 1;
CREATE INDEX IF NOT EXISTS ix_bp_policy_name
    ON proc.bp_policy (policy_name);
CREATE INDEX IF NOT EXISTS ix_bp_policy_type
    ON proc.bp_policy (policy_type);

-- ============================ seed prompts ============================
-- Guarded on prompt_name so re-runs are no-ops (no unique constraint added,
-- so the governance UI stays free to create name variants/versions later).
INSERT INTO proc.bp_prompt (prompt_name, prompt_type, prompt_linked_agents, prompts_desc)
SELECT 'supplier_ranking_justification', 'justification', 'supplier_ranking_agent',
       '{"prompt_template": "Supplier {supplier_name} achieved a final score of {final_score:.2f}.\n{score_breakdown}"}'::jsonb
WHERE NOT EXISTS (SELECT 1 FROM proc.bp_prompt WHERE prompt_name = 'supplier_ranking_justification');

INSERT INTO proc.bp_prompt (prompt_name, prompt_type, prompt_linked_agents, prompts_desc)
SELECT 'rank_by_criteria', 'decomposition', 'supplier_ranking_agent',
       '{"templates": [{"template_id": "rank_by_criteria", "parameters": {"category": null, "criteria": ["price", "delivery", "risk"], "time_period": null, "filters": null}}]}'::jsonb
WHERE NOT EXISTS (SELECT 1 FROM proc.bp_prompt WHERE prompt_name = 'rank_by_criteria');

INSERT INTO proc.bp_prompt (prompt_name, prompt_type, prompt_linked_agents, prompts_desc)
SELECT 'negotiation_message_default', 'message', 'negotiation_agent',
       '{"prompt_template": "{header}\n{details}{context_sections}"}'::jsonb
WHERE NOT EXISTS (SELECT 1 FROM proc.bp_prompt WHERE prompt_name = 'negotiation_message_default');

-- ============================ seed policies ============================
INSERT INTO proc.bp_policy (policy_name, policy_type, policy_desc, policy_linked_agents, policy_details)
SELECT 'WeightAllocationPolicy', 'supplier_ranking', 'Default supplier ranking weights', 'supplier_ranking_agent',
       '{"policy_identifier": "weight_allocation_policy", "rules": {"default_weights": {"price": 0.4, "delivery": 0.3, "risk": 0.2, "payment_terms": 0.1}}}'::jsonb
WHERE NOT EXISTS (SELECT 1 FROM proc.bp_policy WHERE policy_name = 'WeightAllocationPolicy');

INSERT INTO proc.bp_policy (policy_name, policy_type, policy_desc, policy_linked_agents, policy_details)
SELECT 'CategoricalScoringPolicy', 'supplier_ranking', 'Categorical scoring maps', 'supplier_ranking_agent',
       '{"policy_identifier": "categorical_scoring_policy", "rules": {"categorical_maps": {}}}'::jsonb
WHERE NOT EXISTS (SELECT 1 FROM proc.bp_policy WHERE policy_name = 'CategoricalScoringPolicy');

INSERT INTO proc.bp_policy (policy_name, policy_type, policy_desc, policy_linked_agents, policy_details)
SELECT 'NormalizationDirectionPolicy', 'supplier_ranking', 'Per-metric normalization direction', 'supplier_ranking_agent',
       '{"policy_identifier": "normalization_direction_policy", "rules": {"directions": {"price": "lower_is_better", "delivery": "lower_is_better", "risk": "lower_is_better", "payment_terms": "higher_is_better"}}}'::jsonb
WHERE NOT EXISTS (SELECT 1 FROM proc.bp_policy WHERE policy_name = 'NormalizationDirectionPolicy');

INSERT INTO proc.bp_policy (policy_name, policy_type, policy_desc, policy_linked_agents, policy_details)
SELECT 'ContractExpiryOpportunity', 'opportunity', 'Flag contracts nearing expiry', 'opportunity_miner_agent',
       '{"policy_identifier": "contract_expiry_check", "required_fields": ["negotiation_window_days"], "default_conditions": {"negotiation_window_days": 90}}'::jsonb
WHERE NOT EXISTS (SELECT 1 FROM proc.bp_policy WHERE policy_name = 'ContractExpiryOpportunity');

INSERT INTO proc.bp_policy (policy_name, policy_type, policy_desc, policy_linked_agents, policy_details)
SELECT 'PriceBenchmarkVariance', 'opportunity', 'Flag price variance vs benchmark', 'opportunity_miner_agent',
       '{"policy_identifier": "price_variance_check", "required_fields": ["supplier_id", "item_id", "actual_price", "benchmark_price"], "default_conditions": {}}'::jsonb
WHERE NOT EXISTS (SELECT 1 FROM proc.bp_policy WHERE policy_name = 'PriceBenchmarkVariance');

INSERT INTO proc.bp_policy (policy_name, policy_type, policy_desc, policy_linked_agents, policy_details)
SELECT 'VolumeConsolidation', 'opportunity', 'Flag volume consolidation opportunities', 'opportunity_miner_agent',
       '{"policy_identifier": "volume_consolidation_check", "required_fields": ["minimum_volume_gbp"], "default_conditions": {}}'::jsonb
WHERE NOT EXISTS (SELECT 1 FROM proc.bp_policy WHERE policy_name = 'VolumeConsolidation');

INSERT INTO proc.bp_policy (policy_name, policy_type, policy_desc, policy_linked_agents, policy_details)
SELECT 'SupplierRiskAlert', 'opportunity', 'Flag elevated supplier risk', 'opportunity_miner_agent',
       '{"policy_identifier": "supplier_risk_check", "required_fields": ["risk_threshold"], "default_conditions": {}}'::jsonb
WHERE NOT EXISTS (SELECT 1 FROM proc.bp_policy WHERE policy_name = 'SupplierRiskAlert');

INSERT INTO proc.bp_policy (policy_name, policy_type, policy_desc, policy_linked_agents, policy_details)
SELECT 'MaverickSpend', 'opportunity', 'Flag off-contract maverick spend', 'opportunity_miner_agent',
       '{"policy_identifier": "maverick_spend_check", "required_fields": [], "default_conditions": {}}'::jsonb
WHERE NOT EXISTS (SELECT 1 FROM proc.bp_policy WHERE policy_name = 'MaverickSpend');
