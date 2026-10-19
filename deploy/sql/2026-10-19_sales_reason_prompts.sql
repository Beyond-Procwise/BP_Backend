-- The wording of a sales opportunity's reason, as governed prompts (read by
-- src/services/sell_side/reason_prompts.py). The detector supplies the facts; these say them.
-- Guarded on prompt_name so a re-run is a no-op and an edited row is never overwritten.
-- Apply to BOTH bp_testdb and bp_sqldb: without the rows the detector writes no opportunities.

INSERT INTO proc.bp_prompt (prompt_name, prompt_type, prompt_linked_agents, prompts_desc)
SELECT 'sales_reason_relation', 'justification', 'sales_opportunity_detector',
       '{"prompt_template": "Bought {bought_sku}; the distributor states it is {relation_type} {target_sku}.",
         "phrases": {"relation_type": {"replaced_by": "replaced by", "upgrade_of": "upgraded by",
                                       "refill_of": "refilled by", "accessory_of": "has the accessory",
                                       "requires": "requires"}}}'::jsonb
WHERE NOT EXISTS (SELECT 1 FROM proc.bp_prompt WHERE prompt_name = 'sales_reason_relation');

INSERT INTO proc.bp_prompt (prompt_name, prompt_type, prompt_linked_agents, prompts_desc)
SELECT 'sales_reason_end_of_life_dated', 'justification', 'sales_opportunity_detector',
       '{"prompt_template": "{sku} reached {lifecycle_status} on {end_of_life_date}.",
         "phrases": {"lifecycle_status": {"end_of_sale": "end of sale", "default": "end of life"}}}'::jsonb
WHERE NOT EXISTS (SELECT 1 FROM proc.bp_prompt WHERE prompt_name = 'sales_reason_end_of_life_dated');

INSERT INTO proc.bp_prompt (prompt_name, prompt_type, prompt_linked_agents, prompts_desc)
SELECT 'sales_reason_end_of_life_undated', 'justification', 'sales_opportunity_detector',
       '{"prompt_template": "{sku} reached {lifecycle_status} (no date given).",
         "phrases": {"lifecycle_status": {"end_of_sale": "end of sale", "default": "end of life"}}}'::jsonb
WHERE NOT EXISTS (SELECT 1 FROM proc.bp_prompt WHERE prompt_name = 'sales_reason_end_of_life_undated');
