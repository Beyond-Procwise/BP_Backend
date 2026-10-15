-- Which agent answers a question typed on each SpendIQ screen.
--
-- Keys are SpendIQ screen ids (engine.js `current`), values are agent slugs
-- loaded in agent_nick.agents. A screen not listed answers as General, and so
-- does a screen whose agent is not loaded. Sending agents are never used even
-- if listed (src/services/ask_routing.py NOT_ASKABLE).
-- Idempotent. Apply to BOTH bp_testdb and bp_sqldb, before the code that reads it.

BEGIN;

INSERT INTO proc.bp_policy (
    policy_name, policy_type, policy_desc, policy_details,
    policy_linked_agents, policy_status, version,
    created_date, created_by, last_modified_date, last_modified_by
)
SELECT 'AskScreenAgentsPolicy', 'routing',
       'Which agent answers a question asked on each screen. A screen not listed answers as General.',
       jsonb_build_object(
         'policy_identifier', 'ask_screen_agents',
         'rules', jsonb_build_object('screens', jsonb_build_object(
             'negotiations', 'negotiation',
             'quotes',       'quote_comparison',
             'quotebuilder', 'quote_comparison',
             'suppliers',    'supplier_ranking',
             'tprm',         'supplier_ranking',
             'actions',      'discrepancy_detection',
             'invoices',     'discrepancy_detection',
             'pos',          'discrepancy_detection',
             'compliance',   'discrepancy_detection'))),
       '', 1, 1, now(), 'ask_routing', now(), 'ask_routing'
 WHERE NOT EXISTS (
    SELECT 1 FROM proc.bp_policy p
     WHERE p.policy_details->>'policy_identifier' = 'ask_screen_agents'
       AND p.policy_status = 1);

COMMIT;
