-- Extend shadow mode by one week: 2026-10-09 -> 2026-10-16.
--
-- WHY. The observation log (proc.bp_policy_observation) was reviewed on 2026-10-08. 1,296 of the
-- 1,299 shadowed refusals read "no authenticated principal": the callers had no identity because
-- ASK_AUTH_MODE is "off", so the gate would refuse workflow.save, workflow.run, agent.create/
-- update/delete and policy.reload for EVERYONE the day shadow lapses. Seven enrolled actions have
-- no traffic at all, and bp_sqldb has no observations. Not enough evidence to enforce.
--
-- HOW. Every entry already enrolled moves to the new date, whatever the list holds today, so an
-- action added by a later migration (agent.update) is not left behind. An entry already later
-- than the new date is left alone. Idempotent: a second run changes nothing.
BEGIN;

UPDATE proc.bp_policy
   SET policy_details = jsonb_set(
           policy_details, '{rules,shadow_actions}',
           (SELECT jsonb_agg(
                     CASE WHEN (e->>'until')::timestamptz < '2026-10-16T00:00:00Z'::timestamptz
                          THEN jsonb_set(e, '{until}', '"2026-10-16T00:00:00Z"'::jsonb)
                          ELSE e END
                     ORDER BY ord)
              FROM jsonb_array_elements(policy_details->'rules'->'shadow_actions')
                   WITH ORDINALITY AS t(e, ord)),
           true),
       version = version + 1,
       last_modified_by = 'shadow_extend_one_week',
       last_modified_date = now()
 WHERE policy_status = 1
   AND policy_details->>'policy_identifier' = 'shadow_mode'
   AND EXISTS (SELECT 1
                 FROM jsonb_array_elements(policy_details->'rules'->'shadow_actions') x
                WHERE (x->>'until')::timestamptz < '2026-10-16T00:00:00Z'::timestamptz);

COMMIT;
