-- Enrol the remaining newly-gated actions in shadow mode, same expiry.
--
-- Second cut of GA-3 wires eleven more endpoints: workflow save/run, supplier
-- research and enrichment-apply, document upload, extract and promote. Measured
-- against an anonymous caller (ASK_AUTH_MODE is "off" here) every one refuses:
--
--   workflow.save     no authenticated principal
--   workflow.run      no authenticated principal
--   research.web      no authenticated principal
--   supplier.write    role Viewer may not perform write
--   document.upload   role Viewer may not perform write
--   document.promote  role Viewer may not perform write
--   document.extract  role Viewer may not perform compute
--
-- Note the second group: those are not refused for want of a principal, they
-- are refused because an anonymous caller resolves to Viewer and Viewer may not
-- write. Same outcome, different cause, and both would stop document intake and
-- promotion dead -- the pipeline that feeds the entire product.
--
-- So they join the first five under the same 2026-10-09 expiry, and the same
-- bargain: evaluated, recorded, allowed through, and enforcing for real on a
-- date somebody has to look at.
BEGIN;

UPDATE proc.bp_policy
   SET policy_details = jsonb_set(
           policy_details, '{rules,shadow_actions}',
           '[
              {"action": "policy.reload",    "until": "2026-10-09T00:00:00Z"},
              {"action": "prompt.write",     "until": "2026-10-09T00:00:00Z"},
              {"action": "agent.create",     "until": "2026-10-09T00:00:00Z"},
              {"action": "agent.delete",     "until": "2026-10-09T00:00:00Z"},
              {"action": "model.train",      "until": "2026-10-09T00:00:00Z"},
              {"action": "workflow.save",    "until": "2026-10-09T00:00:00Z"},
              {"action": "workflow.run",     "until": "2026-10-09T00:00:00Z"},
              {"action": "research.web",     "until": "2026-10-09T00:00:00Z"},
              {"action": "supplier.write",   "until": "2026-10-09T00:00:00Z"},
              {"action": "document.upload",  "until": "2026-10-09T00:00:00Z"},
              {"action": "document.promote", "until": "2026-10-09T00:00:00Z"},
              {"action": "document.extract", "until": "2026-10-09T00:00:00Z"}
            ]'::jsonb, true
       ),
       version = version + 1,
       last_modified_by = 'shadow_enrol_remaining',
       last_modified_date = now()
 WHERE policy_status = 1
   AND policy_details->>'policy_identifier' = 'shadow_mode';

COMMIT;
