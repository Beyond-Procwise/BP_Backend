-- Who may teach the system a vendor extraction template (POST /vendors/onboard/{sid}/correct and /save).
--
-- extraction_template.write is a configure action, and the gate refuses those unless a policy states
-- effect=allow -- so until this row exists nobody can save a template, an Admin included (and every
-- Admin attempt opens a "clarify policy" decision). Admin, because a learned template steers how later
-- documents of that layout are read. Note the template store behind this route is in-memory and
-- per-process today (src/api/main.py mounts build_router() with no store), so a saved template lasts
-- until the next restart; the class is by intent, not by today's persistence.
-- FOR REVIEW. NOT APPLIED. Idempotent. Apply to BOTH bp_testdb and bp_sqldb.
BEGIN;

INSERT INTO proc.bp_policy (
    policy_name, policy_type, policy_desc, policy_details,
    policy_linked_agents, policy_status, version,
    created_date, created_by, last_modified_date, last_modified_by
)
SELECT 'ExtractionTemplateWriteAuthorityPolicy', 'authority',
       'Who may teach a vendor extraction template through the onboarding form. No agent does this.',
       jsonb_build_object(
         'policy_identifier', 'extraction_template_write_authority',
         'applies_to', jsonb_build_array('extraction_template.write'),
         'required_role', 'Admin',
         'rules', jsonb_build_object('effect', 'allow')),
       '', 1, 1, now(), 'migration:2026-10-22_extraction_template_write_authority', now(), 'migration:2026-10-22_extraction_template_write_authority'
 WHERE NOT EXISTS (
    SELECT 1 FROM proc.bp_policy p
     WHERE p.policy_details->>'policy_identifier' = 'extraction_template_write_authority'
       AND p.policy_status = 1);

COMMIT;
