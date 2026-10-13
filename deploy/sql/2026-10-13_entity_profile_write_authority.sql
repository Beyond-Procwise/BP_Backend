-- Who may change the company profile printed on every quote.
--
-- PUT /entity/profile replaces the selling entity's own details: legal name, registered
-- address, company and VAT numbers, and the bank account customers are told to pay into.
-- entity_profile.write is a configure action, and the gate refuses those unless a policy
-- states effect=allow -- so until this row exists nobody can change the profile, an
-- Admin included. Reading it (GET /entity/profile) needs only a signed-in user and is
-- not governed by this row.
--
-- Admin, because a changed bank account on an outgoing quote is how a payment is
-- misdirected. Idempotent. Apply to BOTH bp_testdb and bp_sqldb.
BEGIN;

INSERT INTO proc.bp_policy (
    policy_name, policy_type, policy_desc, policy_details,
    policy_linked_agents, policy_status, version,
    created_date, created_by, last_modified_date, last_modified_by
)
SELECT 'EntityProfileWriteAuthorityPolicy', 'authority',
       'Who may change the company profile (letterhead and bank details) printed on '
       'every customer quote. No agent changes it.',
       jsonb_build_object(
         'policy_identifier', 'entity_profile_write_authority',
         'applies_to', jsonb_build_array('entity_profile.write'),
         'required_role', 'Admin',
         'rules', jsonb_build_object('effect', 'allow')),
       '', 1, 1, now(), 'sales_quote_builder', now(), 'sales_quote_builder'
 WHERE NOT EXISTS (
    SELECT 1 FROM proc.bp_policy p
     WHERE p.policy_details->>'policy_identifier' = 'entity_profile_write_authority'
       AND p.policy_status = 1);

COMMIT;
