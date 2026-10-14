-- Reseller catalog + sell side: the two numbers it is governed by, and the two
-- irreversible actions it needs a stated permit for.
--
-- LIMITS (read by name through governed_limits.limit, deliberately no applies_to):
--   fuzzy_propose_min      rapidfuzz token_sort_ratio at or above which a catalog
--                          description is PROPOSED as matching a history item.
--                          Proposed, never applied: a person confirms.
--   calibration_min_closed closed (won + lost) opportunities of one type needed
--                          before win_probability is set for that type at all.
--                          Below it the column stays NULL -- an uncalibrated
--                          number is indistinguishable from a measured one.
--
-- PERMITS. The gate refuses transact/communicate unless a policy states
-- effect=allow. Approving an outbound quote commits us to a price; issuing one
-- sends it to a customer. Both need Approver. That nobody approves their own
-- quote is enforced in sell_side.quotes.approve, not stated here: a policy key
-- nothing reads is how this project has shipped rules that governed nothing.
BEGIN;

INSERT INTO proc.bp_policy (
    policy_name, policy_type, policy_desc, policy_details,
    policy_linked_agents, policy_status, version,
    created_date, created_by, last_modified_date, last_modified_by
)
SELECT 'ResellerCatalogLimitPolicy', 'limit',
       'How close a catalog description must be before it is offered as a match '
       'to purchase history, and how much closed history a win probability needs.',
       jsonb_build_object(
         'policy_identifier', 'reseller_catalog',
         'rules', jsonb_build_object(
             'fuzzy_propose_min',      88,
             'calibration_min_closed', 30)),
       '', 1, 1, now(), 'reseller_catalog', now(), 'reseller_catalog'
 WHERE NOT EXISTS (
    SELECT 1 FROM proc.bp_policy p
     WHERE p.policy_details->>'policy_identifier' = 'reseller_catalog'
       AND p.policy_status = 1);

INSERT INTO proc.bp_policy (
    policy_name, policy_type, policy_desc, policy_details,
    policy_linked_agents, policy_status, version,
    created_date, created_by, last_modified_date, last_modified_by
)
SELECT v.name, 'authority', v.descr,
       jsonb_build_object(
         'policy_identifier', v.slug,
         'applies_to', jsonb_build_array(v.action),
         'required_role', 'Approver',
         'rules', jsonb_build_object('effect', 'allow')),
       '', 1, 1, now(), 'reseller_catalog', now(), 'reseller_catalog'
  FROM (VALUES
    ('SalesQuoteApprovalAuthorityPolicy', 'sales_quote_approval_authority',
     'sales_quote.approve',
     'Who may approve an outbound sales quote. The approver is the authenticated '
     'caller and may not be the quote''s author.'),
    ('SalesQuoteIssueAuthorityPolicy', 'sales_quote_issue_authority',
     'sales_quote.issue',
     'Who may issue an approved sales quote to a customer.')
  ) AS v(name, slug, action, descr)
 WHERE NOT EXISTS (
    SELECT 1 FROM proc.bp_policy p
     WHERE p.policy_details->>'policy_identifier' = v.slug
       AND p.policy_status = 1);

COMMIT;
