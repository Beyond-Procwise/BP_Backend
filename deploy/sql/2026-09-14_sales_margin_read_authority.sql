-- Who may see what a sale costs us and what it earns.
--
-- GET /sales/quotes/{id} and the two opportunity reads return unit cost, total
-- cost, margin and margin_pct. Until this migration they asked no gate at all:
-- the router's require_user was the only check, and with ASK_AUTH_MODE=off that
-- check returns no principal rather than refusing -- so cost and margin were
-- readable by anyone who could reach the port.
--
-- margin.read is a READ-class action, and a read with no policy is allowed for
-- every role, Viewer (and so the anonymous caller) included. The restriction
-- therefore lives in required_role here. Buyer is the lowest role that may write
-- a quote, and whoever prices a quote needs its margin to price it.
--
-- The router does not trust the reversible-class default: if this row is absent
-- it refuses margin to everyone rather than falling open (sales.py,
-- _require_margin_permit). Rolling this back withholds margin; it does not
-- expose it.
BEGIN;

INSERT INTO proc.bp_policy (
    policy_name, policy_type, policy_desc, policy_details,
    policy_linked_agents, policy_status, version,
    created_date, created_by, last_modified_date, last_modified_by
)
SELECT 'SalesMarginReadAuthorityPolicy', 'authority',
       'Who may read internal cost and margin on sales quotes and opportunities. '
       'Customers never see these; a Viewer or an unidentified caller is refused.',
       jsonb_build_object(
         'policy_identifier', 'sales_margin_read_authority',
         'applies_to', jsonb_build_array('margin.read'),
         'required_role', 'Buyer',
         'rules', jsonb_build_object('effect', 'allow')),
       '', 1, 1, now(), 'sales_margin_read', now(), 'sales_margin_read'
 WHERE NOT EXISTS (
    SELECT 1 FROM proc.bp_policy p
     WHERE p.policy_details->>'policy_identifier' = 'sales_margin_read_authority'
       AND p.policy_status = 1);

COMMIT;
