-- Who may send a quote to its customer.
--
-- POST /sales/quotes/{id}/send emails the quote's PDF to a contact on the quote's
-- account. sales_quote.send is a communicate action, and the gate refuses those
-- unless a policy states effect=allow -- so until this row exists nobody can send
-- a quote, which is the intended state of a freshly deployed route.
--
-- Sending an approved quote is what issues it, and the route asks sales_quote.issue
-- as well (SalesQuoteIssueAuthorityPolicy, 2026-09-11_reseller_governance.sql). This
-- row uses the same role, Approver, so the two permits cannot disagree about who
-- may put a quote in front of a customer.
--
-- NOT APPLIED when written (2026-10-11): it switches on real email to customers and
-- waits for a yes. Rolling it back stops sending; it sends nothing and unsends nothing.
BEGIN;

INSERT INTO proc.bp_policy (
    policy_name, policy_type, policy_desc, policy_details,
    policy_linked_agents, policy_status, version,
    created_date, created_by, last_modified_date, last_modified_by
)
SELECT 'SalesQuoteSendAuthorityPolicy', 'authority',
       'Who may send an approved sales quote to its customer by email. Sending is '
       'what issues the quote. No agent sends one.',
       jsonb_build_object(
         'policy_identifier', 'sales_quote_send_authority',
         'applies_to', jsonb_build_array('sales_quote.send'),
         'required_role', 'Approver',
         'rules', jsonb_build_object('effect', 'allow')),
       '', 1, 1, now(), 'sales_quote_send', now(), 'sales_quote_send'
 WHERE NOT EXISTS (
    SELECT 1 FROM proc.bp_policy p
     WHERE p.policy_details->>'policy_identifier' = 'sales_quote_send_authority'
       AND p.policy_status = 1);

COMMIT;
