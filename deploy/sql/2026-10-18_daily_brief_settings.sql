-- The daily brief's settings: who reads which view, what each view shows, and the cut-offs.
--
-- Until now the Home screen chose a brief by pattern-matching the NAME of the reader's
-- role in the browser, and every cut-off was a constant in code. This row is where an
-- administrator chooses them instead. The server reads it fresh on every brief, so a
-- change applies at once, and it REFUSES to serve a brief when the row is missing or does
-- not hold together -- it never falls back to a view the code made up.
--
--   view_by_person / view_by_group / view_by_role / default_view
--       which view a reader gets; the most specific that matches wins.
--   views        what each view shows (sections, in order) and whether it covers
--                'everyone' or only the reader's 'own' records.
--   view_order   which view wins when a person's groups point at more than one.
--   the numbers  how old "aged" is, how far ahead "upcoming" looks, how many examples
--                a section names. upcoming_min_value_gbp null = no value floor.
--
-- The values below are a starting point, not a ruling: change them on the settings screen.
-- Idempotent. Apply to BOTH bp_testdb and bp_sqldb, before the code that reads it.

BEGIN;

INSERT INTO proc.bp_policy (
    policy_name, policy_type, policy_desc, policy_details,
    policy_linked_agents, policy_status, version,
    created_date, created_by, last_modified_date, last_modified_by
)
SELECT 'DailyBriefSettingsPolicy', 'daily_brief',
       'Who reads which daily brief, what each one shows, and the cut-offs it uses.',
       jsonb_build_object(
         'policy_identifier', 'daily_brief',
         'rules', $rules$
{
  "default_view": "requester",
  "view_by_role": {"Admin": "manager", "Approver": "manager", "Buyer": "buyer", "Viewer": "requester"},
  "view_by_group": {"PROCWISE_CHIEF_PROCUREMENT_OFFICER": "executive"},
  "view_by_person": {},
  "view_order": ["executive", "manager", "buyer", "requester"],
  "views": {
    "executive": {"scope": "everyone", "sections": ["momentum", "critical_aged", "upcoming"]},
    "manager":   {"scope": "everyone", "sections": ["approvals", "critical_aged", "pipeline", "momentum", "upcoming"]},
    "buyer":     {"scope": "own",      "sections": ["pipeline", "supplier_emails", "opportunities"]},
    "requester": {"scope": "own",      "sections": ["my_requests"]}
  },
  "issue_severities": ["critical", "warning"],
  "critical_aged_severities": ["critical"],
  "momentum_window_days": 7,
  "critical_aged_min_days": 14,
  "critical_aged_top_n": 5,
  "approval_aged_min_days": 7,
  "upcoming_horizon_days": 90,
  "upcoming_top_n": 5,
  "upcoming_min_value_gbp": null,
  "pipeline_top_n": 8,
  "supplier_email_window_days": 14,
  "opportunity_top_n": 5
}
$rules$::jsonb),
       '', 1, 1, now(), 'daily_brief', now(), 'daily_brief'
 WHERE NOT EXISTS (
    SELECT 1 FROM proc.bp_policy p
     WHERE p.policy_details->>'policy_identifier' = 'daily_brief'
       AND p.policy_status = 1);

COMMIT;
