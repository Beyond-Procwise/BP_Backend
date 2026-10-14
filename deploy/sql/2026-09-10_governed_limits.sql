-- P9: the governance limits move out of the environment and into policy.
--
-- Thirty-three values decided how much of a customer's money, how much of their
-- supplier data and how much agent reach the product would allow, and every one
-- of them was an environment variable: changeable with no code change AND no
-- policy edit, versioned by nothing, audited by nothing, and on no governance
-- screen. The classification that produced this set, and the argument for each
-- borderline call, is in docs/superpowers/specs/2026-09-10-p9-environment-limit-classification.md.
--
-- SEVEN ROWS, not thirty-three, grouped by subject. One row per value would be a
-- governance screen nobody can read, which is its own kind of ungoverned.
--
-- DELIBERATELY NO applies_to on any of them. That field is how guardrail.authorize
-- selects the policies it weighs, and these are configuration read BY NAME by the
-- code that needs them -- not authority statements about who may do what. P7 drew
-- that line (giving a scoring-weights table an applies_to would turn it into an
-- authority statement nobody wrote) and P6 followed it. These follow it too.
--
-- Values are today's environment defaults exactly. This migration changes WHERE a
-- limit lives, not what it is; a migration that quietly retuned the product while
-- moving it would be impossible to review.
BEGIN;

INSERT INTO proc.bp_policy (
    policy_name, policy_type, policy_desc, policy_details,
    policy_linked_agents, policy_status, version,
    created_date, created_by, last_modified_date, last_modified_by
)
SELECT v.name, 'limit', v.descr,
       jsonb_build_object('policy_identifier', v.slug, 'rules', v.rules),
       '', 1, 1, now(), 'governed_limits', now(), 'governed_limits'
  FROM (VALUES
    ('PromotionThresholdPolicy', 'promotion_thresholds',
     'What reaches the record of truth. These scores decide which documents are '
     'promoted from _stg into _trgt, which are held for review, and which parent '
     'orders may be proposed at all.',
     jsonb_build_object(
       'promote_min_confidence',  50,
       'promote_min_link_score',  80,
       'promote_review_min',      65,
       'propose_min_link_score',  40,
       'propose_max_candidates',   5,
       'quote_anchor_min_score',  80
     )),

    ('ReconciliationTolerancePolicy', 'reconciliation_tolerances',
     'What counts as a match on money. Widen these and discrepancies stop being '
     'reported; they are the definition of "the invoice agrees with the order".',
     jsonb_build_object(
       'amount_tolerance_pct', 0.01,
       'amount_tolerance_abs', 1.00,
       'tax_tolerance_pct',    0.1
     )),

    ('SupplierIdentityPolicy', 'supplier_identity',
     'When two names are one company. These bands decide which supplier an '
     'invoice is attributed to, which duplicates are merged, and how close a '
     'researched company name must be before it is offered for review.',
     jsonb_build_object(
       'review_low',            82,
       'review_high',           96,
       'sweep_min_score',       88,
       'research_name_match',   85,
       'research_propose_conf', 0.75
     )),

    ('NegotiationBoundsPolicy', 'negotiation_bounds',
     'What an agent may put to a supplier. Commercial bounds on volume, terms, '
     'how hard a first counter may be, and when a price movement is escalated '
     'rather than accepted. An agent allowed to concede 40%% instead of 20%% is a '
     'different agent.',
     jsonb_build_object(
       'max_volume_limit',       1000,
       'max_term_days',           120,
       'max_supplier_replies',      3,
       'first_counter_aggr_pct',  0.12,
       'market_review_pct',       0.2,
       'market_escalation_pct',   0.4,
       'lt_value_pct_per_week',   0.01,
       'cost_of_capital_apr',     0.12
     )),

    ('AgentReachPolicy', 'agent_reach',
     'How far an agent may go before it must stop and answer: how many agents may '
     'be spawned for one workflow, how many rounds of tool-calling any one of them '
     'may take, and how much of a negotiation thread it may read before countering. '
     'neg_thread_transcript_limit is present and null on purpose -- null is a '
     'stated "no limit", which is not the same as the key being absent.',
     jsonb_build_object(
       'max_dynamic_agents',            3,
       'tool_runtime_max_rounds',       6,
       'governed_reasoning_max_rounds', 5,
       'supplier_research_max_rounds',  4,
       'neg_thread_transcript_limit',   null
     )),

    ('ExtractionEffortPolicy', 'extraction_effort',
     'How hard the product tries to be right. These cap the L3 judge''s calls and '
     'seconds per document, so they decide when it stops checking whether a figure '
     'was read correctly. Filed as governance rather than cost control because '
     'extraction accuracy is this product''s first priority, and a cost cap that '
     'quietly lowers it is a decision somebody should have to own.',
     jsonb_build_object(
       'judge_max_calls', 12,
       'judge_budget_s',  25
     )),

    ('AutonomousOperationPolicy', 'autonomous_operation',
     'What the product does unprompted, and how long it keeps what it captures: '
     'the impact floor below which an opportunity is not worth raising, the '
     'retention period for captured data, and whether the duplicate-invoice '
     'detector and outbound supplier research run at all. The last two are '
     'switches rather than numbers, and are here precisely because a switch that '
     'turns a control off is the most consequential setting on the list.',
     jsonb_build_object(
       'opportunity_mining_min_impact',      100,
       'capture_retention_days',              30,
       'duplicate_invoice_detector_enabled', false,
       'supplier_research_enabled',          true
     ))
  ) AS v(name, slug, descr, rules)
 WHERE NOT EXISTS (
    SELECT 1 FROM proc.bp_policy p
     WHERE p.policy_details->>'policy_identifier' = v.slug
       AND p.policy_status = 1
 );

COMMIT;
