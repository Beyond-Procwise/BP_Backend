-- Contract parent-link proposals: the two limits that let them run at all.
--
-- The contract parent-link proposer was built, tested and live-verified on
-- 2026-10-03 and NOTHING CALLED IT -- the cadence question ("may proposals
-- appear in a buyer's queue unprompted?") was unanswered, so the wiring was
-- deliberately left out (specs/2026-10-02-contract-structures-design.md §8).
-- Answered 2026-10-04: yes, on promotion, scoped to the document that just
-- promoted, with a slow corpus-wide backstop.
--
-- Both keys live under AutonomousOperationPolicy because that is where this
-- product keeps "what it does unprompted". governed_limits.limit() RAISES on an
-- absent key, so a database without this migration proposes nothing at all --
-- which is the correct failure, and the reason this is a prerequisite rather
-- than a follow-up. Run on BOTH bp_testdb AND bp_sqldb.
--
-- Idempotent: jsonb || jsonb overwrites the two keys and leaves the rest of the
-- rules, and the row's other columns, untouched.

UPDATE proc.bp_policy
   SET policy_details = jsonb_set(
         policy_details,
         '{rules}',
         (policy_details -> 'rules') || jsonb_build_object(
            -- Whether a contract's parent is proposed without a person asking.
            -- Governs BOTH the promotion hook and the backstop sweep: a flag
            -- that silenced one and left the other writing would be worse than
            -- no flag. A proposal never links anything -- confirm() does, and
            -- only a person calls confirm().
            'contract_parent_proposals_enabled', true,
            -- How often the corpus-wide backstop runs. Hours, not minutes: the
            -- pass is corpus-wide and writes into a queue a person works.
            'contract_parent_proposal_sweep_hours', 24
         )),
       last_modified_date = NOW(),
       last_modified_by   = 'deploy/sql/2026-10-04_contract_parent_proposals.sql',
       version            = COALESCE(version, 1) + 1
 WHERE policy_details ->> 'policy_identifier' = 'autonomous_operation';

-- Verify: both keys present, and the four that were already there still are.
SELECT policy_name,
       policy_details -> 'rules' -> 'contract_parent_proposals_enabled'    AS enabled,
       policy_details -> 'rules' -> 'contract_parent_proposal_sweep_hours' AS sweep_hours,
       jsonb_object_keys(policy_details -> 'rules')                        AS every_rule
  FROM proc.bp_policy
 WHERE policy_details ->> 'policy_identifier' = 'autonomous_operation';
