-- The contract parent-proposal floor and the contested gap, governed.
--
-- They were constants in src/services/contract_links.py (65.0 and 8.0). Every
-- other link threshold lives here, under promotion_thresholds, and
-- governed_limits.limit() RAISES on an absent key -- so a database without this
-- migration proposes no contract parents at all, which is the correct failure.
-- Run on BOTH bp_testdb AND bp_sqldb BEFORE the code that reads them ships.
-- Values unchanged from the constants: this moves them, it does not retune them.
-- Idempotent: jsonb || jsonb overwrites the two keys and leaves every other rule.
UPDATE proc.bp_policy
   SET policy_details = jsonb_set(
         policy_details, '{rules}',
         (policy_details -> 'rules') || jsonb_build_object(
            -- Below this F a contract parent is not worth a person's attention.
            -- 65 is the linking engine's own review band.
            'contract_parent_min_score', 65,
            -- How far apart best and runner-up must be to read "confirm this"
            -- rather than "choose between these" (contested).
            'contract_parent_separation', 8
         )),
       last_modified_date = NOW(),
       last_modified_by   = 'deploy/sql/2026-10-08_contract_parent_thresholds.sql',
       version            = COALESCE(version, 1) + 1
 WHERE policy_details ->> 'policy_identifier' = 'promotion_thresholds';

SELECT policy_details -> 'rules' -> 'contract_parent_min_score'  AS min_score,
       policy_details -> 'rules' -> 'contract_parent_separation' AS separation
  FROM proc.bp_policy
 WHERE policy_details ->> 'policy_identifier' = 'promotion_thresholds';
