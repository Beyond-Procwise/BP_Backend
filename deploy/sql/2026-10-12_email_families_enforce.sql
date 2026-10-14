-- Email families: shadow -> enforce.
--
-- Until now every family recorded what it found and blocked nothing. In enforce the send guard refuses
-- an email whose facts changed since it was written, and one with an open point nobody has answered
-- (a figure software read from a supplier's email, a date with no record behind it). A person clears a
-- point from the approval screen ("Confirm this"), which is what makes this safe to switch on: before
-- that button existed, enforce would have left such an email unsendable from the screen.
--
-- Deliberately NOT part of the bp_sqldb pack (its own decision, per the pack document). Touches only
-- the family rows created by email_assurance_migration that are still in shadow. Idempotent.
-- Reversible: _rollback.sql.

UPDATE proc.bp_policy
   SET policy_details = jsonb_set(policy_details, '{rules,mode}', '"enforce"')
 WHERE policy_type = 'email_family'
   AND created_by = 'email_assurance_migration'
   AND policy_details->'rules'->>'mode' = 'shadow';
