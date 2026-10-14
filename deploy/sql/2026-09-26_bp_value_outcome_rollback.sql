BEGIN;
-- Removes the ledger entirely. Only safe before any real outcome has been recorded:
-- dropping the table discards the value history it exists to keep.
DROP TABLE IF EXISTS proc.bp_value_outcome;
DROP FUNCTION IF EXISTS proc.bp_value_outcome_immutable();
COMMIT;
