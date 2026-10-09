-- Removes exactly the column 2026-10-09_document_type_title_only.sql adds.
-- Roll the code back FIRST: the loader selects this column, and without it the
-- vocabulary query fails and the last good vocabulary is kept.
-- If any of the four types has been activated, dropping the flag lets body
-- mentions of 'guarantee' / 'DPA' relabel untitled contracts again.
BEGIN;

ALTER TABLE proc.bp_document_type DROP COLUMN IF EXISTS title_only;

COMMIT;
