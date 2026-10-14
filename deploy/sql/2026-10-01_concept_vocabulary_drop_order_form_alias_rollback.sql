-- Reverse 2026-10-01_concept_vocabulary_drop_order_form_alias.sql: put "order form"
-- back as an alias of doctype.call_off_contract.
--
-- CAVEAT, the same shape as the other alias rollbacks in this set: this appends the
-- alias wherever it is absent, so running it on a database seeded fresh from the
-- CURRENT src/services/concepts/seed.py -- which no longer carries "order form" --
-- ADDS an alias that database never had, rather than restoring one. It is exact only
-- on a database that actually took the forward migration.
--
-- Running it will also reinstate the 12 false disagreements the forward migration
-- removed, and will make "order form" an accepted upload category again, because the
-- upload gate and the classifier read the same aliases column.
--
-- Append order matters: Task 1's full-column drift test compares alias lists IN
-- ORDER against the seed, so this appends to the end, which is where "order form"
-- sat before it was dropped. If you reinstate it, put it last in seed.py too.
--
-- Idempotent: a second run matches no rows.

BEGIN;

UPDATE proc.bp_document_type
   SET aliases = array_append(aliases, 'order form')
 WHERE concept_code = 'doctype.call_off_contract'
   AND NOT (aliases @> ARRAY['order form']::text[]);

COMMIT;
