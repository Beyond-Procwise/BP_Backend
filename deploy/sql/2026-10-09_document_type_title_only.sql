-- proc.bp_document_type.title_only: a type named by a document's own title,
-- never by mentions in its body.
--
-- Measured 2026-10-09 with the four proposed contract types switched on in a
-- test vocabulary: a contract with no clean title of its own ('THIS AGREEMENT
-- is made on…') read as a DPA, a guaranty or a side letter after TWO body
-- mentions of 'DPA', 'guarantee' or 'side letter' -- ordinary clause wording.
-- A titled contract was never affected. The flag removes such a type from body
-- (tier-2) scoring and leaves title (tier-1) recognition alone.
--
-- Sets the flag on exactly the four proposed types from
-- 2026-10-08_contract_link_vocabulary.sql. All four are status='proposed', so
-- nothing resolves to them and NO live behaviour changes here; the flag is in
-- place before anyone activates them.
--
-- Deploy order: this migration BEFORE the code. The loader selects the column;
-- without it the vocabulary query fails and the last good vocabulary is kept.
--
-- Additive, idempotent, reversible (2026-10-09_document_type_title_only_rollback.sql).
BEGIN;

ALTER TABLE proc.bp_document_type
    ADD COLUMN IF NOT EXISTS title_only boolean NOT NULL DEFAULT false;

COMMENT ON COLUMN proc.bp_document_type.title_only IS
    'When true, only the document''s own title names this type; body mentions '
    'never score it. For types whose name is everyday clause wording (DPA, '
    'guarantee, renewal, side letter).';

UPDATE proc.bp_document_type
   SET title_only = true
 WHERE concept_code IN ('doctype.dpa', 'doctype.side_letter',
                        'doctype.renewal', 'doctype.guaranty')
   AND title_only IS DISTINCT FROM true;

COMMIT;
