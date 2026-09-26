-- 2026-09-26  Release the languages a single bad model reply condemned
-- ---------------------------------------------------------------------------
-- The verdict in bp_translation_language_status is sticky by design: one
-- lang_recognized=false is enough, for ever, per (language, prompt version, model). That is
-- right for a language the model genuinely cannot write -- asked for Elvish it hands the
-- English straight back -- but AgentNick also emits the flag occasionally for a language it
-- plainly can write. French and Dutch were caught by exactly that, and the state is a trap:
-- the verdict gates every later call, so no later reply could ever contradict it.
--
-- The consequences were not cosmetic. Both were dropped from the signed-out language picker
-- (api/routers/i18n.py::_public_available), both reported supported=false in the product,
-- and every string added after the branding stayed English permanently -- French included,
-- which is one of the three languages with human-reviewed copy behind it (1,816 strings).
--
-- The code no longer brands a language whose own reply translated the batch
-- (TranslationService._call, validate.wrote_nothing), so this clears the rows already
-- written under the old rule. It removes them rather than setting recognized=true: no
-- verdict means "not asked yet", which is the truth, and the next batch records a fresh one.
--
-- Deliberately narrow. Verdicts for languages the model really cannot write (x-elvish,
-- x-zorblaxi, chr) are left exactly as they are -- the WHERE clause keeps only rows that
-- have translations to show for themselves, which is the evidence the flag ignored.
--
-- Idempotent. Run against: bp_testdb, bp_sqldb.
-- ---------------------------------------------------------------------------
BEGIN;

DELETE FROM proc.bp_translation_language_status s
 WHERE s.recognized IS FALSE
   AND EXISTS (
       SELECT 1
         FROM proc.bp_translation t
        WHERE t.target_lang = s.target_lang
        HAVING count(*) >= 100
   );

COMMIT;
