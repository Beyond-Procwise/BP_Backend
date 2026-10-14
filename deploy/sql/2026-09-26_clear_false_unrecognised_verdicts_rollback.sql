-- Restores the two verdicts 2026-09-26_clear_false_unrecognised_verdicts.sql removed.
--
-- Roll the code back first: TranslationService._call no longer writes a verdict like this,
-- so putting the rows back while the new code runs re-freezes French and Dutch without
-- anything being able to write them again.
--
-- The prompt version is the one they were branded under. Under any other version the rows
-- are inert, which is why they are spelled out rather than recomputed.
BEGIN;

INSERT INTO proc.bp_translation_language_status
    (target_lang, prompt_version, model, recognized, confidence)
VALUES
    ('fr', 'translate-ui/4+cfg.d5128abb', 'BeyondProcwise/AgentNick:unified', FALSE, 'low'),
    ('nl', 'translate-ui/4+cfg.d5128abb', 'BeyondProcwise/AgentNick:unified', FALSE, 'low')
ON CONFLICT (target_lang, prompt_version, model) DO UPDATE
    SET recognized = FALSE, confidence = 'low';

COMMIT;
