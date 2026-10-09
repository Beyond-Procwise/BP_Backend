-- Restores the original award-commitment pattern where the widened one is in place.
BEGIN;
UPDATE proc.bp_policy
   SET policy_details = jsonb_set(policy_details, '{rules,forbidden_patterns,award_commitment}', to_jsonb($p$we\s+(will|shall|are\s+going\s+to)\s+(award|place\s+the\s+order|issue\s+a\s+(po|purchase\s+order))|guarantee[d]?\s+(the\s+)?(volume|order|business)$p$::text))
 WHERE policy_name IN ('EmailFamily_negotiation_counter', 'EmailFamily_free_prompt', 'EmailFamily_rfq_batch', 'EmailFamily_human_written')
   AND created_by = 'email_assurance_migration'
   AND policy_details->'rules'->'forbidden_patterns'->>'award_commitment' = $p$\bwe(?:\s+will|'ll|\s+shall|\s+are\s+going\s+to|\s+are)\s+(?:award(?:ing)?|plac(?:e|ing)\s+the\s+order|issu(?:e|ing)\s+(?:a|the)\s+(?:po|purchase\s+order)|sign(?:ing)?\s+the\s+(?:contract|order))|\b(?:the\s+)?(?:contract|order|business|deal|work)\s+(?:is|will\s+be)\s+yours\b(?!\s+to\b)|\byou(?:'ve|\s+have)\s+(?:won|got|secured)\s+(?:the\s+|our\s+)?(?:contract|order|business|deal|work)|\b(?:contract|order|business|deal)\s+(?:will\s+be|is\s+being)\s+awarded\s+to\s+you|\bconsider\s+(?:the\s+|this\s+)?(?:order|contract|deal)\s+(?:placed|confirmed|done|agreed|awarded)|\bwe(?:\s+will|'ll)?\s+commit\s+to\s+(?:order|buy|purchas|plac|a\s+(?:minimum\s+)?volume|the\s+volume)|guarantee[ds]?\s+(?:you\s+)?(?:the\s+|a\s+)?(?:volume|order|business|contract)$p$;
COMMIT;
