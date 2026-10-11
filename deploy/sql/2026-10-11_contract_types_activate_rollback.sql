-- Puts the four contract types back to 'proposed' with no pipeline, as
-- 2026-10-08_contract_link_vocabulary.sql seeded them. Nothing resolves or routes
-- to them afterwards. Rows already stored under one of these types keep their
-- recorded type; this changes how NEW documents are read, not history.
BEGIN;

UPDATE proc.bp_document_type
   SET status = 'proposed', pipeline_doc_type = NULL, recorded_at = now()
 WHERE concept_code IN ('doctype.dpa', 'doctype.side_letter',
                        'doctype.renewal', 'doctype.guaranty');

UPDATE proc.bp_concept
   SET status = 'proposed', recorded_at = now()
 WHERE concept_code IN ('doctype.dpa', 'doctype.side_letter',
                        'doctype.renewal', 'doctype.guaranty');

COMMIT;
