BEGIN;
DELETE FROM proc.bp_document_type WHERE concept_code IN
    ('doctype.dpa','doctype.side_letter','doctype.renewal','doctype.guaranty');
DELETE FROM proc.bp_concept WHERE concept_code IN
    ('doctype.dpa','doctype.side_letter','doctype.renewal','doctype.guaranty');
COMMIT;
