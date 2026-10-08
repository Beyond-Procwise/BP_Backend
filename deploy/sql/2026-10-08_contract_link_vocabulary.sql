-- Four document types the vocabulary should know about before anything can resolve to them.
-- status='proposed': counted and visible, never resolved or routed (only 'active' rows
-- resolve). Nick confirms each. No pipeline, like doctype.policy_document.
-- Additive (ON CONFLICT DO NOTHING), idempotent, reversible.
BEGIN;

INSERT INTO proc.bp_concept
    (concept_code, domain, definition, not_to_be_confused_with, status, source, rejection_reason)
VALUES
    ('doctype.dpa', 'DOCUMENT_TYPE',
     'Governs how personal data is processed for a parent agreement; forms part of it.',
     ARRAY['doctype.addendum']::text[], 'proposed', 'seed', NULL),
    ('doctype.side_letter', 'DOCUMENT_TYPE',
     'A separate letter that modifies or waives a term of an agreement it names.',
     ARRAY['doctype.variation']::text[], 'proposed', 'seed', NULL),
    ('doctype.renewal', 'DOCUMENT_TYPE',
     'Extends an agreement past its expiry on terms the original already sets.',
     ARRAY['doctype.variation']::text[], 'proposed', 'seed', NULL),
    ('doctype.guaranty', 'DOCUMENT_TYPE',
     'A third party''s undertaking to answer for a party''s obligations under an agreement.',
     ARRAY[]::text[], 'proposed', 'seed', NULL)
ON CONFLICT (concept_code) DO NOTHING;

INSERT INTO proc.bp_document_type
    (concept_code, role, default_parent_type, execution_mode, aliases, identifiers,
     structural_signals, pipeline_doc_type, status, source)
VALUES
    ('doctype.dpa', 'role.attachment', NULL, NULL,
     ARRAY['dpa','data processing agreement']::text[], '[]'::jsonb, '{}'::text[], NULL, 'proposed', 'seed'),
    ('doctype.side_letter', 'role.variation', NULL, NULL,
     ARRAY['side letter']::text[], '[]'::jsonb, '{}'::text[], NULL, 'proposed', 'seed'),
    ('doctype.renewal', 'role.variation', NULL, NULL,
     ARRAY['renewal agreement']::text[], '[]'::jsonb, '{}'::text[], NULL, 'proposed', 'seed'),
    ('doctype.guaranty', 'role.supporting', NULL, NULL,
     ARRAY['guaranty','guarantee','parent company guarantee']::text[], '[]'::jsonb, '{}'::text[],
     NULL, 'proposed', 'seed')
ON CONFLICT (concept_code) DO NOTHING;

COMMIT;
