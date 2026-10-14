-- Where a contract document's recognised structure is recorded.
--
-- src/services/extraction/dispatch.py:542 says the classification is
-- "Recorded, never acted on" -- it reaches a log line and, on disagreement, a
-- review item, and is never written to the document's row. So nothing could ask
-- "is this a SOW?", and no relationship maths could run. These three columns are
-- the answer to that.
--
-- bp_contract_raw, because it is the ONLY contract table carrying
-- process_monitor_id and source_file -- the per-document identity. bp_contracts
-- and bp_contract_master carry neither, so a classification written there could
-- not be traced to the document that produced it.
--
-- Both tables get the columns because promotion copies the intersection of raw
-- and target columns (promotion.py:858, via information_schema) minus
-- _CONTROL_COLS. None of these three is a control column, so promotion carries
-- them with no code change.
--
-- NULLABLE ON PURPOSE. A page that states no structure stores NULL, never the
-- declared category as a stand-in: that would make every document look as though
-- it had confirmed its own upload category.
--
-- Not a foreign key to proc.bp_concept. A concept can be retired or renamed, and
-- an FK would then either block the retirement or rewrite history on a document
-- that genuinely did read as the old structure. The value is a record of what the
-- classifier concluded at extraction time.
--
-- Additive, idempotent, reversible.
BEGIN;

ALTER TABLE proc.bp_contract_raw
    ADD COLUMN IF NOT EXISTS resolved_doc_type text,
    ADD COLUMN IF NOT EXISTS resolved_role     text,
    ADD COLUMN IF NOT EXISTS type_agreement    text;

ALTER TABLE proc.bp_contracts
    ADD COLUMN IF NOT EXISTS resolved_doc_type text,
    ADD COLUMN IF NOT EXISTS resolved_role     text,
    ADD COLUMN IF NOT EXISTS type_agreement    text;

-- Candidate parents are looked up by structure, so the lookup gets an index.
CREATE INDEX IF NOT EXISTS ix_bp_contracts_resolved_doc_type
    ON proc.bp_contracts (resolved_doc_type);

COMMENT ON COLUMN proc.bp_contracts.resolved_doc_type IS
    'The concept_code the document''s own text resolved to, e.g. doctype.sow. '
    'NULL when the page stated nothing -- never the declared category as a '
    'stand-in.';
COMMENT ON COLUMN proc.bp_contracts.resolved_role IS
    'That structure''s relationship role, e.g. role.master, copied at extraction '
    'time.';
COMMENT ON COLUMN proc.bp_contracts.type_agreement IS
    'agreed | refined | declared_only | evidence_only | disagreed | neither -- '
    'how the page''s reading compared with the uploader''s declaration.';

COMMIT;
