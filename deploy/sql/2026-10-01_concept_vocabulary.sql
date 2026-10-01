-- The document and relationship vocabulary as data rather than as Python dicts.
--
-- Today the list of document types is four words in a dict in
-- src/services/process_monitor_watcher.py:523, and anything else RAISES. The
-- vocabulary is additionally copied into fourteen other module-level maps (see
-- specs/2026-10-01-document-relationship-layer-discovery.md §5.1), so "the list
-- of document types" has no owner. These two tables are that owner.
--
-- Shape follows proc.bp_uom_canonical: aliases on the concept row rather than in
-- a second table, because a second table is a second place to edit one fact.
--
-- status:
--   active   -- usable for resolution
--   proposed -- observed in real data, awaiting human confirmation; NEVER used
--               to resolve, because an unconfirmed guess that silently starts
--               resolving is indistinguishable from a confirmed decision
--   rejected -- confirmed NOT a document type, with the reason recorded
--
-- Additive, idempotent, reversible.
BEGIN;

CREATE TABLE IF NOT EXISTS proc.bp_concept (
    concept_code            text PRIMARY KEY,
    -- Globally unique across domains, so the code carries its domain as a
    -- prefix: role.variation and doctype.variation are different things and
    -- would otherwise collide on this key.
    domain                  text NOT NULL,
    definition              text,
    -- Points at the concepts this one is most often mistaken for. The whole
    -- point of the exercise: 'order form' means one thing under a framework
    -- and another on its own.
    not_to_be_confused_with text[] NOT NULL DEFAULT '{}',
    tenant_id               text NOT NULL DEFAULT 'default',
    status                  text NOT NULL DEFAULT 'proposed',
    source                  text,
    -- Why a rejected row is not a concept. 'fix the uploader' and 'this is
    -- genuinely not a document type' are different problems and look identical
    -- without it.
    rejection_reason        text,
    observed_count          integer NOT NULL DEFAULT 0,
    first_observed_at       timestamptz,
    last_observed_at        timestamptz,
    confirmed_by            text,
    confirmed_at            timestamptz,
    valid_from              timestamptz NOT NULL DEFAULT now(),
    valid_to                timestamptz,
    recorded_at             timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT ck_bp_concept_domain CHECK (domain IN (
        'DOCUMENT_TYPE', 'RELATIONSHIP_ROLE', 'LINK_TYPE',
        'EXECUTION_MODE', 'EVENT_KIND')),
    CONSTRAINT ck_bp_concept_status CHECK (status IN ('active', 'proposed', 'rejected')),
    -- An active concept must say what it means; a proposed one legitimately
    -- does not know yet.
    CONSTRAINT ck_bp_concept_active_has_definition CHECK (
        status <> 'active' OR btrim(coalesce(definition, '')) <> ''),
    CONSTRAINT ck_bp_concept_rejected_has_reason CHECK (
        status <> 'rejected' OR btrim(coalesce(rejection_reason, '')) <> ''),
    CONSTRAINT ck_bp_concept_not_self_confusing CHECK (
        NOT (concept_code = ANY (not_to_be_confused_with)))
);

CREATE INDEX IF NOT EXISTS ix_bp_concept_domain ON proc.bp_concept (domain);
CREATE INDEX IF NOT EXISTS ix_bp_concept_status ON proc.bp_concept (status);

CREATE TABLE IF NOT EXISTS proc.bp_document_type (
    concept_code        text PRIMARY KEY
                        REFERENCES proc.bp_concept (concept_code),
    role                text NOT NULL REFERENCES proc.bp_concept (concept_code),
    default_parent_type text REFERENCES proc.bp_concept (concept_code),
    execution_mode      text REFERENCES proc.bp_concept (concept_code),
    -- Text as it appears on documents. An array rather than a second table,
    -- matching bp_uom_canonical.aliases.
    aliases             text[] NOT NULL DEFAULT '{}',
    -- [{field, pattern, parent_type}]. Matching is on field AND value, never
    -- value alone: a six-digit customer number and a six-digit order number
    -- are not the same identifier.
    identifiers         jsonb NOT NULL DEFAULT '[]',
    structural_signals  text[] NOT NULL DEFAULT '{}',
    -- Which of the four physical pipelines ingests this type. Four table
    -- families exist (invoice, purchase_order, quote, contract); NULL means
    -- the type is recognised but nothing can ingest it yet.
    pipeline_doc_type   text,
    tenant_id           text NOT NULL DEFAULT 'default',
    status              text NOT NULL DEFAULT 'proposed',
    source              text,
    observed_count      integer NOT NULL DEFAULT 0,
    first_observed_at   timestamptz,
    last_observed_at    timestamptz,
    confirmed_by        text,
    confirmed_at        timestamptz,
    valid_from          timestamptz NOT NULL DEFAULT now(),
    valid_to            timestamptz,
    recorded_at         timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT ck_bp_document_type_status CHECK (
        status IN ('active', 'proposed', 'rejected')),
    CONSTRAINT ck_bp_document_type_pipeline CHECK (
        pipeline_doc_type IS NULL OR pipeline_doc_type IN (
            'invoice', 'purchase_order', 'quote', 'contract')),
    CONSTRAINT ck_bp_document_type_identifiers_is_array CHECK (
        jsonb_typeof(identifiers) = 'array')
);

CREATE INDEX IF NOT EXISTS ix_bp_document_type_status
    ON proc.bp_document_type (status);
CREATE INDEX IF NOT EXISTS ix_bp_document_type_pipeline_doc_type
    ON proc.bp_document_type (pipeline_doc_type);
CREATE INDEX IF NOT EXISTS ix_bp_document_type_aliases
    ON proc.bp_document_type USING gin (aliases);

COMMENT ON TABLE proc.bp_concept IS
    'The document and relationship vocabulary. Only status=''active'' rows '
    'resolve; ''proposed'' rows are observed-but-unconfirmed and must never '
    'resolve silently.';
COMMENT ON TABLE proc.bp_document_type IS
    'Per-document-type attributes and aliases. pipeline_doc_type names which '
    'of the four physical pipelines ingests the type; NULL means none yet.';

-- Seed: mirrors src/services/concepts/seed.py. ON CONFLICT DO NOTHING so a re-run
-- never overwrites a row a human has since confirmed or edited.
-- Order matters: bp_document_type's foreign keys need the concepts first.
INSERT INTO proc.bp_concept (concept_code, domain, definition, not_to_be_confused_with, status, source, rejection_reason) VALUES
    ('role.framework', 'RELATIONSHIP_ROLE', 'An umbrella agreement under which later contracts are called off.', '{}'::text[], 'active', 'seed', NULL),
    ('role.master', 'RELATIONSHIP_ROLE', 'Governs a relationship and is itself the contract for the work it covers.', '{}'::text[], 'active', 'seed', NULL),
    ('role.transaction', 'RELATIONSHIP_ROLE', 'Orders, confirms or bills a specific quantity or amount.', '{}'::text[], 'active', 'seed', NULL),
    ('role.variation', 'RELATIONSHIP_ROLE', 'Changes the terms of a document that already exists.', '{}'::text[], 'active', 'seed', NULL),
    ('role.attachment', 'RELATIONSHIP_ROLE', 'Has no force alone; takes effect by being incorporated into another document.', '{}'::text[], 'active', 'seed', NULL),
    ('role.notice', 'RELATIONSHIP_ROLE', 'Communicates a fact or an intention; creates no new obligation by itself.', '{}'::text[], 'active', 'seed', NULL),
    ('role.termination', 'RELATIONSHIP_ROLE', 'Ends a document that is already in force.', '{}'::text[], 'active', 'seed', NULL),
    ('role.supporting', 'RELATIONSHIP_ROLE', 'Evidence or working papers around a relationship; not itself binding.', '{}'::text[], 'active', 'seed', NULL),
    ('link.calls_off', 'LINK_TYPE', 'The source orders work under the target framework.', '{}'::text[], 'active', 'seed', NULL),
    ('link.governed_by', 'LINK_TYPE', 'The source''s terms are set by the target.', '{}'::text[], 'active', 'seed', NULL),
    ('link.incorporates', 'LINK_TYPE', 'The source pulls the target''s terms into itself by reference.', '{}'::text[], 'active', 'seed', NULL),
    ('link.varies', 'LINK_TYPE', 'The source changes the target''s terms.', '{}'::text[], 'active', 'seed', NULL),
    ('link.supersedes', 'LINK_TYPE', 'The source replaces the target in full.', '{}'::text[], 'active', 'seed', NULL),
    ('link.terminates', 'LINK_TYPE', 'The source ends the target.', '{}'::text[], 'active', 'seed', NULL),
    ('link.attaches_to', 'LINK_TYPE', 'The source is an attachment of the target.', '{}'::text[], 'active', 'seed', NULL),
    ('link.references', 'LINK_TYPE', 'The source mentions the target without changing it.', '{}'::text[], 'active', 'seed', NULL),
    ('exec.bilateral', 'EXECUTION_MODE', 'Signed by two parties.', '{}'::text[], 'active', 'seed', NULL),
    ('exec.unilateral', 'EXECUTION_MODE', 'Issued and signed by one party.', '{}'::text[], 'active', 'seed', NULL),
    ('exec.multilateral', 'EXECUTION_MODE', 'Signed by three or more parties.', '{}'::text[], 'active', 'seed', NULL),
    ('exec.incorporated', 'EXECUTION_MODE', 'Not separately executed; takes effect through the document that incorporates it.', '{}'::text[], 'active', 'seed', NULL),
    ('event.signed', 'EVENT_KIND', 'The parties executed the document.', '{}'::text[], 'active', 'seed', NULL),
    ('event.effective', 'EVENT_KIND', 'The document''s terms began to apply.', '{}'::text[], 'active', 'seed', NULL),
    ('event.varied', 'EVENT_KIND', 'The document''s terms were changed.', '{}'::text[], 'active', 'seed', NULL),
    ('event.renewed', 'EVENT_KIND', 'The document''s term was extended on its own renewal terms.', '{}'::text[], 'active', 'seed', NULL),
    ('event.extended', 'EVENT_KIND', 'The document''s term was lengthened other than by renewal.', '{}'::text[], 'active', 'seed', NULL),
    ('event.expired', 'EVENT_KIND', 'The document''s term ran out.', '{}'::text[], 'active', 'seed', NULL),
    ('event.terminated', 'EVENT_KIND', 'The document was ended before its term ran out.', '{}'::text[], 'active', 'seed', NULL),
    ('event.superseded', 'EVENT_KIND', 'The document was replaced by another.', '{}'::text[], 'active', 'seed', NULL),
    ('doctype.framework_agreement', 'DOCUMENT_TYPE', 'Sets terms for future call-offs but orders nothing itself.', ARRAY['doctype.master_agreement']::text[], 'active', 'seed', NULL),
    ('doctype.master_agreement', 'DOCUMENT_TYPE', 'Governs a supplier relationship and is the contract for work done under it.', ARRAY['doctype.framework_agreement', 'doctype.call_off_contract']::text[], 'active', 'seed', NULL),
    ('doctype.sow', 'DOCUMENT_TYPE', 'Defines the deliverables, timescale and price of a specific piece of work under a master agreement.', ARRAY['doctype.call_off_contract', 'doctype.order']::text[], 'active', 'seed', NULL),
    ('doctype.call_off_contract', 'DOCUMENT_TYPE', 'The contract formed when work is ordered under a framework; incorporates the framework''s terms and sets their precedence.', ARRAY['doctype.order', 'doctype.sow', 'doctype.framework_agreement']::text[], 'active', 'seed', NULL),
    ('doctype.order', 'DOCUMENT_TYPE', 'Instructs a supplier to deliver a stated quantity at a stated price.', ARRAY['doctype.call_off_contract']::text[], 'active', 'seed', NULL),
    ('doctype.invoice', 'DOCUMENT_TYPE', 'Demands payment for goods or services supplied.', '{}'::text[], 'active', 'seed', NULL),
    ('doctype.quote', 'DOCUMENT_TYPE', 'Offers a price before any order exists.', '{}'::text[], 'active', 'seed', NULL),
    ('doctype.variation', 'DOCUMENT_TYPE', 'Changes the terms of an existing contract.', ARRAY['doctype.addendum', 'doctype.ccn']::text[], 'active', 'seed', NULL),
    ('doctype.schedule', 'DOCUMENT_TYPE', 'A numbered part of a contract that has no force on its own.', ARRAY['doctype.addendum', 'doctype.sla']::text[], 'active', 'seed', NULL),
    ('doctype.addendum', 'DOCUMENT_TYPE', 'Adds to a contract after signature without replacing it.', ARRAY['doctype.variation', 'doctype.schedule']::text[], 'active', 'seed', NULL),
    ('doctype.ccn', 'DOCUMENT_TYPE', 'Change control note: records an agreed change under a contract''s own change procedure.', ARRAY['doctype.variation']::text[], 'active', 'seed', NULL),
    ('doctype.termination_notice', 'DOCUMENT_TYPE', 'Ends a contract that is in force.', ARRAY['doctype.notice_general']::text[], 'active', 'seed', NULL),
    ('doctype.notice_general', 'DOCUMENT_TYPE', 'Communicates a fact or intention under a contract without ending or changing it.', ARRAY['doctype.termination_notice']::text[], 'active', 'seed', NULL),
    ('doctype.nda', 'DOCUMENT_TYPE', 'Binds the parties to keep information confidential.', '{}'::text[], 'active', 'seed', NULL),
    ('doctype.sla', 'DOCUMENT_TYPE', 'States service levels and remedies; normally a schedule to an agreement rather than a contract alone.', ARRAY['doctype.schedule']::text[], 'active', 'seed', NULL),
    ('doctype.service_agreement', 'DOCUMENT_TYPE', 'Contracts for the supply of a service on stated terms.', ARRAY['doctype.master_agreement', 'doctype.consulting_agreement']::text[], 'active', 'seed', NULL),
    ('doctype.consulting_agreement', 'DOCUMENT_TYPE', 'Contracts for advisory work, usually against time and materials or a retainer.', ARRAY['doctype.service_agreement']::text[], 'active', 'seed', NULL),
    ('doctype.contract_unspecified', 'DOCUMENT_TYPE', 'A contract whose kind the document does not state. Recorded as itself rather than guessed at.', '{}'::text[], 'active', 'seed', NULL),
    ('doctype.policy_document', 'DOCUMENT_TYPE', 'Observed as a contract_type value; what it denotes here is not yet established.', '{}'::text[], 'proposed', 'seed', NULL)
ON CONFLICT (concept_code) DO NOTHING;

INSERT INTO proc.bp_document_type (concept_code, role, default_parent_type, execution_mode, aliases, identifiers, structural_signals, pipeline_doc_type, status, source) VALUES
    ('doctype.framework_agreement', 'role.framework', NULL, 'exec.bilateral', ARRAY['framework agreement', 'framework contract']::text[], '[{"field": "framework_ref", "pattern": "^[A-Z]{2}\\d{4,6}$", "parent_type": null}]'::jsonb, ARRAY['sets terms without ordering', 'names a call-off procedure']::text[], 'contract', 'active', 'seed'),
    ('doctype.master_agreement', 'role.master', NULL, 'exec.bilateral', ARRAY['master agreement', 'msa', 'master service agreement', 'master services agreement']::text[], '[{"field": "contract_id", "pattern": null, "parent_type": null}]'::jsonb, ARRAY['recites the parties and the relationship', 'numbered clauses']::text[], 'contract', 'active', 'seed'),
    ('doctype.sow', 'role.master', 'doctype.master_agreement', 'exec.bilateral', ARRAY['sow', 'statement of work', 'work order', 'task order']::text[], '[{"field": "contract_id", "pattern": null, "parent_type": "doctype.master_agreement"}]'::jsonb, ARRAY['lists deliverables and milestones', 'names the agreement it sits under']::text[], 'contract', 'active', 'seed'),
    ('doctype.call_off_contract', 'role.master', 'doctype.framework_agreement', 'exec.bilateral', ARRAY['call-off contract', 'call off contract', 'call-off', 'order form']::text[], '[{"field": "framework_ref", "pattern": null, "parent_type": "doctype.framework_agreement"}]'::jsonb, ARRAY['lists incorporated documents', 'states an order of precedence']::text[], 'contract', 'active', 'seed'),
    ('doctype.order', 'role.transaction', 'doctype.call_off_contract', 'exec.unilateral', ARRAY['purchase order', 'purchase_order', 'purchaseorder', 'po', 'order']::text[], '[{"field": "po_id", "pattern": "^(?:PO)?\\d{4,10}$", "parent_type": null}]'::jsonb, ARRAY['line items with quantities and a total', 'ship-to address']::text[], 'purchase_order', 'active', 'seed'),
    ('doctype.invoice', 'role.transaction', 'doctype.order', 'exec.unilateral', ARRAY['invoice', 'tax invoice', 'bill']::text[], '[{"field": "invoice_id", "pattern": null, "parent_type": null}, {"field": "po_id", "pattern": null, "parent_type": "doctype.order"}]'::jsonb, ARRAY['amount due and payment terms', 'bill-to address']::text[], 'invoice', 'active', 'seed'),
    ('doctype.quote', 'role.supporting', NULL, 'exec.unilateral', ARRAY['quote', 'quotation', 'estimate', 'price quotation']::text[], '[{"field": "quote_id", "pattern": null, "parent_type": null}]'::jsonb, ARRAY['validity or expiry date', 'prices with no order reference']::text[], 'quote', 'active', 'seed'),
    ('doctype.variation', 'role.variation', NULL, 'exec.bilateral', ARRAY['variation', 'variation form', 'amendment', 'avenant', 'deed of variation']::text[], '[{"field": "amendment_ref", "pattern": null, "parent_type": null}]'::jsonb, ARRAY['names the document it changes', 'states what the change is']::text[], 'contract', 'active', 'seed'),
    ('doctype.schedule', 'role.attachment', NULL, 'exec.incorporated', ARRAY['schedule', 'annex', 'appendix', 'exhibit']::text[], '[]'::jsonb, ARRAY['numbered as part of another document', 'no signature block']::text[], 'contract', 'active', 'seed'),
    ('doctype.addendum', 'role.variation', NULL, 'exec.bilateral', ARRAY['addendum', 'supplemental agreement']::text[], '[]'::jsonb, ARRAY['adds terms after signature', 'names the document it supplements']::text[], 'contract', 'active', 'seed'),
    ('doctype.ccn', 'role.variation', NULL, 'exec.bilateral', ARRAY['ccn', 'change control note', 'change note', 'change request']::text[], '[]'::jsonb, ARRAY['cites the contract''s change procedure', 'states cost and time impact']::text[], 'contract', 'active', 'seed'),
    ('doctype.termination_notice', 'role.termination', NULL, 'exec.unilateral', ARRAY['termination notice', 'notice of termination']::text[], '[]'::jsonb, ARRAY['states a termination date', 'cites a termination clause']::text[], 'contract', 'active', 'seed'),
    ('doctype.notice_general', 'role.notice', NULL, 'exec.unilateral', ARRAY['general notice']::text[], '[]'::jsonb, ARRAY['cites a notice clause', 'creates no new obligation']::text[], NULL, 'active', 'seed'),
    ('doctype.nda', 'role.master', NULL, 'exec.bilateral', ARRAY['nda', 'non-disclosure agreement', 'confidentiality agreement']::text[], '[]'::jsonb, ARRAY['defines confidential information', 'states a confidentiality period']::text[], 'contract', 'active', 'seed'),
    ('doctype.sla', 'role.attachment', NULL, 'exec.incorporated', ARRAY['sla', 'service level agreement']::text[], '[]'::jsonb, ARRAY['service levels with targets', 'remedies or service credits']::text[], 'contract', 'active', 'seed'),
    ('doctype.service_agreement', 'role.master', NULL, 'exec.bilateral', ARRAY['service agreement', 'service contract', 'services agreement']::text[], '[]'::jsonb, ARRAY['describes a service and its term', 'numbered clauses']::text[], 'contract', 'active', 'seed'),
    ('doctype.consulting_agreement', 'role.master', NULL, 'exec.bilateral', ARRAY['consulting', 'consulting agreement', 'consultancy agreement']::text[], '[]'::jsonb, ARRAY['rates or a retainer', 'named consultants or roles']::text[], 'contract', 'active', 'seed'),
    ('doctype.contract_unspecified', 'role.master', NULL, NULL, ARRAY['contract', 'agreement']::text[], '[{"field": "contract_id", "pattern": null, "parent_type": null}]'::jsonb, ARRAY['numbered clauses', 'a signature block']::text[], 'contract', 'active', 'seed'),
    ('doctype.policy_document', 'role.supporting', NULL, NULL, ARRAY['policy']::text[], '[]'::jsonb, '{}'::text[], NULL, 'proposed', 'seed')
ON CONFLICT (concept_code) DO NOTHING;

COMMIT;
