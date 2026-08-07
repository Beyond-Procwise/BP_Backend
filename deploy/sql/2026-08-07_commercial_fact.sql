-- Phase 1b: carry the fact model across the seam.
-- Plan: docs/superpowers/plans/2026-08-07-phase-1b-fact-model-across-the-seam.md
--
-- Four tables. A CommercialFact is one number a document actually stated,
-- together with the evidence for it. Provenance is mandatory and non-empty,
-- enforced here as well as in Pydantic because the backfill and any direct
-- INSERT bypass the model entirely.
--
-- Additive, idempotent, reversible. Applied to bp_sqldb and bp_testdb.
BEGIN;

-- ---------------------------------------------------------------------------
-- 1. The fact itself.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS proc.bp_commercial_fact (
    fact_id             text PRIMARY KEY,
    -- B2 decision: tenant_id on every new table, defaulted to one constant.
    -- RLS is deliberately NOT enabled: there is no second tenant and no tenant
    -- dimension anywhere else in proc, so a policy here would be theatre.
    tenant_id           text NOT NULL DEFAULT 'default',
    fact_type           text NOT NULL,
    concept_code        text,

    -- source
    source_doc_type     text,
    source_doc_pk       text,
    document_id         text,
    document_version    text,
    line_no             integer,

    -- measure semantics (F6). A number's role cannot be inferred from its
    -- field name: the schemas already name unit_price and line_total apart,
    -- and a real bug still booked a total as a unit price. The role is carried
    -- explicitly, and arithmetic_state records whether it was verified.
    measure_role        text,
    basis_uom           text,
    arithmetic_state    text,

    -- economics. NUMERIC throughout, never double precision: float money
    -- reintroduces exactly the rounding drift the benchmark engine removed.
    unit_price          numeric,
    quantity            numeric,
    extended_value      numeric,
    tax_amount          numeric,
    discount_amount     numeric,
    uom                 text,
    uom_normalised      text,
    uom_dimension       text,

    -- currency. The rate is stamped onto the fact, so re-rendering tomorrow
    -- against a refreshed rate table cannot move yesterday's number.
    currency            text,
    base_currency       text,
    extended_value_base numeric,
    fx_rate             numeric,
    fx_rate_date        timestamptz,
    fx_rate_source      text,

    -- commercial identity
    supplier_id         text,
    supplier_name       text,
    buyer_id            text,
    item_reference      text,
    item_description    text,
    category_l1         text,
    category_l2         text,
    category_l3         text,
    category_l4         text,

    -- term
    contract_id         text,
    term_start          timestamptz,
    term_end            timestamptz,
    term_months         integer,
    billing_frequency   text,
    escalator_pct       numeric,
    escalator_basis     text,
    escalator_cap_pct   numeric,

    -- allocation
    cost_centre         text,
    region              text,
    country             text,

    -- grouping
    bundle_group_id     text,
    deal_id             text,

    -- integrity
    value_basis         text NOT NULL DEFAULT 'as_supplied',
    validation_state    text NOT NULL DEFAULT 'unverified',
    reason_codes        text[] NOT NULL DEFAULT '{}',
    confidence          real,

    -- bitemporal
    valid_from          timestamptz NOT NULL DEFAULT now(),
    valid_to            timestamptz,
    recorded_at         timestamptz NOT NULL DEFAULT now(),

    -- The enums are enforced in the database as well as in Pydantic. Anything
    -- that writes here without going through the model still cannot invent a
    -- role or a state.
    CONSTRAINT ck_bp_commercial_fact_measure_role CHECK (
        measure_role IS NULL OR measure_role IN (
            'unit_rate', 'extended_line', 'document_total',
            'quantity', 'tax', 'discount')),
    CONSTRAINT ck_bp_commercial_fact_arithmetic_state CHECK (
        arithmetic_state IS NULL OR arithmetic_state IN (
            'consistent', 'inconsistent',
            'untestable_quantity_one', 'untestable_missing_input')),
    CONSTRAINT ck_bp_commercial_fact_value_basis CHECK (
        value_basis IN ('as_supplied', 'baseline_corrected', 'normalised')),
    CONSTRAINT ck_bp_commercial_fact_validation_state CHECK (
        validation_state IN ('valid', 'invalid', 'unverified', 'indeterminate')),

    -- The two cross-field rules, mirroring the Pydantic model_validator.
    -- A unit rate with no "per what" is not comparable to anything.
    CONSTRAINT ck_bp_commercial_fact_unit_rate_needs_basis CHECK (
        measure_role IS DISTINCT FROM 'unit_rate' OR basis_uom IS NOT NULL),
    -- A priced fact that does not say whether its role was verified is exactly
    -- the ambiguity this phase exists to remove.
    CONSTRAINT ck_bp_commercial_fact_priced_needs_arithmetic CHECK (
        measure_role IS NULL
        OR measure_role NOT IN ('unit_rate', 'extended_line', 'document_total')
        OR arithmetic_state IS NOT NULL)
);

CREATE INDEX IF NOT EXISTS ix_bp_commercial_fact_tenant_measure_role
    ON proc.bp_commercial_fact (tenant_id, measure_role);
CREATE INDEX IF NOT EXISTS ix_bp_commercial_fact_tenant_fact_type
    ON proc.bp_commercial_fact (tenant_id, fact_type);
CREATE INDEX IF NOT EXISTS ix_bp_commercial_fact_supplier_id
    ON proc.bp_commercial_fact (supplier_id);
CREATE INDEX IF NOT EXISTS ix_bp_commercial_fact_contract_id
    ON proc.bp_commercial_fact (contract_id);
CREATE INDEX IF NOT EXISTS ix_bp_commercial_fact_document_version
    ON proc.bp_commercial_fact (document_version);
CREATE INDEX IF NOT EXISTS ix_bp_commercial_fact_source_doc
    ON proc.bp_commercial_fact (source_doc_type, source_doc_pk);

-- ---------------------------------------------------------------------------
-- 2. Provenance. One row per document span behind a fact.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS proc.bp_fact_provenance (
    provenance_row_id bigserial PRIMARY KEY,
    fact_id           text NOT NULL
                          REFERENCES proc.bp_commercial_fact (fact_id)
                          ON DELETE CASCADE,
    tenant_id         text NOT NULL DEFAULT 'default',
    document_id       text NOT NULL,
    doc_type          text,
    extraction_id     text,
    field_path        text NOT NULL,
    page              integer,
    locator           text NOT NULL,
    verbatim_snippet  text,
    model             text,
    confidence        real,
    extracted_at      timestamptz,

    valid_from        timestamptz NOT NULL DEFAULT now(),
    valid_to          timestamptz,
    recorded_at       timestamptz NOT NULL DEFAULT now(),

    -- Provenance that cannot point at a place in a document is not provenance.
    -- Without these, a blank row would satisfy the "at least one child"
    -- trigger below while defeating its entire purpose.
    CONSTRAINT ck_bp_fact_provenance_document_id_non_blank
        CHECK (btrim(document_id) <> ''),
    CONSTRAINT ck_bp_fact_provenance_locator_non_blank
        CHECK (btrim(locator) <> '')
);

CREATE INDEX IF NOT EXISTS ix_bp_fact_provenance_fact_id
    ON proc.bp_fact_provenance (fact_id);
CREATE INDEX IF NOT EXISTS ix_bp_fact_provenance_document_id
    ON proc.bp_fact_provenance (document_id);

-- ---------------------------------------------------------------------------
-- 3. Mandatory provenance, enforced by the database.
--
-- A CHECK constraint CANNOT express this rule. A CHECK sees only the row it is
-- attached to; "at least one row exists in another table" is a cross-table
-- predicate, and a subquery is not permitted in a CHECK. Nor can a plain
-- AFTER INSERT trigger work: the fact must be inserted before its provenance
-- can reference it, so an immediate check would reject every correct
-- insertion. A CONSTRAINT TRIGGER that is DEFERRABLE INITIALLY DEFERRED runs
-- at COMMIT, by which point both rows exist -- which is the only point at
-- which the question is meaningful.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION proc.bp_commercial_fact_require_provenance()
RETURNS trigger AS $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM proc.bp_fact_provenance p WHERE p.fact_id = NEW.fact_id
    ) THEN
        RAISE EXCEPTION
            'commercial fact % has no provenance: a fact with no evidence behind it cannot be committed',
            NEW.fact_id
            USING ERRCODE = '23514';
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS bp_commercial_fact_provenance_required
    ON proc.bp_commercial_fact;
CREATE CONSTRAINT TRIGGER bp_commercial_fact_provenance_required
    AFTER INSERT OR UPDATE ON proc.bp_commercial_fact
    DEFERRABLE INITIALLY DEFERRED
    FOR EACH ROW
    EXECUTE FUNCTION proc.bp_commercial_fact_require_provenance();

-- Closing the other half of the rule: deleting the last provenance row would
-- otherwise leave an unprovenanced fact behind. Skipped when the parent fact
-- is itself being deleted (the ON DELETE CASCADE case), because then there is
-- no fact left to be unprovenanced.
CREATE OR REPLACE FUNCTION proc.bp_fact_provenance_keep_fact_covered()
RETURNS trigger AS $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM proc.bp_commercial_fact f WHERE f.fact_id = OLD.fact_id
    ) AND NOT EXISTS (
        SELECT 1 FROM proc.bp_fact_provenance p WHERE p.fact_id = OLD.fact_id
    ) THEN
        RAISE EXCEPTION
            'removing this row would leave commercial fact % with no provenance',
            OLD.fact_id
            USING ERRCODE = '23514';
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS bp_fact_provenance_keeps_fact_covered
    ON proc.bp_fact_provenance;
CREATE CONSTRAINT TRIGGER bp_fact_provenance_keeps_fact_covered
    AFTER DELETE ON proc.bp_fact_provenance
    DEFERRABLE INITIALLY DEFERRED
    FOR EACH ROW
    EXECUTE FUNCTION proc.bp_fact_provenance_keep_fact_covered();

-- ---------------------------------------------------------------------------
-- 4. Constraints -- commercially material limits that are not prices.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS proc.bp_constraint (
    constraint_id       text PRIMARY KEY,
    tenant_id           text NOT NULL DEFAULT 'default',
    constraint_type     text NOT NULL,
    concept_code        text,

    bound_value         numeric,
    bound_uom           text,
    bound_direction     text,
    bound_currency      text,

    -- Left NULL rather than guessed. "500 users" is not a limit until you know
    -- whether it counts named, concurrent, peak, average or cumulative users.
    bound_basis         text,
    measurement_period  text,
    applies_to_entities  text[] NOT NULL DEFAULT '{}',
    applies_to_documents text[] NOT NULL DEFAULT '{}',
    testability_state   text NOT NULL DEFAULT 'pending_context',

    source_doc_type     text,
    source_doc_pk       text,
    document_id         text,
    contract_id         text,
    effective_from      timestamptz,
    effective_to        timestamptz,

    validation_state    text NOT NULL DEFAULT 'unverified',
    reason_codes        text[] NOT NULL DEFAULT '{}',
    confidence          real,

    valid_from          timestamptz NOT NULL DEFAULT now(),
    valid_to            timestamptz,
    recorded_at         timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT ck_bp_constraint_bound_basis CHECK (
        bound_basis IS NULL OR bound_basis IN (
            'named', 'concurrent', 'peak', 'average', 'cumulative')),
    CONSTRAINT ck_bp_constraint_bound_direction CHECK (
        bound_direction IS NULL OR bound_direction IN (
            'maximum', 'minimum', 'exact')),
    CONSTRAINT ck_bp_constraint_testability_state CHECK (
        testability_state IN ('testable', 'pending_context', 'untestable'))
);

CREATE INDEX IF NOT EXISTS ix_bp_constraint_tenant_type
    ON proc.bp_constraint (tenant_id, constraint_type);
CREATE INDEX IF NOT EXISTS ix_bp_constraint_contract_id
    ON proc.bp_constraint (contract_id);

-- ---------------------------------------------------------------------------
-- 5. Finding -> fact link.
--
-- A join table rather than flat columns on bp_opportunity, because a finding
-- genuinely draws on several documents: measured on bp_opportunity, 301 of 308
-- rows have 3 source documents, one has 4, six have 1. Flattening unit_price
-- onto the opportunity row would force choosing one of three arbitrarily.
-- Facts are independent of findings -- one invoice line can support both an
-- overbilling finding and a duplicate finding -- so the relationship is
-- many-to-many with a role.
--
-- No FK to bp_opportunity: its primary key is opportunity_id, while findings
-- are addressed here by opportunity_ref_id, which carries no unique index.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS proc.bp_finding_fact (
    opportunity_ref_id text NOT NULL,
    fact_id            text NOT NULL
                           REFERENCES proc.bp_commercial_fact (fact_id)
                           ON DELETE CASCADE,
    role               text NOT NULL,
    tenant_id          text NOT NULL DEFAULT 'default',

    valid_from         timestamptz NOT NULL DEFAULT now(),
    valid_to           timestamptz,
    recorded_at        timestamptz NOT NULL DEFAULT now(),

    PRIMARY KEY (opportunity_ref_id, fact_id, role)
);

CREATE INDEX IF NOT EXISTS ix_bp_finding_fact_opportunity_ref_id
    ON proc.bp_finding_fact (opportunity_ref_id);
CREATE INDEX IF NOT EXISTS ix_bp_finding_fact_fact_id
    ON proc.bp_finding_fact (fact_id);

COMMIT;
