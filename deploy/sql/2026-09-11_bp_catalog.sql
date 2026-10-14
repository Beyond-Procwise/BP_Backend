-- Distributor product catalog + landed cost. Gaps 1 and 2 of the reseller capability audit.
-- Spec: docs/superpowers/specs/2026-09-09-reseller-catalog-and-sell-side-design.md
--
-- A catalog is reference data a distributor ASSERTS, not a fact extracted from prose.
-- Nothing here goes through raw -> _stg -> _trgt, and nothing here is judged or grounded:
-- the confidence machinery exists to decide whether a model read a document correctly,
-- and there is no such question to answer about a column in a price file.

BEGIN;

-- PREREQUISITE. proc.bp_supplier is a local base table with 5,028 rows and 5,028 distinct
-- supplier_id, and no declared key. Everything below that points at a supplier needs one.
-- Guarded: ADD CONSTRAINT has no IF NOT EXISTS, and this file must re-apply cleanly.
DO $$
BEGIN
    ALTER TABLE proc.bp_supplier
        ADD CONSTRAINT pk_bp_supplier PRIMARY KEY (supplier_id);
EXCEPTION
    WHEN duplicate_table OR invalid_table_definition THEN NULL;  -- already keyed
END $$;

-- THE CATEGORY REFERENCE CANNOT BE A FOREIGN KEY, and this is not a shortcut.
--
-- proc.bp_category_master is a VIEW over canonical.bp_category, which is a FOREIGN TABLE
-- reaching another database through the uicanvas_srv wrapper. Postgres cannot declare a
-- foreign key to a foreign table, and a constraint added over there would not be enforced
-- here. So bp_catalog_item.unspsc_code is a soft reference, validated by the importer at
-- load time against the 246 live unspsc_code values and rejected on the row if unmatched.
--
-- Worth recording why unspsc_code and not the level ids: category_level_5_id is NOT unique
-- in that view -- 121 distinct values across 246 rows -- so it could not carry the
-- reference even if the plumbing allowed it. unspsc_code has 246 distinct values.


-- One row per catalog file we ingested. Without it, "this distributor sent us nothing"
-- and "the import died on row 4,000" both surface as an absent SKU -- a green zero the
-- system has not earned. content_sha256 is what makes a re-send idempotent.
CREATE TABLE IF NOT EXISTS proc.bp_catalog_source (
    source_id        bigserial PRIMARY KEY,
    tenant_id        text,
    distributor_id   text NOT NULL
        REFERENCES proc.bp_supplier (supplier_id) ON DELETE RESTRICT,
    feed_name        text NOT NULL,
    file_name        text,
    content_sha256   text NOT NULL,
    mapping_profile  text NOT NULL,           -- named column map used; see bp_catalog_mapping
    price_effective  date NOT NULL,           -- the date the DISTRIBUTOR says this file prices
    status           text NOT NULL,           -- imported | partial | failed
    rows_seen        integer NOT NULL DEFAULT 0,
    rows_loaded      integer NOT NULL DEFAULT 0,
    rows_rejected    integer NOT NULL DEFAULT 0,
    error            text,
    imported_by      text,
    -- A retry of a file that failed must be possible, and the UNIQUE below would otherwise
    -- forbid it: one receipt per (distributor, file), carrying the LATEST attempt's outcome.
    -- Found by writing the importer against this table, not by reading it.
    attempt_count    integer NOT NULL DEFAULT 1,
    first_attempt_at timestamptz NOT NULL DEFAULT now(),
    created_date     timestamptz NOT NULL DEFAULT now(),
    UNIQUE (distributor_id, content_sha256)
);

-- A distributor's column headings are their own and they change without warning.
-- The map is a row a human owns, not a heuristic: an unmapped required column is a
-- rejected import, never a guessed one.
CREATE TABLE IF NOT EXISTS proc.bp_catalog_mapping (
    mapping_profile  text NOT NULL,
    distributor_id   text NOT NULL,
    target_column    text NOT NULL,           -- column name in bp_catalog_item
    source_header    text NOT NULL,           -- verbatim heading in the distributor's file
    transform        text,                    -- null | trim_currency | pence_to_major | pack_split
    is_required      boolean NOT NULL DEFAULT FALSE,
    created_date     timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (mapping_profile, target_column)
);


-- The catalog itself, versioned rather than overwritten.
--
-- A distributor reprices monthly. If a row were updated in place, a quote sent in March
-- would silently re-price itself in April and the margin we defended would stop being the
-- margin we quoted. So a price change closes the current row (valid_to = now) and opens a
-- new one -- the same bitemporal shape bp_uom_canonical and bp_fact_provenance already use.
--
-- Every commercial column is nullable ON PURPOSE. A feed that omits stock, or lifecycle,
-- or cost, produces NULL there; it never produces a zero, a FALSE or a guess.
CREATE TABLE IF NOT EXISTS proc.bp_catalog_item (
    catalog_item_id     bigserial PRIMARY KEY,
    source_id           bigint NOT NULL
        REFERENCES proc.bp_catalog_source (source_id) ON DELETE RESTRICT,
    tenant_id           text,

    -- Identity. distributor_sku is the feed's own key and is the only one guaranteed
    -- present; mpn is what actually matches a customer's purchase history, when we have it.
    -- Denormalised from bp_catalog_source deliberately: the partial unique index in ux_...
    -- _current needs it in this row, and a version closed months ago must keep naming its
    -- distributor even if the source receipt is later archived. No FK for that reason.
    distributor_id      text NOT NULL,
    distributor_sku     text NOT NULL,
    mpn                 text,
    manufacturer        text,
    brand               text,
    item_description    text NOT NULL,

    -- Classification. NULL means unclassified, which is a reportable state, not a default
    -- bucket. Soft reference to bp_category_master.unspsc_code -- it CANNOT be an FK; see
    -- the note at the top of this file.
    unspsc_code         text,

    -- Unit economics. pack_size/pack_uom are what let a catalog "box of 10" reconcile with
    -- a history line of "10 each" -- the reconciliation the audit found has no basis today.
    unit_of_measure     text,                 -- soft ref: bp_uom_canonical.uom_code
    pack_size           numeric,
    pack_uom            text,
    currency            char(3) NOT NULL,     -- NOT NULL: see bp_sales_quote_line note

    list_price          numeric(18,4),        -- distributor list / RRP
    cost_price          numeric(18,4),        -- GAP 2. our buy price at this quantity floor
    cost_basis          text,                 -- contract | spot | promotion | unknown

    -- Availability. lead_time_days is meaningful when stock_qty is 0 and useless otherwise;
    -- both are point-in-time and belong to this version of the row.
    availability_status text,                 -- in_stock | backorder | special_order | discontinued
    stock_qty           numeric,
    lead_time_days      integer,

    -- Lifecycle. This is what makes an end-of-life justification defensible instead of a
    -- sales assertion, and it is the single field the audit found no home for anywhere.
    lifecycle_status    text,                 -- active | end_of_sale | end_of_life | superseded
    end_of_sale_date    date,
    end_of_life_date    date,

    valid_from          timestamptz NOT NULL DEFAULT now(),
    valid_to            timestamptz,          -- NULL = current version
    recorded_at         timestamptz NOT NULL DEFAULT now()
);

-- One current version per SKU per distributor. A partial index rather than a plain UNIQUE:
-- history rows (valid_to IS NOT NULL) may repeat the key as often as the price changes.
CREATE UNIQUE INDEX IF NOT EXISTS ux_bp_catalog_item_current
    ON proc.bp_catalog_item (distributor_id, distributor_sku)
    WHERE valid_to IS NULL;

CREATE INDEX IF NOT EXISTS ix_bp_catalog_item_mpn
    ON proc.bp_catalog_item (mpn) WHERE mpn IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_bp_catalog_item_unspsc
    ON proc.bp_catalog_item (unspsc_code);
CREATE INDEX IF NOT EXISTS ix_bp_catalog_item_lifecycle
    ON proc.bp_catalog_item (lifecycle_status) WHERE valid_to IS NULL;
CREATE INDEX IF NOT EXISTS ix_bp_catalog_item_description
    ON proc.bp_catalog_item USING btree (lower(item_description));


-- Volume-break cost. The commonest distributor cost structure there is, and it changes the
-- margin on a quote line the moment quantity crosses a break -- so a margin computed from
-- bp_catalog_item.cost_price alone is wrong at exactly the quantities that matter to a
-- large-value quote. Absent tiers mean flat cost, which is a fact, not a fallback.
CREATE TABLE IF NOT EXISTS proc.bp_catalog_cost_tier (
    catalog_item_id  bigint NOT NULL
        REFERENCES proc.bp_catalog_item (catalog_item_id) ON DELETE CASCADE,
    min_quantity     numeric NOT NULL,
    cost_price       numeric(18,4) NOT NULL,
    currency         char(3) NOT NULL,
    PRIMARY KEY (catalog_item_id, min_quantity)
);


-- Relations the DISTRIBUTOR asserts, and only those.
--
-- "This SKU replaces that one" is in the feed and is defensible to a customer. "Customers
-- who bought this also bought that" is inference, belongs to the affinity work (audit gap 6),
-- and does not go in this table -- if it did, an inferred edge would become indistinguishable
-- from a manufacturer's own succession notice the moment it was queried.
CREATE TABLE IF NOT EXISTS proc.bp_catalog_item_relation (
    relation_id       bigserial PRIMARY KEY,
    from_sku          text NOT NULL,
    to_sku            text NOT NULL,
    distributor_id    text NOT NULL,
    relation_type     text NOT NULL,          -- replaced_by | upgrade_of | refill_of | accessory_of | requires
    asserted_by       text NOT NULL,          -- distributor_feed | manufacturer_notice | human
    source_id         bigint REFERENCES proc.bp_catalog_source (source_id) ON DELETE SET NULL,
    created_date      timestamptz NOT NULL DEFAULT now(),
    UNIQUE (distributor_id, from_sku, to_sku, relation_type)
);

CREATE INDEX IF NOT EXISTS ix_bp_catalog_item_relation_from
    ON proc.bp_catalog_item_relation (distributor_id, from_sku);
CREATE INDEX IF NOT EXISTS ix_bp_catalog_item_relation_to
    ON proc.bp_catalog_item_relation (distributor_id, to_sku);


-- What a catalog SKU resolves to in a customer's purchase history.
--
-- Kept OUT of bp_catalog_item deliberately: a match is a claim about two datasets, it has a
-- confidence and a method, and it can be wrong in a way a catalog row cannot. Separating it
-- means a rejected match leaves a record instead of leaving no trace.
CREATE TABLE IF NOT EXISTS proc.bp_catalog_item_match (
    match_id         bigserial PRIMARY KEY,
    distributor_id   text NOT NULL,
    distributor_sku  text NOT NULL,
    item_id          text NOT NULL,           -- soft ref: bp_*_line_items_trgt.item_id
    match_method     text NOT NULL,           -- mpn_exact | sku_exact | description_fuzzy | human
    confidence       numeric(5,4),            -- NULL for human and exact matches
    status           text NOT NULL DEFAULT 'proposed',  -- proposed | confirmed | rejected
    confirmed_by     text,
    confirmed_at     timestamptz,
    created_date     timestamptz NOT NULL DEFAULT now(),
    UNIQUE (distributor_id, distributor_sku, item_id)
);

CREATE INDEX IF NOT EXISTS ix_bp_catalog_item_match_item
    ON proc.bp_catalog_item_match (item_id) WHERE status = 'confirmed';

COMMIT;
