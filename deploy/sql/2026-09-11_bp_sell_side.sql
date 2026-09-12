-- Sell-side data model: accounts, sales opportunities, outbound quotes. Gap 3.
-- Spec: docs/superpowers/specs/2026-09-09-reseller-catalog-and-sell-side-design.md
-- Apply AFTER 2026-09-11_bp_catalog.sql: bp_sales_opportunity and bp_sales_quote_line reference bp_catalog_item.
--
-- The buy-side model already here answers "what did we pay, and could we have paid less".
-- Its value axis is bp_opportunity.financial_impact_gbp -- money NOT spent. A reseller's
-- axis is money EARNED, and the two are not the same number wearing different labels, so
-- these are separate tables rather than columns bolted onto bp_opportunity.
--
-- The phase vocabulary is NOT invented here. It is the ladder already seeded in the UI at
-- src/lib/processTaxonomy/salesLifecycle.js (Opportunity -> Margin -> Approval, nine
-- sub-processes). These columns are the backend that ladder has never had.

BEGIN;

-- The customer we sell TO.
--
-- Deliberately not a row in bp_supplier. The same legal entity is often both -- we buy from
-- Ingram and we sell to a company that also quotes us -- but a supplier row carries bank
-- details we pay INTO and a risk score about THEIR delivery, and an account carries credit
-- we extend and a probability THEY buy. Conflating them makes both fields ambiguous.
-- also_supplier_id records the overlap without erasing the distinction.
CREATE TABLE IF NOT EXISTS proc.bp_account (
    account_id           text PRIMARY KEY,
    tenant_id            text,
    account_name         text NOT NULL,
    trading_name         text,
    also_supplier_id     text
        REFERENCES proc.bp_supplier (supplier_id) ON DELETE SET NULL,
    registration_number  text,
    vat_number           text,
    country              text,
    default_currency     char(3),
    payment_terms        text,
    credit_limit_amount  numeric(18,2),
    account_owner_email  text,                -- the AM who owns it
    account_status       text NOT NULL DEFAULT 'active',   -- active | dormant | closed
    created_date         timestamptz NOT NULL DEFAULT now(),
    last_modified_date   timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_bp_account_owner
    ON proc.bp_account (account_owner_email);

-- A separate table rather than bp_supplier's contact_name_1 / contact_name_2 columns.
-- That pattern caps the world at two people and has no way to say which of them signs.
CREATE TABLE IF NOT EXISTS proc.bp_account_contact (
    contact_id     bigserial PRIMARY KEY,
    account_id     text NOT NULL REFERENCES proc.bp_account (account_id) ON DELETE CASCADE,
    contact_name   text NOT NULL,
    contact_role   text,
    contact_email  text,
    contact_phone  text,
    is_primary     boolean NOT NULL DEFAULT FALSE,
    created_date   timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_bp_account_contact_account
    ON proc.bp_account_contact (account_id);


-- Which of this account's purchase history we can actually see, and from where.
--
-- A reseller sees their own invoices to a customer completely and the customer's spend with
-- everyone else only if the customer shares it. An upsell case built on the first is a fact;
-- one built on the second is only as good as what was shared. Recording which is which is
-- what stops "they buy nothing from us in networking" from being read as "they buy nothing".
CREATE TABLE IF NOT EXISTS proc.bp_account_history_scope (
    account_id      text NOT NULL REFERENCES proc.bp_account (account_id) ON DELETE CASCADE,
    source_kind     text NOT NULL,            -- our_invoices | customer_shared | third_party
    covers_from     date,
    covers_to       date,
    completeness    text NOT NULL,            -- complete | partial | unknown
    note            text,
    created_date    timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (account_id, source_kind)
);


-- A sell-side opportunity: one catalog SKU we believe one account should be buying.
CREATE TABLE IF NOT EXISTS proc.bp_sales_opportunity (
    sales_opportunity_id  bigserial PRIMARY KEY,
    tenant_id             text,
    account_id            text NOT NULL REFERENCES proc.bp_account (account_id) ON DELETE CASCADE,
    catalog_item_id       bigint REFERENCES proc.bp_catalog_item (catalog_item_id) ON DELETE SET NULL,

    opportunity_type      text NOT NULL,      -- upsell | cross_sell | upgrade | refill | switch_supplier

    -- Value. Native amount plus the currency it is IN -- the audit found currency populated
    -- on 0 of 55,483 existing line items because it lived only on the header. Not repeated here.
    currency              char(3) NOT NULL,
    expected_quantity     numeric,
    expected_revenue      numeric(18,2),
    expected_cost         numeric(18,2),
    expected_margin       numeric(18,2),      -- stored, not derived: the tier that produced it may move
    margin_pct            numeric(7,4),

    -- NULL until bp_sales_quote_outcome holds enough closed quotes to calibrate against.
    -- An uncalibrated 0.5 would be indistinguishable from a measured one.
    win_probability       numeric(5,4),
    win_probability_basis text,               -- calibrated | manual | NULL

    -- The seeded ladder. Verbatim ids from salesLifecycle.js, e.g. 'sales.opportunity' and
    -- 'sales.opportunity.quote-drafted'.
    phase_id              text,
    subprocess_id         text,
    outcome               text NOT NULL DEFAULT 'open',   -- open | won | lost | withdrawn

    detector_type         text,               -- which analysis proposed it
    reason_codes          text[],
    created_date          timestamptz NOT NULL DEFAULT now(),
    last_modified_date    timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_bp_sales_opportunity_account
    ON proc.bp_sales_opportunity (account_id);
CREATE INDEX IF NOT EXISTS ix_bp_sales_opportunity_outcome
    ON proc.bp_sales_opportunity (outcome);


-- The evidence under one opportunity, kept as rows rather than a prose blob so a customer-
-- facing justification can cite a benchmark run or an end-of-life date by id and a reviewer
-- can follow it back. An opportunity with no justification row is not quotable.
CREATE TABLE IF NOT EXISTS proc.bp_sales_justification (
    justification_id      bigserial PRIMARY KEY,
    sales_opportunity_id  bigint NOT NULL
        REFERENCES proc.bp_sales_opportunity (sales_opportunity_id) ON DELETE CASCADE,
    kind                  text NOT NULL,      -- price_gap | benchmark | end_of_life | usage_cadence | coverage_gap
    claim                 text NOT NULL,      -- the sentence shown to the customer
    evidence_ref          text,               -- benchmark run id, catalog_item_id, invoice_id...
    evidence_value        numeric,
    customer_safe         boolean NOT NULL DEFAULT TRUE,  -- FALSE never leaves the building
    created_date          timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_bp_sales_justification_opportunity
    ON proc.bp_sales_justification (sales_opportunity_id);


-- The outbound quote. The artifact this platform has never had -- every existing "quote"
-- row (bp_quote_trgt, 21,049 of them) is a quote a supplier sent US.
CREATE TABLE IF NOT EXISTS proc.bp_sales_quote (
    sales_quote_id     bigserial PRIMARY KEY,
    tenant_id          text,
    quote_ref          text NOT NULL,         -- the reference the customer sees
    account_id         text NOT NULL REFERENCES proc.bp_account (account_id) ON DELETE RESTRICT,
    contact_id         bigint REFERENCES proc.bp_account_contact (contact_id) ON DELETE SET NULL,
    currency           char(3) NOT NULL,

    quote_date         date NOT NULL,
    valid_until        date NOT NULL,         -- NOT NULL: a quote priced off a moving cost
                                              -- feed that never expires is a standing loss

    -- Header totals are stored, not summed on read. A line edited after issue must not
    -- retroactively change what the customer was sent.
    total_ex_tax       numeric(18,2),
    total_cost         numeric(18,2),         -- INTERNAL. never rendered customer-facing.
    total_margin       numeric(18,2),         -- INTERNAL.
    margin_pct         numeric(7,4),          -- INTERNAL.

    phase_id           text,                  -- seeded sales ladder, as above
    subprocess_id      text,
    status             text NOT NULL DEFAULT 'draft',  -- draft | in_review | approved | issued | expired | superseded
    supersedes_id      bigint REFERENCES proc.bp_sales_quote (sales_quote_id) ON DELETE SET NULL,

    -- Set only by the approvals surface, from the authenticated token. Never from a body.
    approved_by        text,
    approved_at        timestamptz,
    issued_at          timestamptz,

    created_by         text,
    created_date       timestamptz NOT NULL DEFAULT now(),
    last_modified_date timestamptz NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, quote_ref)
);

CREATE INDEX IF NOT EXISTS ix_bp_sales_quote_account
    ON proc.bp_sales_quote (account_id);
CREATE INDEX IF NOT EXISTS ix_bp_sales_quote_status
    ON proc.bp_sales_quote (status);


-- One quoted line.
--
-- unit_cost and list_price_at_quote are SNAPSHOTS, copied from bp_catalog_item at draft time
-- and never joined live. The catalog is versioned precisely so the join is possible; taking
-- it anyway would mean a quote sent in March reports April's margin, and the number we
-- defended to a customer would stop being the number on their desk.
CREATE TABLE IF NOT EXISTS proc.bp_sales_quote_line (
    sales_quote_line_id   bigserial PRIMARY KEY,
    sales_quote_id        bigint NOT NULL
        REFERENCES proc.bp_sales_quote (sales_quote_id) ON DELETE CASCADE,
    line_no               integer NOT NULL,
    sales_opportunity_id  bigint REFERENCES proc.bp_sales_opportunity (sales_opportunity_id) ON DELETE SET NULL,
    catalog_item_id       bigint REFERENCES proc.bp_catalog_item (catalog_item_id) ON DELETE RESTRICT,

    distributor_sku       text,
    mpn                   text,
    item_description      text NOT NULL,      -- as quoted, not as re-read from catalog later
    quantity              numeric NOT NULL,
    unit_of_measure       text,
    currency              char(3) NOT NULL,   -- on the LINE, not only the header

    list_price_at_quote   numeric(18,4),
    unit_price            numeric(18,4) NOT NULL,   -- what we sell it for
    discount_pct          numeric(7,4),
    line_total            numeric(18,2) NOT NULL,

    unit_cost             numeric(18,4),      -- INTERNAL. snapshot; tier applied at this qty.
    cost_tier_applied     numeric,            -- the min_quantity break used, for audit
    line_margin           numeric(18,2),      -- INTERNAL.
    line_margin_pct       numeric(7,4),       -- INTERNAL.

    justification_id      bigint REFERENCES proc.bp_sales_justification (justification_id) ON DELETE SET NULL,
    created_date          timestamptz NOT NULL DEFAULT now(),
    UNIQUE (sales_quote_id, line_no)
);

CREATE INDEX IF NOT EXISTS ix_bp_sales_quote_line_quote
    ON proc.bp_sales_quote_line (sales_quote_id);


-- Won or lost, and why. Included in this gap rather than deferred because it is the ONLY
-- thing that can ever populate bp_sales_opportunity.win_probability with a measured number.
-- Ship the model without it and the probability column stays NULL forever by construction.
CREATE TABLE IF NOT EXISTS proc.bp_sales_quote_outcome (
    sales_quote_id   bigint PRIMARY KEY
        REFERENCES proc.bp_sales_quote (sales_quote_id) ON DELETE CASCADE,
    outcome          text NOT NULL,           -- won | lost | expired | withdrawn
    outcome_date     date NOT NULL,
    lost_reason      text,                    -- price | lead_time | incumbent | no_budget | spec | other
    competitor_name  text,
    won_value        numeric(18,2),
    won_margin       numeric(18,2),
    recorded_by      text,
    created_date     timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_bp_sales_quote_outcome_date
    ON proc.bp_sales_quote_outcome (outcome_date);

COMMIT;
