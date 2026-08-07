-- The unit vocabulary as data rather than as a Python dict.
--
-- Phase 1b hand-typed its unit map in src/services/facts/uom.py, having
-- concluded (B3) that no dictionary was needed. That conclusion was drawn from
-- the two databases where the canonical data is absent. Measured against
-- proc.bp_product_master once the uicanvas bridge existed, the hand-typed map
-- covered 63 of 81 products but only 7 of 18 distinct canonical values -- it
-- was already drifting from reference data that predates the phase.
--
-- Two things this table changes:
--
--   * A new unit can be added without a deploy, and
--   * a unit observed but not recognised is RECORDED as 'proposed' rather than
--     silently becoming UOM_UNMAPPED for ever. That is the queue a human
--     confirms from, and it is the reason the drift above went unnoticed:
--     nothing anywhere counted what the normaliser was refusing.
--
-- The seed below is exactly the content of uom.py's _CANONICAL/_ALIASES, and a
-- test asserts the two agree. Duplication between code and data is only safe
-- when something fails loudly the moment they diverge.
--
-- status:
--   active   -- usable for normalisation
--   proposed -- observed in real data, awaiting human confirmation; NEVER used
--               to normalise, because an unconfirmed guess that silently
--               starts resolving is indistinguishable from a real unit
--   rejected -- confirmed NOT a unit (payment terms, scope text)
--
-- Additive, idempotent, reversible.
BEGIN;

CREATE TABLE IF NOT EXISTS proc.bp_uom_canonical (
    uom_code          text PRIMARY KEY,
    tenant_id         text NOT NULL DEFAULT 'default',
    dimension         text,
    aliases           text[] NOT NULL DEFAULT '{}',
    -- Only time units convert to a common basis. Count, mass and length have
    -- no shared denominator to express here.
    factor_days       numeric,
    factor_convention text,
    -- Can a rate legitimately be quoted *per* this unit? Distinguishes a
    -- billing basis from a packaging descriptor.
    is_billing_basis  boolean NOT NULL DEFAULT true,
    status            text NOT NULL DEFAULT 'proposed',
    source            text,
    observed_count    integer NOT NULL DEFAULT 0,
    first_observed_at timestamptz,
    last_observed_at  timestamptz,
    confirmed_by      text,
    confirmed_at      timestamptz,

    valid_from        timestamptz NOT NULL DEFAULT now(),
    valid_to          timestamptz,
    recorded_at       timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT ck_bp_uom_canonical_status CHECK (
        status IN ('active', 'proposed', 'rejected')),
    CONSTRAINT ck_bp_uom_canonical_dimension CHECK (
        dimension IS NULL OR dimension IN (
            'count', 'time', 'mass', 'length')),
    -- An active unit must say what kind of thing it measures; a proposed one
    -- legitimately does not know yet.
    CONSTRAINT ck_bp_uom_canonical_active_has_dimension CHECK (
        status <> 'active' OR dimension IS NOT NULL),
    -- A conversion factor without its convention is an unattributable
    -- assumption -- 'month = 30 days' is a choice, not a fact.
    CONSTRAINT ck_bp_uom_canonical_factor_needs_convention CHECK (
        factor_days IS NULL OR factor_convention IS NOT NULL
        OR dimension <> 'time' OR factor_days IN (1, 7))
);

CREATE INDEX IF NOT EXISTS ix_bp_uom_canonical_status
    ON proc.bp_uom_canonical (status);
CREATE INDEX IF NOT EXISTS ix_bp_uom_canonical_dimension
    ON proc.bp_uom_canonical (dimension);

-- Seed: the active vocabulary, matching uom.py exactly.
-- ON CONFLICT DO NOTHING so a re-run never clobbers a human confirmation.
INSERT INTO proc.bp_uom_canonical
    (uom_code, dimension, aliases, factor_days, factor_convention,
     is_billing_basis, status, source)
VALUES
    -- count
    ('each',     'count', ARRAY['ea','eaches','unit','units'],  NULL, NULL, true, 'active', 'seed'),
    ('case',     'count', ARRAY['cases','cs'],                  NULL, NULL, true, 'active', 'seed'),
    ('pack',     'count', ARRAY['packs','pk'],                  NULL, NULL, true, 'active', 'seed'),
    ('box',      'count', ARRAY['boxes'],                       NULL, NULL, true, 'active', 'seed'),
    ('seat',     'count', ARRAY['seats'],                       NULL, NULL, true, 'active', 'seed'),
    ('licence',  'count', ARRAY['licences','license','licenses','lic'], NULL, NULL, true, 'active', 'seed'),
    ('shipment', 'count', ARRAY['shipments'],                   NULL, NULL, true, 'active', 'seed'),
    -- count, observed in proc.bp_product_master
    ('set',      'count', ARRAY['sets'],                        NULL, NULL, true, 'active', 'bp_product_master'),
    ('sheet',    'count', ARRAY['sheets'],                      NULL, NULL, true, 'active', 'bp_product_master'),
    ('roll',     'count', ARRAY['rolls'],                       NULL, NULL, true, 'active', 'bp_product_master'),
    ('pen',      'count', ARRAY['pens'],                        NULL, NULL, true, 'active', 'bp_product_master'),
    ('module',   'count', ARRAY['modules'],                     NULL, NULL, true, 'active', 'bp_product_master'),
    -- time
    ('hour',    'time', ARRAY['hr','hrs','hours','hourly'],     NULL,  NULL, true, 'active', 'seed'),
    ('day',     'time', ARRAY['days','dy','daily'],                1,  NULL, true, 'active', 'seed'),
    ('week',    'time', ARRAY['weeks','wk','wks','weekly'],        7,  NULL, true, 'active', 'seed'),
    ('month',   'time', ARRAY['mo','mth','mths','months','monthly'], 30, 'CALENDAR_CONVENTION_30D_365D', true, 'active', 'seed'),
    ('quarter', 'time', ARRAY['quarters','qtr','quarterly'],      90, 'CALENDAR_CONVENTION_30D_365D', true, 'active', 'bp_product_master'),
    ('year',    'time', ARRAY['yr','yrs','years','annum','per annum'], 365, 'CALENDAR_CONVENTION_30D_365D', true, 'active', 'seed'),
    -- mass / length
    ('tonne', 'mass',   ARRAY['t','mt','tonnes','tonnes(metric)','metric tonne','ton','tons'], NULL, NULL, true, 'active', 'seed'),
    ('metre', 'length', ARRAY['m','mtr','mtrs','metres','meter','meters'], NULL, NULL, true, 'active', 'seed')
ON CONFLICT (uom_code) DO NOTHING;

-- Service-engagement bases seen in proc.bp_product_master. Recorded so they
-- are visible and countable, but deliberately NOT active: these are lump-sum
-- engagement types, not units. Treating 'retainer' as a unit would let two
-- retainers be compared as if they were rates per identical thing -- exactly
-- the error measure_role exists to prevent. Whether they become an
-- 'engagement' dimension or resolve to extended_line facts is an open
-- modelling decision, and until it is taken they stay refused.
INSERT INTO proc.bp_uom_canonical
    (uom_code, dimension, is_billing_basis, status, source)
VALUES
    ('service',   NULL, false, 'proposed', 'bp_product_master'),
    ('programme', NULL, false, 'proposed', 'bp_product_master'),
    ('retainer',  NULL, false, 'proposed', 'bp_product_master'),
    ('audit',     NULL, false, 'proposed', 'bp_product_master')
ON CONFLICT (uom_code) DO NOTHING;

COMMENT ON TABLE proc.bp_uom_canonical IS
    'The unit vocabulary. Only status=''active'' rows normalise; ''proposed'' '
    'rows are observed-but-unconfirmed and must never resolve silently.';

COMMIT;
