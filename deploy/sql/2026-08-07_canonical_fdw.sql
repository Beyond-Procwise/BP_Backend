-- Canonical master data, read-only, via postgres_fdw onto uicanvas.
--
-- uicanvas is the authoritative source for reference data. BP_Backend runs
-- against bp_testdb (DB_NAME in .env), which holds almost none of it:
--
--                              uicanvas   bp_testdb (before this)
--   category taxonomy          246 rows          0
--   product master             186 rows      absent
--   category -> product map     21 rows      absent
--   contracts                3,051 rows          0
--
-- That gap is why the corpus has had "no category dimension": the dimension
-- exists, in another database, unwired.
--
-- A foreign-data wrapper rather than a copy or a scheduled sync, so there is
-- exactly one place the canonical data lives. A copy would immediately raise
-- the question of which side is right when they disagree, and that question
-- has a cost every time anyone asks it.
--
-- IMPORTANT -- what this deliberately does NOT do:
--   * It does not touch proc.bp_category. That table is a DIFFERENT, degraded
--     thing -- (item_description, category), flat, no hierarchy, no UNSPSC --
--     and silently replacing it would break whatever still reads it. The
--     canonical taxonomy is exposed under a new name instead.
--   * It does not make the data writable. The server carries
--     updatable 'false', so a write through these views is refused by the
--     wrapper rather than quietly modifying the authoritative source.
--
-- Credentials are NOT stored in this file. Apply with psql variables:
--
--   set -a && . ./.env && set +a
--   PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" \
--     -d bp_testdb -v ON_ERROR_STOP=1 \
--     -v fdw_host="$DB_HOST" -v fdw_port="$DB_PORT" \
--     -v fdw_user="$DB_USER" -v fdw_password="$DB_PASSWORD" \
--     -f deploy/sql/2026-08-07_canonical_fdw.sql
--
-- Idempotent and reversible.
BEGIN;

CREATE EXTENSION IF NOT EXISTS postgres_fdw;

-- The RDS cluster endpoint, so the mapping follows a failover rather than
-- pinning to whichever instance happened to be the writer today.
CREATE SERVER IF NOT EXISTS uicanvas_srv
    FOREIGN DATA WRAPPER postgres_fdw
    OPTIONS (host :'fdw_host', port :'fdw_port', dbname 'uicanvas', updatable 'false');

CREATE USER MAPPING IF NOT EXISTS FOR CURRENT_USER
    SERVER uicanvas_srv
    OPTIONS (user :'fdw_user', password :'fdw_password');

-- Foreign tables live in their own schema so it is obvious at a glance that
-- these are not local data. Dropped and re-imported to stay idempotent: the
-- schema holds no data of its own, only pointers, so nothing is lost. CASCADE
-- takes the dependent views with it and they are recreated below in the same
-- transaction.
DROP SCHEMA IF EXISTS canonical CASCADE;
CREATE SCHEMA canonical;

COMMENT ON SCHEMA canonical IS
    'Read-only foreign tables onto uicanvas (authoritative master data). '
    'Nothing here is local; do not write.';

IMPORT FOREIGN SCHEMA proc
    LIMIT TO (bp_category, bp_products, bp_category_product_mapping,
              supplier, bp_contracts)
    FROM SERVER uicanvas_srv
    INTO canonical;

-- Named views in proc, so callers do not need to know the data is remote.
-- The _master suffix distinguishes the canonical article from the local
-- same-named tables that are empty or degraded.
CREATE OR REPLACE VIEW proc.bp_category_master AS
    SELECT * FROM canonical.bp_category;

CREATE OR REPLACE VIEW proc.bp_product_master AS
    SELECT * FROM canonical.bp_products;

CREATE OR REPLACE VIEW proc.bp_category_product_map AS
    SELECT * FROM canonical.bp_category_product_mapping;

CREATE OR REPLACE VIEW proc.bp_supplier_master AS
    SELECT * FROM canonical.supplier;

CREATE OR REPLACE VIEW proc.bp_contract_master AS
    SELECT * FROM canonical.bp_contracts;

COMMENT ON VIEW proc.bp_category_master IS
    'Authoritative 5-level category taxonomy with UNSPSC codes, from uicanvas. '
    'NOT the same as proc.bp_category, which is a flat (item_description, category) table.';
COMMENT ON VIEW proc.bp_product_master IS
    'Authoritative product master from uicanvas, including unit_of_measure -- '
    'the canonical unit vocabulary the extraction corpus lacks.';

COMMIT;
