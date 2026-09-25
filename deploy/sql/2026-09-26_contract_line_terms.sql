-- 2026-09-26 Contract line terms: the rate card a contract sets, line by line.
--
-- A contract is extracted today as a header only (extraction_schemas/contract.yaml has
-- line_items: null), so nothing records what it allows per line — the day rate, the price
-- cap, the item it includes at no charge. The Document match view (UI BACKEND_GAPS.md,
-- "Document match") measures an invoice against exactly that, and without it every
-- purchase is measured against its PO instead.
--
-- Two tables, mirroring every other document type's raw -> promoted pair. A contract
-- promotes straight to proc.bp_contracts (promotion.py _RAW_TO_STG), so its lines promote
-- straight to proc.bp_contract_line_items — there is no _stg/_trgt layer to mirror.
--
--   term_basis  'rate'     unit_price is charged per unit (day rate, per seat)
--               'cap'      unit_price is the most that may be charged (per unit, or for the
--                          line when no quantity applies)
--               'included' the item is covered at no extra charge (freight, support)
--
-- Additive and idempotent. Run BEFORE shipping the contract.yaml line_items block:
-- load_all_schemas() checks every db_column against information_schema at start-up.
BEGIN;

CREATE TABLE IF NOT EXISTS proc.bp_contract_line_items_raw (
    line_raw_id      BIGSERIAL PRIMARY KEY,
    raw_id           BIGINT  NOT NULL,
    line_number      INTEGER NOT NULL,
    item_description TEXT,
    term_basis       TEXT,
    unit_price       NUMERIC,
    unit_of_measure  TEXT,
    currency         VARCHAR,
    qualifier        TEXT,
    UNIQUE (raw_id, line_number)
);
CREATE INDEX IF NOT EXISTS idx_contract_line_raw_raw
    ON proc.bp_contract_line_items_raw (raw_id);

CREATE TABLE IF NOT EXISTS proc.bp_contract_line_items (
    contract_line_id   TEXT PRIMARY KEY,          -- "<contract_id>-L<line_number>" (promotion.py)
    contract_id        TEXT NOT NULL,
    line_number        INTEGER,
    item_description   TEXT,
    term_basis         TEXT,
    unit_price         NUMERIC,
    unit_of_measure    TEXT,
    currency           VARCHAR,
    qualifier          TEXT,
    created_date       TIMESTAMP DEFAULT now(),
    created_by         TEXT,
    last_modified_by   TEXT,
    last_modified_date TIMESTAMP,
    CONSTRAINT bp_contract_line_items_basis_chk
        CHECK (term_basis IS NULL OR term_basis IN ('rate', 'cap', 'included'))
);
-- Every read is "the terms of this contract" (gateway spendiq.match.ts).
CREATE INDEX IF NOT EXISTS ix_bp_contract_line_items_contract_id
    ON proc.bp_contract_line_items (contract_id);

-- promotion.py upserts a contract with ON CONFLICT (contract_id), which needs a unique
-- key that bp_contracts never had: the first contract ever extracted (2026-09-25) failed
-- to promote with "no unique or exclusion constraint matching the ON CONFLICT
-- specification", and its lines with it. The table was empty on both databases.
CREATE UNIQUE INDEX IF NOT EXISTS ix_bp_contracts_contract_id
    ON proc.bp_contracts (contract_id);

COMMIT;
