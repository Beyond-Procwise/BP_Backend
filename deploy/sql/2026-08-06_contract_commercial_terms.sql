-- 2026-08-06 Phase 1a: commercial terms on the contract record.
--
-- Phase 0 established that escalator, term length, billing cadence, amendment
-- reference and document version exist nowhere in this codebase — not in a
-- schema, not in a column, not in a grep. Every Phase 3.2 check (escalator
-- conformance, rate drift against the governing amendment, co-termination) and
-- the Phase 1b CommercialFact Term group need them.
--
-- Columns are added to BOTH bp_contract_raw and bp_contracts because contract
-- promotion goes raw -> bp_contracts directly with no _stg layer
-- (promotion.py:29 maps "contract" to that pair).
--
-- Numeric precision follows the existing bp_contracts convention
-- (total_contract_value is numeric(18,2)). Percentages get (9,4) so a rate of
-- 3.125% survives without rounding.
--
-- NOT converted, NOT derived: term_months records a term the document STATES in
-- months. A term stated in years is left NULL here — converting it would be
-- arithmetic performed by the extractor, and the derived value belongs to
-- Phase 1b where it can carry its own basis and provenance.
BEGIN;

ALTER TABLE proc.bp_contract_raw
    ADD COLUMN IF NOT EXISTS term_months        NUMERIC(9,2),
    ADD COLUMN IF NOT EXISTS billing_frequency  TEXT,
    ADD COLUMN IF NOT EXISTS escalator_pct      NUMERIC(9,4),
    ADD COLUMN IF NOT EXISTS escalator_basis    TEXT,
    ADD COLUMN IF NOT EXISTS escalator_cap_pct  NUMERIC(9,4),
    ADD COLUMN IF NOT EXISTS amendment_ref      TEXT,
    ADD COLUMN IF NOT EXISTS document_version   TEXT;

ALTER TABLE proc.bp_contracts
    ADD COLUMN IF NOT EXISTS term_months        NUMERIC(9,2),
    ADD COLUMN IF NOT EXISTS billing_frequency  TEXT,
    ADD COLUMN IF NOT EXISTS escalator_pct      NUMERIC(9,4),
    ADD COLUMN IF NOT EXISTS escalator_basis    TEXT,
    ADD COLUMN IF NOT EXISTS escalator_cap_pct  NUMERIC(9,4),
    ADD COLUMN IF NOT EXISTS amendment_ref      TEXT,
    ADD COLUMN IF NOT EXISTS document_version   TEXT;

-- The amendment chain is walked parent-first by every integrity check.
CREATE INDEX IF NOT EXISTS ix_bp_contracts_parent_contract_id
    ON proc.bp_contracts (parent_contract_id);

COMMIT;
