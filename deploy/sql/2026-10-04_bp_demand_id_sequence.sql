-- 2026-10-04  One id for a demand, minted by the database.
--
-- THE RULING (SpendIQ Demand Intake, engine.js): a draft has ONE id, the server mints it, and it
-- is the demand's own id from the first turn to the award.
--
-- What it replaces, and why a sequence rather than the clock:
--
--   · the intake conversation minted 'DRAFT-' || last 6 digits of Date.now() in the browser,
--   · submitting minted 'DM-' || last FOUR digits — a space that wraps every ten seconds,
--   · and the gateway minted a third, 'DM-' || last 6 digits, because the controller passes the
--     PAYLOAD to createDemand and the payload carries no id, so the browser's id never arrived.
--     The demand the register showed and the row that was stored had different ids, and the
--     follow-up POST /spendiq/demand/<id>/messages then updated WHERE demand_id = an id no row
--     had: nought rows changed, no error, every message lost.
--
-- A clock-derived id has no per-user namespace and wraps; every other id in proc comes from a
-- sequence, and this one now does too.
--
-- STARTS AT 3000, clear of the five live rows in bp_sqldb (DM-2041 … DM-2055, highest suffix
-- 2055). bp_testdb holds none. Nothing existing can collide.
--
-- Idempotent. Safe to re-run: CREATE SEQUENCE IF NOT EXISTS leaves an advanced sequence where it
-- is rather than rewinding it onto ids already handed out.

BEGIN;

CREATE SEQUENCE IF NOT EXISTS proc.bp_demand_id_seq START WITH 3000 INCREMENT BY 1;

-- The register reads every demand that is not a draft, so the status is worth an index: a draft
-- row exists for every conversation anybody starts, finished or not.
CREATE INDEX IF NOT EXISTS ix_bp_demand_status ON proc.bp_demand (status);

COMMIT;
