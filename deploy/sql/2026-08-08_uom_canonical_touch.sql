-- Keep proc.bp_uom_canonical.recorded_at true on every write.
--
-- Running processes cache the unit vocabulary and notice changes by probing
-- (count, max(recorded_at)) over the active rows. That probe is only as
-- reliable as recorded_at, and relying on every writer to remember to set it
-- is the weakest link in the chain: a confirmation applied by hand, or by a
-- future script that forgets, would leave recorded_at stale and the change
-- would silently not propagate until the TTL expired. The symptom -- "I
-- confirmed the unit and it did not take effect" -- would be blamed on the
-- cache rather than on the missing timestamp.
--
-- A BEFORE trigger removes the requirement to remember.
--
-- Idempotent and reversible.
BEGIN;

CREATE OR REPLACE FUNCTION proc.bp_uom_canonical_touch()
RETURNS trigger AS $$
BEGIN
    NEW.recorded_at := now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS bp_uom_canonical_touch ON proc.bp_uom_canonical;
CREATE TRIGGER bp_uom_canonical_touch
    BEFORE INSERT OR UPDATE ON proc.bp_uom_canonical
    FOR EACH ROW
    EXECUTE FUNCTION proc.bp_uom_canonical_touch();

COMMIT;
