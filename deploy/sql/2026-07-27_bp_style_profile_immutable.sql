BEGIN;
-- deploy/sql/2026-07-27_bp_style_profile_immutable.sql
--
-- Phase 1. Invariant 5: profiles are immutable once approved.
--
-- Enforced in the database rather than in the repository, because the repository is not
-- the only thing that will ever hold a connection to this table. A migration, a support
-- script or a psql session at 2am is exactly when this rule matters most, and none of
-- them import Python.
--
-- What is frozen: profile_json, and the identity of the row (user_ref, intent, version).
-- What stays mutable: is_active and state, because standing a profile down to SUPERSEDED
-- is the normal, intended lifecycle. Freezing those would make approval impossible.
--
-- SUPERSEDED is covered as well as APPROVED. A superseded profile was approved once, and
-- the draft rows that cite it must keep pointing at the rules that actually produced them.
-- Editing history is worse than deleting it, because it still looks trustworthy.

CREATE OR REPLACE FUNCTION proc.bp_style_profile_freeze_approved()
RETURNS TRIGGER AS $$
BEGIN
    IF OLD.state NOT IN ('APPROVED', 'SUPERSEDED') THEN
        RETURN NEW;
    END IF;

    IF NEW.profile_json IS DISTINCT FROM OLD.profile_json THEN
        RAISE EXCEPTION
            'profile_json is immutable once approved (profile_id=%, state=%). '
            'Compile a new version and approve it instead.',
            OLD.profile_id, OLD.state
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;

    IF NEW.user_ref IS DISTINCT FROM OLD.user_ref
       OR NEW.intent IS DISTINCT FROM OLD.intent
       OR NEW.version IS DISTINCT FROM OLD.version THEN
        RAISE EXCEPTION
            'the identity of an approved profile is immutable (profile_id=%)',
            OLD.profile_id
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;

    -- Approval is a one-way door: an approved profile cannot be walked back to DRAFT.
    IF NEW.state NOT IN ('APPROVED', 'SUPERSEDED') THEN
        RAISE EXCEPTION
            'an approved profile cannot return to % (profile_id=%)',
            NEW.state, OLD.profile_id
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS tr_bp_style_profile_freeze_approved ON proc.bp_style_profile;
CREATE TRIGGER tr_bp_style_profile_freeze_approved
    BEFORE UPDATE ON proc.bp_style_profile
    FOR EACH ROW
    EXECUTE FUNCTION proc.bp_style_profile_freeze_approved();

COMMIT;
