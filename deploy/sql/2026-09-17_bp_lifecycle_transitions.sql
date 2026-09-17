BEGIN;
-- deploy/sql/2026-09-17_bp_lifecycle_transitions.sql
--
-- Findings and opportunities move only along declared edges. ADR 0003 §3.5, ruling D9.
--
-- Before this, both tables accepted any status from any status. A CHECK constraint
-- limits which VALUES a column may hold; it says nothing about which MOVES are legal.
-- So a realised opportunity could be sent back to identified, a finding one person had
-- resolved could be silently re-resolved by a second person, and the automatic writers
-- overruled people: the PO reconciler resolved findings a person had ignored, and the
-- miner sync could mark a realised opportunity rejected.
--
-- WHY IN THE DATABASE. The Node gateway (spendiq.service.ts resolveDiscrepancy) writes
-- bp_extraction_discrepancy.status too, with no from-state check. A guard in Python
-- would leave that writer, and every psql session, uncovered.
--
-- ONE COPY OF THE RULES. proc.bp_lifecycle_transition holds every legal move. The
-- trigger reads it; so do src/services/lifecycle.can_apply and the Python bulk writers
-- (they filter on it, so a statement never trips over a row it should have skipped).
-- A same-state write is not a move and never consults the table.
--
-- THE REFUSAL. SQLSTATE BP409 -- class "BP" is unused by Postgres -- so a caller can
-- tell "this finding already moved" apart from an outage and say so to the person.
--
-- Idempotent: CREATE ... IF NOT EXISTS, ON CONFLICT DO NOTHING, triggers dropped first.
-- Reversible: 2026-09-17_bp_lifecycle_transitions_rollback.sql.

CREATE TABLE IF NOT EXISTS proc.bp_lifecycle_transition (
    object_type text NOT NULL,
    from_state  text NOT NULL,
    to_state    text NOT NULL,
    note        text,
    created_at  timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT pk_bp_lifecycle_transition PRIMARY KEY (object_type, from_state, to_state),
    CONSTRAINT ck_bp_lifecycle_transition_not_a_loop CHECK (from_state <> to_state)
);

INSERT INTO proc.bp_lifecycle_transition (object_type, from_state, to_state, note) VALUES
    -- Opportunity: forward along identified -> negotiation -> agreed -> realised, skipping
    -- ahead allowed (an agreement reached offline). Closed or rejected from any stage not
    -- yet final. realised, closed and rejected are final: a realised saving is a claim
    -- that money was saved, and unwinding it silently would falsify the reported figure.
    ('opportunity', 'identified',  'negotiation', 'forward'),
    ('opportunity', 'identified',  'agreed',      'forward, skipping negotiation'),
    ('opportunity', 'identified',  'realised',    'forward, skipping ahead'),
    ('opportunity', 'identified',  'closed',      'closed, incl. retired by the miner'),
    ('opportunity', 'identified',  'rejected',    'rejected'),
    ('opportunity', 'negotiation', 'agreed',      'forward'),
    ('opportunity', 'negotiation', 'realised',    'forward, skipping agreed'),
    ('opportunity', 'negotiation', 'closed',      'closed'),
    ('opportunity', 'negotiation', 'rejected',    'rejected'),
    ('opportunity', 'agreed',      'realised',    'forward'),
    ('opportunity', 'agreed',      'closed',      'closed'),
    ('opportunity', 'agreed',      'rejected',    'rejected'),
    -- Finding (proc.bp_extraction_discrepancy): open is the only state that closes. A
    -- person may re-open a resolved or ignored finding -- decision_engine.execute does
    -- this on purpose -- but closing it a DIFFERENT way goes through open first, so the
    -- change of mind is a visible step rather than an overwrite. superseded is final.
    ('finding', 'open',     'resolved',   'settled'),
    ('finding', 'open',     'ignored',    'set aside'),
    ('finding', 'open',     'superseded', 'replaced by a newer finding'),
    ('finding', 'resolved', 'open',       're-opened by a person'),
    ('finding', 'ignored',  'open',       're-opened by a person')
ON CONFLICT DO NOTHING;

CREATE OR REPLACE FUNCTION proc.bp_lifecycle_guard()
RETURNS TRIGGER AS $$
-- TG_ARGV: 0 = object_type in bp_lifecycle_transition, 1 = state column, 2 = key column.
DECLARE
    frm text := to_jsonb(OLD) ->> TG_ARGV[1];
    dst text := to_jsonb(NEW) ->> TG_ARGV[1];
    key text := to_jsonb(OLD) ->> TG_ARGV[2];
BEGIN
    IF frm IS NOT DISTINCT FROM dst THEN
        -- Only reached for a finding whose resolution is being rewritten while it stays
        -- closed (the WHEN clause admits nothing else): the second person's click.
        RAISE EXCEPTION '% % is already %; re-open it before closing it again',
            TG_ARGV[0], key, frm
            USING ERRCODE = 'BP409';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM proc.bp_lifecycle_transition
         WHERE object_type = TG_ARGV[0] AND from_state = frm AND to_state = dst
    ) THEN
        RAISE EXCEPTION '% % is %; it cannot move to %', TG_ARGV[0], key, frm, dst
            USING ERRCODE = 'BP409';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS tr_bp_opportunity_lifecycle ON proc.bp_opportunity;
CREATE TRIGGER tr_bp_opportunity_lifecycle
    BEFORE UPDATE OF stage ON proc.bp_opportunity
    FOR EACH ROW
    WHEN (OLD.stage IS DISTINCT FROM NEW.stage)
    EXECUTE FUNCTION proc.bp_lifecycle_guard('opportunity', 'stage', 'opportunity_id');

-- A finding also fires when it stays resolved/ignored but WHO closed it, HOW, or with
-- WHAT value changes. Other writes to a closed finding (query_sent_at, notes, promotion
-- outcome) are not its resolution and pass untouched.
DROP TRIGGER IF EXISTS tr_bp_extraction_discrepancy_lifecycle ON proc.bp_extraction_discrepancy;
CREATE TRIGGER tr_bp_extraction_discrepancy_lifecycle
    BEFORE UPDATE ON proc.bp_extraction_discrepancy
    FOR EACH ROW
    WHEN (OLD.status IS DISTINCT FROM NEW.status
          OR (OLD.status IN ('resolved', 'ignored')
              AND (OLD.resolved_by, OLD.resolution_action, OLD.resolved_value)
                  IS DISTINCT FROM (NEW.resolved_by, NEW.resolution_action, NEW.resolved_value)))
    EXECUTE FUNCTION proc.bp_lifecycle_guard('finding', 'status', 'discrepancy_id');

-- Verify after applying (expects 17 rows, then two triggers):
--   SELECT object_type, count(*) FROM proc.bp_lifecycle_transition GROUP BY 1;
--   SELECT tgname FROM pg_trigger WHERE tgname LIKE 'tr_bp_%_lifecycle';
-- And the behaviour itself:
--   tests/guardrails/test_lifecycle_transitions.py (PROCWISE_TEST_LIVE_DB=1)

COMMIT;
