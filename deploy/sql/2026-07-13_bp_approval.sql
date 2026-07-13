-- proc.bp_approval — the approvals decision record.
--
-- ApprovalsAgent used to write to `proc.approvals` and read thresholds from a
-- bare `approval_policies` table. NEITHER TABLE EXISTS in bp_sqldb. Both calls
-- sat inside bare `except: logger.exception(...)` blocks, so the agent reported
-- SUCCESS while every approval it "stored" was silently discarded and every
-- threshold silently fell back to a hardcoded 1000. It has never worked.
--
-- Naming follows the bp_ convention (indexes ix_bp_<table>_<cols>).

CREATE TABLE IF NOT EXISTS proc.bp_approval (
    approval_id      BIGSERIAL PRIMARY KEY,

    -- What is being approved. deal_id is the platform's grouping key; the other
    -- two are optional because an approval can hang off a quote/RFQ or a finding.
    deal_id          TEXT,
    rfq_id           TEXT,
    finding_id       TEXT,
    supplier_id      TEXT,

    -- The money question.
    amount           NUMERIC(18, 2),
    currency         TEXT,
    threshold        NUMERIC(18, 2),

    -- The verdict. 'approve' | 'require_approval' | 'escalate' | 'deny'
    decision         TEXT NOT NULL,
    decision_reason  TEXT,

    -- Provenance: WHICH governed policy produced the threshold, and what facts
    -- the decision was computed from. A decision with no grounding is a guess;
    -- this column is what makes it checkable.
    policy_id        BIGINT,
    policy_name      TEXT,
    grounding        JSONB NOT NULL DEFAULT '{}'::jsonb,

    -- Human-in-the-loop.
    status           TEXT NOT NULL DEFAULT 'pending',   -- pending | actioned | overridden
    actioned_by      TEXT,
    actioned_at      TIMESTAMPTZ,

    workflow_id      TEXT,
    created_by       TEXT NOT NULL DEFAULT 'system',
    created_date     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_bp_approval_deal_id     ON proc.bp_approval (deal_id);
CREATE INDEX IF NOT EXISTS ix_bp_approval_status      ON proc.bp_approval (status);
CREATE INDEX IF NOT EXISTS ix_bp_approval_decision    ON proc.bp_approval (decision);
CREATE INDEX IF NOT EXISTS ix_bp_approval_finding_id  ON proc.bp_approval (finding_id);
CREATE INDEX IF NOT EXISTS ix_bp_approval_created     ON proc.bp_approval (created_date DESC);


-- The governed approval thresholds the agent reads. These live in the existing
-- bp_policy governance table (policy_type='approval') rather than a new bespoke
-- table, so they are editable through the existing governance UI/API and are
-- versioned/audited like every other policy.
--
-- Seeded as a single default gate. An approval threshold is a business rule, so
-- it MUST be human-authored data, never a constant in the agent.
INSERT INTO proc.bp_policy
    (policy_name, policy_type, policy_desc, policy_details,
     policy_linked_agents, policy_status, version, created_by, last_modified_by)
SELECT
    'ApprovalThresholdPolicy',
    'approval',
    'Spend authority gate: amounts at or below the threshold auto-approve; above it escalates.',
    '{"policy_identifier": "approval_threshold",
      "rules": {"default_threshold_gbp": 10000,
                "currency": "GBP",
                "on_at_or_below": "approve",
                "on_above": "escalate"}}'::jsonb,
    'approvals_agent',
    1,
    1,
    'system',
    'system'
WHERE NOT EXISTS (
    SELECT 1 FROM proc.bp_policy WHERE policy_name = 'ApprovalThresholdPolicy'
);
