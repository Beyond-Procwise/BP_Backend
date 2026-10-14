-- email_agent: inbound replies a person must look at before anything is drafted or sent against them.
--
-- NOT applied anywhere. PACK (a). Today's only screen is the payment-detail-change screen (a reply asking for new or changed
-- bank/payment details). The row names the reply by its message id and dispatch id (a pointer into the mailbox) and records WHICH
-- signals fired and the keywords that matched. It never stores text from the email, so a bank detail in the reply cannot be copied here.
--
-- While a flag is 'open' or 'confirmed_fraud' the drafting layer refuses to draft against that supplier's thread and the send guard
-- refuses to send on it. Only 'cleared' lifts it, and clearing needs approver-class authority (see the router).

BEGIN;

CREATE TABLE IF NOT EXISTS email_agent.bp_inbound_flag (
    flag_id             BIGSERIAL   PRIMARY KEY,
    kind                TEXT        NOT NULL,                       -- payment_detail_change, sender_not_verified, injection_suspected (more screens may be added)
    workflow_id         TEXT,
    unique_id           TEXT,                                       -- the dispatch the reply answers
    supplier_id         TEXT,
    response_message_id TEXT,                                       -- the reply's own Message-ID: where to find it in the mailbox
    kinds               JSONB       NOT NULL,                       -- which signals fired
    terms               JSONB       NOT NULL,                       -- keywords that matched (never email text)
    status              TEXT        NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'confirmed_fraud', 'cleared')),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    decided_by          TEXT,
    decided_at          TIMESTAMPTZ,
    note                TEXT
);

-- Screening the same message twice (the watcher inserts a reply and the interaction agent updates it) must not raise it twice.
CREATE UNIQUE INDEX IF NOT EXISTS ix_bp_inbound_flag_message
    ON email_agent.bp_inbound_flag (kind, workflow_id, response_message_id) WHERE response_message_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_bp_inbound_flag_blocking
    ON email_agent.bp_inbound_flag (workflow_id, supplier_id) WHERE status <> 'cleared';
REVOKE ALL ON email_agent.bp_inbound_flag FROM PUBLIC;

COMMENT ON TABLE email_agent.bp_inbound_flag IS
    'Inbound replies a person must review (suspected payment-detail change). No email text is stored. open/confirmed_fraud block drafting and sending; only cleared lifts it.';

COMMIT;
