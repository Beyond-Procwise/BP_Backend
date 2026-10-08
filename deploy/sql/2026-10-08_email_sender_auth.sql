-- email_agent: who an inbound reply is really from, and the switches that decide when a doubtful sender holds a reply.
--
-- NOT applied anywhere. PACK (a). One row per checked reply: the SPF / DKIM / DMARC results read from an Authentication-Results header stamped by a
-- TRUSTED receiver (a forged header inside the message is ignored), the From domain, whether it is a domain on the supplier master, and the
-- verdict. It stores domains and result words, never an address or any header text.
--
-- A held reply becomes a row in bp_inbound_flag (kind 'sender_not_verified'), which blocks drafting and sending on its thread until an approver
-- clears it. WHAT holds is the EmailSenderAuthRules policy row:
--   mode                      shadow = record only, hold nothing; enforce = hold per the switches below
--   trusted_authserv_ids      the receiving systems whose Authentication-Results we believe. ASSUMPTION, NOT VERIFIED: 'amazonses.com' is what
--                             Amazon SES stamps on inbound mail; no real message was available to check. If it is wrong every reply reads as
--                             "nothing stamped", which holds nothing under the defaults below.
--   hold_on_fail              a hard SPF/DKIM/DMARC failure with no DMARC pass holds the reply (ON)
--   hold_on_missing           no result at all, or only soft ones, holds it (OFF until real mail has been seen)
--   hold_on_domain_mismatch   an authenticated sender whose domain is not the supplier's holds it (OFF until real mail has been seen)
-- A missing or invalid row means the check does not run at all.

BEGIN;

CREATE TABLE IF NOT EXISTS email_agent.bp_inbound_auth (
    auth_id             BIGSERIAL   PRIMARY KEY,
    workflow_id         TEXT,
    unique_id           TEXT,
    supplier_id         TEXT,
    response_message_id TEXT,
    spf                 TEXT        NOT NULL,
    dkim                TEXT        NOT NULL,
    dmarc               TEXT        NOT NULL,
    verdict             TEXT        NOT NULL CHECK (verdict IN ('authenticated', 'failed', 'missing', 'inconclusive')),
    from_domain         TEXT,
    domain_match        BOOLEAN,                      -- NULL = could not be compared (no domain, or none known for the supplier)
    reasons             JSONB       NOT NULL,
    held                BOOLEAN     NOT NULL,
    trusted_headers     INTEGER     NOT NULL,
    ignored_headers     INTEGER     NOT NULL,         -- Authentication-Results lines from servers we do not trust
    checked_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS ix_bp_inbound_auth_message
    ON email_agent.bp_inbound_auth (workflow_id, response_message_id) WHERE response_message_id IS NOT NULL;
REVOKE ALL ON email_agent.bp_inbound_auth FROM PUBLIC;

INSERT INTO proc.bp_policy
    (policy_name, policy_type, policy_desc, policy_details,
     policy_linked_agents, policy_status, version, created_by, created_date)
SELECT 'EmailSenderAuthRules', 'email_sender_auth',
 'Which receivers'' authentication results to trust for an inbound reply, and when a doubtful sender holds it for a person.',
 $json${
  "policy_identifier": "email_sender_auth_rules",
  "required_role": "Admin",
  "rules": {
    "mode": "enforce",
    "trusted_authserv_ids": ["amazonses.com"],
    "hold_on_fail": true,
    "hold_on_missing": false,
    "hold_on_domain_mismatch": false
  }
}$json$::jsonb,
 '', 1, 1, 'email_assurance_migration', now()
WHERE NOT EXISTS (SELECT 1 FROM proc.bp_policy WHERE policy_name = 'EmailSenderAuthRules');

COMMIT;
