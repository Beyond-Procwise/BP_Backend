-- Email settings part 1: the per-customer email policy, the internal-email outbox, and a
-- Security reviewer role that may only inspect with a new `inspect` action class.
--
-- `inspect` is for looking at how mail is configured and what was read (mailbox.view). It is
-- NOT part of `read`, because every role has `read`. It is listed as a reversible class so
-- the guardrail decides it by role alone (a class in neither list is treated as irreversible
-- and would be raised for human review instead of answered). Idempotent.
BEGIN;

INSERT INTO proc.bp_policy
    (policy_name, policy_type, policy_desc, policy_details, policy_status, version,
     created_by, created_date, last_modified_by, last_modified_date)
SELECT 'email_policy', 'setting',
       'How this company''s mailbox is used: internal sends, who may be written to, what is read.',
       '{"policy_identifier":"email_policy",
         "rules":{"internal_send":"with_user_approval","company_domains":[],
                  "external_recipients":"known_suppliers_only",
                  "read_scope":"replies_and_suppliers","on_disconnect":"hold",
                  "access_level":"read_send","notices_from_mailbox":"off",
                  "poll_seconds":120,"restriction_group":"Procwise Mailboxes"},
         "history":[]}'::jsonb,
       1, 1, 'migration:2026-10-16-email-settings', now(),
       'migration:2026-10-16-email-settings', now()
 WHERE NOT EXISTS (SELECT 1 FROM proc.bp_policy
                    WHERE policy_name = 'email_policy' AND policy_status = 1);

CREATE TABLE IF NOT EXISTS proc.bp_internal_email (
    internal_email_id BIGSERIAL PRIMARY KEY,
    author            TEXT NOT NULL,
    kind              TEXT NOT NULL CHECK (kind IN ('forward','reply','notice')),
    mailbox_address   TEXT,
    recipients        JSONB NOT NULL,
    subject           TEXT NOT NULL,
    body              TEXT NOT NULL,
    attachments       JSONB NOT NULL DEFAULT '[]',
    source_ref        TEXT,
    status            TEXT NOT NULL DEFAULT 'pending'
                      CHECK (status IN ('pending','approved','sent','held','declined')),
    held_reason       TEXT,
    content_hash      TEXT NOT NULL,
    approved_by       TEXT,
    approved_at       TIMESTAMPTZ,
    sent_at           TIMESTAMPTZ,
    message_id        TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_bp_internal_email_author_status
    ON proc.bp_internal_email (lower(author), status);

-- Security reviewer role; Admin gains `inspect`; `inspect` is a reversible class.
UPDATE proc.bp_policy
   SET policy_details = jsonb_set(
         jsonb_set(
           jsonb_set(policy_details, '{rules,roles,SecurityReviewer}',
                     '{"rank":1,"allow":["inspect"]}'::jsonb, true),
           '{rules,roles,Admin,allow}',
           -- COALESCE: a missing Admin.allow must never make jsonb_set (strict) NULL the row.
           CASE WHEN COALESCE(policy_details #> '{rules,roles,Admin,allow}', '[]'::jsonb) ? 'inspect'
                THEN COALESCE(policy_details #> '{rules,roles,Admin,allow}', '[]'::jsonb)
                ELSE COALESCE(policy_details #> '{rules,roles,Admin,allow}', '[]'::jsonb) || '["inspect"]'::jsonb END,
           true),
         '{rules,reversible_classes}',
         CASE WHEN COALESCE(policy_details #> '{rules,reversible_classes}', '[]'::jsonb) ? 'inspect'
              THEN COALESCE(policy_details #> '{rules,reversible_classes}', '[]'::jsonb)
              ELSE COALESCE(policy_details #> '{rules,reversible_classes}', '[]'::jsonb) || '["inspect"]'::jsonb END,
         true),
       version = version + 1,
       last_modified_by = 'migration:2026-10-16-email-settings',
       last_modified_date = now()
 WHERE policy_status = 1
   AND policy_details->>'policy_identifier' = 'role_definition'
   AND NOT (policy_details #> '{rules,roles,SecurityReviewer}' = '{"rank":1,"allow":["inspect"]}'::jsonb
            AND COALESCE(policy_details #> '{rules,roles,Admin,allow}', '[]'::jsonb) ? 'inspect'
            AND COALESCE(policy_details #> '{rules,reversible_classes}', '[]'::jsonb) ? 'inspect');

UPDATE proc.bp_policy
   SET policy_details = jsonb_set(
           policy_details, '{rules,group_to_role}',
           COALESCE(policy_details #> '{rules,group_to_role}', '{}'::jsonb)
           || '{"PROCWISE_SECURITY_REVIEWER":"SecurityReviewer"}'::jsonb, true),
       version = version + 1,
       last_modified_by = 'migration:2026-10-16-email-settings',
       last_modified_date = now()
 WHERE policy_status = 1
   AND policy_details->>'policy_identifier' = 'role_assignment'
   AND NOT (policy_details #> '{rules,group_to_role}' ? 'PROCWISE_SECURITY_REVIEWER');

COMMIT;
