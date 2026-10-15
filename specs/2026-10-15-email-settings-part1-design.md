# Email settings, part 1 — policy, restrictions screen, security view, internal approvals

**Date:** 2026-10-15 · **Status:** design, awaiting Nick's review
**Builds on:** `specs/2026-10-15-m365-shared-mailbox-design.md` (built on branches
`m365-mailbox` / `m365-mailbox-ui`, not merged). Mock-ups:
https://claude.ai/artifact/PA7WnSrUuSH5GcwyCnJ7R7 (Admin, Security reviewer, Approval).
**Later parts:** part 2 Google Workspace, part 3 IMAP — each its own spec and plan.

## 1. What Nick asked for (rulings, 2026-10-15)

1. How a customer's mailbox may be used is **configured in Settings at implementation**, per
   customer — not fixed in code. Same page for Microsoft 365 now, Google and IMAP later.
2. The shared mailbox may send **inside the customer's organisation** — forwards, internal
   replies, and the product's own notices — **only after the user approves**. This is part of
   the customer's policy.
3. The settings page is **visual**: admins see each restriction being applied; the customer's
   **security team** sees the same through a new **read-only Security reviewer role**.
4. Microsoft 365 signs in with a **certificate**, not a password. Customer-registered apps are
   **not** in scope, so only a customer's Microsoft administrator may complete the connection.
5. Google: dedicated account + admin-trusted app (part 2). IMAP: login stored encrypted in the
   secrets store, shown as the weaker method (part 3).

## 2. Scope of part 1

In: the email policy (store, versions, history), the restrictions model and screen, the
Security reviewer role and view, the evidence report, internal-send approval, recipient rules,
read-scope rule, certificate login, the administrator-only connection check, and turning the
existing Microsoft 365 code into the first provider behind a provider interface.
Out: Google, IMAP (parts 2, 3); customer-registered apps; the wider Admin Configuration hub
(the page sits at the existing Setup → Integrations → Email entry and moves into the hub when
that is built).

## 3. The email policy

One policy per customer, stored as a `proc.bp_policy` row (`policy_identifier = 'email_policy'`),
read by name through one reader (`services/mail_policy.py`). **A missing value raises**; nothing
falls back to a code default (same rule as `governed_limits`). The implementation seed writes the
defaults below so a new customer starts in the safe state.

| Setting | Values | Seeded default |
|---|---|---|
| `internal_send` | `off` · `with_user_approval` | `with_user_approval` |
| `company_domains` | list of domains counted as "inside the company" | the default mailbox's domain, edited by the admin |
| `external_recipients` | `known_suppliers_only` · `any` | `known_suppliers_only` |
| `read_scope` | `replies_and_suppliers` · `whole_inbox` | `replies_and_suppliers` |
| `on_disconnect` | `hold` · `procwise_address` | `hold` |
| `access_level` | `read_send` · `read_send_change` | `read_send` (drives the generated IT script) |
| `notices_from_mailbox` | `off` · `with_user_approval` | `off` |
| `poll_seconds` | 60–3600 | 120 |
| `restriction_group` | name, `[A-Za-z0-9 ._-]{1,64}` | `Procwise Mailboxes` |

- **Versions:** every save writes a new version number; the row keeps the current version and
  each save appends a history entry `{version, changed_by, changed_at, field, before, after}`.
  A save that cannot write its history entry does not apply.
- **Who:** writes need the existing `mailbox.bind` (configure) action; reads need a new
  `mailbox.view` (read) action granted to Admin and Security reviewer.
- `on_disconnect = procwise_address` is the admin's explicit choice and is recorded like any other
  change; it replaces the fixed "hold forever" rule from the mailbox build.
- The size limit is not a setting: it comes from the provider (Microsoft: 3 MB per message).

## 4. Rules applied at send time

Every outgoing email from a company mailbox is checked against the **current** policy at the
moment it is sent, not when it was drafted. Each recipient is classified:

- **Internal:** domain in `company_domains`.
- **Known supplier:** the address belongs to a supplier contact in the supplier master.
- **Other.**

| Email | Outcome |
|---|---|
| any internal recipient, `internal_send = off` | held: "Emails to colleagues are switched off in your email policy" |
| any internal recipient, `with_user_approval`, no valid approval | held, and an approval request goes to the author (§6) |
| any "other" recipient, `known_suppliers_only` | held: "{address} is not a known supplier" |
| over the provider size limit | held: "Too large to send from your Microsoft 365 mailbox (limit 3 MB)" |
| provider unavailable / mailbox not proven | held, as today |

Held emails keep their reason, appear to the sender, and the admin is alerted (existing alert
path). Mixed internal + supplier recipients need the approval (internal rule wins).

## 5. Read scope

With `read_scope = replies_and_suppliers` the poller lists message ids **with sender and
thread headers only** and fetches the full message only when it is a reply to an email we sent
(matches our thread / Message-ID records) or the sender is a known supplier contact. Everything
else in the shared inbox is never downloaded. `whole_inbox` keeps today's behaviour. The setting
applies to both the attachment reader and reply matching.

## 6. Internal-send approval

- An internal email (forward, internal reply, or a product notice when `notices_from_mailbox` is
  on) is created as a draft and an approval request is written to the existing
  `proc.bp_approval` table, assigned to **the user who wrote it** (for a notice: the user it was
  produced for).
- The approval binds to exactly what the user saw (recipients, subject, body, attachments) using
  the existing approval hash; any edit invalidates it.
- The Approvals screen shows it as in the mock-up: From (shared mailbox), To with an "Internal"
  tag, type, subject, body preview, attachments, and **Approve and send / Edit / Don't send**,
  with the line "Your company's policy: emails inside the company leave the shared mailbox only
  after you approve them."
- Security and health alerts about the email connection itself **always** come from the Procwise
  address, never through the mailbox they report on.

## 7. Restrictions model (what the screen draws)

`GET /admin/integrations/mail/restrictions` (action `mailbox.view`) returns a list computed on the
server from live state; the UI only draws it.

```
{ id, title, explanation,          # plain English, as in the mock-up
  guarantee: "provider" | "procwise" | "policy",
  status: "in_force" | "failed" | "not_proven",
  last_checked_at, evidence: [{check, answer, result, at}] }
```

Part-1 restrictions for Microsoft 365:

| id | Guarantee | Proven by |
|---|---|---|
| `no_mail_permissions` | provider | the app's Microsoft identity holds no mail permissions (token `roles` claim empty of `Mail.*`) |
| `mailbox_scope` | provider | per mailbox: allowed read succeeds, control read returns 403 |
| `actions_limited` | procwise | the Graph path allowlist (list, read, send only) |
| `access_level` | policy | `access_level`; matches the script last generated |
| `internal_approval` | policy | `internal_send` |
| `external_suppliers_only` | policy | `external_recipients` |
| `read_scope` | policy | `read_scope` |
| `on_disconnect` | policy | `on_disconnect` |

A failed provider check turns its line red, holds mail from the affected mailbox, alerts the
admin, and is recorded with the time and the provider's answer.

## 8. Security reviewer role

- New product role **Security reviewer**: Cognito group `PROCWISE_SECURITY_REVIEWER`, mapped in
  the `role_assignment` policy (both databases — bp_sqldb only with Nick's OK) and in the
  gateway's `user-roles.ts` (they change together).
- Allowed: `mailbox.view` and the existing audit-log read. Nothing else; never `configure`.
- The Email page in this role is the mock-up's Security view: read-only banner, summary tiles,
  latest proof step by step, change history, evidence report. No edit control is rendered, and
  every write endpoint refuses the role on the server.

## 9. Evidence report

`GET /admin/integrations/mail/evidence.pdf` (`mailbox.view`): a dated PDF of the restrictions,
the latest proof steps with the provider's answers, the policy (current version and every
setting), the mailboxes, and the change history. Generated from the same data as the screen; the
download itself is audited.

## 10. Microsoft 365 changes

- **Certificate login:** the app authenticates with a certificate (client assertion signed with
  the private key held in Secrets Manager) instead of a client secret. `docs/m365-setup.md` gains
  the certificate steps and rotation; the secret path is removed.
- **Administrator-only connection:** the sign-in id_token must carry a directory-role claim
  (`wids`) for Global Administrator, Privileged Role Administrator, Cloud Application
  Administrator or Application Administrator; missing or other roles → refused
  (`not_admin`). If Microsoft omits the claim by default, the app registration enables it as an
  optional claim — verified in the live run.
- **Provider interface:** `mail_providers/base.py` defines `connect`, `prove_restrictions`,
  `list_new`, `fetch`, `send`, `size_limit`. The existing `mail_m365` package becomes the
  Microsoft implementation; the screen, policy and send rules talk only to the interface.

## 11. Failure handling

- Policy unreadable → every send from a company mailbox is held ("email policy unavailable"),
  never sent on assumptions. Customers that have never connected a mailbox are unaffected (SES,
  as today).
- Approval request cannot be written → the email stays a draft with that reason.
- Restrictions endpoint cannot reach the provider → lines show `not_proven` with the time of the
  last successful proof; nothing is shown as proven that was not.

## 12. Testing and release gate

- Unit + API tests through the real app for: every policy rule in §4 (including a policy change
  between draft and send), read scope never downloading out-of-scope mail, approval invalidated
  by an edit, Security reviewer refused on every write route, policy save without history refused.
- "Prove the guard fails" for each send rule and the read-scope filter.
- Live test account (extends the existing runbook): certificate login; a non-admin Microsoft user
  refused at connection; each restriction broken on purpose and seen red; an internal forward held
  until approved, then sent; evidence report downloaded and checked against the screen.

## 13. Open items for the plan

- Where supplier contact addresses live today (supplier master table/column) — confirm before
  coding §4.
- Whether `proc.bp_policy` already keeps a version column usable for §3, or the history lives in
  the row's JSON.
- PDF generation: reuse the report builder's renderer if it can produce this layout.
