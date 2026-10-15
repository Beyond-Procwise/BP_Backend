# Microsoft 365 shared mailbox — design

**Date:** 2026-10-15 · **Status:** awaiting Nick's review · **Author:** Claude with Nick
**Home in the product:** Admin Configuration → Integrations → Email
(the hub itself is a separate build: `WINDOW-BRIEF-admin-configuration.md`)

## 1. What we are building, and why

Today every supplier email leaves through our own SES account, from our address, and replies
reach us only if they happen to be routed to our relay (procwise.co.uk's MX is Hostinger, so
most do not). Customers expect supplier correspondence to come from **their** procurement
mailbox and to live in it.

This build lets a customer's IT admin connect **Microsoft 365, company-wide**, so the product
**sends and receives** through one or more **shared mailboxes** the admin chooses (for example
`procurement@customer.com`), one of them the default.

**Nick's rulings (2026-10-15):** Microsoft 365 first; company-wide admin approval, not
per-user sign-in; send **and** receive; shared mailboxes, not each buyer's own.

**Success means:**
1. An admin connects M365 and adds a shared mailbox without anyone at Procwise touching
   their tenant, and without the product ever holding a user's password.
2. The product can provably reach **only** the mailboxes IT allowed, and shows that proof.
3. A supplier email sent from the product appears in the shared mailbox's Sent Items and
   the supplier's reply is picked up, matched and (if it has attachments) read.
4. When access breaks, nothing is sent from a different address; the admin is told.

**Out of scope:** Gmail/Google Workspace, IMAP, per-buyer mailboxes, delegated (signed-in
user) access, the Admin Configuration hub shell, the negotiation reply-text fault.

## 2. How the permission model works

### 2.1 One Procwise app, consented by each customer
We register **one multi-tenant app** ("Procwise Mail") in our own Microsoft Entra directory.
A customer admin approves it through Microsoft's **admin consent** page. Approval creates the
app's identity (service principal) inside the customer's tenant and tells us their tenant id.

The product then authenticates as itself (OAuth client-credentials): our app id + our secret
→ a short-lived token for that customer's tenant. **There are no refresh tokens and no user
passwords**; the only long-lived credential is our own app secret, held in AWS Secrets
Manager. This is the token style `src/services/style/graph_source.py` already implements.

### 2.2 Scoping to the chosen mailboxes — the important part
A Microsoft "application" mail permission granted through Entra covers **every mailbox in
the tenant**. That is too broad, and permissions in Entra and Exchange are **additive** —
restricting in Exchange does nothing if Entra already granted tenant-wide access.

So:
- The app registration **requests no mail permissions in Entra.** Consent only creates the
  service principal.
- Mail access is granted **in Exchange Online**, using *RBAC for Applications*, scoped to a
  mail-enabled security group (suggested name "Procwise Mailboxes"). Adding a shared mailbox
  to the product = adding it to that group.
- The product generates the exact PowerShell for the admin, with their group and our app's
  ids filled in: create the management scope on group membership, register the service
  principal in Exchange, and assign `Application Mail.Send` and `Application Mail.ReadWrite`
  with that scope.

Microsoft documents both mechanisms; the exact cmdlet sequence and the additivity behaviour
are **to be verified against a live tenant** before release (see §9).

### 2.3 Proof before use (existing control, reused)
`proc.bp_mailbox_binding` already refuses to go active without `scope_verified_at`, and
`MailboxBindingRepository.verify_scope` already runs a probe that must be **denied** on a
control mailbox. For each shared mailbox the admin adds:

1. **Allowed read** — list the mailbox's Inbox folder. Must succeed. (Proves Exchange scope
   includes it.)
2. **Control read** — the same call against a control mailbox the admin names (normally
   their own). Must be **denied** (401/403/404). Success means access is wider than claimed:
   the mailbox is **refused**, with the reason shown in plain English.
3. **Test send** — an explicit "Send test email" button sends one message from the shared
   mailbox to the admin. Sending cannot be probed without sending, so this is the admin's
   deliberate act, not automatic. The mailbox may be the default only after one test send
   succeeds.

The evidence (timestamps, control address, HTTP outcomes) is written to `scope_evidence_ref`
and to the audit trail (`agent_actions.record_action`).

## 3. Admin flow (what the admin sees)

Admin Configuration → Integrations → **Email**. Until the hub exists, the same panel is
opened from the Home avatar menu "Setup", shown only when the server says the user may
configure (the `canConfigure` pattern used by `BriefSettings`).

1. **Not connected** → "Connect Microsoft 365" → Microsoft consent page → back to the panel,
   now "Connected to *customer tenant*, approved by *admin* on *date*".
2. **Restrict access** → the panel shows the generated PowerShell and a checklist. Nothing is
   usable until step 3 passes.
3. **Add shared mailbox** → address + control mailbox → the product runs the allowed/control
   probes → ✓ or a plain reason ("We could also read *control@…*, so access is not
   restricted yet. Run the commands above, wait up to 30 minutes, try again.").
4. **Send test email** → ✓ → **Make default**.
5. Per mailbox: health (OK / degraded / access withdrawn), last checked, last mail received,
   remove. Panel-level: **Disconnect** (stops all use; tells the admin to remove the
   enterprise app in Entra for a full revoke — we cannot do that for them).

## 4. Components

### 4.1 Backend (BP_Backend) — all new files unless noted

| Unit | Purpose |
|---|---|
| `src/services/mail_m365/consent.py` | Builds the admin-consent URL; signs/verifies `state` (HMAC, carries principal + expiry, 10-minute life, single use via nonce in `bp_admin_config`); records the tenant on callback. |
| `src/services/mail_m365/graph_client.py` | The only code that talks to Graph: token (client credentials, per-tenant cache until expiry), list Inbox, get message MIME (`/messages/{id}/$value`), send MIME (`/sendMail`), reply. Extends/reuses `graph_source.py` HTTP + `egress` (`Purpose.MAILBOX`). |
| `src/services/mail_m365/setup_script.py` | Generates the PowerShell text from tenant id, app ids, group name. Pure function. |
| `src/services/mail_m365/sender.py` | `GraphSender.send(binding, mime) -> EmailSendResult`. |
| `src/services/mail_m365/source.py` | `GraphMailSource` implementing mail-intake's `MailSource.iter_messages(binding, since)`; yields `RawMessage(raw=MIME, provider_message_id=Graph id, received_at, source_ref)`. |
| `src/services/style/mailbox.py` (extend) | org-level bindings, `is_default`, probe evidence; no parallel store. |
| `src/services/email_service.py` (change) | Transport choice in one place — see §5.1. |
| `src/api/routers/mail_integration.py` | Endpoints in §4.3. |

### 4.2 Data

- **`proc.bp_admin_config`, key `mail_integration`** (existing store):
  `{provider: "m365", tenant_id, tenant_name, consented_at, consented_by, status: connected|disconnected, pending_state_nonces: [...]}`.
- **`proc.bp_mailbox_binding`** (existing table), migration:
  - `role` CHECK gains `send_receive` (on top of `relay`/`intake` from the mail-intake branch —
    this migration must run after it).
  - New `is_default BOOLEAN NOT NULL DEFAULT FALSE`; partial unique index
    `ix_bp_mailbox_binding_default` on `((1)) WHERE is_default AND is_active` (one default).
  - Org-level bindings use `user_ref = 'org'`.
  - `credential_ref` = ARN of **our** app secret (`{client_id, client_secret}`); the tenant id
    comes from `mail_integration`, not from the secret (`resolve_credentials` changes
    accordingly — today it expects tenant_id inside the secret).
  - New `last_test_send_at TIMESTAMPTZ`; default requires it non-null (CHECK).
- Migrations applied to bp_testdb first, bp_sqldb only with Nick's OK.

### 4.3 Endpoints (BP_Backend, `require_user` + `endpoint_gate.require("mailbox.bind")`, which already exists as a `configure` action — Admin only)

| Method + path | Does |
|---|---|
| `GET /admin/integrations/mail` | Status: connection, mailboxes, health, `canConfigure`. |
| `POST /admin/integrations/mail/m365/connect` | Returns `{authorization_url}`. |
| `GET /admin/integrations/mail/m365/callback` | Microsoft redirects the browser here. No bearer token — authenticity is the signed `state`. Records tenant, redirects to the UI panel with `?mail=connected` or `?mail_error=<code>`. |
| `GET /admin/integrations/mail/m365/setup-script` | PowerShell text. |
| `POST /admin/integrations/mail/mailboxes` `{address, control_mailbox}` | Create binding + run probes; returns ✓ or reason. |
| `POST /admin/integrations/mail/mailboxes/{id}/verify` | Re-run probes. |
| `POST /admin/integrations/mail/mailboxes/{id}/test-send` | Send test to the calling admin. |
| `PATCH /admin/integrations/mail/mailboxes/{id}` `{is_default}` | Set default. |
| `DELETE /admin/integrations/mail/mailboxes/{id}` | Deactivate. |
| `DELETE /admin/integrations/mail` | Disconnect: deactivate every binding, status disconnected. |

Every write is audited with who / when / before → after.

### 4.4 UI (beyond_procwise_ui)

- `modules/MailConnection/mailboxService.js` keeps its exported shape but calls the endpoints
  above (direct to BP_Backend, `VITE_AI_API_URL`, like `BriefSettings`). Contract tests stay;
  the mock moves into test fixtures.
- New `IntegrationsEmailPanel` (Microsoft 365 only in v1) styled like `BriefSettings` /
  `modalStyles.js`, built to drop into the Admin Configuration hub. The existing modal's
  Google and IMAP options are hidden, not deleted.
- The Email compose "From" row shows the real default mailbox.

### 4.5 Configuration

New settings: `M365_CLIENT_ID`, `M365_CLIENT_SECRET_ARN`, `PUBLIC_BASE_URL` (https; builds the
redirect URI `PUBLIC_BASE_URL/admin/integrations/mail/m365/callback`), `MAIL_STATE_SIGNING_KEY`
(Secrets Manager), `M365_POLL_SECONDS` (default 120). Missing config → the panel says
"Microsoft 365 is not configured on this server", never a stack trace.

## 5. Data flow

### 5.1 Sending
`EmailDispatchService.send_draft` → `EmailService.send_email` (unchanged callers). Inside,
one decision:

- **Org has an active, healthy default M365 mailbox** → build the same MIME message as today
  (our `Message-ID`, `X-Procwise-*` headers, attachments), `From` = the shared mailbox,
  send via `GraphSender` (`sendMail` with MIME, saved to Sent Items).
- **M365 connected but the default mailbox is unusable** (revoked, degraded, missing) →
  raise `MailboxUnavailable`. The dispatch marks the draft **held — company mailbox
  unavailable**, records an alert for admins, sends nothing. **No silent SES fallback.**
- **M365 not connected** → SES exactly as today.

Replies to a supplier use the Graph reply path when we hold the original Graph message id,
so threading stays native in Outlook. Whether Exchange preserves our own `Message-ID` on MIME
send is **to be verified live**; if it does, it also cures the SES Message-ID rewrite that
forced domain-scored reply matching (commit 338d2961).

### 5.2 Receiving
Scheduler job `mail-m365-poll` (own lane, `M365_POLL=off` disables) every `M365_POLL_SECONDS`:
for each active binding → `GraphMailSource.iter_messages(binding, since=cursor)` → the
**mail-intake pipeline** (raw MIME stored, safety gate, dedup, attachment hand-off to
extraction) and the **email watcher's reply matching** for the body. The cursor is ours
(mail-intake's per-binding cursor); we never move, delete, flag or mark-read anything in the
customer's mailbox. Graph ids are the dedup key alongside the content hash.

Dependency: receiving needs branch `mail-intake-v2` merged to Development first.

### 5.3 Health
The existing `style-mailbox-health` job covers `send_receive` bindings: Inbox list OK → OK;
401/403 → REVOKED (binding deactivated, admin alert, default cleared); network/5xx →
DEGRADED (sending held, receiving retried).

## 6. Error handling

| Situation | Behaviour |
|---|---|
| Admin cancels/declines consent | Back to panel, "Not connected — approval was not given." |
| `state` missing, forged, expired or reused | Callback refuses, records a security event, nothing stored. |
| Consent by a non-admin Microsoft account | Microsoft refuses; we show its error code text. |
| Control mailbox readable | Mailbox refused (`ScopeVerificationFailed`), reason shown. |
| Exchange change not yet propagated (can take ~30 min) | Probe says so; "try again" — not a failure state. |
| Token request fails (secret expired/rotated) | All bindings DEGRADED, sending held, admin alert naming the cause. |
| Graph throttling (429) | Honour `Retry-After`; poll skips the cycle, send retries up to the existing retry budget then holds. |
| Message larger than Graph's send limit | Draft held with the size reason. |
| Disconnect while drafts queued | Queued drafts held, not sent via SES. |

## 7. Security notes
- No customer passwords or refresh tokens exist anywhere in the system.
- Our app secret: Secrets Manager only, never logged; rotation documented; certificate
  credential is the follow-up hardening.
- The callback is the only unauthenticated endpoint; it accepts nothing but a signed,
  single-use, short-lived `state` and Microsoft's tenant id.
- Graph client allowlist of paths: messages list/get/`$value`, mailFolders list, `sendMail`,
  `reply`. A test asserts no other path is reachable (extends the existing no-`sendMail`
  test, which must change because sending is now intended for `send_receive` bindings only).
- `exemplar_source` bindings stay read-only: send is refused unless role is `send_receive`.

## 8. Testing
- Unit, against a stubbed Graph transport: consent URL + state sign/verify/expiry/replay;
  setup script text; probe outcomes (allowed ok/control denied → active; control readable →
  refused); transport choice in `send_email` (connected/held/not connected); MIME round-trip;
  poll cursor and no-write-to-mailbox.
- **Prove the guard fails:** run the probe against a stub where the control mailbox is
  readable and watch activation refused; remove the hold and watch a test go red when a
  revoked mailbox would fall back to SES.
- UI: contract tests on `mailboxService.js` against the real response shapes; render test of
  the panel states.
- **Live:** on a Microsoft 365 developer tenant — consent, run the generated script, add a
  shared mailbox, control probe denied, test send lands in Sent Items, a reply with a PDF is
  read end to end, revoke in Entra → mailbox flagged and sending held.

## 9. Prerequisites only Nick / IT can supply
1. An **https public address** for the product (Microsoft will only redirect to https).
2. **Access to Procwise's Microsoft Entra** to register the "Procwise Mail" app (we will write
   click-by-click steps) and store its secret in Secrets Manager.
3. A **Microsoft 365 test tenant** (the free developer programme is fine) with one shared
   mailbox, to prove §2.2 and §5.1 for real.
4. Approval to **merge `mail-intake-v2`** (receiving depends on it).

Code for §4 can be built and tested against stubs before 1–3 arrive; it is not "done" until
the live test in §8 passes.

## 10. Build order
1. Migration + binding extensions + `graph_client` (stubbed tests).
2. Consent + callback + setup script + endpoints.
3. Probes + test send + default.
4. Sending switch in `email_service` with the hold.
5. Receiving (after mail-intake merge).
6. UI panel + real `mailboxService`.
7. Live tenant test.
