# Microsoft 365 shared mailbox: setup and release runbook

Procwise sends supplier email from, and reads replies in, a shared mailbox the customer
chooses. Procwise holds **no customer password or refresh token**. Its only credential is
its own app secret (a Secrets Manager ARN). The Entra app requests **no mail permissions**;
access is granted in Exchange (RBAC for Applications) and is limited to members of one
mail-enabled security group. Procwise never moves, deletes, flags or marks-read anything in
the customer's mailbox. Graph paths used: `/users/{m}/mailFolders...`, `/users/{m}/messages...`
(list, get, `$value`) and `/users/{m}/sendMail` only.

## 0. Server settings (environment)

| Variable | Meaning |
|---|---|
| `M365_CLIENT_ID` | Application (client) ID of "Procwise Mail". |
| `M365_CLIENT_SECRET_ARN` | Secrets Manager ARN of `{"client_id": "...", "client_secret": "..."}`. |
| `PUBLIC_BASE_URL` | The server's public **https** address. Microsoft returns the admin to `PUBLIC_BASE_URL/admin/integrations/mail/m365/callback`; a non-https value makes Connect fail. |
| `MAIL_STATE_SIGNING_KEY_ARN` | ARN of a secret holding 48 random bytes (at least 32) that signs the single-use consent `state`. |
| `M365_POLL_SECONDS` | Poll interval for connected mailboxes (default 120, minimum 30). |
| `M365_POLL` | `on` (default) or `off`; `off` stops the poll job being registered. The job is a no-op until Microsoft 365 is configured and a mailbox is connected. |
| `M365_TRUSTED_ASSERTER` | Optional. Host whose `Authentication-Results` the poll job trusts for sender-auth verdicts. Unset (default) = verdict unknown; Graph delivers the message as the mailbox holds it and nothing vouches for it. |
| `STYLE_MAILBOX_HEALTH_INTERVAL_MINUTES` | Existing health-check interval (default 60). Revocation is noticed within one interval. |

If any of the first four is unset, `GET /admin/integrations/mail` reports not configured and
the other routes answer 503 "Microsoft 365 is not configured on this server."

## 1. Register "Procwise Mail" (once, by Procwise)

1. Entra admin centre -> App registrations -> New registration.
2. Supported account types: **Accounts in any organizational directory (multitenant)**.
3. Redirect URI, platform **Web**: `PUBLIC_BASE_URL/admin/integrations/mail/m365/callback`.
4. **API permissions: remove the default `User.Read` and add nothing.** Any permission granted
   in Entra *adds* to the Exchange scoping and would widen access (the probe in step 3(e)
   below proves this is caught).
5. Certificates and secrets -> New client secret, 24 months.
6. Secrets Manager: store `{"client_id": "...", "client_secret": "..."}`; put its ARN in
   `M365_CLIENT_SECRET_ARN`. Put the application ID in `M365_CLIENT_ID`.
7. Create a second secret holding 48 random bytes (for example `openssl rand -base64 48`) and
   put its ARN in `MAIL_STATE_SIGNING_KEY_ARN`.
8. Set `PUBLIC_BASE_URL`, restart the service.

## 2. Customer admin steps (Settings -> Integrations -> Email)

Only users who may `mailbox.bind` (Admin) can use these routes.

1. **Connect Microsoft 365.** The admin is sent to Microsoft's admin-approval page for
   "Procwise Mail". It lists *no* mail permissions (that is expected and is what to point a
   cautious IT team at). They approve as a Global/tenant administrator and are returned to
   Procwise (`?view=settings&mail=connected`).
2. In Exchange admin, create a **mail-enabled security group** called "Procwise Mailboxes" and
   add each shared mailbox Procwise may use. Adding or removing a mailbox later = changing
   this group.
3. Copy the script from the panel (`GET /admin/integrations/mail/m365/setup-script`, optional
   `?service_principal_object_id=<Object ID>`). Before running, set the two variables at the
   top: `$ServicePrincipalId` (Entra -> Enterprise applications -> Procwise Mail -> Object ID)
   and `$TestMailbox` (a shared mailbox in the group). It registers the service principal,
   creates a management scope limited to the group, grants `Application Mail.Send` and
   `Application Mail.ReadWrite` within that scope only, and ends with
   `Test-ServicePrincipalAuthorization`.
4. **Wait up to 30 minutes** for Exchange to apply it.
5. Add the shared mailbox in the panel with a **control mailbox** (normally the admin's own,
   not in the group). Procwise reads the shared mailbox, then tries the control mailbox and
   must be refused.
6. Test-send (sends to the signed-in admin), then Make default.

Routes (all under `/admin/integrations/mail`, all `mailbox.bind` except the callback):

| Route | Purpose |
|---|---|
| `GET ` | status (`canConfigure`) |
| `POST /m365/connect` | returns `authorization_url` (login.microsoftonline.com/organizations/v2.0/adminconsent) |
| `GET /m365/callback` | Microsoft's redirect; no bearer token, trusts only the signed single-use `state` |
| `GET /m365/setup-script` | the PowerShell (plain text) |
| `POST /mailboxes` | `{address, control_mailbox}`: read it, then prove the control is denied |
| `POST /mailboxes/{id}/verify` | `{control_mailbox}`: re-run the proof |
| `POST /mailboxes/{id}/test-send` | send a test to the signed-in admin |
| `PATCH /mailboxes/{id}` | `{is_default: true}` |
| `DELETE /mailboxes/{id}` | stop using this mailbox |
| `DELETE ` | Disconnect Microsoft 365 |
| `GET /mail/sender` | any signed-in user: where supplier email will leave from |

### Refusals (body `{"detail": {"code", "message"}}`)

| code | HTTP | Meaning / admin action |
|---|---|---|
| `not_connected` | 409 | Connect Microsoft 365 first. |
| `duplicate` | 409 | That mailbox is already connected. |
| `same_as_control` | 422 | Control mailbox must differ from the shared mailbox. |
| `cannot_read` | 422 | Could not read the mailbox: is it in the group? Allow up to 30 min. |
| `over_scoped` | 422 | The control mailbox was also readable: access is wider than chosen. Fix the grant (and remove any Entra mail permission), wait, retry. |
| `inconclusive` | 503 | Microsoft gave no clear answer. Nothing changed; retry. |
| `not_found` | 404 | That mailbox is not connected. |

Addresses are trimmed and lower-cased before storing and comparing. The proof counts only an
**HTTP 403** on the control mailbox as "restricted"; 401 (our own token) and 404 (unknown
address) are inconclusive, never a pass. Test-send maps Throttled to 503, access denied to
403 and unreachable to 502 (nothing sent).

Callback failures redirect to `?view=settings&mail_error=<code>` with a code from a fixed
set: `malformed`, `bad_signature`, `expired`, `reused` (state problems), `not_granted`,
`bad_tenant`, `tenant_mismatch`, `access_denied`, or `consent_failed` for anything else.
**`tenant_mismatch`**: a callback for a different tenant while already connected is refused;
the admin must Disconnect first. A repeated callback with the same state is `reused`.

### Sending behaviour (important to tell customers)

- Only **supplier dispatch** uses the company mailbox. Procwise's own alerts, digests and
  support mail stay on SES.
- Once the company has test-sent from its mailbox, Procwise **never silently falls back to
  SES**. If the mailbox is disconnected, removed, revoked or unhealthy, supplier email is
  **held** (with the reason) until fixed. This was confirmed by the product owner.
- `GET /mail/sender` returns `{mode, address, can_send, reason}`: `ses` (never test-sent),
  `m365` (sending from `address`), or `held` (`can_send: false`, with `reason`).
- Health: `can_read` of the Inbox: 2xx -> OK; **403/404 -> REVOKED** (mailbox deactivated, drafts
  held); **401 (Microsoft rejected our token) -> DEGRADED, never REVOKED**; 429/5xx/network ->
  DEGRADED.

## 3. Live-tenant runbook (the release gate)

Nothing here can be proven without a real tenant. Use a Microsoft 365 **developer tenant**.
Record each result with a timestamp and the HTTP status where one is given. Do not release
until every step passes.

Prerequisites: step 1 done on a server with an https `PUBLIC_BASE_URL`; one shared mailbox
(S); the tester's own mailbox as control (C); a supplier test account.

- **(a) Connect.** `POST /m365/connect`, open the URL, approve as tenant admin. Expect
  redirect with `mail=connected`.
- **(b) Entra check.** Enterprise applications -> Procwise Mail -> Permissions: **no mail
  permissions** listed.
- **(c) Script.** Create the group with S; run the script; wait up to 30 min.
  `Test-ServicePrincipalAuthorization` shows InScope for S only (and not for C). Note whether
  the Exchange cmdlets in the script ran as written; fix the script otherwise.
- **(d) Add with a control mailbox.** `POST /mailboxes {address: S, control_mailbox: C}` ->
  verified. **Record which HTTP status Exchange RBAC returns for C** (the probe needs a
  read of C to be refused with **403**). If it is **404**, that is a **blocking finding**:
  the proof would report `inconclusive` for every customer, and the design must be revisited
  before release.
- **(e) Negative: over-scoping.** In Entra temporarily grant `Mail.Read` (application,
  admin-consented) to Procwise Mail. Re-run `POST /mailboxes/{id}/verify` with control C ->
  must be refused `over_scoped` (422). Remove the permission again; verify passes again.
  Record the status C returned in each case.
- **(f) Test send.** `POST /mailboxes/{id}/test-send` -> mail in the admin's inbox, and a copy
  in S's Sent Items.
- **(g) Real send.** Make S the default. Send a real RFQ draft -> it arrives *from* S.
  Record: whether the received `Message-ID` is the one Procwise set (our Message-ID domain is
  Procwise's, so Graph may replace it), and whether the sent message has the expected
  In-Reply-To/References when it is a reply.
- **(h) Replies.** Reply from the supplier account with a PDF attached. Confirm the watcher
  matches the reply to the RFQ and mail-intake reads the PDF. Confirm in Outlook that a reply
  Procwise sent (MIME with `In-Reply-To`/`References`) **threads** under the original
  conversation; if not, add `/messages/{id}/reply` to the Graph allow-list as a follow-up.
- **(i) Paging.** Put more than 50 messages in the Inbox (the poll pages at 50) and run a
  poll. Page 2 must load. The `nextLink` is accepted only if it continues the exact same
  `/users/{S}/mailFolders/inbox/messages` path; if Graph returns a differently-encoded path
  (for example `%40`), page 2 is refused and mail beyond 50 is not read: a blocking finding.
- **(j) Revocation.** Remove Procwise Mail in Entra (or remove S from the group). Within one
  health interval (`STYLE_MAILBOX_HEALTH_INTERVAL_MINUTES`, default 60) the mailbox goes
  REVOKED (403/404). A queued draft is **held**, not sent via SES. `GET /mail/sender` shows
  `mode: held` with a reason.
- **(k) 401 vs revocation.** Temporarily break `M365_CLIENT_SECRET_ARN`'s secret (wrong
  value); the next health check must give DEGRADED, **not** REVOKED, and the mailbox stays
  active. Restore the secret.
- **(l) Tenant mismatch.** While connected, complete consent from a second tenant -> redirect
  with `mail_error=tenant_mismatch`; the first connection is untouched. Disconnect, then
  connect the second tenant.
- **(m) Disconnect.** `DELETE /admin/integrations/mail`: mailboxes deactivated; supplier email
  is held (not SES) because the company had test-sent.

## 4. Rotation

Client secret (24-month life): create a new secret in Entra beside the old one, update the
Secrets Manager value (the ARN does not change), confirm a test-send and a health check pass,
then delete the old secret in Entra. Procwise caches credentials briefly, so allow about ten
minutes. Rotate the state-signing key the same way (in-flight consent links, valid for ten
minutes, fail as `bad_signature`; the admin just clicks Connect again). A certificate
credential in place of the client secret is the planned follow-up.
