# Microsoft 365 shared mailbox: setup and release runbook

Procwise sends supplier email from, and reads replies in, a shared mailbox the customer
chooses. Procwise holds **no customer password or refresh token**. Its only credential is
its own app certificate (the private key lives only in Secrets Manager, referenced by ARN;
there is no client secret). The Entra app requests **no mail permissions**;
access is granted in Exchange (RBAC for Applications) and is limited to members of one
mail-enabled security group. Procwise never moves, deletes, flags or marks-read anything in
the customer's mailbox. Graph paths used: `/users/{m}/mailFolders...`, `/users/{m}/messages...`
(list, get, `$value`) and `/users/{m}/sendMail` only.

## 0. Server settings (environment)

| Variable | Meaning |
|---|---|
| `M365_CLIENT_ID` | Application (client) ID of "Procwise Mail". |
| `M365_CLIENT_CERT_ARN` | Secrets Manager ARN of the app certificate: `{"client_id": "...", "private_key_pem": "-----BEGIN PRIVATE KEY-----...", "thumbprint_sha1_b64url": "..."}`. Microsoft sign-in uses a client assertion signed with this key; client secrets are not supported. |
| `PUBLIC_BASE_URL` | The server's public **https** address. Microsoft returns the admin to `PUBLIC_BASE_URL/admin/integrations/mail/m365/callback` (after the approval page and again after the sign-in); a non-https value makes Connect fail. |
| `APP_BASE_URL` | Optional. The screen's own address when the UI runs on a different origin from the API (the usual case). After Microsoft, the admin's browser lands on `APP_BASE_URL/?view=settings&mail=...`. Unset = `PUBLIC_BASE_URL`. |
| `MAIL_STATE_SIGNING_KEY_ARN` | ARN of a secret holding 48 random bytes (at least 32) that signs the single-use consent and sign-in `state` (and derives the sign-in nonce). |
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
   The same URI serves both returns (approval page, then sign-in); no second URI is needed.
4. **API permissions: remove the default `User.Read`; add only Microsoft Graph -> Delegated ->
   `openid` ("Sign users in").** `openid` lets the admin sign in once so Procwise can learn,
   from a token Microsoft signs, which organisation approved it; it reads no mail and no
   profile. Add **no** mail permission: any permission granted in Entra *adds* to the
   Exchange scoping and would widen access (the probe in step 3(e) below proves this is
   caught). Leave "ID tokens" under Authentication -> Implicit grant **unticked** (the code is
   redeemed server-side).
   **Manifest: set `"groupMembershipClaims": "DirectoryRole"`** so Microsoft puts the signer's
   directory roles (`wids`) in the id_token. Procwise only accepts a connection from a Global
   Administrator, Privileged Role Administrator, Cloud Application Administrator or
   Application Administrator; without this setting every sign-in is refused as `not_admin`.
5. Make a certificate (for example `openssl req -x509 -newkey rsa:2048 -nodes -days 730
   -keyout key.pem -out cert.pem -subj "/CN=Procwise Mail"`). Certificates and secrets ->
   Certificates -> Upload `cert.pem`. Create **no** client secret.
6. Secrets Manager: store `{"client_id": "...", "private_key_pem": "<key.pem contents>",
   "thumbprint_sha1_b64url": "..."}` (the SHA-1 thumbprint as base64url, no padding:
   `openssl x509 -in cert.pem -outform DER | openssl dgst -sha1 -binary | basenc --base64url | tr -d '='`);
   put its ARN in `M365_CLIENT_CERT_ARN`. Put the application ID in `M365_CLIENT_ID`. Delete
   `key.pem` from disk.
7. Create a second secret holding 48 random bytes (for example `openssl rand -base64 48`) and
   put its ARN in `MAIL_STATE_SIGNING_KEY_ARN`.
8. Set `PUBLIC_BASE_URL`, restart the service.

## 2. Customer admin steps (Settings -> Integrations -> Email)

Only users who may `mailbox.bind` (Admin) can use these routes.

1. **Connect Microsoft 365.** The admin is sent to Microsoft's admin-approval page for
   "Procwise Mail". It lists *no* mail permissions (that is expected and is what to point a
   cautious IT team at). They approve as a Global/tenant administrator, then Microsoft asks
   them to **sign in once** (scope `openid` only). Procwise redeems that sign-in server-side
   with its own certificate, verifies the id_token (Microsoft's signature from its published keys,
   audience = our client id, issuer = the token's own tenant, not expired, our nonce, and the
   person holds Global Administrator, Privileged Role Administrator, Cloud Application
   Administrator or Application Administrator in the token's `wids` claim; anyone else gets
   `mail_error=not_admin`) and
   records the tenant **only** from the verified `tid` claim. The `tenant` Microsoft appends
   to the approval return is ignored as evidence: it is a URL parameter anyone can edit, and
   trusting it would let one customer's admin connect their Procwise to another customer's
   tenant. The admin then lands on `APP_BASE_URL/?view=settings&mail=connected`. Until a
   default mailbox is chosen, supplier email is **held** (it does not go out from Procwise).
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
| `GET /m365/callback` | Microsoft's two redirects (approval, then sign-in `code`); no bearer token; trusts only the signed single-use `state` and the id_token it verifies itself |
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
`bad_tenant`, `tenant_mismatch`, `access_denied`, `wrong_account` (the admin signed in to a
different organisation from the one named on the approval return), `signin_failed` (the
code could not be redeemed or the id_token failed verification), or `consent_failed` for
anything else. **`tenant_mismatch`**: a verified tenant different from the one already
connected is refused; the admin must Disconnect first. A repeated callback with the same
state (either step) is `reused`. Every refusal writes an audit row
(`phase=authorize`, `action_type=mailbox.consent_callback`, `status=denied`) holding only
the stage and that code, never the query text; the raw cause is in the server log.

### Sending behaviour (important to tell customers)

- Only **supplier dispatch** uses the company mailbox. Procwise's own alerts, digests and
  support mail stay on SES.
- Once the company has test-sent from its mailbox, Procwise **never silently falls back to
  SES**. If the mailbox is disconnected, removed, revoked or unhealthy, supplier email is
  **held** (with the reason) until fixed. This was confirmed by the product owner.
- `GET /mail/sender` returns `{mode, address, can_send, reason, max_attachment_bytes}`: `ses`
  (never connected, or never test-sent), `m365` (sending from `address`), or `held`
  (`can_send: false`, with `reason`). Reasons are plain sentences for buyers (for example
  "Access to procurement@... was withdrawn in Microsoft 365, so this email is held"); the
  health codes and Microsoft's own error text go to the server log only.
- **Size limit.** Graph's `sendMail` takes the message base64-encoded in a request of about
  4 MB, so Procwise holds any email whose raw MIME (attachments included) is over
  **3,000,000 bytes** with "This email is too large to send from your Microsoft 365 mailbox
  (limit 3 MB including attachments)"; an HTTP 413 from Graph gives the same reason.
  `max_attachment_bytes` is 3,000,000 in `m365` mode so compose blocks earlier. Larger
  emails via upload sessions are a follow-up.
- A customer that never connected Microsoft 365 is sent via SES **without the binding table's
  new columns being read**, so this feature cannot hold email on a database the migration
  has not reached. The migrations (mail-intake x2, then `2026-10-15-m365-mailbox.sql`) are
  still a hard deploy gate.
- Health: `can_read` of the Inbox: 2xx -> OK; **403/404 -> REVOKED** (mailbox deactivated, drafts
  held); **401 (Microsoft rejected our token) -> DEGRADED, never REVOKED**; 429/5xx/network ->
  DEGRADED. A successful Inbox poll (every `M365_POLL_SECONDS`) lifts an active DEGRADED
  mailbox back to OK, so one blip does not hold email for an hour; REVOKED is never lifted.
- A REVOKED mailbox stays on the admin's list labelled "Access withdrawn in Microsoft 365"
  (with Remove only) until an admin removes it; the panel shows why email is held.

## 2b. Email policy (Settings -> Integrations -> Email)

One policy row per company (`proc.bp_policy`, name `email_policy`) says how the mailbox may
be used. Read: Admin and Security reviewer (`GET /admin/integrations/mail/policy`). Change:
Admin only (`PATCH` the same path with `{"version": <the version you read>, "changes": {...}}`;
a stale version gets 409 and nothing is overwritten). Every applied change writes one history
entry per changed field (who, when, before, after) and an audit row in the same transaction;
if either cannot be written the change is not applied. If the policy row is missing or has
an invalid value the routes answer 503 `policy_unavailable` and mail that depends on it is
**held**: there are no defaults in code, so an outage never looks like "nothing is
restricted". The defaults below come only from the seed migration
`scripts/migrations/2026-10-16-email-settings.sql`.

| Setting | In plain words | Default |
|---|---|---|
| `internal_send` | Emails to colleagues (forwards, internal replies). `off`, or `with_user_approval`: they wait until the person who wrote them approves. | `with_user_approval` |
| `company_domains` | Which addresses count as "inside the company" (lower-cased, no `@`). | none (empty) |
| `external_recipients` | Emails outside the company. `known_suppliers_only` (anything else is held with the reason) or `any`. | `known_suppliers_only` |
| `read_scope` | Which incoming mail is opened. `replies_and_suppliers` (replies to emails Procwise sent, and mail from known suppliers) or `whole_inbox`. | `replies_and_suppliers` |
| `on_disconnect` | If the connection breaks. `hold` (emails wait) or `procwise_address` (send from the Procwise address). A temporary outage or unreadable policy always holds. | `hold` |
| `access_level` | What your IT has granted in Microsoft: `read_send` or `read_send_change`. This is a statement shown on the Email page and in the evidence report, and it fills the PowerShell script; it does not change Microsoft's grant. | `read_send` |
| `notices_from_mailbox` | Procwise's own notices (digests, product notices) sent from the mailbox. `off` or `with_user_approval`. | `off` |
| `poll_seconds` | How often the inbox is checked. A whole number, 60 to 3600 (send a JSON integer). | `120` |
| `restriction_group` | Name of the mail-enabled security group that limits Procwise (letters, numbers, spaces, `. _ -`, max 64). | `Procwise Mailboxes` |

**Seed state refuses every save until a company domain exists.** With `internal_send` on (or
notices on) and no `company_domains`, any save is refused ("Add at least one company domain
before allowing emails to colleagues"). The first step for a new customer is therefore to
add the company domain(s) (or switch `internal_send` off in the same save).

Emails to colleagues (`/mail/internal`): the author is the email on their own sign-in token,
never a field in the request. Only the author can approve, edit or decline; an Admin or
another buyer who calls approve gets 403 `not_author`. The policy is re-checked at approve
time: if `internal_send` has been switched off since, the approval is refused (422
`policy_off`). Each address is classified by its lower-cased bare address, so
`" Finance <FINANCE@YourCompany.com>"` is internal when `yourcompany.com` is listed.

`GET /admin/integrations/mail/restrictions` lists every restriction and how it is guaranteed
(Microsoft enforces and we prove it; our own code enforces; or the policy says so), each
`in_force`, `failed` or `not_proven`. `GET /admin/integrations/mail/evidence.pdf` is the same
list, the policy and its history as a dated PDF (the download is audited). **Provider answers
are cached for 120 seconds per company (a failed or unknown check for 5 seconds)**, so after
changing a permission in Entra wait up to two minutes, or restart the service, before
expecting a line to change.

Digest and product notices are queued for approval only when `VALUE_DIGEST_SENT_AS` is a real
user's sign-in email (they approve it like any email to a colleague). If it is anything else
the notice is sent from the Procwise address as before. Alerts about the email connection
itself always come from the Procwise address.

## 2c. Security reviewer

Add people to the Cognito group **`PROCWISE_SECURITY_REVIEWER`**. It maps to the role
`SecurityReviewer` (rank 1, allowed action class `inspect` only, i.e. `mailbox.view`).

A reviewer **can**: see the Email page's read-only view, read the policy and its history,
read the restrictions list, and download the evidence PDF.

A reviewer **cannot**: change the policy, connect, add or remove a mailbox, send or approve
anything, or read any commercial screen. Because `inspect` is the only class they hold, even
"read" is not granted: this is deliberate (spec section 8). Every write route answers 403 to
a reviewer, the refusal is audited, and the gateway's agent-policy endpoints also refuse
reviewers.

Things the administrator must know:

- **Keep reviewers only in the reviewer group.** Someone in two groups acts as the
  higher-ranked role; on a tie the role that is not the fallback (Viewer) wins. So
  Viewer + Reviewer acts only as a Reviewer (they lose read access), and
  Buyer + Reviewer acts as a Buyer (they lose the security view).
- **The UI's route-access configuration (outside this repo) must include
  `PROCWISE_SECURITY_REVIEWER`**, or reviewers cannot reach Home at all.
- The migration also adds `inspect` to Admin and to the guardrail's reversible classes so
  the read is decided by role alone.

## 3. Live-tenant runbook (the release gate)

Nothing here can be proven without a real tenant. Use a Microsoft 365 **developer tenant**.
Record each result with a timestamp and the HTTP status where one is given. Do not release
until every step passes.

Prerequisites: step 1 done on a server with an https `PUBLIC_BASE_URL`; one shared mailbox
(S); the tester's own mailbox as control (C); a supplier test account.

- **(a) Connect.** `POST /m365/connect`, open the URL, approve as tenant admin, then sign in
  when asked. Expect the browser on `APP_BASE_URL` with `mail=connected`, and `tenant_id` equal
  to the developer tenant's ID. Record whether Microsoft showed a second consent prompt for
  "Sign you in" (it should not, with `openid` on the registration).
- **(a2) Negative: forged tenant.** Start Connect again (Disconnect first), approve, and before
  the browser follows the approval return, edit its `tenant=` to a **different** tenant id.
  Expect Microsoft to refuse the sign-in for that tenant, or `mail_error=wrong_account`;
  never `mail=connected`, and `tenant_id` never the edited value.
- **(b) Entra check.** Enterprise applications -> Procwise Mail -> Permissions: **no mail
  permissions** listed (only "Sign users in").
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
  poll. All of them must load. Procwise pages with its own `$top=50&$skip=N` URLs on the same
  `$filter`/`$orderby` and never follows Graph's `nextLink`; record whether Graph honoured
  `$skip` with that filter (if it does not, mail beyond 50 is missed: a blocking finding).
- **(i2) Size.** Send a draft with a 2.5 MB attachment from S: it is held with the 3 MB size
  reason and nothing reaches Graph. Record the largest attachment that does send.
- **(j) Revocation.** Remove Procwise Mail in Entra (or remove S from the group). Within one
  health interval (`STYLE_MAILBOX_HEALTH_INTERVAL_MINUTES`, default 60) the mailbox goes
  REVOKED (403/404). A queued draft is **held**, not sent via SES. `GET /mail/sender` shows
  `mode: held` with a reason.
- **(k) 401 vs revocation.** Temporarily break `M365_CLIENT_CERT_ARN`'s secret (a key whose
  certificate is not uploaded to the app); the next health check must give DEGRADED, **not**
  REVOKED, and the mailbox stays active. Restore the secret.
- **(l) Tenant mismatch.** While connected, complete consent and sign-in from a second tenant
  -> redirect with `mail_error=tenant_mismatch`; the first connection is untouched.
  Disconnect, then connect the second tenant.
- **(m) Disconnect.** `DELETE /admin/integrations/mail`: mailboxes deactivated; supplier email
  is held (not SES) because the company had test-sent.

Email settings part 1 additions (same tenant, same record-keeping):

- **(n0) `wids` arrives.** Complete (a) as a real tenant administrator. If it ends in
  `mail_error=not_admin` for a Global Administrator, the id token carries no `wids`: fix the
  manifest (`"groupMembershipClaims": "DirectoryRole"`) or add `wids` as an optional id-token
  claim, and repeat. Record which of the two worked. **Unverified until this passes.**
- **(n) Certificate sign-in.** (a) succeeds using `M365_CLIENT_CERT_ARN`, and Entra ->
  Certificates and secrets shows one certificate and **no client secret**.
- **(o) Non-administrator refused.** A tenant user without an admin role completes Connect
  -> redirected with `mail_error=not_admin`; nothing is connected.
- **(p) Restrictions turn red and green.** Grant `Mail.Read` (application, admin-consented)
  in Entra, then wait **up to 2 minutes (answers are cached 120 s; or restart the service)**
  and open the Email page / `GET /restrictions`: the "no mail permissions" line is red
  (`failed`). Revoke it, wait again: green. Record both times.
- **(q) Read scope.** With `read_scope` = `replies_and_suppliers`, send an unrelated internal
  email to the shared mailbox. It is never downloaded; the poll log shows it skipped.
- **(r) Forward needs approval.** Forward a supplier email to a colleague: it waits in
  approvals; approve it as its author; it arrives from the shared mailbox. A second user (an
  Admin too) calling approve gets 403 and nothing is sent.
- **(s) Policy beats a pending approval.** With one pending, set `internal_send` to `off`;
  approving it is refused (422 `policy_off`).
- **(t) Security reviewer.** An account only in `PROCWISE_SECURITY_REVIEWER` reaches Home,
  sees the Security view, cannot change anything (every write 403), and downloads the
  evidence PDF; the PDF matches what the screen shows.

### Open item (not built): a failed permissions check does not hold mail

Spec section 7 says a failed provider check should also hold mail from the affected mailbox
and alert the admin. **That is not built.** Today a failed `no_mail_permissions` check only
turns its line red. (A REVOKED or unhealthy mailbox is already held by the `on_disconnect`
rule.) Until it is built, step (p) shows red while mail keeps flowing; do not treat the red
line as a block.

### Known limits

- A queued internal email can stay `approved` if the process dies between the claim and the
  send/hold (there is no sweeper), and also if recording a hold fails during a database outage.
  Look for old `approved` rows in `proc.bp_internal_email`.
- A permission change can show stale for up to 120 seconds; there is no force-refresh.
- Several minor items are deferred in the build ledger (for example the evidence PDF's
  `access_level` wording needs the product owner's sign-off).

## 4. Rotation

Certificate: upload the new certificate in Entra beside the old one, update the Secrets
Manager value with the new key and thumbprint (the ARN does not change), confirm a test-send
and a health check pass, then delete the old certificate in Entra. Procwise caches credentials briefly, so allow about ten
minutes. Rotate the state-signing key the same way (in-flight consent links, valid for ten
minutes, fail as `bad_signature`; the admin just clicks Connect again).
