# UI briefing — people, notifications and messaging (2026-10-22)

For the session that owns `../beyond_procwise_ui` (and the Node gateway `../beyond-procwaise-Api`, which must forward these paths).
Backend branch: `people-messaging` in BP_Backend. Every endpoint below exists and is tested; nothing here needs a backend change.
Full detail of behaviour and limits: `specs/2026-10-21-people-messaging-notifications-status.md`.

## What the user gets
1. **Bell** in the top bar with an unread count and a list. Click an item -> go to its `link`, mark read.
2. **Inbox / messages**: direct chats, groups, and a discussion thread on a PO, contract, supplier, RFQ, deal, requisition, invoice or approval.
3. **Recipient picker** when composing: "Your manager", "Your reporting line", "Department head", "Your reports", "Already in this thread", plus name/email search. Each suggestion carries a reason; show it as a chip.
4. **Notification settings** (per person) and an **Out of office** switch (pick a cover + end date).
5. **Admin: People & approvals** screen: org chart editing, "who approves what", who each event notifies.
6. **Admin: onboarding org chart** — start from a template, or import from the company's own system.

## Endpoints
Auth as everywhere else (bearer). Own-data routes only ever act on the signed-in person.

### Bell / notifications
- `GET /notifications?unread_only=&limit=&before_id=` -> `{items, unread, next_before_id}`. Item fields include title, body, link, event_type, priority, read.
- `GET /notifications/unread-count` -> number. Poll every 30-60 s as a fallback to the socket.
- `POST /notifications/{id}/read`, `POST /notifications/read-all`
- `GET|PUT /notifications/preferences` — channels (`email` only today), muted event types, quiet hours `HH:MM`. Mandatory events (approval requests) cannot be muted: the API refuses, so grey those switches out.
- `GET /people/me`, `PUT /people/me/delegate` `{delegate_subject|null, delegate_until}` — out of office. A delegate needs an end date in the future.

### Live signals
`WS /ws/user?token=<jwt>` — frames carry `{type: "message"|"notification"|"conversation", ids}` and nothing else. On a frame, refetch over HTTP. Reconnect with backoff; keep the polling fallback.

### Messaging (prefix `/messages`)
- `GET /messages/conversations` (unread counts), `GET /messages/conversations/{id}`, `GET .../{id}/messages` (newest first)
- `POST /messages/direct` `{subject}`, `POST /messages/groups`, `POST /messages/entity-thread` `{entity_type, entity_id}`
- `POST /messages/conversations/{id}/messages` `{text, mentions?}`; mention syntax in text: `@[Name](subject)`
- `POST .../{id}/read`, `PUT .../{id}/mute`, `POST .../{id}/participants`, `POST .../{id}/leave`
- `PUT /messages/{message_id}`, `DELETE /messages/{message_id}` — own messages only.
- `GET /messages/suggest-recipients?q=&conversation_id=` — picker. `POST /messages/suggestions/feedback` — call it when a suggested person is used or dismissed (this is how suggestions are measured).
- Rules the UI must respect: a conversation not yours is a **404** (same as missing). A thread on an item that already has one returns **409 "ask someone in it to add you"** — show that sentence, offer "request to join" copy, do not retry. Mentioning someone outside a direct/group chat is refused. A mute never silences a @mention. Attachments are metadata only (no upload yet) — do not show an attach button.
- The server may rewrite a message body that names an internal table or route; render what it returns.

### Admin (all gated; a refusal is 403 with a reason — show it)
- `GET /admin/people`, `GET|PUT /admin/people/{subject}` (only fields sent change; fields: display_name, email, title, manager_subject, department_id, is_department_head, roles, approval_limit [-1 = unlimited], is_active, delegate_*). Validation failures are 422 with a sentence. People are deactivated, never deleted.
- `GET /admin/people/{subject}/approver?amount=` — dry run for a "who would approve this?" preview.
- `GET /admin/people/starter-chart` — small-company template (the signed-in admin as Owner, unlimited authority) -> pre-fill an editable table.
- `POST /admin/people/import` `{people: [...], dry_run, deactivate_missing}` — **all or nothing**. Always call with `dry_run: true` first and show created / updated / deactivated; on 422 the detail lists every problem (unknown manager, loop, bad limit...). `deactivate_missing` is a separate, clearly worded checkbox; the server refuses if it would leave no active Admin. Rows may arrive in any order (managers are applied first).
- `GET /admin/routing-rules`, `PUT|DELETE /admin/routing-rules/{event_type}` `{targets, notify_actor}` — who each event notifies. Targets come from the list the GET returns.
- Roles on a person are a routing mirror only, not permissions. Say so in the screen.

### Connected systems (ERP / requisition tool) — informational for the integrations screen
`POST /integrations/events` `{event_type, external_id, source, requester_subject?, amount?, version?, link?}` with an admin-level credential. Allowed types: requisition.created, requisition.submitted, po.issued, po.changed, invoice.received. A repeat of the same (type, source, external_id, version) is accepted with `duplicate: true`. People never see sender-written text; the sentence is generated. Small companies have no requisition system: the **approval itself** is the request and already notifies the manager.
A pull connector for an HR/directory product (like the Microsoft 365 mailbox connection) is NOT built yet; when it is, it feeds the same import, so the screen can offer "Connect" later beside "Import".

### Graph (admin, advisory — build last, behind a flag)
`/admin/graph/*` (proposals to confirm/reject, tier switch, acceptance, bottlenecks, who-knows). Default OFF. Suggestions are advisory and must always show their reason and basis (`asserted | derived | inferred`). `DELETE /people/me/activity` = "erase my activity" button in privacy settings.

## Look and feel
Reuse the existing popover/modal styles and the Midnight Canvas tokens (all colour lives in `midnight.css`); do not add new colour values.

## Suggested order
1. Bell + list + unread poll (value on day one: approvals already notify managers once people exist).
2. People & approvals admin with starter chart + import (without people, nobody is notified).
3. Notification settings + out of office.
4. Messaging inbox, thread panel, picker.
5. Graph screens.

## Known limits to design around
- Nobody is notified until `bp_person` has people. Empty state on the bell should say how to fix it for admins.
- Email notifications are off until an admin turns the channel on server-side; the settings screen should show "email" as a channel the user can opt into, not as something guaranteed.
- One notification per message (no grouping of a busy chat) — group visually in the UI if needed.
