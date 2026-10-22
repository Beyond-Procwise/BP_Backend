# UI packet — people, notifications and messaging (2026-10-22, v2)

For the session that owns `../beyond_procwise_ui`. A copy of this file sits in that repo as
`ui-packet-people-messaging.md`. Backend: BP_Backend branch `Development` (merged from
`people-messaging`, pushed to origin as `people-messaging`). Every endpoint below exists, is
tested (161 unit tests, live DB tests behind `PROCWISE_TEST_LIVE_DB=1`) and the delivery chain
was proven on the running local service on 2026-10-22: an event in the outbox became a
notification row for the requester's manager within one drain cycle. **Nothing in this packet
needs a backend change.** Behaviour detail and limits: BP_Backend
`specs/2026-10-21-people-messaging-notifications-status.md`.

## How the UI reaches these endpoints
- Same base as the other BP_Backend calls: `import.meta.env.VITE_AI_API_URL` (local
  `.env.development.local` -> `http://localhost:8000`). The Node gateway is **not** involved;
  no gateway change is needed.
- Bearer auth as everywhere else (`setAuthToken` in `src/services/api.js`). Own-data routes only
  ever act on the signed-in person; the `subject` is the Cognito `sub`.
- Live signals: `WS /ws/user?token=<jwt>` — build the URL with the existing `withWsToken()`
  helper (it already does this for `/ws/session/{id}`). Frames are
  `{type: "message"|"notification"|"conversation", ids: [...]}` and nothing else; on a frame,
  refetch over HTTP. Reconnect with backoff and keep the HTTP polling as the fallback.
- Per the UI repo's CLAUDE.md, build against mock data shaped exactly like the responses below,
  behind the existing service interface, and leave a TODO at the integration seam.

## What to build, in order

### 1. Bell: real notifications (day-one value)
Today `engine.js` (`liveNotifs()`, around line 5388) fills the bell from `GET /spendiq/alerts`
(latest 5, read state client-side only). Change it to read the person's own notifications and
keep the alerts feed as a second tab or section — do not drop it.
- `GET /notifications?unread_only=&limit=&before_id=` -> `{items, unread, next_before_id}`.
  Item: `{id, event_id, reason, why, title, body, link, priority, read, created_at}`.
  `reason` is the routing target that picked this person (manager, approver, requester,
  mention...) and `why` is the sentence for it ("You can approve this amount"); show `why` as a
  small chip.
- `GET /notifications/unread-count` -> number. Poll every 30–60 s; the WS frame
  `{type:"notification"}` triggers an immediate refetch.
- Click an item -> `POST /notifications/{id}/read` then navigate to `link` (a relative path,
  e.g. `/approvals/123`, `/messages/45`). `POST /notifications/read-all` for the "mark all read"
  action. An id that is not yours is a 404: treat it as already gone.
- Empty state for admins: "Nobody is notified until people are added — go to People &
  approvals." For everyone else: "No notifications."

### 2. Admin: People & approvals (without people, nobody is notified)
New admin screen (beside Configuration at `/config`). Refusals are 403 with a sentence — show it.
- `GET /admin/people` -> list. `GET|PUT /admin/people/{subject}`; PUT changes only the fields
  sent: `display_name, email, title, manager_subject, department_id, is_department_head, roles,
  approval_limit (-1 = unlimited), is_active, delegate_subject, delegate_until`. 422 carries a
  sentence (manager loop, self-manager, unknown manager, bad limit). People are deactivated
  (`is_active:false`), never deleted — no delete button.
- Say on the screen: roles here are a routing mirror only, not permissions.
- "Who would approve?" preview: `GET /admin/people/{subject}/approver?amount=`.
- Onboarding: `GET /admin/people/starter-chart` -> a small-company template (signed-in admin as
  Owner, unlimited) to pre-fill an editable table. Import with
  `POST /admin/people/import {people:[...], dry_run, deactivate_missing}` — **all or nothing**.
  Always send `dry_run:true` first and show created / updated / deactivated counts; on 422 the
  detail lists every problem. `deactivate_missing` is a separate, clearly worded checkbox; the
  server refuses an import that would leave no active Admin. Rows may be in any order.
  Offer "Connect a directory" as a disabled tile ("coming later") beside "Import" — the pull
  connector is not built.
- Routing: `GET /admin/routing-rules` -> `{rules: [{event_type, priority, mandatory, targets,
  notify_actor, customised}]}`; `customised:false` means the built-in default applies.
  `PUT /admin/routing-rules/{event_type} {targets, notify_actor}`; `DELETE` reverts to the
  default. Allowed targets: manager, chain_of_command, approver, entity_owner, requester,
  category_owner, team, watcher (an unknown one is a 422 with a sentence).

### 3. Notification settings and out of office (in Settings)
- `GET /notifications/preferences` -> `{preferences: {channels, quiet_hours, muted_events},
  mandatory_events, events}`; `PUT` sends any subset of those three keys (null clears one).
  `channels` is a list and only `"email"` exists today: show it as an opt-in, not a guarantee,
  since an admin must turn the channel on server-side. `quiet_hours` is `["HH:MM", "HH:MM"]`.
  `events` lists every event type to offer a mute switch for; the ones in `mandatory_events`
  (approval requests) cannot be muted — the API refuses with 422, so grey those switches out.
- `GET /people/me` -> `{subject, known, person}` where `person` has the fields of an admin
  people row (manager_subject, department_id, roles, approval_limit, delegate_subject,
  delegate_until...) or is null when the admin has not added this person yet (`known:false`,
  show "you are not in the org chart yet").
  `PUT /people/me/delegate {delegate_subject|null, delegate_until}` — the cover needs an end
  date in the future; null clears it.

### 4. Messaging: inbox, thread panel, composer, recipient picker (prefix `/messages`)
- Inbox: `GET /messages/conversations` -> items `{id, type: direct|group|entity_thread, title,
  entity_type, entity_id, last_message_at, muted, unread, participants}`.
- Thread: `GET /messages/conversations/{id}` -> the conversation plus `participants`;
  `GET .../{id}/messages?limit=&before_id=` -> `{messages, next_before_id}`, newest first, each
  `{id, sender, sender_name, body, mentions, attachments, reply_to_id, created_at, deleted,
  edited_at}`. `POST .../{id}/read {message_id?}` (default: all).
- Compose: `POST /messages/direct {to}` (one per pair; returns the existing one),
  `POST /messages/groups {title, members}`, `POST /messages/entity-thread {entity_type,
  entity_id, title?}` with entity_type in po | contract | supplier | rfq | deal | requisition |
  invoice | approval. Put a "Discuss" button on PO, contract, supplier, RFQ, deal, invoice and
  approval detail panes.
- Send: `POST .../{id}/messages {body, mentions?, reply_to_id?}` -> the stored message; mention
  syntax inside the body is `@[Name](subject)`; max 20 mentions; never yourself. The server may
  rewrite a body that names an internal table or route — render what it returns, not what was
  typed.
- `PUT .../{id}/mute {muted}`, `POST .../{id}/participants {subject}`, `POST .../{id}/leave`,
  `PUT /messages/{message_id} {body}`, `DELETE /messages/{message_id}` — own messages only;
  a deleted message comes back with an empty body and `deleted:true` (show "message deleted").
- Picker: `GET /messages/suggest-recipients?q=&conversation_id=` -> `{suggested: [{label, tier,
  people: [{subject, name, reason}]}], search: [{subject, name, reason}]}`. Labels today:
  "Your manager", "Your reporting line", "Your department head", "Your direct reports",
  "Already in this conversation"; `tier` is `asserted` (or `derived` once the graph switch is
  on). Show each person as a chip under its label with the reason. Whenever a suggested person is
  used or dismissed call `POST /messages/suggestions/feedback {subject, tier, outcome:
  accepted|dismissed}` — this is how suggestions are measured.
- Rules the UI must respect:
  - A conversation that is not yours is a **404**, identical to one that does not exist.
  - Starting a thread on an item that already has one returns **409** with "ask someone in it to
    add you": show that sentence, offer "copy a request to join", do not retry.
  - Mentioning someone outside a direct/group chat is refused (422); in an entity thread an
    @mention invites them.
  - A mute never silences an @mention.
  - Attachments are metadata only, no upload endpoint: **no attach button**.
  - One notification per message, no server-side grouping: collapse a busy chat visually.
- Replace the hard-coded `TEAM` list and `detectMentions` in `engine.js` with the picker; the
  Tag chips should insert `@[Name](subject)`.

### 5. Graph screens (last, behind a flag, admin only)
`/admin/graph/*`: proposals to confirm/reject (`GET /admin/graph/proposals`,
`POST .../{id}/confirm|reject`), tier switch (`GET|PUT /admin/graph/tiers`; "on" needs acceptance
evidence unless `force:true`, show the server's sentence), `acceptance`, `bottlenecks`,
`who-knows`, `who-has-acted`, `who-can-act`, `refresh`. Default OFF. Every suggestion must show
its `reason` and `basis` (asserted | derived | inferred). Privacy settings get an "Erase my
activity" button -> `DELETE /people/me/activity` (confirm first; it also removes derived
connections).

### Informational (integrations screen copy only)
`POST /integrations/events` is for a connected ERP or requisition tool, with an admin-level
credential: `{event_type, external_id, source, requester_subject?, amount?, version?, link?}`,
types requisition.created | requisition.submitted | po.issued | po.changed | invoice.received.
A repeat is accepted with `duplicate:true`. People never see sender-written text. Small
companies have no requisition system: the approval itself is the request and already notifies
the manager. No UI to build beyond a paragraph and the endpoint name.

## Look and feel
Reuse the existing popover/modal styles and the Midnight Canvas tokens (all colour lives in
`midnight.css`); add no new colour values. The bell popover keeps its current anatomy
(`notif-item`, `notif-tag`); the inbox follows the existing pane + detail layout.

## Done when
1. Bell shows the signed-in person's notifications with reason chips, unread count from the
   endpoint, read-on-click, read-all, WS refetch, polling fallback, both empty states.
2. People & approvals admin: list/edit, approver preview, starter chart, dry-run import with
   problem list, routing table. All 403/422 sentences surfaced verbatim.
3. Settings: notification preferences (mandatory greyed), quiet hours, out of office.
4. Inbox, thread panel, composer with picker + mentions, "Discuss" on the seven item panes,
   409/404 handling as above, feedback call on every used/dismissed suggestion.
5. Graph screens behind a flag.
Report back: files changed, what changed visually, mock data used, anything left for integration.
