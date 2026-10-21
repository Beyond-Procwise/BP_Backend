# People, messaging & notifications — build status (2026-10-21)
Branch `people-messaging` (worktree ../BP_Backend-people), local, unpushed. Audit: docs/audit/messaging_kg_gap_report.md (main checkout, gitignored).
Decisions taken (user: "go with your recommendations"): single-tenant with tenant_id enforced on new tables; hierarchy admin-entered in proc.bp_person; email-only channel; backend only; derivation defaults N=8/M=3/90d/half-life 6w, retention 12m.

## Built (Phase 1 + 3 core) — code in src/services/people/, SQL in sql/bp_people.sql, sql/bp_notification.sql
- bp_person, bp_routing_rule; HierarchyService (chain, cycles, approver-by-amount, dept-head/admin fallback, delegation incl. loops); RecipientResolver (rule-driven, precedence, delegation, permission filter, dedupe, actor exclusion, safe default if the rule source fails).
- bp_domain_event outbox (emit in the action's own transaction), event catalogue, NotificationService (idempotent per event+user, prefs, mutes, quiet hours, mandatory approvals, per-channel retry), drain worker with MAX_ATTEMPTS.
- 28 unit tests; Postgres path proven on bp_testdb (tables applied there; bp_sqldb NOT migrated).

## NOT built yet
- Wiring: nothing emits events yet; no worker scheduled; PgRuleSource for bp_routing_rule; can_view permission adapter; bp_policy_notification -> bp_notification unification; per-user WS channel.
- Admin endpoints to edit bp_person / routing rules / notification prefs; notification list/read APIs.
- Phase 2 messaging, Phase 4 graph/activity/derivation/queries, Phase 5 UI (other repos) + seed org chart + docs.

## Update — approvals wired (same day)
- Hooks (src/services/agent_policy/approvals.py `_announce`, via people/wiring.emit_safe: savepoint + swallow, DOMAIN_EVENTS=off disables):
  `approval.requested` from open_case (not _insert_case, which is pinned byte-for-byte — replay/live-conflict cases do NOT announce),
  `approval.approved|rejected` from act(), both inside the approval's own transaction. Value-free sentences.
- Worker: BackendScheduler job `domain-event-drain` every 15 s (people/wiring.drain_once). In-app only; email channel not wired.
- Routing defaults changed: requested -> requester's manager (named deciders already get bp_policy_notification); approved/rejected -> requester.
- Proven live on bp_testdb: open_case -> manager notified; act(approve) -> requester notified; rows cleaned up.
- NOT hooked: approval_store.record_approval / record_report_refusal (human dispatch/round/report sign-offs — no requester on record),
  revoke, timeouts (sweep), and REQUISITIONS: there is no requisition-submission code in this backend (only an extracted doc type),
  so `requisition.submitted` has no emitter until a requisition entry point exists (gateway/UI or a new endpoint).
- can_view is same-tenant-active only (TODO real visibility). bp_person is empty on every DB until populated, so nobody is notified in practice yet.
- Baseline: 5 agent_policy tests fail with or without this branch (scheduler lane + 4 scrub-through-app).

## Update — endpoints built (same day)
src/api/routers/people.py (registered in the authenticated router list), logic in src/services/people/admin.py.
- Admin (gate policy.read / policy.write): GET/PUT /admin/people[/{subject}], GET /admin/people/{subject}/approver?amount= (dry run),
  GET/PUT/DELETE /admin/routing-rules[/{event_type}]. PUT changes only the fields sent; manager loops, self-manager,
  unknown manager/delegate, bad limits, unknown targets/events are refused (422/404).
- Own data (signed-in only, always the caller's subject): GET /notifications, /notifications/unread-count, POST /notifications/{id}/read
  (someone else's id = 404), /notifications/read-all, GET/PUT /notifications/preferences (channels: email only; mandatory events cannot be muted),
  GET /people/me, PUT /people/me/delegate (out of office).
- No endpoint emits events or deletes people (deactivate with is_active=false). Roles in bp_person are a routing mirror, NOT authorisation.
- Tests: 54 (services/people + tests/api/test_people_router.py); gate ratchet passes unchanged. Proven live on bp_testdb through TestClient (rows cleaned).

## Update — Phase 2 messaging built (same day)
Tables (sql/bp_messaging.sql, applied to bp_testdb only): bp_conversation (direct|group|entity_thread), bp_conversation_participant, bp_message.
Code: src/services/people/{messaging,realtime,db}.py, src/api/routers/messaging.py (+ WS /ws/user in routers/ws.py).
- Access: a conversation is visible to current participants only; anyone else gets 404 (same as "does not exist"). Entity threads
  (po|contract|supplier|rfq|deal|requisition|invoice|approval) are invite-only: this backend has no per-item visibility rule (bp_deal has no creator),
  so a thread cannot be opened by "anyone who can see the PO". One thread per item; a second person gets 409 "ask someone in it to add you".
  A participant's @mention in an entity thread invites the person; in direct/group an outsider mention is refused. Direct = one per pair.
- Mentions: @[Name](subject) in the text and/or `mentions` list; max 20; never yourself. Edit/delete own messages only (soft delete blanks the text).
- Notifications: message.received (explicit = other active, unmuted participants) and comment.mentioned (mentions; a mute never silences a mention),
  through the same outbox/worker. Titles carry NO message text. The resolver's can_view for conversations re-checks participation at notify time
  (someone who left before the worker ran is not notified — proven by mutation).
- Real-time: WS /ws/user?token= pushes {type:message|notification|conversation, ids} only (own manager, not the session one — a shared one would
  let any user subscribe to another's channel). Single uvicorn worker assumed; clients also poll /messages/conversations and /notifications/unread-count.
- Recipient picker: GET /messages/suggest-recipients?q=&conversation_id= -> labelled asserted groups (manager, reporting line, department head,
  direct reports, already-in-thread) + name/email search. Graph tiers 2-4 come with Phase 4.
- Tests: tests/services/people/test_messaging_pure.py, test_messaging_live.py (PROCWISE_TEST_LIVE_DB=1; 11 tests, 3 guards proven red by mutation),
  tests/api/test_messaging_router.py. End-to-end over HTTP+WS on bp_testdb verified.
- Not done: attachments are metadata only (no upload/storage endpoint); one notification per message (no coalescing of a busy chat);
  no search over message text; no activity-store rows yet (Phase 4: message.sent verb, metadata only); output-safety may rewrite a message body
  that names an internal table/route; UI (inbox, thread panel, picker, bell) lives in the UI repo.

## Update — Phase 4 people knowledge graph built (same day)
Code: src/services/people/graph/{model,store,writer,sync,activity,derive,proposed,queries,service}.py, src/api/routers/people_graph.py,
sql/bp_graph.sql (adds bp_person.title, bp_activity, bp_proposed_edge; applied to bp_testdb only — DEPLOY ORDER: the code now SELECTs bp_person.title,
so migrate before deploying).
Governing principle + admission rule are verbatim in the module docstring of graph/model.py (copy into docs/people_knowledge_graph.md when docs/ is writable in git).
- Three layers: GRAPH = asserted facts (hr / rbac / user_action edges, each with source, source_ref, asserted_by, valid_from, valid_to; changes close+open);
  ACTIVITY STORE (bp_activity) = behaviour, metadata only; EMBEDDINGS = content only, may only RE-ORDER people the graph already returned
  and may only PROPOSE edges via bp_proposed_edge (confirmed by a person -> source=inferred, confirmed_by).
- validate_edge is the single gate: unknown type, wrong source for the type, missing source_ref/asserted_by/valid_from, bad labels, missing required
  props, MESSAGED/VIEWED/COLLABORATED_ON/MENTIONED... (refused by name) all raise AdmissionError and store nothing. Same contract suite runs on the in-memory
  store and on real Neo4j.
- Graph store: Neo4j (the app's existing connection). Own nodes: User, Title, Role, Permission, Department, Team (spec labels) + EntityType + Entity stubs keyed
  type:id, all with pw_tenant/pw_key, so the existing Supplier/PurchaseOrder nodes other code queries are never touched. Entity stubs are NOT yet linked to the
  legacy nodes (follow-up). Nothing has been synced into the shared graph yet: first service.refresh() writes Users/Roles/Permissions for tenant 'default'.
- Sync (idempotent, daily job `people-graph-refresh` + 60 s `people-graph-decisions`, PEOPLE_GRAPH=off disables): HR from bp_person (REPORTS_TO, HOLDS_TITLE,
  MEMBER_OF, DELEGATES_TO with its own window, HAS_AUTHORITY max_amount/unlimited); RBAC from bp_role_assignment + RoleDefinitionPolicy (HAS_ROLE, GRANTS, ON) —
  only DIRECT DB assignments; roles that arrive via Cognito groups at runtime are not mirrored; decisions from approval.* events AND history (bp_approval, policy
  approval cases) — one edge per action, idempotent by event_id, with duration_ms.
- Derivation (daily): WORKS_WITH (>= N=8 addressed interactions across >= M=3 distinct entities in 90 days; direct chat with no entity never counts) and
  ACTIVE_IN (decision edges only, outcome-weighted). Half-life 6 weeks, floor 0.2, each dismissed suggestion x0.8. Rebuilt every run, source=derived (advisory).
- Queries (admin-gated, each result has reason + basis asserted|derived|inferred, all take as_of): who_can_act, who_has_acted, who_knows_about,
  find_bottlenecks, suggest_recipients (tier 1 asserted always; tiers 2-4 only when the tenant switch is ON). Switch default OFF; turning it on needs tier-1
  acceptance evidence (>=30 shown, >=20% accepted) unless an admin forces it. Picker logs suggestion.shown; POST /messages/suggestions/feedback logs accepted/dismissed.
- Privacy: DELETE /people/me/activity and /admin/graph/users/{subject}?pseudonymise= remove activity rows and derived edges; asserted decision edges stay (de-identified
  when pseudonymised). Retention default 365 days (pruned by the daily job).
- NOT done: embedding-based proposer (the queue + confirm flow exist; nothing writes proposals yet); linking Entity stubs to legacy Supplier/PO nodes; Team nodes
  (no source table); WATCHES/OWNS/ASSIGNED/SIGNED/ESCALATED have a validated write path but no product action emits them yet; RBAC via Cognito groups; UI
  ("My network", reason chips); docs/people_knowledge_graph.md (docs/ is gitignored); bp_sqldb migrations.

## Update — email channel built (same day)
src/services/people/email_channel.py; wired in people/wiring.build_service. OFF unless NOTIFICATION_EMAIL=on (first outward-facing piece).
Sends via EmailService/SES from ses_default_sender to bp_person.email, only for users whose preferences list "email"; text = value-free event sentence + reason,
link only when APP_BASE_URL is set. No address -> status skipped_no_address (recorded, never retried); mail refusal -> failed + retried (MAX_ATTEMPTS). Quiet hours defer as before.
6 tests; skip handler and already-sent guard each proven red by mutation. NOT done: no live send (needs a yes + SES sender check), no digest/coalescing, M365 company mailbox not used (that path is supplier-only).

## Update — graph stubs linked to existing Supplier / PurchaseOrder nodes (same day)
src/services/people/graph/legacy_link.py + GraphStore.entity_keys/link_legacy; runs inside the daily `people-graph-refresh` (result under `legacy_links`).
`(Entity "supplier:<id>")-[:REPRESENTS]->(:Supplier {supplier_id})` and `(Entity "po:<id>")-[:REPRESENTS]->(:PurchaseOrder {po_id})`.
Older node is only matched, never created or changed; links only on exactly one match; zero = missing, several = ambiguous (reported, left alone, retried next run).
The link carries no pw_tenant and the older node has none, so asserted-edge queries/deletes never see or touch it. Live graph has a uniqueness constraint on Supplier.supplier_id.
Proven on real Neo4j: made-up ids reported missing, real SUP-CondorLogisticsUkLtd + PO000001 linked, then removed; Supplier/PO/relationship counts identical before/after. 10 tests (memory + Neo4j); ambiguity guard and closed-map check each proven red by mutation.
NOTE: only `supplier:` stubs exist today (from approval history/derivation). Nothing creates `po:` stubs yet — PO linking is ready but idle until a PO-keyed fact is written. Not yet used by queries (e.g. who_knows_about a supplier -> its POs/invoices).
