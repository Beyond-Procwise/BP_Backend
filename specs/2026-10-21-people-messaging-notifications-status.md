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
