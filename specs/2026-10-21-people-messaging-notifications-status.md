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
