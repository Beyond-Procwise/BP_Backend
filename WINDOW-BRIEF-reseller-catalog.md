# Window 3 — Reseller Catalog, Cost and Sell-Side (stage 2)

**Replaces** the 2026-09-11 brief. Stage 1 is finished: the spec, 10-task plan and build were merged to `Development` as `243ed5b` and pushed to origin on 2026-09-12. Facts below come from memory notes dated 2026-09-12 to 2026-10-15; re-verify against the code before relying on them.

## What already exists (do not rebuild)
- Catalog importer: human-owned column map, versioned rows, per-row rejects, receipts.
- SKU-to-history matching: matches are "proposed"; a person confirms or rejects.
- Sell side: accounts, opportunities, outbound quotes (cost snapshot, supersede, submit/approve/issue), self-approval bar, customer view behind an allowlist and leak guard, outcomes, win-probability calibration (NULL until 30 closed deals per type), daily scheduler job, `/catalog` and `/sales` routers.
- Quote builder backend, on origin up to `71fe390a`: cost split, `/internal`, `/settings/quote`, stable line ids, customer PDF, `POST /spendiq/quotes/{id}/link`, margin floor enforced at approval.
- Migrations applied on BOTH `bp_testdb` and `bp_sqldb`. Rollback files exist and have never been run.

## Still open
1. **Pricing values (needs Nick).** In policy `sales_quote_pricing`, `validity_days`, `payment_terms_default` and the discount corridor are still null. Tax (20%) and margin floor (15%) are set on both DBs.
2. **UI Sales mode.** Not wired. The quote builder screen's cost display is broken until the UI reads costs from `/internal`. Owner: the UI session (beyond-procwise-ui-89), waiting on its user.
3. **Cost-tier and relation-feed importers.** Not built, because no feed shape exists. Get a sample feed from Nick before designing.
4. **Opportunity detector.** Not built.
5. **Matching yield.** 0 exact matches on the seeded corpus: history item ids are synthetic `ITM…` and no line table has an MPN column. Decide whether to match another way or wait for real data.
6. **Permits.** `sales_quote.approve` and `sales_quote.issue` need role Approver (a governance default Nick may change).

## Do
1. Ask Nick for the three pricing values and a sample cost/relation feed.
2. Check the state of origin/Development against `243ed5b` and `71fe390a` before starting.
3. Use `superpowers:brainstorming`, then `superpowers:writing-plans`, for the importers and the detector. Keep the design to one proposal, not option matrices.

## Traps
- Tables take the `bp_` prefix; indexes `ix_bp_<table>_<cols>`.
- Tests run with `./venv/bin/python` and need `.env` loaded; DB tests need `PROCWISE_TEST_LIVE_DB=1` or they silently skip. `.env` points at `bp_testdb`, not `bp_sqldb`.
- With `ASK_AUTH_MODE=off` every gated write is refused (no principal means Viewer). Demoing writes needs a real token or a local walk-only principal, never committed.
- The sell side uses `sell_side._db.transactional_conn()` because `get_conn()` is autocommit.
- `2026-09-09_stated_permits.sql` has no re-run guard. Never run it twice.
- Failing tests already present, not ours: governed_limits x2, policy_action_vocabulary x2, send_path_gate check_3.
- Never bare `git stash`. Never push to `main`. Remove only worktrees you created.
