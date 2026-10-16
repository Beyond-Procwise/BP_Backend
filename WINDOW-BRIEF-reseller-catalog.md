# Window 3 — Reseller Catalog, Cost and Sell-Side (stage 2)

**Replaces** the 2026-09-11 brief. Stage 1 is finished: the spec, 10-task plan and build were merged to `Development` as `243ed5b` and pushed to origin on 2026-09-12. Facts below come from memory notes dated 2026-09-12 to 2026-10-15; re-verify against the code before relying on them.

## What already exists (do not rebuild)
- Catalog importer: human-owned column map, versioned rows, per-row rejects, receipts.
- SKU-to-history matching: matches are "proposed"; a person confirms or rejects.
- Sell side: accounts, opportunities, outbound quotes (cost snapshot, supersede, submit/approve/issue), self-approval bar, customer view behind an allowlist and leak guard, outcomes, win-probability calibration (NULL until 30 closed deals per type), daily scheduler job, `/catalog` and `/sales` routers.
- Quote builder backend, on origin up to `71fe390a`: cost split, `/internal`, `/settings/quote`, stable line ids, customer PDF, `POST /spendiq/quotes/{id}/link`, margin floor enforced at approval.
- Migrations applied on BOTH `bp_testdb` and `bp_sqldb`. Rollback files exist and have never been run.

## Still open
1. **Pricing values (needs Nick).** In policy `sales_quote_pricing`, the discount corridor is still null on both DBs. Set 2026-10-16: `validity_days` = 30 on both DBs; `payment_terms_default` = "30 days from invoice" on bp_testdb only (bp_sqldb pending Nick's yes). Tax (20%) and margin floor (15%) are set on both DBs.
2. **UI Sales mode.** Not wired. The quote builder screen's cost display is broken until the UI reads costs from `/internal`. Owner: the UI session (beyond-procwise-ui-89), waiting on its user.
3. **Cost-tier and relation-feed importers.** Not built, because no feed shape exists. Get a sample feed from Nick before designing.
4. **Opportunity detector.** Not built.
5. **Matching yield.** 0 exact matches on the seeded corpus: history item ids are synthetic `ITM…` and no line table has an MPN column. Decide whether to match another way or wait for real data.
6. **Permits.** `sales_quote.approve` and `sales_quote.issue` need role Approver (a governance default Nick may change).

## Do
1. Ask Nick for the discount corridor (one maximum %, or a % per category with a default), whether to apply payment terms to bp_sqldb, and a sample cost/relation feed.
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

## Stage 3 candidates: what a reseller needs that is not built (review 2026-10-16)
Judgement from the spec, the table list and a keyword search of the sell-side code, not a line-by-line audit. Re-check each item before designing it.

**Quote correctness (build first)**
1. Customer-specific and contract pricing: price lists, agreed prices, per-customer discounts. Nothing found; quotes price from list and cost with one company-wide corridor.
2. Special bids and distributor deal registration: deal-specific, time-limited cost. No cost tier is tied to a deal or an expiry.
3. Live stock and lead-time check when a quote is issued. Availability is stored but not re-checked.
4. Bundles, kits and required add-ons. `requires` is distributor-asserted only and nothing enforces it on a quote.
5. Recurring and licence items: terms, seats, renewals, co-terming. Check whether the code only mentions them or models them.
6. Freight, installation and services as quote lines. Almost no handling found.

**After a quote is won**
7. Customer order, then purchase order to the distributor. The loop stops at the outcome. Build this after 1 and 2.
8. Delivery, customer invoicing, credit notes and returns.
9. Back-end rebates. A spec non-goal, so margin is front-end only and understated. Seam: `cost_basis` and the bitemporal shape.

**Customer side**
10. Customer credit check and per-customer payment terms. The 30-day default applies to everyone.
11. Quote lifecycle: automatic expiry and re-quote, chasing quotes with no answer.

**Data feeding it**
12. Cost-tier and relation feed importers (also open item 3 above).
13. Category classification: 0 of 4,845 history items resolve to a category, so per-category rules have nothing to work with.
14. Matching yield (also open item 5 above).

**Order proposed:** 1 and 2, then 7, then 3 and 10. Items 12 to 14 wait on real data from Nick.
