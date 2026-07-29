# Negotiation advice for buyers — design

**Date:** 2026-07-29
**Branch:** Development
**Status:** approved, ready for implementation planning

## 1. What this is

A buyer opens a deal and gets ranked, grounded negotiation advice: which levers
to pull, the specific play, why it applies to *this* supplier, what it costs in
trade-offs, and what to do next. The buyer can challenge the framing, ask for
other options, and tell the agent things the data does not know — and the advice
re-ranks.

The negotiation intelligence for this already exists and is unreachable. This
design unlocks it, grounds it in each deal's evidence, and puts a conversation
in front of it.

Out of scope: changing how negotiation rounds are conducted with suppliers
(`NegotiationAgent`'s existing behaviour is preserved unchanged), and the two
open summary-agent defects, which are tracked separately (§14).

## 2. Diagnosis

`_resolve_playbook_context()` (`negotiation_agent.py:7328`) produces exactly what
a buyer needs — per play: `lever`, `play`, `score`, `policy_alignment`,
`performance_signals`, `market_signals`, `rationale`, `trade_offs`. The playbook
behind it (`src/resources/reference_data/negotiation_playbook.json`, loaded at
`negotiation_agent.py:8140`) is real and specific: 4 supplier types × 5
negotiation styles × 3 lever categories, e.g. *"Demand tiered volume discounts
(10% at 100 units, 20% at 500 units)"*.

Measured behaviour:

| Input given | Plays returned |
|---|---|
| `deal_id` only — what the dashboard has today | **0** |
| `supplier_type` only | **0** |
| `supplier_type` + `negotiation_style` | 5, ranked |
| both + performance/market signals | 5, **re-ranked** (Operational rises to 1.4) |

Two causes:

**It is gated on two classifications nothing computes.**
`_resolve_playbook_context` returns `{"plays": [], "lever_priorities": []}` unless
both `supplier_type` and `negotiation_style` arrive in `context.input_data`. They
are read from input only — no code derives them — so the dashboard can never
obtain a single play.

**Without signals the ranking is meaningless.** `_score_supplier_performance`
(`:8294`) and `_score_market_context` (`:8345`) both return `(0.0, [])` on an
empty dict, and both dicts are read from `input_data` with `{}` as the default.
Every score then collapses to 1.00–1.01 — which is just the playbook file's line
order.

Meanwhile the surface the buyer actually sees,
`negotiate_dashboard.negotiation_strategy()` (`:230`), is a static template:

```python
"leveragePoints": {"rationale": "Cost justification and benchmark variance",
                   "approach": "Volume commitment and term levers"},
"counterStrategy": {"action": "Request a detailed cost breakdown for leverage"},
```

Identical for every deal and supplier. `_supplier_insights()` (`:273`) is three
more constants with two `if` branches. There are no tips or next-steps fields.

## 3. Why this shape

| Capability | Where it lives | State |
|---|---|---|
| Playbook: 4 types × 5 styles × 3 levers, with plays | `negotiation_playbook.json` | live, unreachable |
| Play scoring + rationale + trade-offs | `negotiation_agent.py:7328-7420` | live, gated |
| Performance / market scorers | `negotiation_agent.py:8294`, `:8345` | live, never fed |
| Trade-off hints per lever | `TRADE_OFF_HINTS` | live |
| Type / style vocabularies | `_normalise_supplier_type`, `_normalise_negotiation_style` | live |
| Negotiate dashboard + endpoint | `negotiate_dashboard.py:415`, `routers/negotiate.py:24` | live, static strategy |
| Benchmark price position | `src/services/benchmark/` | live |
| Invoice-vs-PO variance | overbilling detector in `opportunity_miner_agent.py` | live |
| Governed policy as data | `proc.bp_policy`, `PolicyEngine` | live |

So this is classification, grounding and a conversation — not new intelligence.

## 4. Supplier classification — suggested, never imposed

The playbook's four keys are the Kraljic quadrants, and every input needed
already exists.

| Signal | Source |
|---|---|
| Spend for this supplier | `bp_deal_overview.invoice_total` / `po_total` / `quote_total`, and the supplier's total across `bp_invoice_trgt` |
| Alternative suppliers for the deal's items | distinct `bp_quote_trgt.supplier_id` quoting the same `bp_quote_line_items_trgt.item_description` |
| Criticality / risk | `bp_supplier.risk_score`, `is_preferred_supplier` (populated for 5,000 of 5,019 suppliers) |

**Measured against the live database, three candidate signals were rejected as
unavailable:**

- **Category** — there is no category dimension at volume. `proc.bp_category`
  (`item_description → category`) holds **0 rows**, `bp_analysis_summary.category`
  holds **6**, and neither `bp_purchase_order_trgt` nor `bp_deal_overview` has a
  `category` column at all. (This also means
  `requirement_service.seed_context()`, which queries
  `bp_purchase_order_trgt WHERE lower(coalesce(category,'')) = lower(%s)`, can
  only ever raise and return `{}` — noted in §14.)
- **Competing quotes on the deal** — only **1 of 5,038** deals has quotes from
  more than one supplier; 5,037 have exactly one.
- **`bp_supplier.supplier_type`** — despite the name, it is a business-type
  taxonomy (Consulting, Retailer, Manufacturer, Distributor, Wholesaler, Service
  Provider), **not** Kraljic. `_normalise_supplier_type` returns `None` for every
  one of those values, so it must not be used as the quadrant source.

Item-level supplier overlap is therefore the only viable competitiveness measure,
and it is strong: 5,019 distinct item descriptions, with suppliers per item at
min 1, p25 20, **median 23**, p75 26, max 41.

| Spend | Alternatives | Quadrant |
|---|---|---|
| high | many | `Leverage` |
| high | few | `Strategic` |
| low | many | `Transactional` |
| low | few | `Bottleneck` |

The quadrant is presented **as a suggestion with its reasons and a confidence**,
e.g. *"Leverage — £340k spend across 2 years, 6 alternative suppliers in this
category, risk score 0.2."* The buyer can override it, and the advice re-ranks
immediately.

Thresholds for "high spend" and "many alternatives" are governed data in
`proc.bp_policy`, not constants, so they are tunable per organisation without a
deploy. Seed defaults are taken from the live distribution — high spend at the
deal-value p90 (£98,175; median is £4,180), many alternatives at the
suppliers-per-item median (23). These are **testdata-derived**: the corpus is
almost entirely generated, so the seeds are a starting point to be retuned
against real spend, which is exactly why they are data and not constants.

When spend or alternatives cannot be determined, no quadrant is guessed: the
buyer is asked to choose, with the partial evidence shown. A wrong quadrant sends
every downstream play in the wrong direction, so silence is safer than a guess.

## 5. Style selection

Derived from the quadrant plus deal evidence, also governed data:

| Quadrant | Evidence | Style |
|---|---|---|
| `Leverage` | price above benchmark | `Competitive` |
| `Strategic` | preferred supplier | `Collaborative` |
| `Bottleneck` | continuity is the exposure | `Principled` |
| `Transactional` | — | `Competitive` |

Also a suggestion, also overridable, and the buyer can request a comparison
against another style (§7).

## 6. Signals

Both dicts are populated using the keys the scorers already read, so no scorer
changes:

| Dict | Keys read (existing) | Source |
|---|---|---|
| `supplier_performance` | `on_time_delivery`, `on_time`, `delivery_score`, `otif` | delivery dates on `bp_invoice_trgt` vs `bp_purchase_order_trgt` |
| `supplier_performance` | quality / discrepancy metrics | `bp_extraction_discrepancy`, invoice-vs-PO variance findings |
| `market_context` | `supply_risk`, `supply_risk_level` | alternative-supplier count, `bp_supplier.risk_score` |
| `market_context` | `demand_trend`, `demand` | volume trend across the deal chain |

A signal that cannot be computed is omitted rather than defaulted, so it
contributes nothing instead of contributing a fabricated nudge.

## 7. Grounding — the part that makes it advice

Each play carries a **precondition** tested against this deal's data. *"Leverage
competitor quotes to pressure pricing"* is strong advice when competing quotes
exist and damaging when they do not — a buyer who bluffs a competing quote that
does not exist loses credibility with the supplier.

| Play family | Precondition |
|---|---|
| Competitive tension | ≥2 quote suppliers on the deal, or ≥2 alternative suppliers for its items. Since only 1 of 5,038 deals has multiple quote suppliers, this almost always resolves through item overlap — and a deal with alternatives but no second quote is precisely the *groundwork* case below |
| Volume / tiering | volume or spend above the tiering threshold |
| Price challenge | benchmark position known for the item or category |
| Overbilling recovery | invoice-vs-PO variance found on this deal |
| Continuity / dual-source | alternative suppliers exist |

Plays are shown in three states:

- **Ready** — precondition holds; evidence shown inline.
- **Groundwork** — precondition fails but is achievable. Shown, clearly marked
  not-yet-usable, with what would unlock it: *"Needs a second quote — 3
  alternative suppliers available in this category."* Often the most valuable
  advice before a negotiation.
- **Not applicable** — precondition cannot be met for this deal; suppressed.

Ready plays rank above groundwork.

### The conversation

The buyer can:

- **Ask for more plays on a lever** — deeper into the ranked list for that lever.
- **Ask for a different lever** — Commercial / Operational / Risk.
- **Compare another style** — re-run under a second style and see both side by
  side before committing.
- **Override the quadrant or style** — advice re-ranks.
- **Add information the data lacks** — *"there are really only two suppliers who
  can do this at volume"* contradicts the computed alternative count, moves the
  quadrant from `Leverage` toward `Bottleneck`, and changes the advice. Likewise
  *"we can't move off them before March"* or *"the budget ceiling is £300k"*.

### Buyer-stated facts stay labelled as stated

A fact supplied in conversation is stored as buyer-stated and never merged into
measured data. Each play shows what it rests on:

```
spend:        £340,000   (invoices)
alternatives: 2          (stated by you)
benchmark:    +8.4%      (benchmark engine)
```

Three reasons: advice remains explicable months later; when advice proves wrong
you can tell whether the data or the assumption was at fault; and a stated fact
can be withdrawn, after which the advice reverts cleanly to measured evidence.

### The LLM's job is narrow

AgentNick writes the *why* for this specific supplier and deal from the grounded
facts, and interprets the buyer's conversational input into a
classification override or a stated fact. It does **not** choose levers, rank
plays, or invent preconditions — those stay deterministic and auditable.
Deterministic spine, LLM surface, as in extraction and the requirements design.

## 8. Module layout

`negotiation_agent.py` is ~540KB — the god-class problem already recorded for
this repo. Adding a buyer-facing advisory conversation to it makes that worse.

- **New** `src/services/negotiation_advice/` — `classification.py` (quadrant +
  style), `signals.py` (the two dicts), `grounding.py` (preconditions and
  states), `ranking.py` (play resolution and scoring, extracted from
  `_resolve_playbook_context`).
- **`NegotiationAgent`** calls the extracted ranking module. Its behaviour is
  unchanged — verified by its existing tests — and it keeps
  `_append_playbook_recommendations` for supplier-facing copy.
- **New agent** `negotiation_advisor`, registered in `agent_definitions.json`
  with its own slug so it appears in the Agent Workspace alongside the other
  fourteen. Thin: it owns the conversation and delegates to the modules.

Extraction is confined to the code being changed; no unrelated refactoring.

## 9. Persistence

**New — `proc.bp_negotiation_advice`**: one row per advice session —
`advice_id`, `deal_id`, `supplier_id`, `quadrant`, `quadrant_source`
(`computed` | `buyer`), `quadrant_confidence`, `style`, `style_source`,
`signals` (JSONB), `plays` (JSONB, with state and evidence per play),
`created_by`, `created_at`, `updated_at`.

**New — `proc.bp_negotiation_advice_fact`**: buyer-stated facts —
`advice_id`, `fact_key`, `fact_value`, `stated_by`, `stated_at`, `withdrawn_at`.
Kept separate from measured signals by construction, so a stated fact can never
silently become data.

`bp_` prefix and `ix_bp_*` indexes per project convention. Additive and
idempotent. `deal_id` is read only — it remains owned by the stored procedure.

## 10. API surface

- `GET /negotiate/{deal_id}/advice` — suggested quadrant and style with reasons
  and confidence, populated signals, and ranked plays each with state, evidence
  and trade-offs.
- `POST /negotiate/{deal_id}/advice/message` — one conversational turn: more
  plays, another lever, compare a style, override a classification, or state a
  fact. Returns the re-ranked advice.
- `DELETE /negotiate/{deal_id}/advice/fact/{fact_key}` — withdraw a stated fact;
  advice reverts to measured evidence.

`negotiate_dashboard.negotiation_strategy()` keeps its computed
`highLevelSummary`, `currentStandpoint`, `preferredOutcome` and
`supplierInsights`. Two fields carrying three constant strings —
`leveragePoints` (`rationale`, `approach`) and `counterStrategy` (`action`) — are
replaced by a `plays` list, since that is the surface the buyer reads. An advice
failure degrades `plays` to `[]` rather than breaking the rest of the payload,
which is unrelated to advice.

## 11. Testing

- **Classification** — each quadrant reached from its spend/alternatives
  combination; thresholds read from policy, not hardcoded; indeterminate inputs
  produce no guess but a prompt to choose.
- **Gate closed today, open after** — `deal_id` alone yields 0 plays against the
  current code and ranked plays after classification is wired.
- **Signals change ranking** — populated dicts re-rank versus empty; an
  uncomputable signal is absent, not defaulted.
- **Preconditions** — a single-quote deal marks competitive-tension plays as
  groundwork with the unlocking condition; a deal with variance surfaces
  overbilling recovery as ready; not-applicable plays are suppressed.
- **No bluffing** — no play presented as ready claims evidence the deal lacks.
- **Buyer-stated facts** — override changes the quadrant and the advice;
  provenance is labelled `stated by you`; withdrawal reverts the advice.
- **NegotiationAgent unchanged** — its existing suite passes against the
  extracted ranking module.
- **Honest emptiness** — a deal with no usable evidence returns groundwork and an
  explicit "not enough evidence yet", never generic filler.

## 12. Corpus reality

`proc.supplier_response` holds **7 rows**, so almost no deal has negotiation
history to summarise. This design is unaffected: advice for a negotiation that
has not happened yet needs spend, alternative suppliers, benchmark position and
price variance, all of which exist at volume today. It does mean the *history*
sections of any negotiation summary will be honestly thin for most deals.

## 13. Risks

| Risk | Handling |
|---|---|
| Wrong quadrant misdirects all advice | Suggested with reasons and confidence, buyer-overridable; no guess when indeterminate (§4) |
| Generic plays read as tailored advice | Precondition per play; ready vs groundwork states with inline evidence (§7) |
| Buyer bluffs a lever the deal cannot support | Nothing is marked ready without its evidence; groundwork states what is missing (§7) |
| Stated facts corrupt measured data | Stored in a separate table, labelled, withdrawable (§7, §9) |
| Extraction regresses negotiation rounds | `NegotiationAgent` behaviour pinned by its existing tests (§8, §11) |
| Thresholds wrong for this organisation | Governed data in `bp_policy`, tunable without deploy (§4) |

## 14. Still open, tracked separately

Two summary-agent defects found in the same investigation, not addressed here:

1. **Portfolio persona summaries are built from an empty fact sheet.**
   `gather_portfolio_context()` returns `totals` / `top_suppliers` /
   `currency_mix`, but `_summary_facts()` (`deal_summary.py:147`) reads only
   `documents` and `discrepancies`, so the model receives
   `{"document_count": 0, "documents": [], "discrepancies": []}`. All 26
   portfolio summaries in `bp_summary` read "Not available" while their own
   stored snapshot holds the real figures.
2. **Persona framing is overridden by a deal-shaped fixed format.** The base
   prompt appended after the persona says *"summary of the deal"* and *"Respond
   in EXACTLY this format"* with only `Key Outcomes` and `Conclusion`, so a
   persona asking for approach, levers or next steps cannot produce them.

3. **`requirement_service.seed_context()` cannot succeed.** It queries
   `proc.bp_purchase_order_trgt WHERE lower(coalesce(category, '')) = lower(%s)`,
   but that table has no `category` column. The call is wrapped in a
   `try/except` returning `{}`, so it fails silently — which is why the "prior
   suppliers for this category" context has never appeared anywhere. This
   matters to the requirements design
   (`2026-07-29-requirements-scope-agents-design.md`), whose Zone 1 lists prior
   suppliers as inherited context.

Also minor: `src/scripts/end_to_end_demo.py:42` calls
`agent.generate_negotiation_strategy(...)`, which does not exist on
`NegotiationAgent` — that script raises `AttributeError`.

## 15. Not doing

- No change to how negotiation rounds are run with suppliers.
- No auto-selection of a quadrant when evidence is indeterminate.
- No merging of buyer-stated facts into measured data.
- No LLM-chosen levers or LLM-invented preconditions.
