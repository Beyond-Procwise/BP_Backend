# Skill scenario run 2 (2026-10-22, one edge scenario per skill, live on bp_testdb through control.reason)

Evidence for grading, not graded. 26 skills. 15 of 26 edge prompts reached no skill: edge scenarios are written as mid-conversation follow-ups ('Two contracts have never been read for clauses.') with no routable subject, so the router stays silent by design and the model worked from the tools alone (contract_pipeline, invoice_reconciliation, similar_items were still the tools it chose). One model fault to note: demand-aggregation asked get_policy for a config threshold (similar_items.min_similarity) and reported 'no governed policy' as a finding.


26 scenarios; kinds edge; one orchestrator run each. Grade each answer against its expect line.

## award-recommendation / e4 (edge)

**Prompt:** Only one bidder responded.

**Expect:** Escalates as a sole bidder; no recommendation without approval route

**Routed to:** skill:award-recommendation · rounds 3 · 7.0s

**Tools:** screening_summary({"supplier": "bidder"})

**Answer:**

The supplier name or ID "bidder" is not on record. Please provide the correct supplier name or ID to proceed with screening and compliance checks. Without a valid supplier identifier, no further evaluation can be performed.

---

## category-strategy / e1 (edge)

**Prompt:** One supplier has 70 percent of our facilities spend. Is that a problem?

**Expect:** Dependence alert above the configured share, who and how much from the cube, whether a second source exists on record; not a verdict on its own

**Routed to:** no skill · rounds 4 · 12.4s

**Tools:** get_corpus_facts({"query": "total spend by supplier"}); get_corpus_facts({"query": "total spend by supplier"})

**Answer:**

The supplier Harbourline Trading 13 has invoiced spend of INR 102,142,166.94. The total invoiced spend in INR is INR 979,891,069.92. 

No policy or threshold for supplier concentration is found in the corpus. The figure of 70% is not supported by the data.

---

## contract-clause-review / e2 (edge)

**Prompt:** The termination clause says we can leave on 90 days notice. When is the deadline?

**Expect:** Reads the expiry from the register, quotes the clause, computes the deadline only from the dates the clause and register state, keeps the configured buffer, hands the renew question to renewal-and-exit-strategy

**Routed to:** no skill · rounds 2 · 3.6s

**Tools:** obligation_tracker({"contract": "Contract-2025-001"})

**Answer:**

The contract is not in the register, so the obligation calendar and clause details are unavailable. Please provide the contract's expiry date and the exact wording of the termination clause (including the notice period) so I can calculate the deadline.

---

## contract-portfolio-review / e2 (edge)

**Prompt:** Two contracts have never been read for clauses.

**Expect:** Decision date unknown because the clause was not read; flagged; request to read or supply the clauses; not left out of the pipeline

**Routed to:** no skill · rounds 2 · 10.1s

**Tools:** contract_pipeline({"detail": 10, "expires_within_days": 180, "status": "active"})

**Answer:**

Two contracts have never been read for clauses:  
- **C01151**: "Facilitate Next-Generation Web Services" (Supplier: S3345, USD 504,277.45, expires 2026-10-22)  
- **C01076**: "Productize Cross-Media Web Services" (Supplier: S1065, GBP 63,388.52, expires 2026-10-22)  

Both are active, non-auto-renewing, and have no clauses stored for review. Their decision basis is "clauses never read for this contract."

---

## demand-aggregation / e1 (edge)

**Prompt:** MRO spend has poor descriptions. Can we match items?

**Expect:** Matches only at configured confidence with evidence; otherwise listed for review, no saving counted

**Routed to:** no skill · rounds 5 · 14.4s · UNVERIFIED FIGURES

**Tools:** similar_items({"limit": 10, "min_similarity": 0.6, "description": "MRO item"}); get_policy({"query": "similar_items.min_similarity"})

**Answer:**

There is no governed policy for `similar_items.min_similarity`. The default minimum similarity threshold used by the system is 0.6, as observed in the tool's behavior.

Since no policy defines this value, and you requested that no figures be estimated or assumed, I cannot provide a specific number.

The result remains:  
No items with similar descriptions were found for "MRO item" at the default similarity threshold of 0.6 across your PO, invoice, and quote lines.

To proceed:
- You may request a lower threshold (e.g., 0.4) if allowed by your system's default behavior.
- Alternatively, provide a more specific item description to improve matching.

Would you like to try with a different description or request a lower threshold?

Note: these figures are not from a tool result or from your request, so they are not verified: 0.6, 0.4.

---

## incoterms-selection / e2 (edge)

**Prompt:** EXW from India but our buyer cannot export.

**Expect:** Says EXW is wrong; recommends FCA

**Routed to:** skill:incoterms-selection · rounds 6 · 28.1s

**Tools:** get_policy({"type": "preferred incoterm"}); get_policy({"slug": "preferred incoterm"}); get_policy({"query": "preferred incoterm"}); list_calculations({"name": "landed_cost.total_cost"})

**Answer:**

### Step 2: Confirm transport mode and destination

Transport mode and destination are missing. Without them, no valid Incoterm can be selected.

Call `get_corpus_facts` to check if the purchase or PO includes this data.

---

## maverick-spend-review / e2 (edge)

**Prompt:** Half the invoice lines carry no PO reference at all.

**Expect:** Pattern sized per currency; our gap (unrecorded orders) named as the likely cause to check; not a verdict on the suppliers or buyers

**Routed to:** no skill · rounds 2 · 7.0s

**Tools:** invoice_reconciliation({"supplier": "supplier_name_or_id"})

**Answer:**

Please provide the correct supplier name or supplier ID so I can check the invoice reconciliation for that supplier.

---

## negotiation-planning / e3 (edge)

**Prompt:** SaaS renewal with 40 percent uplift.

**Expect:** Uses governed bounds; asks for usage and alternatives; no invented benchmark

**Routed to:** skill:renewal-and-exit-strategy · rounds 5 · 23.5s · UNVERIFIED FIGURES

**Tools:** search_contracts({"query": "SaaS renewal", "expires_within_days": 365}); get_corpus_facts({"query": "top suppliers for SaaS"})

**Answer:**

The corpus contains 3,510 suppliers with invoices for SaaS-related services, but no specific contract or supplier is identified for renewal. No contract details, expiry date, notice period, or supplier name were found in the query or in the corpus.

To proceed, please provide:
- The **supplier name** or **contract identifier** (e.g., "Falconridge Systems 15" or "Cloud Infrastructure SaaS Agreement")
- The **expiry date** (YYYY-MM-DD)
- The **auto-renewal term** (yes/no, and duration if yes)
- The **notice period** (in days) and **clause reference** (e.g., "Clause 14.2")
- The **current annual value** and **currency**
- The **governing law** and **payment terms**

Without these, I cannot compute the notice deadline, assess lead time, or evaluate the renewal path.

Note: these figures are not from a tool result or from your request, so they are not verified: 14.2.

---

## opportunity-pipeline-review / e2 (edge)

**Prompt:** A lever has been open for nine months with no movement.

**Expect:** Stalled beyond the configured age; escalated if its recorded value is above the configured level; carrying skill named

**Routed to:** no skill · rounds 2 · 8.0s

**Tools:** get_opportunity_bucket({"supplier_id": "SUPP123", "item_id": "ITEM456"})

**Answer:**

I need to find the correct supplier and item identifiers before proceeding. Let me search for the supplier and item details using a more general query. I'll start by looking up the supplier by name or other identifying information. Could you please provide the supplier name or item description?

---

## overbilling-investigation / e2 (edge)

**Prompt:** They billed 120 units; we ordered 100 and there is no receipt on record.

**Expect:** Quantity above ordered on record; received not on record is our gap, not proof of non-delivery; asks the user for delivery evidence

**Routed to:** no skill · rounds 2 · 2.3s

**Tools:** invoice_reconciliation({"supplier": "supplier_name_or_id", "start": "2025-04-01", "end": "2026-10-22"})

**Answer:**

Please provide the correct supplier name or supplier_id to proceed with the invoice reconciliation.

---

## payment-terms-optimisation / e2 (edge)

**Prompt:** Half of our invoices have no payment term recorded.

**Expect:** Share with no term stated; those suppliers' contract terms read where stored; the gap is a data finding, not assumed terms

**Routed to:** skill:payment-terms-optimisation · rounds 4 · 16.5s · UNVERIFIED FIGURES

**Tools:** build_spend_cube({"by": "payment_terms", "start": "2025-10-23", "end": "2026-10-22", "currency": "GBP"}); build_spend_cube({"by": "payment_terms", "currency": "GBP", "end": "2026-10-22", "start": "2025-10-23"})

**Answer:**

- **Scope**: All suppliers, period: 2025-10-23 to 2026-10-22, currency: GBP (with as-billed amounts in AED, EUR, INR, USD).  
- **Total spend**: £17,640,482.17 (invoiced).  
- **Payment terms recorded**:  
  - **"(not recorded)"**: £17,641,457.17 (2,231 invoices, 100.0% share).  
  - **"30 days — due 30 Jan 2026"**: £-975.00 (1 invoice, -0.0% share).  

> **Finding**: 100% of invoices have no payment term recorded. This is a systemic gap in data capture.  
> **Action**: Investigate why terms are missing and enforce recording at invoice entry.  
> **Note**: The negative amount (£-975.00) suggests a credit note or error; it is excluded from analysis.

Note: these figures are not from a tool result or from your request, so they are not verified: 2025.

---

## price-benchmarking / e2 (edge)

**Prompt:** We have exactly one previous invoice for this item.

**Expect:** Below the minimum peer count; no verdict; says what evidence would be enough

**Routed to:** no skill · rounds 3 · 8.9s

**Tools:** get_corpus_facts({"query": "open findings"})

**Answer:**

The corpus contains 5,235 open findings, with 4,218 classified as critical. The most common issue is "quantity_invoiced_above_po" (3,281 findings), followed by "line_amount_not_qty_x_price" (715) and "invoices_exceed_po_total" (496). No specific invoice for the item in question is identified in the summary.

To locate the exact invoice, please provide the item description or supplier name so I can query the corpus with those parameters.

---

## pricing-mechanisms / e3 (edge)

**Prompt:** A rebate tier means buying more units costs more in total.

**Expect:** Detects cliff / negative marginal price; restructure as incremental tiers

**Routed to:** skill:pricing-mechanisms · rounds 5 · 12.4s

**Tools:** get_corpus_facts({"query": "top suppliers, total spend, invoice history, contract terms"}); get_corpus_facts({"query": "total spend, contracts, POs, purchase orders"}); get_corpus_facts({"query": "top 10 items by spend, total spend, category breakdown"})

**Answer:**

The corpus does not contain item-level spend data or category breakdowns. The `get_corpus_facts` tool returned no item or category details, only supplier-level spend and currency totals.

Therefore, I cannot determine whether a rebate tier structure causes buying more units to cost more in total, as no item or volume data is available.

To proceed, please provide:
- The specific item or category in question.
- The volume forecast and current price.
- The proposed rebate tier structure (thresholds, rebate base, marginal prices).

Only then can I evaluate the structure for monotonicity and compliance with the pricing mechanisms framework.

---

## procurement-policy-compliance / e1 (edge)

**Prompt:** PO for 9,999.99 GBP.

**Expect:** At or below limit: approve under policy; no rounding up

**Routed to:** no skill · rounds 2 · 11.2s

**Tools:** check_policy({"term_months": 12, "action_type": "purchase_order", "total_commitment": 9999.99, "currency": "GBP", "requester": "unkno)

**Answer:**

The PO for 9,999.99 GBP is **compliant** with the approval threshold policy.  

- **Total commitment**: 9,999.99 GBP  
- **Policy threshold**: 10,000 GBP  
- **Verdict**: Compliant (within limit)  
- **Required approval**: None (below threshold)  
- **Additional notes**:  
  - No related orders were provided, so split-order check could not be performed.  
  - Several rules (e.g., competitive bid, delegation, preferred supplier) are not governed and thus not assessed.  

No further approval is required.

---

## quote-comparison / e2 (edge)

**Prompt:** The two quotes are within 1 percent of each other.

**Expect:** Close call inside the configured band; the open points decide; escalates to the category lead

**Routed to:** no skill · rounds 5 · 12.3s

**Tools:** get_policy({"query": "price_variance_threshold"}); get_policy({"query": "reconciliation_tolerance_policy"})

**Answer:**

The `reconciliation_tolerance_policy` returns the following tolerance thresholds:

- **tax_tolerance_pct**: 0.1
- **uplift_tolerance_pp**: 0.25
- **amount_tolerance_abs**: 1.0
- **amount_tolerance_pct**: 0.01

These are the exact values from the policy. The 1% difference between the quotes exceeds the `amount_tolerance_pct` of 0.01 (i.e., 1%), as 1% is equal to 0.01 in decimal form.

Therefore, the 1% difference is **at the threshold** of the `amount_tolerance_pct` defined in the policy. Since the policy does not specify whether a difference equal to the threshold is allowed or requires action, and no other policy governs quote comparison variance, **the difference is within the defined tolerance limit**.

No further action is required based on this policy.

---

## renewal-and-exit-strategy / e3 (edge)

**Prompt:** Renewing again takes us beyond the policy maximum term.

**Expect:** Checks policy via check_policy; may need a re-tender; escalates

**Routed to:** skill:renewal-and-exit-strategy · rounds 5 · 21.7s

**Tools:** get_policy({"type": "contract_renewal"}); get_policy({"slug": "renewal-max-term"}); get_policy({"agent": "contract_renewal"}); get_policy({"query": "renewal maximum term"})

**Answer:**

The policy check for "renewal maximum term" returned no match. The available policies do not include a specific rule for renewal maximum term. Therefore, there is no governed policy that defines the maximum allowed term for contract renewal.

Since no policy exists to govern the maximum term for renewal, the decision to renew beyond the current term is not restricted by a policy. However, the user's concern about exceeding the policy maximum term indicates that there may be an internal guideline or threshold that is not captured in the governed policies.

I will now proceed to assess the contract renewal decision based on the available information and the skill's default thresholds. I will check the contract's expiry date, auto-renewal terms, and notice period to determine the timeline for action.

I will use `obligation_tracker` to retrieve the contract's details, including the expiry date, auto-renewal flag, renewal term, and the notice clause. I will need the contract ID or title words to proceed.

Since the user has not provided the contract ID or title, I will ask for it to proceed with the analysis.

---

## rfx-design / e2 (edge)

**Prompt:** One bidder quotes EXW in USD, another DAP in GBP. How do we compare?

**Expect:** Fixes like-for-like terms in the invitation or states the bids.normalise rule with user-supplied rates; no 1:1 conversion

**Routed to:** skill:incoterms-selection · rounds 3 · 11.3s

**Tools:** get_policy({"type": "preferred incoterm"}); get_policy({"query": "preferred incoterm"})

**Answer:**

There is no governed policy for a preferred Incoterm. We proceed without a policy override.

Now, to compare the two bids — EXW (USD) and DAP (GBP) — we must:

1. Confirm the transport mode, origin, destination, cargo value, and whether we have forwarder contracts or cargo insurance.
2. Use `list_calculations` to check the required inputs for `landed_cost.total_cost`.
3. Run the calculation for both EXW and DAP, using the same lane, volume, and currency conversion rate.

But first, we need the following missing inputs:

- Transport mode (sea, air, road, rail, multimodal)
- Origin and destination countries
- Cargo value (in USD or GBP)
- Whether we have forwarder contracts that beat the supplier’s freight
- Whether we are VAT registered
- Whether our own cargo policy covers transit
- Duty rate (user-supplied, since `duty_calc` is unavailable)
- Freight and insurance amounts (user-supplied, since `get_freight_quote` is unavailable)

Please provide:
- Transport mode
- Origin and destination countries
- Cargo value (in either USD or GBP)
- Whether we have forwarder contracts
- Whether we are VAT registered
- Whether our cargo policy covers transit
- Duty rate (user-supplied)
- Freight and insurance amounts (user-supplied, in the respective currencies)

Without these, we cannot compute a valid landed-cost comparison.

---

## savings-methodology / e3 (edge)

**Prompt:** EUR/USD swing created an apparent saving.

**Expect:** Needs constant rate from user or tool; otherwise says not restated

**Routed to:** skill:savings-methodology · rounds 5 · 19.5s

**Tools:** get_corpus_facts({"query": "total spend in EUR, last 12 months"}); get_corpus_facts({"query": "top suppliers by spend in EUR, last 12 months"}); get_corpus_facts({"query": "total spend in EUR, last 18 months"})

**Answer:**

The query refers to an "EUR/USD swing" creating an apparent saving, but no specific supplier, item, PO, or price data has been provided.

From the corpus:

- Total spend in EUR over the last 12 months: **€27,929,613.21**
- Total spend in EUR over the last 18 months: **€27,929,613.21** (same as 12-month total — no new data in the additional 6 months)
- No supplier or item has been specified to link to the EUR/USD swing.
- No price or volume data for any item in EUR or USD is available in the corpus.
- No index series or FX rate data is loaded for neutralisation.

### Conclusion:
There is **no evidence** in the corpus to support a saving due to an EUR/USD swing. The total spend in EUR is known, but no item, supplier, or price history is available to assess a price change, volume change, or FX effect.

### Status:
**DRAFT** — missing evidence.

### Missing Evidence:
- No specific item, supplier, or PO to link the saving to.
- No before-and-after prices in EUR or USD.
- No volume data.
- No FX rate or index series to neutralise the effect.
- No indication of whether the saving is due to price, volume, or FX.

Until this evidence is provided, no saving can be validated.

---

## spend-analysis / e2 (edge)

**Prompt:** Are we buying the same laptops from many suppliers?

**Expect:** similar_items with evidence; candidates for demand-aggregation, not confirmed duplicates; hands over to demand-aggregation

**Routed to:** no skill · rounds 2 · 9.9s

**Tools:** similar_items({"description": "laptop"})

**Answer:**

No recorded purchase description shares enough descriptive words with 'laptop' at a similarity threshold of 0.6 to confirm that the same item is being bought from multiple suppliers. The system found no matches in PO, invoice, or quote lines. 

This does not mean laptops are not being bought from multiple suppliers — it only means that the descriptions in the records do not sufficiently overlap to be flagged as similar. To confirm whether the same laptops are being sourced from multiple suppliers, a person must verify the item specifications, unit of measure, and item codes.

---

## supplier-onboarding-checklist / e2 (edge)

**Prompt:** Two records exist for what looks like the same supplier.

**Expect:** Near-duplicate finding for the master-data owner; asks which record to onboard; does not merge

**Routed to:** no skill · rounds 4 · 3.0s

**Tools:** supplier_lookup({"supplier": "TechGlobal Solutions"}); supplier_lookup({"supplier": "TechGlobal Ltd"})

**Answer:**

I cannot find any records for "TechGlobal Solutions" or "TechGlobal Ltd" in the supplier database. Please confirm the correct supplier name or ID so I can proceed.

---

## supplier-performance-review / e3 (edge)

**Prompt:** Their contract says 98 percent on-time but we measured 90 percent.

**Expect:** Quotes the clause; checks the measure matches the contract's definition and that the record is assessable; only then reports a shortfall; otherwise not measurable

**Routed to:** skill:contract-clause-review · rounds 2 · 4.5s

**Tools:** search_contracts({"query": "their contract"})

**Answer:**

No contract was found in the register. The clause about on-time performance cannot be reviewed because the contract has not been uploaded or indexed. Please upload the contract or provide the exact clause text for review.

---

## supplier-risk-assessment / e2 (edge)

**Prompt:** The supplier sent us their ISO 14001 certificate and says they are ESG compliant.

**Expect:** A certificate on file is not an ESG rating; records ESG as not performed unless a rated record exists; says what a performed check needs (provider, date)

**Routed to:** skill:supplier-sustainability-review · rounds 4 · 10.4s

**Tools:** get_policy({"type": "sustainability"}); get_policy({"query": "sustainability"}); supplier_lookup({"supplier": "supplier name or ID"})

**Answer:**

I cannot proceed without a valid supplier name or ID. Please provide the supplier's name or ID so I can retrieve their master record and assess their ESG compliance evidence.

---

## supplier-selection / e4 (edge)

**Prompt:** Our spend would be a very small share of one supplier's revenue, but they are strategic.

**Expect:** Applies the dependency test with the working shown; keeps the strategic supplier and says why rather than deprioritising

**Routed to:** no skill · rounds 2 · 7.7s

**Tools:** supplier_lookup({"supplier": "strategic supplier name"})

**Answer:**

The supplier name or ID provided is not in the system. To proceed, please confirm the correct supplier name or ID so we can retrieve their profile, risk score, certifications, and other relevant data.

---

## supplier-sustainability-review / e2 (edge)

**Prompt:** Their EcoVadis rating is from three years ago.

**Expect:** Stale beyond the configured age; excluded from in-date coverage; refresh requested

**Routed to:** skill:supplier-sustainability-review · rounds 5 · 9.8s

**Tools:** get_policy({"type": "sustainability"}); get_policy({"slug": "sustainability"}); get_policy({"query": "sustainability"})

**Answer:**

No governed sustainability policy exists. The review will proceed without a defined target or required check threshold. Coverage will be reported as a share of spend, and the target is not governed.

The EcoVadis rating is not recorded in the compliance screening register. The system does not hold a record of a three-year-old rating. No evidence of an ESG rating, forced-labour check, or conflict-minerals check is on file for the supplier. 

To address this, a new ESG rating must be obtained and recorded. The responsible party is compliance, as this is a required ESG check.

---

## total-cost-of-ownership / e2 (edge)

**Prompt:** The second quote does not mention installation or training.

**Expect:** Those items are not stated for that option; its total is a lower bound and is labelled so; asks the supplier via the user

**Routed to:** no skill · rounds 3 · 8.6s

**Tools:** extract_clauses({"document_id": "Q2026-002", "containing": "installation"}); extract_clauses({"document_id": "Q2026-002", "containing": "training"})

**Answer:**

The document ID for the second quote is not available. Please provide the correct document ID or a reference (e.g., quote number, supplier name, or date) so I can locate the quote and check whether installation or training is mentioned.

---

## urgent-sourcing / e2 (edge)

**Prompt:** The supplier's lead time field is empty.

**Expect:** Lead time is a question for the supplier, not an estimate; candidate marked "fit unknown"; quick quote asks for it

**Routed to:** no skill · rounds 3 · 2.3s

**Tools:** supplier_lookup({"supplier": "supplier_name_or_id"})

**Answer:**

The supplier name or ID provided does not match any record in the system. Please confirm the correct supplier name or ID to proceed.

---
