# Skills and tools library: the 125-row sheet against what exists (2026-10-22)

Source: the three screenshots of the library sheet (SC-001 to SC-125: 50 skills, 75 tools, 12 domains).
"Built" means it exists on local Development under that name or an equivalent; "equivalent" names the
thing that already does the job; "dormant" means a drafted skill with `status: dormant` (checked, not
routed, not listed, not usable); "cannot" means the platform holds no data or no action for it yet.

## 0. Shared / cross-cutting
| SC | Name | Kind | Status |
|---|---|---|---|
| 001 | query_erp | Tool | equivalent: get_corpus_facts, open_orders, invoice_status, supplier_items (read the same records) |
| 002 | supplier_lookup | Tool | built |
| 003 | parse_document | Tool | cannot as an agent tool yet (the extraction pipeline exists for uploads; no on-demand tool) |
| 004 | web_search | Tool | exists in the research loop only (services/supplier_enrichment), governed by egress policy |
| 005 | web_fetch | Tool | exists in the research loop only, as fetch_url |
| 006 | send_message | Tool | cannot (a send); supplier-message-sending drafted dormant |
| 007 | Procurement Policy & Compliance | Skill | built: procurement-policy-compliance |
| 008 | Brand / House Style | Skill | dormant, built: brand-house-style |

## 1. Category & sourcing strategy
| 009 | build_spend_cube | Tool | built (no category dimension; shown summary) |
| 010 | classify_spend | Tool | built |
| 011 | commodity_index_lookup | Tool | built |
| 012 | should_cost_model | Tool | equivalent: calculation cost.should_cost |
| 013 | savings_tracker | Tool | built |
| 014 | Category Strategy | Skill | built |
| 015 | Make-vs-Buy Analysis | Skill | dormant, built: make-vs-buy-analysis |
| 016 | Demand Aggregation | Skill | built |
| 017 | Savings Methodology | Skill | built |

## 2. Sourcing & supplier selection
| 018 | supplier_discovery | Tool | built |
| 019 | create_rfx | Tool | cannot (an action); rfx-issuance drafted dormant |
| 020 | run_e_auction | Tool | cannot (an action); e-auction-execution drafted dormant |
| 021 | normalise_bids | Tool | equivalent: calculation bids.normalise |
| 022 | weighted_scoring | Tool | equivalent: calculation scoring.weighted_total |
| 023 | tco_calc | Tool | equivalent: compare_total_cost and tco.total_cost |
| 024 | RFx Design | Skill | built |
| 025 | Supplier Selection | Skill | built |
| 026 | Auction Strategy | Skill | built as e-auction-design |
| 027 | Negotiation Planning | Skill | built |
| 028 | Award Recommendation | Skill | built |

## 3. Contract management
| 029 | extract_clauses | Tool | built |
| 030 | search_contracts | Tool | built |
| 031 | redline_compare | Tool | built |
| 032 | e_signature | Tool | cannot (a signing) |
| 033 | obligation_tracker | Tool | built |
| 034 | Contract Drafting | Skill | dormant, built: contract-drafting |
| 035 | Clause Risk Review | Skill | built as contract-clause-review |
| 036 | Incoterms Selection | Skill | built |
| 037 | Pricing Mechanisms | Skill | built |
| 038 | Renewal & Exit Strategy | Skill | built |

## 4. Purchase-to-pay
| 039 | create_requisition | Tool | cannot (an action) |
| 040 | search_catalogue | Tool | cannot (no catalogue data held) |
| 041 | create_po | Tool | cannot (an action); purchase-order-placement drafted dormant |
| 042 | record_goods_receipt | Tool | cannot (an action); goods-receipt-recording drafted dormant |
| 043 | invoice_ocr | Tool | cannot as an agent tool (extraction runs on upload) |
| 044 | three_way_match | Tool | equivalent: invoice_reconciliation |
| 045 | detect_duplicate_payments | Tool | built as duplicate_invoices (suspected pairs by amount and date; a suspicion, never a verdict) |
| 046 | early_payment_discount_calc | Tool | built as calculation finance.early_payment_decision |
| 047 | route_approval | Tool | cannot (an action); approval-routing drafted dormant |
| 048 | Guided Buying | Skill | built as buying-channel-guidance |
| 049 | Tail Spend Management | Skill | built |
| 050 | P2P Exception Handling | Skill | dormant, built: p2p-exception-handling (overbilling and invoice-query cover the read side) |

## 5. Supplier relationship & risk
| 051 | onboard_supplier | Tool | cannot (an action); supplier-onboarding-checklist covers the checks |
| 052 | screen_sanctions | Tool | built (reads the register) |
| 053 | financial_health_score | Tool | built (reads the register) |
| 054 | esg_rating_lookup | Tool | built (reads the register) |
| 055 | monitor_risk_events | Tool | built (reads the register) |
| 056 | map_tier_n_suppliers | Tool | built as supplier_parents (one level from the master; deeper tiers not held) |
| 057 | calc_supplier_scorecard | Tool | built |
| 058 | Supplier Segmentation | Skill | built |
| 059 | Risk Assessment & Mitigation | Skill | built as supplier-risk-assessment |
| 060 | Business Continuity Planning | Skill | built as supply-continuity-planning |
| 061 | Supplier Development | Skill | built as supplier-development-plan |
| 062 | QBR Preparation | Skill | built as supplier-performance-review |

## 6. Demand & supply planning
| 063 | statistical_forecast | Tool | built as calculation forecast.exponential_smoothing |
| 064 | forecast_accuracy | Tool | built as calculation forecast.accuracy |
| 065 | run_mrp | Tool | cannot (no BOM or inventory data held) |
| 066 | check_capacity | Tool | cannot (no capacity data held) |
| 067 | simulate_scenario | Tool | cannot (no planning model held) |
| 068 | Demand Sensing | Skill | dormant, built |
| 069 | S&OP / IBP Process | Skill | dormant, built: sop-ibp-process |
| 070 | Consensus Forecasting | Skill | dormant, built |
| 071 | Constrained Supply Planning | Skill | dormant, built |

## 7. Inventory management
| 072 | abc_xyz_classify | Tool | built as calculation inventory.abc_xyz |
| 073 | eoq_calc | Tool | built as calculation inventory.eoq |
| 074 | safety_stock_calc | Tool | built as calculation inventory.safety_stock |
| 075 | reorder_point_calc | Tool | built as calculation inventory.reorder_point |
| 076 | inventory_turns_calc | Tool | built as calculation inventory.turns |
| 077 | detect_slob | Tool | cannot (no stock-on-hand data held); the calculation needs the user's stock list |
| 078 | Inventory Policy Setting | Skill | dormant, built (ordering-and-inventory covers ordering patterns) |
| 079 | SLOB Management | Skill | dormant, built |
| 080 | Multi-Echelon Optimisation | Skill | dormant, built |

## 8. Logistics & trade
| 081 | get_freight_quote | Tool | cannot (no carrier connection) |
| 082 | book_shipment | Tool | cannot (an action) |
| 083 | optimise_route | Tool | cannot (no routing engine) |
| 084 | track_shipment | Tool | cannot (no carrier connection) |
| 085 | plan_load | Tool | cannot (no dimension data held) |
| 086 | landed_cost_calc | Tool | equivalent: calculation landed_cost.total_cost |
| 087 | hs_code_lookup | Tool | cannot (no tariff data) |
| 088 | duty_calc | Tool | cannot (no tariff data; the landed-cost calculation takes a stated duty rate) |
| 089 | Mode Selection | Skill | dormant, built |
| 090 | Carrier Procurement | Skill | dormant, built |
| 091 | Trade Compliance | Skill | dormant, built |
| 092 | Network Design | Skill | dormant, built |

## 9. Warehousing
| 093 | query_wms | Tool | cannot (no WMS connection) |
| 094 | optimise_slotting | Tool | cannot |
| 095 | optimise_pick_path | Tool | cannot |
| 096 | schedule_cycle_counts | Tool | cannot (an action on WMS data) |
| 097 | Warehouse Layout Design | Skill | dormant, built |
| 098 | Labour Planning | Skill | dormant, built |
| 099 | Returns / Reverse Logistics | Skill | dormant, built |

## 10. Quality
| 100 | log_inspection | Tool | cannot (a write; no inspection register yet) |
| 101 | log_non_conformance | Tool | cannot (a write; no NCR register yet) |
| 102 | calc_supplier_ppm | Tool | built as calculation quality.ppm |
| 103 | Root Cause Analysis | Skill | dormant, built |
| 104 | 8D Problem Solving | Skill | dormant, built: eight-d-problem-solving |
| 105 | CAPA Management | Skill | dormant, built |
| 106 | Supplier Audit | Skill | dormant, built |

## 11. Sustainability & compliance
| 107 | scope3_emissions_calc | Tool | built as calculation sustainability.scope3 (factors supplied, never from memory) |
| 108 | send_esg_survey | Tool | cannot (a send) |
| 109 | forced_labour_screen | Tool | built (reads the register) |
| 110 | conflict_minerals_check | Tool | built (reads the register) |
| 111 | Sustainable Sourcing | Skill | dormant, built (supplier-sustainability-review covers the review side) |
| 112 | CSRD / ESG Reporting | Skill | dormant, built: csrd-esg-reporting |
| 113 | Ethical Sourcing Policy | Skill | dormant, built |

## 12. Performance & reporting
| 114 | calc_kpis | Tool | partly: calc_supplier_scorecard (OTIF, PPV), savings_tracker; a cross-cutting KPI tool is not built |
| 115 | render_chart | Tool | built |
| 116 | render_kpi_card | Tool | built |
| 117 | build_pptx | Tool | built |
| 118 | build_docx | Tool | built |
| 119 | build_xlsx | Tool | built |
| 120 | export_file | Tool | cannot (PDF conversion needs a converter the host lacks) |
| 121 | Control Tower Monitoring | Skill | dormant, built |
| 122 | Executive Supply Chain Reporting | Skill | dormant, built (procurement-reporting covers procurement's own results) |
| 123 | Market Analysis | Skill | dormant, built |
| 124 | Chart & KPI Design Standards | Skill | dormant, built |
| 125 | PowerPoint Pack Builder | Skill | dormant, built |

## Totals
- Skills: 50 in the sheet. 20 built under the sheet's name or an equivalent; 30 drafted dormant (all done 2026-10-22 evening; 86 skills on disk, 36 dormant incl. the six action playbooks).
  (The 56 skills on disk also include ones the sheet does not name: supplier-offboarding,
  contract-amendment-review, specification-writing, price-increase-challenge, supplier-communication-
  drafting, delivery-issue-handling, dispute-and-claims, procurement-process-guide, framework-call-off,
  policy-exception-request, budget-commitment-tracking, vendor-master-data-quality, invoice-query-
  handling, business-case, rebates-and-incentives, sla-design, price-benchmarking, quote-comparison,
  overbilling-investigation, contract-portfolio-review, opportunity-pipeline-review, maverick-spend-
  review, urgent-sourcing, payment-terms-optimisation, spend-analysis, and the six dormant action skills.)
- Tools: 75 in the sheet. 38 built or equivalent (the 12 calculations and data tools above were built on 2026-10-22);
  37 cannot be built honestly on this platform (actions, sends, or data the platform does not hold).
