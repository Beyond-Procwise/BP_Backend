# Test Dataset Answer Key

7070 instances planted across 30 defect types.

Negative controls are the important half: any finding against them is a
false positive, and the scenario fails.

| Ref | Defect | Kind | Target | Planted | Test cases |
|---|---|---|---|---|---|
| D01 | Duplicate invoice, exact resubmission | true_positive | 180 | 180 | A7, B5 |
| D02 | Near-duplicate invoice | true_positive | 120 | 120 | B5 |
| D03 | PO to invoice unit-price mismatch | true_positive | 340 | 340 | B1 |
| D04 | PO to invoice quantity mismatch | true_positive | 210 | 210 | B2 |
| D05 | Invoice with no purchase order | true_positive | 620 | 620 | B3 |
| D06 | Invoice exceeds PO beyond tolerance | true_positive | 260 | 260 | B4 |
| D07 | Approval bypass | true_positive | 145 | 145 | B4, E3 |
| D08 | Split PO to evade threshold | true_positive | 75 | 75 | B4 |
| D09 | Award not to lowest compliant quote | true_positive | 400 | 400 | C5 |
| D10 | Wide unit-price spread for same item | true_positive | 520 | 520 | D1 |
| D11 | Tail-spend consolidation opportunity | true_positive | 340 | 340 | D2 |
| D12 | Single-source concentration | true_positive | 160 | 160 | D3 |
| D13 | Expired insurance certificate | true_positive | 310 | 310 | D4 |
| D14 | Lapsed ESG certification | true_positive | 275 | 275 | D4 |
| D15 | Supplier near-duplicate names | true_positive | 240 | 240 | C1 |
| D16 | Contract obligation breached | true_positive | 90 | 90 | D5 |
| D17 | Contract obligation expiring soon | true_positive | 130 | 130 | D5 |
| D18 | Auto-renewal notice window missed | true_positive | 45 | 45 | D5 |
| D19 | Payment terms breach | true_positive | 230 | 230 | B4 |
| D20 | Currency and total mismatch | true_positive | 95 | 95 | A2, A6 |
| D21 | Missing required extraction fields | true_positive | 380 | 380 | A8 |
| D22 | Services line with no quantity | negative_control | 800 | 800 | A5, B6 |
| D23 | Legitimate credit note | negative_control | 190 | 190 | B6 |
| D24 | Contracted price increase within index | negative_control | 150 | 150 | D6 |
| D25 | Justified sole source | negative_control | 60 | 60 | C5, D3 |
| D26 | Genuinely distinct near-name suppliers | negative_control | 80 | 80 | C1 |
| D27 | Cost-centre budget overrun | true_positive | 85 | 85 | E2 |
| D28 | Cross-entity price inconsistency | true_positive | 220 | 220 | E4 |
| D29 | Supplier onboarded separately per entity | true_positive | 140 | 140 | E4 |
| D30 | Justified cross-entity price difference | negative_control | 180 | 180 | E5 |
