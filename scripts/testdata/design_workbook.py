"""Generate the reviewable Excel design workbook for the BP_Backend / SpendIQ test dataset.

Read-only against the live databases. It reads the real relation inventory from
bp_sqldb AND uicanvas, and the real 5-level category taxonomy from
uicanvas.proc.bp_category, so the workbook describes the system as it actually is
rather than as anyone remembers it. It writes nothing to any database.

    .venv/bin/python -m scripts.testdata.design_workbook
"""
from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "testdata" / "BP_TestData_Design.xlsx"

ACCENT = "1D4ED8"
HEAD_FILL = PatternFill("solid", fgColor="1D4ED8")
BAND_FILL = PatternFill("solid", fgColor="F1F5F9")
NOTE_FILL = PatternFill("solid", fgColor="FEF3C7")
NEG_FILL = PatternFill("solid", fgColor="FEE2E2")
_T = Side(style="thin", color="CBD5E1")
BORDER = Border(left=_T, right=_T, top=_T, bottom=_T)

TARGET_SUPPLIERS = 5000

# Tables that are backups, dated snapshots or scratch copies - out of test scope.
_EXCLUDE = re.compile(
    r"(_bkp|_bkup|_backup|_old\d*$|_test$|_stage$|june12|_\d{6}$|may_4th|_27_04|"
    r"_new\d*$|_excel$|_agent_\d+$)",
    re.I,
)


def _excluded(name: str) -> bool:
    return bool(_EXCLUDE.search(name))


# --------------------------------------------------------------------------
# Designed content
# --------------------------------------------------------------------------

DECISIONS = [
    ("Destination", "New bp_testdb AND uicanvas_test",
     "Both databases are cloned structurally and seeded coherently. Switch worlds via DB_NAME in .env "
     "and the gateway's connection string. Neither live database is written to.",
     "Agreed 24 Jul"),
    ("Database scope", "Both bp_sqldb and uicanvas, linked",
     "The same 5,000 suppliers and the same 5-level taxonomy exist on both sides, joined by an explicit "
     "ID crosswalk, so a supplier is the same entity whichever screen you open.",
     "Agreed 24 Jul"),
    ("Ingestion depth", "Hybrid: bulk rows + golden document set",
     "Bulk volume inserted as rows for scale, plus ~60 real files genuinely extracted so the ingestion "
     "path is exercised.",
     "Agreed 24 Jul"),
    ("Planted defects", "Yes, with a published answer key",
     "Every detector gets something to find, and a ground-truth file states exactly what was planted, "
     "including cases that must NOT be flagged.",
     "Agreed 24 Jul"),
    ("Scale & history", "3.5 years, ~40k documents, ~GBP 480M spend",
     "Jan 2023 to Jul 2026, enough depth for year-on-year trends and seasonality.",
     "Agreed 24 Jul"),
    ("Test focus", "All four areas, plus multi-entity",
     "Extraction accuracy, three-way match, deals/ranking/negotiation, opportunities/compliance, and "
     "the multi-entity area that the organisation decision added. 32 test cases - see Test Scenarios.",
     "Agreed 24 Jul"),
    ("Category taxonomy", "The real 5-level tree from uicanvas.proc.bp_category",
     "6 families, 19 / 54 / 114 / 242 levels below, with UNSPSC code, ESG impact, spend classification, "
     "risk rating, owner, audit frequency and policy coverage carried through.",
     "Corrected 24 Jul"),
    ("Supplier ID convention", "Keep both, add a crosswalk",
     "bp_sqldb uses name-derived ids (SUP-AbcMedia); uicanvas uses sequential ids (SI000001). The test "
     "data reproduces both and adds a mapping table, so the mismatch is testable rather than hidden.",
     "Agreed 24 Jul"),
    ("Buying organisation", "Multi-entity: 6 legal entities under one group",
     "A group parent, 6 buying entities in different countries and base currencies, a 5-level business "
     "unit tree (~400 units) and 500 cost centres with real budgets and per-cost-centre approval "
     "thresholds. Your schema already models all of this - see the Organisation tab.",
     "Agreed 24 Jul"),
    ("Data isolation test", "Included, expected to fail initially",
     "Test E6 checks whether a user in one entity can see another entity's data. Your notes record that "
     "/workflows/ask has no authorization and user_id does not scope retrieval, so this should fail "
     "today. It is a measurable target, not a defect in the dataset.",
     "Agreed 24 Jul"),
]

# level, id pattern, name, country, currency, share of spend, cost centres, what it exercises
ORG_ENTITIES = [
    ("Group", "ORG-GRP", "Beyond Procurement Group plc", "United Kingdom", "GBP", "100% (roll-up)", "-",
     "Group-level roll-up, consolidated reporting, non-trading parent. Owns no cost centres directly"),
    ("Entity", "ORG-UK", "Beyond Procurement UK Ltd", "United Kingdom", "GBP", "42%", 185,
     "Largest entity. UK VAT, postcode-to-region derivation, base-currency reporting"),
    ("Entity", "ORG-DE", "Beyond Procurement Deutschland GmbH", "Germany", "EUR", "16%", 92,
     "EU VAT, comma decimal separators, FX conversion into group GBP"),
    ("Entity", "ORG-US", "Beyond Procurement North America Inc", "United States", "USD", "18%", 98,
     "US date formats, state/ZIP addresses, dollar-symbol disambiguation"),
    ("Entity", "ORG-IE", "Beyond Procurement Ireland Ltd", "Ireland", "EUR", "12%", 61,
     "Eircode formats, EU VAT, shared-service overlap with UK"),
    ("Entity", "ORG-IN", "Beyond Procurement India Pvt Ltd", "India", "INR", "8%", 42,
     "GSTIN, lakh/crore magnitudes, offshore professional services"),
    ("Entity", "ORG-AE", "Beyond Procurement Middle East FZE", "United Arab Emirates", "AED", "4%", 22,
     "TRN, free-zone entity, smallest entity for tail-behaviour tests"),
]

ORG_HIERARCHY = [
    ("Group parent", 1, "ORG-GRP", "Non-trading holding company",
     "Consolidated spend must equal the sum of all six entities, with no double counting"),
    ("Legal entities", 6, "ORG-XX", "Buying companies, each with its own base currency and tax regime",
     "Cross-entity comparison, FX consolidation, entity-scoped access control"),
    ("Business unit L1", 6, "BU-1xxxx", "Function: Operations, Sales, Finance, Corporate, Technology, Supply Chain",
     "Matches the convention already present in your cost_centre data"),
    ("Business unit L2", 40, "BU-2xxxx", "Region within function: Europe, North America, Middle East, LATAM, APAC",
     "Regional spend analysis and roll-up"),
    ("Business unit L3", 120, "BU-3xxxx", "Department",
     "Departmental budget ownership"),
    ("Business unit L4", 240, "BU-4xxxx", "Sub-department",
     "Granular attribution of spend"),
    ("Business unit L5", 400, "BU-5xxxx", "Team, each with a named head and email",
     "Lowest level of BU attribution; feeds approval routing"),
    ("Cost centres", 500, "CC0000xx", "6-level cost centre with budget, threshold and category link",
     "Budget vs actual vs forecast, per-cost-centre approval thresholds, category linkage"),
]

# ref, area, name, what it proves, data setup, pass criterion, defect refs, golden refs
SCENARIOS = [
    ("A1", "A. Extraction accuracy", "Header fields across all six families",
     "The engine reads the right supplier, buyer, document number, dates and totals regardless of layout",
     "24 clean golden documents, one per family per document type",
     "100% of required header fields match the expected-values file byte-for-byte after normalisation",
     "-", "G01-G24"),
    ("A2", "A. Extraction accuracy", "Line items complete and arithmetic reconciles",
     "No dropped, duplicated or phantom lines; line totals sum to the document total",
     "Golden documents including a page-break split table and a 100+ line document",
     "Line count matches expected; sum of line totals equals document net total to the penny",
     "D20", "G27, G28"),
    ("A3", "A. Extraction accuracy", "Quantity greater than one (regression guard)",
     "unit_price x qty is booked as the line total, and the line total is NOT booked as unit price",
     "Invoices with qty 2-40. Your live corpus is entirely qty=1, which is what hid this bug before",
     "unit_price x qty = line_total on every line; no line where unit_price equals line_total when qty>1",
     "-", "G25"),
    ("A4", "A. Extraction accuracy", "Five-level category assignment",
     "A line description resolves to the correct L1-L5 path, not just a top-level guess",
     "Line descriptions drawn from the real leaf taxonomy across all six families",
     "At least 90% of lines assigned to the correct L5 leaf; 100% to the correct L1 family",
     "-", "G01-G24"),
    ("A5", "A. Extraction accuracy", "Absent data stays NULL",
     "Lump-sum services lines legitimately have no quantity or unit price and must not be fabricated",
     "800 services lines with no qty or unit price, correct as issued",
     "qty and unit_price are NULL, line_total populated, and no value is invented",
     "D22", "G26"),
    ("A6", "A. Extraction accuracy", "Multi-currency and FX conversion",
     "Foreign-currency documents convert correctly at the document-date rate",
     "Documents in 11 currencies including comma-decimal, zero-decimal and $-ambiguous cases",
     "Converted GBP total re-derives exactly from bp_fx_rates at the document date",
     "D20", "G34-G37, G46"),
    ("A7", "A. Extraction accuracy", "Duplicate and amended re-upload",
     "Re-uploading the same file is recognised; re-uploading a corrected file supersedes",
     "Two golden files re-uploaded, one identical and one with a corrected total",
     "doc_action='duplicate' for the identical file, 'updated' for the corrected one; spend not doubled",
     "D01", "G55, G56"),
    ("A8", "A. Extraction accuracy", "Missing required field routes to review",
     "An incomplete document is queued for a human rather than silently promoted",
     "380 documents missing a required field such as supplier VAT or PO reference",
     "Row lands in extraction_review_queue; nothing reaches _trgt with the field absent",
     "D21", "G57"),
    ("B1", "B. Three-way match", "Over-billing on unit price",
     "An invoice billed above the PO unit price is caught with the correct dilution figure",
     "340 invoices with unit price 3-40% above the matching PO line",
     "Every case found, and the reported GBP delta matches the answer key exactly",
     "D03", "-"),
    ("B2", "B. Three-way match", "Quantity variance",
     "An invoice for more units than the PO authorised is caught",
     "210 invoices with quantity above the PO quantity",
     "All 210 found, none missed, delta correct",
     "D04", "-"),
    ("B3", "B. Three-way match", "Spend with no purchase order",
     "Off-contract, unapproved spend surfaces rather than passing through",
     "620 invoices with no PO chain behind them",
     "All 620 surfaced and aggregated correctly by category and supplier",
     "D05", "-"),
    ("B4", "B. Three-way match", "Tolerance breach and approval bypass",
     "Spend above the PO tolerance, or above the approval threshold with no approver, is flagged",
     "260 tolerance breaches, 145 missing approvals, 75 split POs just under threshold",
     "All three patterns found; split POs grouped as one finding, not three unrelated ones",
     "D06, D07, D08", "-"),
    ("B5", "B. Three-way match", "Duplicate invoice not double-counted",
     "A resubmitted invoice does not inflate reported spend",
     "180 exact duplicates and 120 near-duplicates differing by one character",
     "Exact duplicates blocked; near-duplicates flagged for review; total spend unaffected by both",
     "D01, D02", "G55"),
    ("B6", "B. Three-way match", "NEGATIVE - legitimate cases stay silent",
     "Proves the detectors discriminate rather than flag everything",
     "190 genuine credit notes and 800 lump-sum services lines, all correct as issued",
     "ZERO findings raised. Any finding here is a false positive and fails the scenario",
     "D22, D23", "G26, G33"),
    ("C1", "C. Deals & ranking", "Competing quotes group into one deal",
     "Quotes answering the same requirement land in the same deal",
     "2-5 competing quotes per requirement across 8,000 deals",
     "Every competing quote maps to the correct deal; no quote left orphaned unintentionally",
     "-", "-"),
    ("C2", "C. Deals & ranking", "Unrelated documents do NOT merge",
     "Directly tests your open deal mis-grouping bug, where a GBP 81k supplier was pulled into a GBP 638 deal",
     "Deliberate near-miss pairs: same period, similar value, different supplier and category",
     "ZERO incorrect merges. Every planted near-miss stays in its own deal",
     "-", "-"),
    ("C3", "C. Deals & ranking", "Quote revision supersedes, not duplicates",
     "A v2 quote replaces v1 rather than being counted twice",
     "Revision chains on a share of deals, including v3 in a few cases",
     "Only the latest version counts toward deal value; superseded versions retained but excluded",
     "-", "G41"),
    ("C4", "C. Deals & ranking", "Supplier ranking orders correctly",
     "Competing quotes are ranked on the stated criteria, deal by deal",
     "Deals with quotes differing on price, lead time, ESG certification and risk score",
     "Ranking order matches the answer key for every deal; no deal ranked with an empty input",
     "-", "-"),
    ("C5", "C. Deals & ranking", "Award to the wrong supplier is flagged",
     "An award 4-25% above the lowest compliant quote surfaces with the money left on the table",
     "400 deals awarded away from the lowest compliant quote, with no justification on file",
     "All 400 flagged, foregone saving correct; the 60 justified sole-source cases NOT flagged",
     "D09, D25", "-"),
    ("C6", "C. Deals & ranking", "Negotiation round runs end to end",
     "A negotiation can be opened, a supplier reply recorded, and the session state advanced",
     "Multi-round negotiations with supplier responses across several deals",
     "Session state advances correctly through rounds; responses attach to the right round and deal",
     "-", "-"),
    ("D1", "D. Opportunities & compliance", "Price variance opportunity",
     "The same item bought at widely different prices is identified with real addressable spend",
     "520 item-supplier pairs with over 25% unit price spread",
     "Opportunity raised with correct addressable spend and target price, to the penny",
     "D10", "-"),
    ("D2", "D. Opportunities & compliance", "Tail-spend consolidation",
     "Fragmented low-value buying is identified as a consolidation opportunity",
     "1,160 one-off suppliers and 340 fragmented category-supplier pairs",
     "Opportunity raised per category with correct supplier count and spend",
     "D11", "-"),
    ("D3", "D. Opportunities & compliance", "Single-source concentration",
     "A category dependent on one supplier is surfaced as a supply risk",
     "160 category-supplier pairs holding over 80% of category spend with no competitive quotes",
     "All 160 flagged; the 60 justified sole-source cases NOT flagged",
     "D12, D25", "-"),
    ("D4", "D. Opportunities & compliance", "Supplier compliance failures",
     "Expired insurance and lapsed ESG certification surface against active suppliers",
     "310 expired insurance certificates, 275 lapsed ESG certifications, all on trading suppliers",
     "All 585 surfaced on the compliance screen with the correct supplier and expiry date",
     "D13, D14", "-"),
    ("D5", "D. Opportunities & compliance", "Contract obligations",
     "Breached and imminent obligations appear, read from contract prose",
     "600 contracts, 4,200 obligations, of which 90 breached, 130 due within 60 days, 45 renewal risks",
     "All 265 surfaced with the correct clause reference and due date; none invented",
     "D16, D17, D18", "G51-G53"),
    ("D6", "D. Opportunities & compliance", "NEGATIVE - contracted rises stay silent",
     "A price rise permitted by a CPI clause must not be reported as variance",
     "150 contracted CPI uplifts, each with the clause present in the contract text",
     "ZERO variance findings. Proves the engine reads the contract rather than only the numbers",
     "D24", "G53"),
    ("E1", "E. Multi-entity", "Spend rolls up without loss or double counting",
     "Group total equals the sum of entities, business units and cost centres",
     "6 entities, 400 business units, 500 cost centres, every document attributed",
     "Group total = sum of entities = sum of BU L1 = sum of cost centres, to the penny. Zero orphans",
     "-", "-"),
    ("E2", "E. Multi-entity", "Cost-centre budget overrun",
     "Spending above an allocated budget is detected at cost-centre level",
     "85 cost centres where actual_spend_ytd exceeds budget_allocated_annual",
     "All 85 surfaced with the correct overrun amount and the responsible cost-centre manager",
     "D27", "-"),
    ("E3", "E. Multi-entity", "Approval threshold is per cost centre",
     "Bypass is judged against that cost centre's limit, not one global number",
     "145 approval bypasses spread across cost centres with thresholds from GBP 5k to GBP 250k",
     "Each bypass judged against its own spend_threshold_limit; no false positive from a global default",
     "D07", "-"),
    ("E4", "E. Multi-entity", "Cross-entity price inconsistency",
     "The same supplier charging different entities different prices for the same item is found",
     "220 supplier-item pairs priced differently across two or more entities, plus 140 suppliers "
     "onboarded separately per entity with no group-level agreement",
     "All 360 surfaced with the correct spread and group-level addressable spend",
     "D28, D29", "-"),
    ("E5", "E. Multi-entity", "NEGATIVE - justified entity differences stay silent",
     "Price differences explained by currency, region, volume tier or Incoterms are not variance",
     "180 cross-entity price differences, each with a documented commercial reason",
     "ZERO findings. Proves the engine accounts for entity context rather than comparing raw numbers",
     "D30", "-"),
    ("E6", "E. Multi-entity", "Entity data isolation (EXPECTED TO FAIL)",
     "A user scoped to one entity must not retrieve another entity's documents or spend",
     "Users assigned per entity across the RBAC model, with distinctive data in each entity",
     "A user in ORG-UK retrieves ZERO ORG-DE records. Expected to FAIL today: /workflows/ask has no "
     "authorization and user_id does not scope retrieval. Turns green when that work lands",
     "-", "-"),
]

DEFECTS = [
    ("D01", "Duplicate invoice, exact resubmission", 180, "True positive", "A7, B5",
     "Same supplier, invoice number and total, uploaded twice",
     "Second copy marked duplicate, not counted in spend"),
    ("D02", "Near-duplicate invoice", 120, "True positive", "B5",
     "Same supplier, total and date; reference differs by one character",
     "Flagged for review, not auto-blocked"),
    ("D03", "PO to invoice unit-price mismatch", 340, "True positive", "B1",
     "Invoice unit price 3-40% above the matching PO line",
     "Over-billing finding with the exact GBP delta"),
    ("D04", "PO to invoice quantity mismatch", 210, "True positive", "B2",
     "Invoice quantity exceeds PO quantity",
     "Quantity variance finding"),
    ("D05", "Invoice with no purchase order", 620, "True positive", "B3",
     "Invoice raised with no PO chain in the period",
     "Maverick spend finding, aggregated by category"),
    ("D06", "Invoice exceeds PO beyond tolerance", 260, "True positive", "B4",
     "Invoice total 5-22% above PO total, above the 5% tolerance",
     "Tolerance breach requiring approval"),
    ("D07", "Approval bypass", 145, "True positive", "B4",
     "Invoice above threshold with no matching row in bp_approval",
     "Control failure finding naming the missing approver role"),
    ("D08", "Split PO to evade threshold", 75, "True positive", "B4",
     "Two or three POs to one supplier in one week, each just under threshold",
     "Split-purchase finding grouping the sibling POs"),
    ("D09", "Award not to lowest compliant quote", 400, "True positive", "C5",
     "Award 4-25% above the lowest compliant quote, no justification",
     "Award-variance finding with the foregone saving"),
    ("D10", "Wide unit-price spread for same item", 520, "True positive", "D1",
     "Same catalogue item across suppliers with over 25% spread",
     "Opportunity with addressable spend and target price"),
    ("D11", "Tail-spend consolidation opportunity", 340, "True positive", "D2",
     "Category with many one-off suppliers and low average order value",
     "Consolidation opportunity with supplier count and spend"),
    ("D12", "Single-source concentration", 160, "True positive", "D3",
     "Category-supplier pair over 80% of category spend, no competitive quotes",
     "Single-source risk finding"),
    ("D13", "Expired insurance certificate", 310, "True positive", "D4",
     "insurance_expiry_date in the past on a trading supplier",
     "Compliance finding, supplier flagged non-compliant"),
    ("D14", "Lapsed ESG certification", 275, "True positive", "D4",
     "ISO14001 / SA8000 / EcoVadis false where the contract requires it",
     "ESG non-conformance finding"),
    ("D15", "Supplier near-duplicate names", 240, "True positive", "C1",
     "240 clusters of 2-3 rows, e.g. Techworld Ltd / Tech World Limited / TECHWORLD LTD.",
     "Close-call match queued for human confirm, never auto-merged"),
    ("D16", "Contract obligation breached", 90, "True positive", "D5",
     "Obligation past its due date with no evidence of completion",
     "Breached obligation with clause reference"),
    ("D17", "Contract obligation expiring soon", 130, "True positive", "D5",
     "Obligation due within 60 days of the reporting date",
     "Upcoming obligation in the action centre"),
    ("D18", "Auto-renewal notice window missed", 45, "True positive", "D5",
     "Contract auto-renews and the notice window closes within 30 days",
     "Renewal risk finding with the notice deadline"),
    ("D19", "Payment terms breach", 230, "True positive", "B4",
     "Invoice paid outside the contracted payment terms",
     "Terms breach finding with days beyond terms"),
    ("D20", "Currency and total mismatch", 95, "True positive", "A2, A6",
     "Line totals inconsistent with the stated document total after conversion",
     "Arithmetic discrepancy raised before promotion"),
    ("D21", "Missing required extraction fields", 380, "True positive", "A8",
     "Document lacking a required field such as supplier VAT or PO reference",
     "Row lands in extraction_review_queue, not silently promoted"),
    ("D22", "Services line with no quantity", 800, "NEGATIVE control", "A5, B6",
     "Lump-sum services lines with no qty or unit price - correct as issued",
     "MUST NOT be flagged. Any finding here is a false positive"),
    ("D23", "Legitimate credit note", 190, "NEGATIVE control", "B6",
     "Negative-value invoice correctly issued against an earlier invoice",
     "MUST NOT be flagged. Must reduce net spend correctly"),
    ("D24", "Contracted price increase within index", 150, "NEGATIVE control", "D6",
     "Unit price rises in line with a contractual CPI uplift clause",
     "MUST NOT be flagged as variance. Tests the engine reads the contract"),
    ("D25", "Justified sole source", 60, "NEGATIVE control", "C5, D3",
     "Single-source award with documented justification and approval on file",
     "MUST NOT be flagged. Tests justification is honoured"),
    ("D26", "Genuinely distinct near-name suppliers", 80, "NEGATIVE control", "C1",
     "Similar names but different legal entities with different VAT numbers",
     "MUST NOT be auto-merged. Tests the VAT/registration guard"),
    ("D27", "Cost-centre budget overrun", 85, "True positive", "E2",
     "actual_spend_ytd above budget_allocated_annual on a live cost centre",
     "Overrun surfaced with the amount and the responsible manager"),
    ("D28", "Cross-entity price inconsistency", 220, "True positive", "E4",
     "One supplier charging two or more entities different prices for the same item, no reason on file",
     "Finding with the spread and group-level addressable spend"),
    ("D29", "Supplier onboarded separately per entity", 140, "True positive", "E4",
     "The same legal supplier holds a separate record in two or more entities, no group agreement",
     "Fragmented-negotiating-power finding proposing a group-level agreement"),
    ("D30", "Justified cross-entity price difference", 180, "NEGATIVE control", "E5",
     "Price differs across entities for a documented reason: currency, region, volume tier or Incoterms",
     "MUST NOT be flagged. Tests that entity context is read, not just the numbers"),
]

FAMILIES = [
    "IT & Technology", "Marketing & Media", "Facilities & Real Estate",
    "Professional Services", "Logistics & Supply Chain", "Office & Administrative Supplies",
]

GOLDEN_EDGE = [
    ("Facilities & Real Estate", "Invoice", "PDF", "Quantity greater than one",
     "Your whole corpus is qty=1, which hid a 2x unit-price bug. This is the regression guard",
     "unit_price x qty equals line_total; line_total never booked as unit_price"),
    ("Professional Services", "Invoice", "PDF", "Services lump sum, no quantity",
     "Legitimate absence of qty and unit_price must not be fabricated",
     "qty and unit_price NULL, line_total populated"),
    ("Logistics & Supply Chain", "Quote", "PDF", "Rate card with version-history table",
     "Grand total sits in a version-history table, not the section sub-total",
     "total_amount read from the version table, quote_id from the slash-format reference"),
    ("IT & Technology", "Invoice", "PDF", "Table split across a page break",
     "Line-item table continues on page 2 with a repeated header",
     "All lines captured once; header row not read as a line"),
    ("IT & Technology", "Quote", "PDF", "Over 100 line items",
     "Long-table extraction and pagination",
     "Full line count captured; totals reconcile"),
    ("Office & Administrative Supplies", "Invoice", "XLSX", "Native spreadsheet invoice",
     "Exercises the xlsx parser path",
     "Header and lines extracted without a PDF render"),
    ("Office & Administrative Supplies", "Purchase Order", "CSV", "Flat CSV purchase order",
     "Exercises the csv parser path",
     "Header inferred, lines typed correctly"),
    ("Marketing & Media", "Invoice", "PDF", "VAT-inclusive totals",
     "Gross-stated document where net must be derived",
     "Net spend excludes VAT; tax_amount populated"),
    ("Facilities & Real Estate", "Invoice", "PDF", "VAT-exclusive with reverse charge",
     "Reverse-charge wording must not be read as a zero total",
     "Net total correct; tax treatment noted"),
    ("IT & Technology", "Invoice", "PDF", "Credit note, negative values",
     "Negative amounts must survive extraction and reduce spend",
     "Negative line and document totals preserved"),
    ("Logistics & Supply Chain", "Invoice", "PDF", "EUR with comma decimals",
     "European number formatting",
     "1.234,56 read as 1234.56 and converted at the document-date rate"),
    ("IT & Technology", "Invoice", "PDF", "USD with dollar-symbol ambiguity",
     "$ could be USD or AUD; supplier country resolves it",
     "Currency resolved from the supplier address, not guessed"),
    ("Logistics & Supply Chain", "Invoice", "PDF", "JPY zero-decimal currency",
     "No minor units; must not divide by 100",
     "Amount preserved exactly, no phantom decimals"),
    ("Office & Administrative Supplies", "Invoice", "PDF", "Ambiguous date format",
     "03/04/2025 could be March or April; locale resolves it",
     "Date resolved from the supplier country convention"),
    ("Professional Services", "Invoice", "PDF", "Supplier and buyer blocks adjacent",
     "The known supplier/buyer swap failure mode",
     "supplier_name is the issuer, not the recipient"),
    ("Facilities & Real Estate", "Quote", "PDF", "Two-column layout",
     "Multi-column page geometry",
     "Reading order correct, no interleaved text"),
    ("Marketing & Media", "Quote", "PDF", "Revision v2 supersedes v1",
     "Version collapse in deal clustering",
     "v2 supersedes v1; only one version counts toward the deal"),
    ("IT & Technology", "Purchase Order", "PDF", "PO amendment supersedes original",
     "Amended PO must replace, not duplicate",
     "Amendment linked to the original; value not double counted"),
    ("Facilities & Real Estate", "Invoice", "PDF", "Partial delivery invoice",
     "Invoice covers part of a PO",
     "Partial match against PO; remainder still open"),
    ("Office & Administrative Supplies", "Invoice", "PDF", "Consolidated across several POs",
     "One invoice referencing multiple POs",
     "All PO references captured and matched"),
    ("Logistics & Supply Chain", "Invoice", "PDF", "Freight surcharges and fuel levy",
     "Surcharge lines must not be read as goods",
     "Surcharges classified as charges, not items"),
    ("Facilities & Real Estate", "Quote", "PDF", "Unit price to four decimals",
     "Precision must not be rounded away",
     "Four-decimal unit price preserved through to _trgt"),
    ("IT & Technology", "Invoice", "PDF", "Thousands separators in totals",
     "1,234,567.89 formatting",
     "Parsed as 1234567.89, not 1.234"),
    ("Office & Administrative Supplies", "Invoice", "PDF", "Zero-value promotional line",
     "A genuine zero-price line",
     "Zero retained; line not dropped"),
    ("Marketing & Media", "Invoice", "PDF", "Percentage discount line",
     "Discount applied after sub-total",
     "Discount reduces net total and reconciles"),
    ("Professional Services", "Contract", "PDF", "Obligation-rich prose contract",
     "Contract obligations live in prose, not tables",
     "Obligations, parties, due dates and clause refs extracted"),
    ("Professional Services", "Contract", "PDF", "Auto-renewal and notice clause",
     "Renewal risk detection",
     "Renewal date and notice window extracted"),
    ("IT & Technology", "Contract", "PDF", "CPI uplift clause",
     "Feeds negative control D24",
     "Uplift clause found so contracted rises are not flagged as variance"),
    ("IT & Technology", "Purchase Order", "PDF", "PO referencing a quote number",
     "PO-to-quote linkage, your currently open quote_number gap",
     "Quote reference captured and linked to the quote record"),
    ("Marketing & Media", "Invoice", "PDF", "Scanned appearance, low contrast",
     "OCR-quality degradation path",
     "Either extracted correctly or routed to review, never silently wrong"),
    ("Office & Administrative Supplies", "Invoice", "PDF", "Rotated landscape page",
     "Page orientation handling",
     "Text and tables read in the correct orientation"),
    ("Logistics & Supply Chain", "Invoice", "PDF", "Exact re-upload of an earlier golden file",
     "Content-hash dedup on a real upload",
     "doc_action='duplicate'; no second spend record"),
    ("Logistics & Supply Chain", "Invoice", "PDF", "Amended re-upload with corrected total",
     "Same document, corrected figure",
     "doc_action='updated'; prior record superseded"),
    ("Facilities & Real Estate", "Invoice", "PDF", "Missing supplier VAT number",
     "Required-field gap",
     "Routed to the extraction review queue, not promoted"),
    ("Professional Services", "Invoice", "PDF", "Long wrapping descriptions",
     "Descriptions wrapping across three lines",
     "Description reassembled, not split into phantom lines"),
    ("Marketing & Media", "Quote", "PDF", "Multi-currency lines in one document",
     "Lines in GBP and EUR within one quote",
     "Per-line currency respected; single converted total"),
]

VOLUMES = [
    ("Suppliers", "-", "-", "-", "-", 5000, "bp_supplier (both DBs) + crosswalk"),
    ("Requirements", 1720, 1850, 1780, 650, 6000, "bp_requirement"),
    ("Quotes", 3980, 4290, 4180, 1550, 14000, "bp_quote_raw / _stg / _trgt"),
    ("Purchase Orders", 2280, 2450, 2380, 890, 8000, "bp_purchase_order_raw / _stg / _trgt"),
    ("Invoices", 3280, 3520, 3420, 1280, 11500, "bp_invoice_raw / _stg / _trgt"),
    ("Contracts", 170, 185, 175, 70, 600, "bp_contracts / bp_contract_raw"),
    ("Documents total", 11430, 12295, 11935, 4440, 40100, "All three tiers"),
    ("Quote line items", 29800, 32100, 31300, 11800, 105000, "bp_quote_line_items_*"),
    ("PO line items", 15900, 17150, 16650, 6300, 56000, "bp_po_line_items_*"),
    ("Invoice line items", 22700, 24400, 23700, 9200, 80000, "bp_invoice_line_items_*"),
    ("Contract obligations", 1190, 1295, 1225, 490, 4200, "bp_contract_obligation"),
    ("Line items total", 69590, 74945, 72875, 27790, 245200, "All line-item tables"),
    ("Deals", 2290, 2470, 2400, 840, 8000, "bp_deal + bp_deal_document_map"),
    ("Catalogue items", "-", "-", "-", "-", 5000, "item / bp_products, mapped to L5 leaves"),
    ("Legal entities", "-", "-", "-", "-", 6, "buyer_org_id on every document"),
    ("Business units", "-", "-", "-", "-", 400, "business_unit (5 levels, currently 0 rows)"),
    ("Cost centres", "-", "-", "-", "-", 500, "cost_centre (6 levels, currently 500 junk rows)"),
    ("Users", "-", "-", "-", "-", 240, "users / roles_and_access, scoped per entity"),
    ("Invoiced spend (GBP m)", 120.0, 135.0, 152.0, 73.0, 480.0, "Net of VAT, at bp_fx_rates"),
]

SUPPLIER_TIERS = [
    ("Strategic", 120, "2.4%", "38%", "60-140", "Contracted, preferred, multi-year, quarterly review",
     "Contracts, obligations, rankings, negotiations, TPRM records"),
    ("Core", 700, "14.0%", "41%", "12-45", "Repeat purchasing, some contracted, annual review",
     "Regular quote/PO/invoice chains, supplier reviews"),
    ("Tail", 3020, "60.4%", "18%", "2-9", "Occasional, no contract, transactional",
     "Sparse chains, maverick invoices, consolidation opportunities"),
    ("One-off", 1160, "23.2%", "3%", "1", "Single transaction, never used again",
     "One document each, feeds tail-spend rationalisation"),
]

VERIFICATION = [
    ("V01", "Row counts", "Every in-scope table reaches its planned row count", "Blocks sign-off"),
    ("V02", "Referential integrity", "No orphan line items, deal maps or document references", "Blocks sign-off"),
    ("V03", "Cross-database coherence", "Every supplier and category resolves identically on both sides via the crosswalk", "Blocks sign-off"),
    ("V04", "Arithmetic", "Line totals sum to document totals on every non-defect document", "Blocks sign-off"),
    ("V05", "FX", "Every converted total re-derives from bp_fx_rates at the document date", "Blocks sign-off"),
    ("V06", "Taxonomy integrity", "Every line resolves to a valid L1-L5 path in the real L1-L5 tree", "Blocks sign-off"),
    ("V07", "Organisation roll-up", "Group spend equals the sum of entities, business units and cost centres, to the penny", "Blocks sign-off"),
    ("V08", "Screen coverage", "Every UI screen's backing query returns a non-empty result", "Blocks sign-off"),
    ("V09", "Endpoint coverage", "Every FastAPI route and gateway route returns 200 with a non-empty payload", "Blocks sign-off"),
    ("V10", "Scenario A-D true positives", "Each planted defect is found by its detector", "Scored per scenario"),
    ("V11", "Scenario A-D negative controls", "No negative control produces a finding", "Blocks sign-off"),
    ("V12", "Golden set extraction", "Extraction of the 60 golden files matches expected values", "Scored as accuracy %"),
    ("V13", "Determinism", "Two builds with the same seed produce identical checksums", "Blocks sign-off"),
    ("V14", "Isolation", "bp_sqldb and uicanvas row counts unchanged before and after the build", "Blocks sign-off"),
]

COVERAGE_NOTES = {
    "bp_supplier": ("Supplier master", "Generated, 5,000 rows, all 51 columns, both databases"),
    "supplier": ("Supplier master (uicanvas)", "Generated, same 5,000 entities under SI-style ids"),
    "bp_supplier_alias": ("Entity resolution", "Generated from the near-duplicate clusters"),
    "bp_supplier_enrichment": ("Supplier research", "Generated for strategic and core suppliers"),
    "bp_supplier_ranking": ("Supplier ranking", "Generated per deal with competing quotes"),
    "bp_supplier_review": ("Supplier review HITL", "Generated from close-call name matches"),
    "bp_supplier_name_reject": ("Entity resolution guard", "Generated from the negative-control name pairs"),
    "bp_tprm_supplier": ("Third-party risk", "Generated for strategic and core suppliers"),
    "bp_category": ("Category taxonomy", "Copied verbatim from uicanvas: the real 5-level tree, verbatim"),
    "category": ("Category hierarchy", "Copied verbatim, parent/child and full path preserved"),
    "category_denorm": ("Flattened taxonomy", "Rebuilt from bp_category"),
    "category_mapping": ("Legacy category crosswalk", "Copied verbatim"),
    "cat_product_mapping": ("Product to L5 mapping", "Generated for all 5,000 catalogue items"),
    "bp_category_product_mapping": ("Product to L5 mapping", "Generated for all 5,000 catalogue items"),
    "item": ("Catalogue", "Generated, 5,000 items mapped to L5 leaves with realistic prices"),
    "bp_products": ("Observed products", "Generated from the line items actually issued"),
    "bp_fx_rates": ("Currency conversion", "Copied verbatim from live"),
    "bp_policy": ("Policy governance", "Copied verbatim from live"),
    "bp_prompt": ("Prompt governance", "Copied verbatim from live"),
    "bp_admin_config": ("Admin configuration", "Copied verbatim from live"),
    "bp_vendor_extraction_profiles": ("Vendor extraction hints", "Copied verbatim from live"),
    "bp_complaince_metric_prty_lkup": ("Compliance metric lookup", "Copied verbatim from live"),
    "procurement_patterns": ("Pattern store", "Copied verbatim from live"),
    "bp_opportunity": ("Opportunities", "Produced by running the opportunity engine, not seeded"),
    "bp_detection_finding": ("Compliance & detection", "Produced by running the detectors, not seeded"),
    "bp_extraction_discrepancy": ("Discrepancy detection", "Produced by running the detectors, not seeded"),
    "bp_discrepancy_data": ("Discrepancy detection", "Produced by running the detectors, not seeded"),
    "bp_contracts": ("Contracts", "Generated, 600 contracts for strategic and core suppliers"),
    "bp_contract_obligation": ("Obligations", "Generated, 4,200 obligations incl. breached and expiring"),
    "bp_contract_obligation_party": ("Obligations", "Generated, buyer and supplier parties per obligation"),
    "bp_contract_obligation_run": ("Obligation runs", "Generated, one run per contract"),
    "bp_quote_evaluation": ("Quote evaluation", "Generated for every multi-quote deal"),
    "bp_requirement": ("Requirements", "Generated, 6,000 requirements heading each chain"),
    "bp_demand": ("Demand planning", "Generated by category and period"),
    "bp_approval": ("Approvals", "Generated: present for compliant spend, absent for bypass defects"),
    "bp_decision": ("Decision layer", "Generated, grounded decisions with HITL states"),
    "bp_action": ("Action centre", "Generated, open, in-progress and closed"),
    "bp_agent_actions": ("Agent audit log", "Generated, one trail per document and agent run"),
    "bp_deal": ("Deals", "Generated, 8,000 deals"),
    "bp_deal_document_map": ("Deal linkage", "Generated, every document mapped or deliberately orphaned"),
    "bp_deal_proposal": ("Deal clustering", "Generated: pending, confirmed and rejected"),
    "extraction_review_queue": ("Extraction HITL", "Generated from the missing-field defects"),
    "process_monitor": ("Process monitor", "Generated, one row per document with a realistic doc_action mix"),
    "negotiation_sessions": ("Negotiate", "Generated, multi-round negotiations"),
    "users": ("Access control", "Generated, 240 users scoped per entity for the E6 isolation test"),
    "roles_and_access": ("Access control", "Copied verbatim, extended with per-entity test users"),
    "business_unit": ("Organisation hierarchy",
                      "Generated, 400 units across 5 levels. Currently EMPTY in your live database"),
    "cost_centre": ("Cost centres",
                    "Generated, 500 centres with real budgets, thresholds and category links. "
                    "Replaces the 500 placeholder rows currently present"),
    "bp_contracts": ("Contracts",
                     "Generated, 600 contracts attributed to entity, business unit and cost centre"),
}


# --------------------------------------------------------------------------
# Live reads
# --------------------------------------------------------------------------

def _conn(db: str):
    load_dotenv(ROOT / ".env")
    return psycopg2.connect(
        host=os.getenv("DB_HOST"), dbname=db, user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"), port=os.getenv("DB_PORT"), connect_timeout=25)


def read_taxonomy() -> list[tuple]:
    """The real 5-level taxonomy, straight from uicanvas.proc.bp_category."""
    conn = _conn("uicanvas")
    cur = conn.cursor()
    cur.execute("""
        select category_level_1, category_level_2, category_level_3,
               category_level_4, category_level_5, unspsc_code, esg_impact,
               category_status, spend_classification, category_risk_rating,
               audit_frequency, policy_coverage
        from proc.bp_category
        order by 1,2,3,4,5
    """)
    rows = cur.fetchall()
    conn.close()

    # Apportion 5,000 primary supplier assignments across the leaves,
    # weighted so larger families carry proportionally more suppliers.
    n = len(rows)
    fam_counts = Counter(r[0] for r in rows)
    weights = [1.0 / fam_counts[r[0]] * fam_counts[r[0]] for r in rows]  # uniform per leaf
    total_w = sum(weights)
    exact = [w * TARGET_SUPPLIERS / total_w for w in weights]
    floors = [int(v) for v in exact]
    rem = TARGET_SUPPLIERS - sum(floors)
    order = sorted(range(n), key=lambda i: exact[i] - floors[i], reverse=True)
    for i in order[:rem]:
        floors[i] += 1

    return [(*r[:5], floors[i], r[5], r[6], r[7], r[8], r[9], r[10], r[11])
            for i, r in enumerate(rows)]


def read_scope() -> list[tuple]:
    """Every relation across every database and schema."""
    out = []
    for db in ["bp_sqldb", "uicanvas", "ses"]:
        try:
            conn = _conn(db)
        except Exception as exc:
            out.append((db, "?", "UNREACHABLE", "-", "-", "No", str(exc)[:60], ""))
            continue
        cur = conn.cursor()
        cur.execute("""
            select table_schema, table_name, table_type
            from information_schema.tables
            where table_schema not in ('information_schema','pg_catalog')
              and table_schema not like 'pg_temp%' and table_schema not like 'pg_toast%'
            order by table_schema, table_name
        """)
        for schema, name, ttype in cur.fetchall():
            if ttype == "VIEW":
                out.append((db, schema, name, "view", "derived", "Yes",
                            "Deals and pipeline views",
                            "Recreated with the schema; populated by its base tables"))
                continue
            try:
                cur.execute(f'select count(*) from "{schema}"."{name}"')
                live = cur.fetchone()[0]
            except Exception:
                conn.rollback()
                live = "n/a"
            if _excluded(name):
                out.append((db, schema, name, "table", live, "No",
                            "Backup / dated snapshot / scratch copy",
                            "Not seeded. Structure cloned so nothing breaks, left empty"))
                continue
            area, how = COVERAGE_NOTES.get(
                name, ("Supporting table", "Generated in proportion to the documents it serves"))
            out.append((db, schema, name, "table", live, "Yes", area, how))
        conn.close()
    return out


# --------------------------------------------------------------------------
# Workbook plumbing
# --------------------------------------------------------------------------

def _sheet(wb: Workbook, title: str, headers: list[str], rows: list[tuple], widths: list[int],
           intro: str | None = None, total_cols: tuple[int, ...] = (),
           highlight: tuple[int, str] | None = None) -> None:
    ws = wb.create_sheet(title)
    r = 1
    if intro:
        c = ws.cell(row=1, column=1, value=intro)
        c.font = Font(italic=True, color="475569", size=10)
        c.alignment = Alignment(wrap_text=True, vertical="center")
        ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(headers))
        ws.row_dimensions[1].height = 30
        r = 3

    head = r
    for i, h in enumerate(headers, start=1):
        c = ws.cell(row=head, column=i, value=h)
        c.font = Font(bold=True, color="FFFFFF", size=10)
        c.fill = HEAD_FILL
        c.alignment = Alignment(vertical="center", wrap_text=True)
        c.border = BORDER
    ws.row_dimensions[head].height = 30

    for j, row in enumerate(rows):
        rr = head + 1 + j
        flag = highlight and highlight[1] in str(row[highlight[0]])
        for i, v in enumerate(row, start=1):
            c = ws.cell(row=rr, column=i, value=v)
            c.alignment = Alignment(vertical="top", wrap_text=True)
            c.border = BORDER
            c.font = Font(size=10, bold=bool(flag))
            if flag:
                c.fill = NEG_FILL
            elif j % 2 == 1:
                c.fill = BAND_FILL

    last = head + len(rows)
    if total_cols:
        tr = last + 1
        tc = ws.cell(row=tr, column=1, value="TOTAL")
        tc.font = Font(bold=True, size=10)
        tc.border = BORDER
        for i in range(2, len(headers) + 1):
            c = ws.cell(row=tr, column=i)
            c.border = BORDER
            c.font = Font(bold=True, size=10)
            if i in total_cols:
                col = get_column_letter(i)
                c.value = f"=SUM({col}{head + 1}:{col}{last})"

    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w

    ref = f"A{head}:{get_column_letter(len(headers))}{last}"
    tbl = Table(displayName="T_" + re.sub(r"\W", "", title), ref=ref)
    tbl.tableStyleInfo = TableStyleInfo(name="TableStyleLight1", showRowStripes=False)
    ws.add_table(tbl)
    ws.freeze_panes = ws.cell(row=head + 1, column=1)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    taxonomy = read_taxonomy()
    scope = read_scope()

    in_scope = sum(1 for r in scope if r[5] == "Yes")
    fam_leaves = Counter(r[0] for r in taxonomy)
    distinct_l5 = len({r[4] for r in taxonomy})
    leaf_desc = f"{len(taxonomy)} leaf rows ({distinct_l5} distinct L5 names)"

    wb = Workbook()
    wb.remove(wb.active)

    # ---- Overview ----
    ws = wb.create_sheet("Overview")
    ws.column_dimensions["A"].width = 32
    ws.column_dimensions["B"].width = 112
    lines = [
        ("BP_Backend / SpendIQ test dataset", "Design for review, revision 2, 24 July 2026"),
        ("", ""),
        ("What this is",
         "A proposal for a complete synthetic test dataset: 5,000 suppliers across the real 5-level "
         "category taxonomy, six buying entities, roughly 40,000 documents and 3.5 years of history, seeded into "
         "BOTH databases, with 32 numbered test cases that prove the product actually works."),
        ("Status", "DESIGN ONLY. Nothing has been built or written to any database."),
        ("", ""),
        ("What changed in revision 2",
         f"Revision 1 covered only bp_sqldb and used the wrong category table. The real scope is "
         f"{len(scope)} relations across three databases, and the real taxonomy is the 5-level, "
         f"{len(taxonomy)}-leaf tree in uicanvas.proc.bp_category - not the 49-row lookup in bp_sqldb."),
        ("", ""),
        ("Decisions", "All choices agreed. Nothing outstanding."),
        ("Test Scenarios", f"The {len(SCENARIOS)} numbered test cases across five areas. Start here."),
        ("Organisation / Org Hierarchy", "The six buying entities and the business-unit and cost-centre depth beneath them."),
        ("Category Taxonomy", f"The real tree, read live: 6 families, {leaf_desc}, with UNSPSC and governance attributes."),
        ("Full Scope", f"All {len(scope)} relations across both databases. {in_scope} in test scope, {len(scope) - in_scope} deliberately excluded."),
        ("Suppliers", "Tier shape, and how the 5,000 map onto both ID conventions."),
        ("Volumes", "Documents, line items, deals, catalogue and spend by year."),
        ("Planted Defects", "30 defect types mapped to the test cases. Six are negative controls."),
        ("Golden Documents", "The 60 real files that genuinely run through extraction."),
        ("Verification", "14 checks; 12 block sign-off."),
        ("", ""),
        ("Key finding, needs your decision",
         "Your two databases disagree. bp_sqldb has no 5-level taxonomy and no category link on "
         "suppliers; uicanvas has the full tree but demo-grade transactions. Supplier IDs use two "
         "conventions (SUP-AbcMedia vs SI000001). The test data reproduces both and adds a crosswalk, "
         "so the mismatch becomes testable - but this looks like a real integration gap, not just a "
         "test-data problem."),
        ("Key safety point",
         "Data goes into new bp_testdb and uicanvas_test databases. The generator refuses to run "
         "against bp_sqldb or uicanvas. Switching is one line in .env plus the gateway connection string."),
        ("Key limitation",
         "Bulk data is inserted as rows, not extracted from documents - extracting 40,000 documents on "
         "your local model would take weeks. The 60 golden documents cover the extraction path properly."),
    ]
    for i, (a, b) in enumerate(lines, start=1):
        ca, cb = ws.cell(row=i, column=1, value=a), ws.cell(row=i, column=2, value=b)
        ca.font = Font(bold=True, size=14 if i == 1 else 10, color=ACCENT if i == 1 else "1F2933")
        cb.font = Font(size=10, italic=(i == 1))
        cb.alignment = Alignment(wrap_text=True, vertical="top")
        if a.startswith("Key") or a == "What changed in revision 2":
            ca.fill = cb.fill = NOTE_FILL
        ws.row_dimensions[i].height = 46 if len(b) > 150 else (30 if len(b) > 90 else 16)

    _sheet(wb, "Decisions",
           ["Topic", "Decision", "What it means", "Status", "Your comments"],
           [(a, b, c, d, "") for a, b, c, d in DECISIONS], [22, 34, 74, 20, 32],
           "Six were agreed in conversation. The last two are assumptions - please confirm or change them.")

    _sheet(wb, "Test Scenarios",
           ["Ref", "Area", "Test case", "What it proves", "Data setup", "Pass criterion",
            "Defect refs", "Golden refs", "Your comments"],
           [(a, b, c, d, e, f, g, h, "") for a, b, c, d, e, f, g, h in SCENARIOS],
           [7, 26, 34, 50, 50, 56, 14, 14, 30],
           "32 test cases across five areas. Four are NEGATIVE tests (B6, C2, D6, E5) shaded red - they "
           "pass only when the product stays silent. Those are the ones that separate a working detector "
           "from one that flags everything.",
           highlight=(5, "ZERO"))

    _sheet(wb, "Organisation",
           ["Level", "ID pattern", "Name", "Country", "Base currency", "Share of spend",
            "Cost centres", "What it exercises", "Your comments"],
           [(*e, "") for e in ORG_ENTITIES], [10, 14, 38, 22, 14, 14, 12, 52, 30],
           "One group parent and six buying legal entities, each with its own base currency and tax "
           "regime. Every document, deal and contract is attributed to an entity, a business unit and a "
           "cost centre. Your schema already supports all of this: business_unit has 5 levels but is "
           "EMPTY today, and cost_centre has 6 levels populated with 500 placeholder rows.",
           total_cols=(7,))

    _sheet(wb, "Org Hierarchy",
           ["Layer", "Count", "ID pattern", "What it is", "What it exercises", "Your comments"],
           [(*h, "") for h in ORG_HIERARCHY], [20, 10, 14, 60, 56, 30],
           "The full organisational depth beneath the six entities. Cost centres carry real budgets "
           "(budget_allocated_annual, actual_spend_ytd, forecast_spend_annual), their own approval "
           "threshold (spend_threshold_limit) and a link to a category leaf - all columns that already "
           "exist in your schema and drive test cases E2 and E3.")

    _sheet(wb, "Category Taxonomy",
           ["L1 Family", "L2", "L3", "L4", "L5 leaf", "Suppliers (primary)", "UNSPSC",
            "ESG impact", "Status", "Spend class", "Risk rating", "Audit freq", "Policy cover",
            "Your comments"],
           [(*t, "") for t in taxonomy],
           [26, 24, 26, 28, 30, 14, 12, 12, 14, 13, 12, 12, 12, 28],
           f"Read live from uicanvas.proc.bp_category - this is your real taxonomy, not an invention. "
           f"6 families, {leaf_desc}. Family sizes: "
           + ", ".join(f"{k} {v}" for k, v in fam_leaves.most_common())
           + ". 'Suppliers (primary)' is each supplier's main leaf and sums to 5,000; most also carry "
             "one or two secondary categories.",
           total_cols=(6,))

    _sheet(wb, "Full Scope",
           ["Database", "Schema", "Relation", "Type", "Rows today", "In test scope?",
            "Product area", "How it gets filled", "Your comments"],
           [(*s, "") for s in scope], [14, 13, 40, 9, 12, 14, 32, 54, 28],
           f"All {len(scope)} relations across bp_sqldb, uicanvas and ses. {in_scope} are in test scope; "
           f"{len(scope) - in_scope} are backups, dated snapshots or scratch copies, cloned structurally "
           "but left empty. Note the tables marked 'Produced by running the detectors, not seeded' - "
           "those must be EMPTY at the start, or the tests prove nothing.")

    _sheet(wb, "Suppliers",
           ["Tier", "Suppliers", "% of suppliers", "% of spend", "Documents each",
            "Relationship character", "What it exercises", "Your comments"],
           [(*t, "") for t in SUPPLIER_TIERS], [14, 12, 14, 12, 16, 44, 50, 30],
           "A deliberate Pareto shape so spend analytics and consolidation opportunities have real signal. "
           "Every supplier exists in BOTH databases: as SUP-<Name> in bp_testdb and SI###### in "
           "uicanvas_test, joined by a crosswalk table. Distributions for supplier type, legal structure, "
           "Incoterms and country are matched to the existing uicanvas.proc.supplier master (239 countries).",
           total_cols=(2,))

    _sheet(wb, "Volumes",
           ["What", "2023", "2024", "2025", "2026 (Jan-Jul)", "Total", "Where it lands", "Your comments"],
           [(*v, "") for v in VOLUMES], [24, 12, 12, 12, 16, 14, 44, 30],
           "Volumes grow year on year with seasonality. 2026 is a partial year to July, hence the lower figures.")

    _sheet(wb, "Planted Defects",
           ["Ref", "Defect", "Count", "Type", "Test cases", "How it is planted",
            "Expected outcome", "Your comments"],
           [(*d, "") for d in DEFECTS], [8, 34, 10, 18, 14, 54, 48, 30],
           "21 true positives your detectors SHOULD find, and 5 negative controls (shaded red) they must "
           "NOT flag. All exported to a machine-readable answer key.",
           total_cols=(3,), highlight=(3, "NEGATIVE"))

    golden = []
    n = 1
    for fam in FAMILIES:
        for dtype in ["Quote", "Purchase Order", "Invoice", "Contract"]:
            golden.append((f"G{n:02d}", fam, dtype, "PDF", "Baseline, clean document",
                           "Establishes the happy path for this family and document type",
                           "All required fields extracted, totals reconcile, L1-L5 category assigned", ""))
            n += 1
    for fam, dtype, fmt, edge, why, expect in GOLDEN_EDGE:
        golden.append((f"G{n:02d}", fam, dtype, fmt, edge, why, expect, ""))
        n += 1
    _sheet(wb, "Golden Documents",
           ["Ref", "Family", "Document type", "Format", "What makes it hard", "Why it matters",
            "Expected extraction result", "Your comments"],
           golden, [8, 30, 16, 10, 34, 50, 52, 30],
           f"{len(golden)} real files, uploaded and genuinely extracted. Each has a hand-checked "
           "expected-values file, so this doubles as a permanent extraction regression suite you can "
           "re-run after any model or pipeline change.")

    _sheet(wb, "Verification",
           ["Ref", "Check", "What must be true", "Severity", "Your comments"],
           [(*v, "") for v in VERIFICATION], [8, 28, 76, 20, 32],
           "Run by a single verify command after the build. Anything marked 'Blocks sign-off' must pass "
           "before the dataset is usable.")

    wb.save(OUT)
    print(f"written: {OUT}")
    print(f"  relations inventoried: {len(scope)}  (in scope {in_scope}, excluded {len(scope) - in_scope})")
    print(f"  taxonomy leaves: {len(taxonomy)}  supplier apportionment: {sum(t[5] for t in taxonomy)}")
    print(f"  test cases: {len(SCENARIOS)}  defects: {len(DEFECTS)}  golden docs: {len(golden)}")


if __name__ == "__main__":
    main()
