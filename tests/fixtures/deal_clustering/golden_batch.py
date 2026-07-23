"""The real Analysis Set_190726 batch, transcribed as extracted rows.

Ground truth: tests/Test Documents_SpendIQ6/quotes/** (WINNER filenames) and the
spec's worked figures. These are the *rows extraction would produce*, so the
clustering core can be exercised with no database.

Four events, 12 distinct bids (base quote references), 28 quote rows once every
negotiation round (V1/V2/V3 (BAFO)) is counted separately:

  Freight (01_Freight)        Swift (winner, null supplier_id), Condor, Meridian
                               Freight -- 3 rounds each = 9 rows.
  IT-MSA (05_Managed_Services) Synapse (winner, 3 rounds), PrimeOps (2 rounds),
                               Fortis (1 round) = 6 rows. Tightest price spread
                               (~1.07x) and near-identical descriptions among the
                               four events -- highest-confidence cluster.
  Consultancy (03)             Meridian Consulting (winner, 3 rounds), Apex
                               (2 rounds), Vantage (2 rounds) = 7 rows.
  Platform/SaaS (06)           Orbis (winner, 2 rounds), ClearPath (2 rounds),
                               NexusFlow (2 rounds) = 6 rows. Widest price spread
                               (~1.27x) and mild description drift -- lowest-
                               confidence of the four real events.

  9 + 6 + 7 + 6 = 28 quote rows; 3 + 3 + 3 + 3 = 12 bids.

Fortis (IT-MSA) and NexusFlow (Platform) each carry a couple of extra generic
IT-contract tokens (seats/licence/support/annual/3-year/IT) so they are the
strongest CROSS-event pair -- deliberately still far below any within-event
pair, and far above the near-zero Freight<->IT overlap.
"""
from __future__ import annotations
import copy

_FX_GBP_USD = 1.27

# --- Shared descriptions -----------------------------------------------------
# Freight: identical text across all three bidders (same lane, same scope).
_FREIGHT_DESC = "London (Heathrow) to Edinburgh full truckload FTL freight haulage lane"

# IT-MSA: identical for the winner + closest competitor; Fortis adds a couple of
# generic contract tokens (shared with Platform) without touching the IT-specific
# core tokens (endpoint/service desk/24x7/3000 users).
_ITMSA_DESC = ("Managed IT endpoint and service desk support, 24x7 coverage, "
               "3000 users, annual contract, seats, licence, support")
_ITMSA_DESC_FORTIS = _ITMSA_DESC + ", 3-year platform roadmap"

# Consultancy: identical day-rate engagement scope across all three bidders.
_CONSULT_DESC = "Business Analyst professional services day rate engagement"

# Platform: identical for the winner + closest competitor; NexusFlow adds a
# couple of generic contract tokens (shared with IT-MSA) without touching the
# platform-specific core tokens (SaaS/platform/licence/subscription).
_PLATFORM_DESC = "SaaS platform licence, 3-year term, annual subscription, seats, support, users"
_PLATFORM_DESC_NEXUSFLOW = _PLATFORM_DESC + ", IT endpoint integration"


def _rounds(base, supplier, buyer, desc, qty, prices, dates, region="England"):
    """Expand one supplier's bid into its negotiation-round quote rows + lines.

    `prices`/`dates` are parallel lists of length 1-3; round suffixes are
    assigned in order ("" for V1, " (V2)" for round 2, " (V3 (BAFO))" for a
    third round) so src.services.version_collapse.collapse_versions groups
    them back into a single bid.
    """
    suffixes = ["", " (V2)", " (V3 (BAFO))"][:len(prices)]
    rows, lines = [], {}
    for suffix, price, date in zip(suffixes, prices, dates):
        qid = base + suffix
        total = round(price * qty, 2)
        rows.append({
            "quote_id": qid, "supplier_id": supplier, "buyer_id": buyer,
            "currency": "GBP", "total_amount": total,
            "converted_amount_usd": round(total * _FX_GBP_USD, 2),
            "quote_date": date, "country": "GB", "region": region,
        })
        lines[qid] = [{"item_description": desc, "quantity": qty, "unit_price": price}]
    return rows, lines


_BUYER = "Assurity Ltd"

# --- Freight event: Swift (winner, null supplier_id), Condor, Meridian Freight.
#     Prices ~£872-945, qty 1, spread ~1.08x. Tight V1 round-synchrony (~1-2 days apart).
_SWIFT_Q, _SWIFT_L = _rounds(
    "SDP-Q-44120", None, _BUYER, _FREIGHT_DESC, 1,
    [905.00, 884.00, 872.34],
    ["2024-05-06", "2024-05-16", "2024-05-27"], region="Scotland")
_CONDOR_Q, _CONDOR_L = _rounds(
    "CL-2024-0771", "SUP-CondorLogistics", _BUYER, _FREIGHT_DESC, 1,
    [930.00, 910.00, 898.00],
    ["2024-05-07", "2024-05-17", "2024-05-28"], region="Scotland")
_MERIDIAN_FREIGHT_Q, _MERIDIAN_FREIGHT_L = _rounds(
    "MFS-Q-3391", "SUP-MeridianFreight", _BUYER, _FREIGHT_DESC, 1,
    [945.00, 928.00, 915.00],
    ["2024-05-08", "2024-05-18", "2024-05-29"], region="Scotland")

# --- IT-MSA event: Synapse (winner), PrimeOps, Fortis.
#     Prices £272,000-291,000, qty 1, spread ~1.07x -- the tightest of the four events.
_SYNAPSE_Q, _SYNAPSE_L = _rounds(
    "SYN-Q-8820", "SUP-SynapseIT", _BUYER, _ITMSA_DESC, 1,
    [291000.00, 281000.00, 272000.00],
    ["2024-05-10", "2024-05-24", "2024-06-07"])
_PRIMEOPS_Q, _PRIMEOPS_L = _rounds(
    "POM-Q-5510", "SUP-PrimeOpsManaged", _BUYER, _ITMSA_DESC, 1,
    [285000.00, 277000.00],
    ["2024-05-12", "2024-05-26"])
_FORTIS_Q, _FORTIS_L = _rounds(
    "FSM-Q-7742", "SUP-FortisServiceMgmt", _BUYER, _ITMSA_DESC_FORTIS, 1,
    [279000.00],
    ["2024-05-14"])

# --- Consultancy event: Meridian Consulting (winner), Apex, Vantage.
#     Day-rate £680-780, 60-day engagement, spread ~1.15x. V1 span 39 days
#     (2024-04-01 -> 2024-05-10) per spec round-synchrony table.
_MERIDIAN_CONSULTING_Q, _MERIDIAN_CONSULTING_L = _rounds(
    "MCG-Q-1204", "SUP-MeridianConsulting", _BUYER, _CONSULT_DESC, 60,
    [780.00, 750.00, 680.00],
    ["2024-04-01", "2024-04-15", "2024-04-29"])
_APEX_Q, _APEX_L = _rounds(
    "APX-Q-6631", "SUP-ApexDigitalTransformation", _BUYER, _CONSULT_DESC, 60,
    [770.00, 705.00],
    ["2024-04-25", "2024-05-09"])
_VANTAGE_Q, _VANTAGE_L = _rounds(
    "VAP-Q-9903", "SUP-VantageAdvisory", _BUYER, _CONSULT_DESC, 60,
    [760.00, 715.00],
    ["2024-05-10", "2024-05-24"])

# --- Platform/SaaS event: Orbis (winner), ClearPath, NexusFlow.
#     3-year annual subscription value £165,000-210,000, spread ~1.27x -- the
#     widest of the four events. V1 span 28 days (2024-05-01 -> 2024-05-29).
_ORBIS_Q, _ORBIS_L = _rounds(
    "ORB-Q-2290", "SUP-OrbisPlatform", _BUYER, _PLATFORM_DESC, 1,
    [190000.00, 175000.00],
    ["2024-05-01", "2024-05-15"])
_CLEARPATH_Q, _CLEARPATH_L = _rounds(
    "CPS-Q-3380", "SUP-ClearPathSystems", _BUYER, _PLATFORM_DESC, 1,
    [210000.00, 165000.00],
    ["2024-05-15", "2024-05-29"])
_NEXUSFLOW_Q, _NEXUSFLOW_L = _rounds(
    "NXF-Q-4471", "SUP-NexusFlowPlatform", _BUYER, _PLATFORM_DESC_NEXUSFLOW, 1,
    [200000.00, 178000.00],
    ["2024-05-29", "2024-06-12"])

_QUOTES: list[dict] = [
    *_SWIFT_Q, *_CONDOR_Q, *_MERIDIAN_FREIGHT_Q,
    *_SYNAPSE_Q, *_PRIMEOPS_Q, *_FORTIS_Q,
    *_MERIDIAN_CONSULTING_Q, *_APEX_Q, *_VANTAGE_Q,
    *_ORBIS_Q, *_CLEARPATH_Q, *_NEXUSFLOW_Q,
]
_QUOTE_LINES: dict[str, list[dict]] = {
    **_SWIFT_L, **_CONDOR_L, **_MERIDIAN_FREIGHT_L,
    **_SYNAPSE_L, **_PRIMEOPS_L, **_FORTIS_L,
    **_MERIDIAN_CONSULTING_L, **_APEX_L, **_VANTAGE_L,
    **_ORBIS_L, **_CLEARPATH_L, **_NEXUSFLOW_L,
}

# --- The 5 awarded POs: one per event winner, plus the Caldwell orphan (a real
#     PO with no corresponding quotes in this batch -- must not be swept into
#     any cluster). Amounts follow each winner's final (BAFO/last-round) price.
_PURCHASE_ORDERS = [
    {"po_id": "PO-2024-0091", "supplier_name": "Swift Distribution Partners Ltd",
     "supplier_id": None, "converted_amount_usd": 1107.87, "order_date": "2024-06-10",
     "expected_delivery_date": "2024-07-01"},
    {"po_id": "PO-2024-0128", "supplier_name": "Synapse IT Services Ltd",
     "supplier_id": "SUP-SynapseIT", "converted_amount_usd": 345440.00, "order_date": "2024-06-12",
     "expected_delivery_date": "2024-07-15"},
    {"po_id": "PO-2024-0145", "supplier_name": "Orbis Platform Solutions Ltd",
     "supplier_id": "SUP-OrbisPlatform", "converted_amount_usd": 222250.00, "order_date": "2024-06-14",
     "expected_delivery_date": "2024-08-01"},
    {"po_id": "PO-2024-0114", "supplier_name": "Meridian Consulting Ltd",
     "supplier_id": "SUP-MeridianConsulting", "converted_amount_usd": 51816.00, "order_date": "2024-06-11",
     "expected_delivery_date": "2024-07-20"},
    {"po_id": "PO-2024-0163", "supplier_name": "Caldwell Building Contractors Ltd",
     "supplier_id": "SUP-Caldwell", "converted_amount_usd": 415000.00, "order_date": "2024-06-20",
     "expected_delivery_date": "2024-09-01"},  # orphan: no quotes in batch
]

# --- PO line items, keyed by po_id. Each mirrors the winner's final quote line
#     (same description/quantity/unit_price) so later award-detection has real
#     line data to correlate against; Caldwell (orphan) is unrelated building work.
_PO_LINES = {
    "PO-2024-0091": [{"item_description": _FREIGHT_DESC, "quantity": 1, "unit_price": 872.34}],
    "PO-2024-0128": [{"item_description": _ITMSA_DESC, "quantity": 1, "unit_price": 272000.00}],
    "PO-2024-0145": [{"item_description": _PLATFORM_DESC, "quantity": 1, "unit_price": 175000.00}],
    "PO-2024-0114": [{"item_description": _CONSULT_DESC, "quantity": 60, "unit_price": 680.00}],
    "PO-2024-0163": [{"item_description": "Structural remedial building works - Phase 1",
                       "quantity": 1, "unit_price": 415000.00}],
}

# --- 7 invoices, all carrying po_id (spec: 7/7 populated). Some POs are billed
#     across two instalments; each pair sums back to the PO's converted amount.
_INVOICES = [
    {"invoice_id": "INV-2024-0091-1", "po_id": "PO-2024-0091",
     "converted_amount_usd": 1107.87, "invoice_date": "2024-07-02"},
    {"invoice_id": "INV-2024-0128-1", "po_id": "PO-2024-0128",
     "converted_amount_usd": 172720.00, "invoice_date": "2024-07-16"},
    {"invoice_id": "INV-2024-0128-2", "po_id": "PO-2024-0128",
     "converted_amount_usd": 172720.00, "invoice_date": "2024-10-16"},
    {"invoice_id": "INV-2024-0145-1", "po_id": "PO-2024-0145",
     "converted_amount_usd": 222250.00, "invoice_date": "2024-08-05"},
    {"invoice_id": "INV-2024-0114-1", "po_id": "PO-2024-0114",
     "converted_amount_usd": 25908.00, "invoice_date": "2024-07-25"},
    {"invoice_id": "INV-2024-0114-2", "po_id": "PO-2024-0114",
     "converted_amount_usd": 25908.00, "invoice_date": "2024-08-25"},
    {"invoice_id": "INV-2024-0163-1", "po_id": "PO-2024-0163",
     "converted_amount_usd": 415000.00, "invoice_date": "2024-09-05"},
]

# Negative control (repeat commodity buying, product-wide): four correlated pairs that must
# NOT merge because each anchors its own PO+invoice. Includes the Gomez name-mismatch.
_NC_QUOTES = [
    {"quote_id": "128234", "supplier_id": "SUP-DuncanLlc", "buyer_id": "Assurity Ltd",
     "currency": "GBP", "total_amount": 1385.0, "converted_amount_usd": 1758.95},
    {"quote_id": "136586", "supplier_id": "SUP-PerryLtd", "buyer_id": "Assurity Ltd",
     "currency": "GBP", "total_amount": 1249.0, "converted_amount_usd": 1586.23},
    {"quote_id": "102494", "supplier_id": "SUP-DixonReynoldsAndSolomon", "buyer_id": "Assurity Ltd",
     "currency": "GBP", "total_amount": 638.0, "converted_amount_usd": 810.26},
    {"quote_id": "104683", "supplier_id": "SUP-GomezGoodAndCross", "buyer_id": "Assurity Ltd",
     "currency": "GBP", "total_amount": 655.0, "converted_amount_usd": 831.85},
]
_NC_QUOTE_LINES = {
    "128234": [{"item_description": "Staedtler Ballpoint Pen Black Ink, 10 per Pack", "quantity": 100, "unit_price": 13.85},
               {"item_description": "Faber-Castell A4 Ruled Notebook, White Cover", "quantity": 100, "unit_price": 11.69}],
    "136586": [{"item_description": "Staedtler Ballpoint Pen", "quantity": 100, "unit_price": 12.49},
               {"item_description": "Faber-Castell A4 Ruled", "quantity": 100, "unit_price": 10.10}],
    "102494": [{"item_description": "Copier paper A4 80gsm", "quantity": 50, "unit_price": 12.76}],
    "104683": [{"item_description": "Copier paper A4 80 gsm ream", "quantity": 50, "unit_price": 13.10}],
}
_NC_POS = [
    {"po_id": "PO-128234", "supplier_name": "Duncan LLC", "supplier_id": "SUP-DuncanLlc"},
    {"po_id": "PO-136586", "supplier_name": "Perry Ltd", "supplier_id": "SUP-PerryLtd"},
    {"po_id": "PO-102494", "supplier_name": "Dixon Reynolds and Solomon", "supplier_id": "SUP-DixonReynoldsAndSolomon"},
    {"po_id": "PO-104683", "supplier_name": "Gomez, Good and Cross Trading Ltd", "supplier_id": None},
]
_NC_PO_LINES = {p["po_id"]: _NC_QUOTE_LINES[p["po_id"].split("-", 1)[1]] for p in _NC_POS}
_NC_INVOICES = [{"invoice_id": f"INV-{p['po_id']}", "po_id": p["po_id"],
                 "converted_amount_usd": None, "invoice_date": "2024-07-01"} for p in _NC_POS]

def quotes(): return copy.deepcopy(_QUOTES)
def quote_lines(): return copy.deepcopy(_QUOTE_LINES)
def purchase_orders(): return copy.deepcopy(_PURCHASE_ORDERS)
def po_lines(): return copy.deepcopy(_PO_LINES)
def invoices(): return copy.deepcopy(_INVOICES)
def negative_control_quotes(): return copy.deepcopy(_NC_QUOTES)
def negative_control_pos(): return copy.deepcopy(_NC_POS)
def negative_control_quote_lines(): return copy.deepcopy(_NC_QUOTE_LINES)
def negative_control_po_lines(): return copy.deepcopy(_NC_PO_LINES)
def negative_control_invoices(): return copy.deepcopy(_NC_INVOICES)
