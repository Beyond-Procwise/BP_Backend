"""Deal clustering --- turning a batch of bids into sourcing events.

Delegates to ``src.services.deal_clustering``.
"""
from __future__ import annotations

from datetime import date

from src.services import deal_clustering as _dc
from src.services.requirement_similarity import rivalry_score as _rivalry

from ..contract import COUNT, RECORD, ROWS, SCORE_100, TEXT, Output, Term
from ..registry import GoldenVector, formula

_OWNER = "linkage"
_FROM = date(2026, 9, 5)

_A = {"quote_id": "Q-1", "supplier_id": "SUP-1", "buyer_id": "BUY-1", "currency": "GBP",
      "converted_amount_usd": 100000.0, "quote_date": "2025-05-01"}
_B = {"quote_id": "Q-2", "supplier_id": "SUP-2", "buyer_id": "BUY-1", "currency": "GBP",
      "converted_amount_usd": 107000.0, "quote_date": "2025-05-02"}
_LA = [{"item_description": "London Heathrow to Edinburgh full truckload FTL freight",
        "quantity": 120, "unit_price": 833.33}]
_LB = [{"item_description": "London Heathrow to Edinburgh full truckload FTL freight",
        "quantity": 120, "unit_price": 891.67}]
# Built with the comparator passed explicitly. Going through the registry here
# would re-enter registration while this very module is still importing, and
# hand the fixture a half-built registry.
_MATRIX = _dc.pairwise_matrix([_A, _B], {"Q-1": _LA, "Q-2": _LB}, scorer=_rivalry)


@formula(
    "deal_clustering.cluster_confidence",
    version="1.0.0",
    owner=_OWNER,
    purpose="Confidence that every member of a cluster belongs to the same sourcing event",
    effective_from=_FROM,
    inputs=[
        Term("cluster", ROWS, "the bids in one proposed cluster"),
        Term("matrix", RECORD, "pairwise rivalry results keyed by frozenset of quote ids"),
    ],
    output=Output("float", SCORE_100, "minimum pairwise correlation x 100, 1dp"),
    notes=(
        "A singleton returns 100.0 -- maximum confidence from no evidence at all "
        "(gap report D-9). Callers are expected to treat a one-bid event separately; "
        "that is a convention, not something this function enforces."
    ),
    golden=[
        GoldenVector(inputs={"cluster": [_A, _B], "matrix": _MATRIX}, expected=93.2),
        GoldenVector(inputs={"cluster": [_A], "matrix": _MATRIX}, expected=100.0,
                     note="pins D-9 so a later fix is a visible version bump"),
    ],
)
def cluster_confidence(cluster: list, matrix: dict) -> float:
    return _dc.cluster_confidence(cluster, matrix)


@formula(
    "deal_clustering.awarded_po",
    version="1.0.0",
    owner=_OWNER,
    purpose="Which purchase order a bid won, by continuity scoring rather than name matching",
    effective_from=_FROM,
    inputs=[
        Term("bid", RECORD, "the collapsed bid"),
        Term("purchase_orders", ROWS, "candidate POs"),
        Term("po_lines", RECORD, "PO line items keyed by po_id", required=False),
        Term("bid_lines", ROWS, "the bid's line items", required=False),
        Term("min_score", SCORE_100, "F floor for calling it an award",
             minimum=0.0, maximum=100.0, required=False),
    ],
    output=Output("tuple", SCORE_100, "(winning po_id or None, its F)"),
    notes=(
        "**Under active concurrent development.** `awarded_po_scored` was an argmax "
        "over every candidate PO when this formula was registered; it now delegates to "
        "`src.services.resolution`, which is being built in a parallel workstream and "
        "appears to address the separation gap this formula's registration flagged "
        "(gap report D-6: with S pinned to 1.0, a bid matching four POs equally well "
        "reported the same confidence as one matching exactly one). The vector below "
        "holds across both implementations because it pins the no-award case, which "
        "neither changes. **Re-snapshot this formula's vectors once that work lands** "
        "-- the current set does not pin the award path."
    ),
    golden=[
        GoldenVector(
            inputs={"bid": _A,
                    "purchase_orders": [{"po_id": "PO-4001", "supplier_id": "SUP-100",
                                         "converted_amount_usd": 12500.0, "currency": "GBP",
                                         "order_date": "2025-03-01"}],
                    "po_lines": {"PO-4001": _LA}, "bid_lines": _LA, "min_score": None},
            expected=(None, 0.0),
            note="different supplier and amount -- no award, and NOT a nearest match",
        ),
    ],
)
def awarded_po(bid, purchase_orders, po_lines=None, bid_lines=None, min_score=None):
    if min_score is None:
        return _dc.awarded_po_scored(bid, purchase_orders, po_lines or {}, bid_lines or [])
    return _dc.awarded_po_scored(
        bid, purchase_orders, po_lines or {}, bid_lines or [], float(min_score)
    )
