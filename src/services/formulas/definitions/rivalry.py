"""Rivalry correlation --- the "are these two quotes competing for the same
business" question, which is the opposite signature to continuity linking.

Delegates to ``src.services.requirement_similarity``. Golden vectors snapshotted
2026-09-05; the same-event pair reproduces the calibration band the module's own
tuning notes describe (91-96% for a tight rival pair).
"""
from __future__ import annotations

from datetime import date

from src.services import requirement_similarity as _rs

from ..contract import LABEL, RATIO, RECORD, ROWS, Output, Term
from ..registry import GoldenVector, formula

_OWNER = "linkage"
_FROM = date(2026, 9, 5)

_A = {"quote_id": "Q-1", "supplier_id": "SUP-1", "buyer_id": "BUY-1", "currency": "GBP",
      "converted_amount_usd": 100000.0, "quote_date": "2025-05-01"}
_B = {"quote_id": "Q-2", "supplier_id": "SUP-2", "buyer_id": "BUY-1", "currency": "GBP",
      "converted_amount_usd": 107000.0, "quote_date": "2025-05-02"}
_C = dict(_B, quote_id="Q-3", converted_amount_usd=630000.0)
_LA = [{"item_description": "London Heathrow to Edinburgh full truckload FTL freight",
        "quantity": 120, "unit_price": 833.33}]
_LB = [{"item_description": "London Heathrow to Edinburgh full truckload FTL freight",
        "quantity": 120, "unit_price": 891.67}]
_LC = [{"item_description": "Managed IT endpoint service desk 24x7 3000 users",
        "quantity": 3000, "unit_price": 210.0}]


@formula(
    "rivalry.correlation",
    version="1.0.0",
    owner=_OWNER,
    purpose="How strongly two bids look like rivals for the same sourcing event",
    effective_from=_FROM,
    inputs=[
        Term("bid_a", RECORD, "one collapsed bid"),
        Term("bid_b", RECORD, "the other collapsed bid"),
        Term("lines_a", ROWS, "bid A's line items", required=False),
        Term("lines_b", ROWS, "bid B's line items", required=False),
    ],
    output=Output("dict", RATIO,
                  "score_link's full result plus `correlation` = F/100 in [0,1]"),
    notes=(
        "The quote_rival profile deliberately carries NO supplier_id and NO exact-amount "
        "signal: divergence on those is the signature of a competitive event, not a "
        "defect. p0=0.03 / alpha=0.55 were calibrated against "
        "tests/fixtures/deal_clustering/golden_batch.py."
    ),
    golden=[
        GoldenVector(
            inputs={"bid_a": _A, "bid_b": _B, "lines_a": _LA, "lines_b": _LB},
            expected={"F": 93.2294, "correlation": 0.9323, "decision": "auto_link"},
            note="same freight lane, 7% price spread -- a textbook rival pair",
        ),
        GoldenVector(
            inputs={"bid_a": _A, "bid_b": _C, "lines_a": _LA, "lines_b": _LC},
            expected={"F": 0.0642, "correlation": 0.0006, "decision": "block_or_exception"},
            note="freight vs IT: near-zero without any hard cutoff",
        ),
    ],
)
def rivalry_correlation(bid_a, bid_b, lines_a=None, lines_b=None) -> dict:
    return _rs.rivalry_score(bid_a, bid_b, lines_a or [], lines_b or [])


@formula(
    "rivalry.description_overlap",
    version="1.0.0",
    owner=_OWNER,
    purpose="Token Jaccard over two bids' aggregated line descriptions",
    effective_from=_FROM,
    inputs=[Term("lines_a", ROWS, required=False), Term("lines_b", ROWS, required=False)],
    output=Output("tuple", RATIO, "(score 0-1, status)"),
    golden=[
        GoldenVector(inputs={"lines_a": _LA, "lines_b": _LB}, expected=(1.0, "OK")),
        GoldenVector(inputs={"lines_a": _LA, "lines_b": _LC}, expected=(0.0, "CONFLICT")),
    ],
)
def description_overlap(lines_a=None, lines_b=None):
    return _rs.cmp_desc_overlap(lines_a or [], lines_b or [])


@formula(
    "rivalry.volume_agreement",
    version="1.0.0",
    owner=_OWNER,
    purpose="Quantity-total ratio between two bids",
    effective_from=_FROM,
    inputs=[Term("lines_a", ROWS, required=False), Term("lines_b", ROWS, required=False)],
    output=Output("tuple", RATIO, "(score 0-1, status)"),
    notes=(
        "Services deals carry no quantity at all (lump-sum lines), so MISSING is the "
        "normal outcome for them rather than an anomaly."
    ),
    golden=[
        GoldenVector(inputs={"lines_a": _LA, "lines_b": _LB}, expected=(1.0, "OK")),
        GoldenVector(inputs={"lines_a": [{"item_description": "x"}], "lines_b": _LB},
                     expected=(0.5, "MISSING")),
    ],
)
def volume_agreement(lines_a=None, lines_b=None):
    return _rs.cmp_volume(lines_a or [], lines_b or [])


@formula(
    "rivalry.price_proximity",
    version="1.0.0",
    owner=_OWNER,
    purpose="Graded price closeness between two bids (proximity, never equality)",
    effective_from=_FROM,
    inputs=[Term("row_a", RECORD), Term("row_b", RECORD)],
    output=Output("tuple", RATIO, "(min/max ratio, status)"),
    golden=[
        GoldenVector(inputs={"row_a": _A, "row_b": _B},
                     expected=(0.9345794392523364, "OK")),
        GoldenVector(inputs={"row_a": _A, "row_b": _C},
                     expected=(0.15873015873015872, "CONFLICT"),
                     note="6.3x apart contributes ~0 but is never a hard reject"),
    ],
)
def price_proximity(row_a, row_b):
    return _rs.cmp_price_prox(row_a, row_b)
