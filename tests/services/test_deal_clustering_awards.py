"""Award detection as a set problem.

A purchase order is placed with one supplier, so it can be won by one quote.
Picking each quote's favourite PO independently cannot honour that.
"""
from src.services import deal_clustering as dc


def _scorer(table):
    """A stand-in for score_link driven by a {(quote_id, po_id): F} table, so a
    test can state the evidence directly instead of crafting documents."""
    def scorer(bid, po, profile, bid_lines, po_lines, *args, **kwargs):
        return {"F": table.get((bid["quote_id"], po["po_id"]), 0.0)}
    return scorer


def _bids(*ids):
    return [{"quote_id": i} for i in ids]


def _pos(*ids):
    return [{"po_id": i} for i in ids]


def test_one_purchase_order_is_won_by_one_quote():
    """Both quotes score well against the same PO. Only the stronger one wins it;
    the other is not awarded, because there is nothing left to award."""
    awards = dc.awarded_pos(
        _bids("Q1", "Q2"), _pos("PO-1"), {}, {}, min_score=60.0,
        scorer=_scorer({("Q1", "PO-1"): 90.0, ("Q2", "PO-1"): 85.0}),
    )

    assert awards["Q1"]["po_id"] == "PO-1"
    assert awards["Q2"]["po_id"] is None


def test_the_set_beats_each_quote_taking_its_own_favourite():
    """Q1 prefers PO-1, but Q1 is the only quote that can take PO-2 at all.
    Handing Q1 its favourite strands Q2. Taken together, crossing them over
    carries more evidence."""
    awards = dc.awarded_pos(
        _bids("Q1", "Q2"), _pos("PO-1", "PO-2"), {}, {}, min_score=60.0,
        scorer=_scorer({
            ("Q1", "PO-1"): 92.0, ("Q1", "PO-2"): 88.0,
            ("Q2", "PO-1"): 90.0,
        }),
    )

    assert awards["Q1"]["po_id"] == "PO-2"
    assert awards["Q2"]["po_id"] == "PO-1"


def test_a_score_below_the_gate_is_never_a_candidate():
    awards = dc.awarded_pos(
        _bids("Q1"), _pos("PO-1"), {}, {}, min_score=60.0,
        scorer=_scorer({("Q1", "PO-1"): 59.9}),
    )

    assert awards["Q1"]["po_id"] is None


def test_a_decisive_award_carries_a_margin_and_a_close_one_is_contested():
    decisive = dc.awarded_pos(
        _bids("Q1"), _pos("PO-1", "PO-2"), {}, {}, min_score=60.0,
        scorer=_scorer({("Q1", "PO-1"): 98.0, ("Q1", "PO-2"): 61.0}),
    )
    assert decisive["Q1"]["margin"] > 0.0
    assert decisive["Q1"]["contested"] is False

    tied = dc.awarded_pos(
        _bids("Q1"), _pos("PO-1", "PO-2"), {}, {}, min_score=60.0,
        scorer=_scorer({("Q1", "PO-1"): 80.0, ("Q1", "PO-2"): 80.0}),
    )
    assert tied["Q1"]["margin"] == 0.0
    assert tied["Q1"]["contested"] is True


def test_an_explicitly_awarded_po_is_not_up_for_competition():
    """A PO that cites a bid's reference is a declared fact. It must not be
    handed to a different quote by scoring."""
    awards = dc.awarded_pos(
        _bids("Q2"), _pos("PO-1", "PO-2"), {}, {}, min_score=60.0,
        scorer=_scorer({("Q2", "PO-1"): 95.0, ("Q2", "PO-2"): 70.0}),
        exclude_targets=("PO-1",),
    )

    assert awards["Q2"]["po_id"] == "PO-2"


def test_a_single_bid_still_answers_the_old_question():
    """awarded_po_scored keeps its contract for the one-bid callers."""
    po_id, f = dc.awarded_po_scored(
        {"quote_id": "Q1"}, _pos("PO-1", "PO-2"), {}, [], min_score=60.0,
        scorer=_scorer({("Q1", "PO-1"): 70.0, ("Q1", "PO-2"): 91.0}),
    )

    assert (po_id, f) == ("PO-2", 91.0)


# --- the same rule, through the batch orchestrator --------------------------
_QUOTES = [
    # Two sourcing events, two bidders each. Within an event the bids describe the
    # same requirement (rivals); across events they do not (never merged).
    {"quote_id": "Q1", "supplier_id": "SUP-A", "converted_amount_usd": 1000.0,
     "currency": "USD", "quote_date": "2026-01-05", "po_id": None},
    {"quote_id": "Q1b", "supplier_id": "SUP-B", "converted_amount_usd": 1000.0,
     "currency": "USD", "quote_date": "2026-01-05", "po_id": None},
    {"quote_id": "Q2", "supplier_id": "SUP-C", "converted_amount_usd": 1000.0,
     "currency": "USD", "quote_date": "2026-01-06", "po_id": None},
    {"quote_id": "Q2b", "supplier_id": "SUP-D", "converted_amount_usd": 1000.0,
     "currency": "USD", "quote_date": "2026-01-06", "po_id": None},
]
_WIDGET = [{"item_description": "widget alpha", "quantity": 10, "unit_price": 100.0}]
_GADGET = [{"item_description": "gadget beta", "quantity": 10, "unit_price": 100.0}]
_QUOTE_LINES = {"Q1": _WIDGET, "Q1b": _WIDGET, "Q2": _GADGET, "Q2b": _GADGET}
_POS = [{"po_id": "PO-1", "supplier_id": "SUP-A", "converted_amount_usd": 1000.0,
         "currency": "USD", "order_date": "2026-01-10"}]
_PO_LINES = {"PO-1": _WIDGET}


def _awards_from_batch(**overrides):
    kwargs = dict(quotes=_QUOTES, quote_lines=_QUOTE_LINES, purchase_orders=_POS,
                  po_lines=_PO_LINES, invoices=[])
    kwargs.update(overrides)
    res = dc.cluster_batch(**kwargs)
    won = {}
    for proposal in res["proposals"]:
        for member in proposal["members"]:
            if member["doc_type"] == "po":
                won.setdefault(member["doc_pk"], []).append(proposal)
    assert won, "fixture attached no purchase order at all"
    return res, won


def test_a_batch_never_hands_one_purchase_order_to_two_events(monkeypatch):
    """Two quotes that are not rivals, each scoring well against the same order.
    Only one can have won it: an order attached to two sourcing events is a
    phantom deal, and every downstream spend figure double-counts it."""
    monkeypatch.setattr(dc, "score_link", _scorer({("Q1", "PO-1"): 95.0,
                                                   ("Q2", "PO-1"): 90.0}))
    res, won = _awards_from_batch()

    assert len(res["proposals"]) == 2, "fixture must produce two separate events"
    for po_id, proposals in won.items():
        assert len(proposals) == 1, f"{po_id} attached to {len(proposals)} events"


def test_the_attached_order_carries_how_forced_its_award_was(monkeypatch):
    """The stored evidence says not just 'continuity, F=95' but how much better
    that was than the next best assignment. A reviewer needs the second number."""
    monkeypatch.setattr(dc, "score_link", _scorer({("Q1", "PO-1"): 95.0,
                                                   ("Q2", "PO-1"): 90.0}))
    res, won = _awards_from_batch()

    po_members = [m for p in res["proposals"] for m in p["members"]
                  if m["doc_type"] == "po"]
    assert po_members, "fixture attached no purchase order"
    evidence = po_members[0]["match_evidence"]
    assert evidence["linked_by"] == "continuity"
    assert evidence["margin"] > 0.0
    assert evidence["contested"] is False
