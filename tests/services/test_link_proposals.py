"""Proposing a parent for the documents that never named one.

An invoice that cites a purchase order is matched against that order and nothing
else — the citation is a declared fact and is not up for scoring. An invoice that
cites nothing has never been matched against anything at all: 1,964 of them sit
in _trgt today linked to no order, so no three-way match ever runs and their
spend hangs off no purchase.

This is where the resolution layer earns the half of itself that no caller uses.
Several unreferenced invoices from one supplier compete for that supplier's
orders, an order can only absorb the value it authorised, and the margin says
whether the evidence actually chose an order or merely tolerated one.

Nothing here writes a link. A proposal is a suggestion carrying its own margin,
routed to a human — inferring a parent and stamping it on the document would be
fabricating the reference the document does not carry.
"""
import pytest

from src.services import link_proposals as lp


def _scorer(table):
    """A stand-in for score_link driven by a {(doc_pk, po_id): F} table, so a
    test can state the evidence directly instead of crafting documents."""
    def scorer(doc, po, profile, doc_lines=None, po_lines=None, *args, **kwargs):
        return {"F": table.get((doc["invoice_id"], po["po_id"]), 0.0)}
    return scorer


def _invoices(*specs):
    """(invoice_id, amount) pairs, or bare ids for an invoice with no amount."""
    out = []
    for spec in specs:
        if isinstance(spec, tuple):
            out.append({"invoice_id": spec[0], "converted_amount_usd": spec[1]})
        else:
            out.append({"invoice_id": spec})
    return out


def _pos(*specs):
    out = []
    for spec in specs:
        if isinstance(spec, tuple):
            out.append({"po_id": spec[0], "converted_amount_usd": spec[1],
                        "remaining_capacity": spec[1]})
        else:
            out.append({"po_id": spec})
    return out


def _by_doc(proposals):
    return {p.doc_pk: p.po_id for p in proposals}


# ---------------------------------------------------------------------------
# What gets proposed
# ---------------------------------------------------------------------------
def test_the_one_order_that_fits_is_proposed():
    proposals = lp.propose_links(
        _invoices("INV-1"), _pos("PO-1"), min_score=65.0,
        scorer=_scorer({("INV-1", "PO-1"): 74.0}),
    )

    assert _by_doc(proposals) == {"INV-1": "PO-1"}
    assert proposals[0].F == 74.0


def test_an_order_below_the_floor_is_never_a_candidate():
    """The floor is the review band. Below it there is no evidence worth putting
    in front of a person, and a proposal nobody should act on is noise."""
    proposals = lp.propose_links(
        _invoices("INV-1"), _pos("PO-1"), min_score=65.0,
        scorer=_scorer({("INV-1", "PO-1"): 64.9}),
    )

    assert proposals == []


def test_an_order_carries_as_many_invoices_as_the_evidence_gives_it():
    """N:1 with nothing to run out: both invoices belong to PO-1 on the evidence,
    and both are proposed for it. The set decides one order per document, not one
    document per order."""
    proposals = lp.propose_links(
        _invoices(("INV-1", 600.0), ("INV-2", 600.0)),
        _pos(("PO-1", 700.0), ("PO-2", 700.0)), min_score=65.0,
        scorer=_scorer({("INV-1", "PO-1"): 92.0, ("INV-1", "PO-2"): 88.0,
                        ("INV-2", "PO-1"): 90.0}),
    )

    assert _by_doc(proposals) == {"INV-1": "PO-1", "INV-2": "PO-1"}


def test_many_invoices_may_share_one_order():
    """N:1 is the normal shape. Two invoices billing against one order is
    ordinary part-billing, not a competition, so both are proposed."""
    proposals = lp.propose_links(
        _invoices(("INV-1", 300.0), ("INV-2", 300.0)), _pos(("PO-1", 1000.0)),
        min_score=65.0,
        scorer=_scorer({("INV-1", "PO-1"): 90.0, ("INV-2", "PO-1"): 88.0}),
    )

    assert _by_doc(proposals) == {"INV-1": "PO-1", "INV-2": "PO-1"}


# ---------------------------------------------------------------------------
# What an order can absorb
# ---------------------------------------------------------------------------
def test_two_invoices_that_together_outrun_an_order_are_both_still_proposed():
    """Each fits on its own; together they claim more than the order authorised.
    That is over-billing, and it is the three-way match's finding to raise once
    the documents are linked — not a reason to withhold the parent from either.

    Enforcing it here cost 108 of 253 true parents when measured against
    documents whose real order is known."""
    proposals = lp.propose_links(
        _invoices(("INV-1", 600.0), ("INV-2", 600.0)), _pos(("PO-1", 1000.0)),
        min_score=65.0,
        scorer=_scorer({("INV-1", "PO-1"): 90.0, ("INV-2", "PO-1"): 85.0}),
    )

    assert _by_doc(proposals) == {"INV-1": "PO-1", "INV-2": "PO-1"}
    # Each fits on its own, and each says so.
    assert [p.within_order_value for p in proposals] == [True, True]


def test_an_invoice_larger_than_the_whole_order_is_still_proposed_for_it():
    """An invoice that alone bills more than its order authorised is not in the
    wrong deal — it is over-billing, which is a finding the three-way match
    raises, and it can only raise it once the document is linked to the order at
    all. Refusing to propose the parent is how that over-billing stays invisible.

    Not hypothetical: 617 of the 10,251 same-currency invoice/order pairs in
    bp_testdb bill beyond what the order authorised. Dropping them would cost 6%
    of true parents, and precisely the 6% a buyer most wants to see."""
    proposals = lp.propose_links(
        _invoices(("INV-BIG", 5000.0), ("INV-2", 200.0)), _pos(("PO-1", 1000.0)),
        min_score=65.0,
        scorer=_scorer({("INV-BIG", "PO-1"): 95.0, ("INV-2", "PO-1"): 70.0}),
    )

    assert _by_doc(proposals) == {"INV-BIG": "PO-1", "INV-2": "PO-1"}


def test_a_proposal_says_what_the_document_bills_and_what_the_order_has_left():
    """The value evidence reaches the reviewer as a fact about the proposal, which
    is the whole of what it is for here. A person seeing 'bills 5,000 against an
    order of 1,000' needs no threshold to know what they are looking at."""
    proposals = lp.propose_links(
        _invoices(("INV-BIG", 5000.0), ("INV-1", 600.0)), _pos(("PO-1", 1000.0)),
        min_score=65.0,
        scorer=_scorer({("INV-BIG", "PO-1"): 95.0, ("INV-1", "PO-1"): 90.0}),
    )

    big = next(p for p in proposals if p.doc_pk == "INV-BIG")
    ok = next(p for p in proposals if p.doc_pk == "INV-1")
    assert (big.claim, big.order_remaining, big.within_order_value) == (5000.0, 1000.0, False)
    assert (ok.claim, ok.order_remaining, ok.within_order_value) == (600.0, 1000.0, True)


def test_an_invoice_with_no_amount_claims_nothing_and_is_still_proposed():
    """Absence is not evidence. A document we could not read an amount from is
    not a document claiming the order's whole value, and it is not disqualified
    from having a parent either."""
    proposals = lp.propose_links(
        _invoices("INV-1", ("INV-2", 900.0)), _pos(("PO-1", 1000.0)),
        min_score=65.0,
        scorer=_scorer({("INV-1", "PO-1"): 90.0, ("INV-2", "PO-1"): 88.0}),
    )

    assert _by_doc(proposals) == {"INV-1": "PO-1", "INV-2": "PO-1"}


def test_an_order_with_no_value_bounds_nothing():
    """A purchase order whose total we do not hold cannot say how much is left of
    it, so it constrains nobody rather than constraining everybody to zero."""
    proposals = lp.propose_links(
        _invoices(("INV-1", 600.0), ("INV-2", 600.0)), _pos("PO-1"),
        min_score=65.0,
        scorer=_scorer({("INV-1", "PO-1"): 90.0, ("INV-2", "PO-1"): 85.0}),
    )

    assert _by_doc(proposals) == {"INV-1": "PO-1", "INV-2": "PO-1"}


# ---------------------------------------------------------------------------
# How forced the proposal was — what routing acts on
# ---------------------------------------------------------------------------
def test_a_proposal_with_no_rival_is_decisive():
    proposals = lp.propose_links(
        _invoices("INV-1"), _pos("PO-1"), min_score=65.0,
        scorer=_scorer({("INV-1", "PO-1"): 74.0}),
    )

    assert proposals[0].routing == "suggested"
    assert proposals[0].margin_normalised == 1.0


def test_two_orders_the_evidence_cannot_separate_are_contested():
    """The same score against two orders is not a match, it is a coin toss, and
    a person is told so rather than handed an arbitrary winner as though it were
    a finding."""
    proposals = lp.propose_links(
        _invoices("INV-1"), _pos("PO-1", "PO-2"), min_score=65.0,
        scorer=_scorer({("INV-1", "PO-1"): 74.0, ("INV-1", "PO-2"): 74.0}),
    )

    assert len(proposals) == 1
    assert proposals[0].routing == "contested"
    assert proposals[0].margin == 0.0
    assert proposals[0].alternatives == ("PO-2",)


def test_a_clear_winner_over_a_rival_is_still_decisive():
    proposals = lp.propose_links(
        _invoices("INV-1"), _pos("PO-1", "PO-2"), min_score=65.0,
        scorer=_scorer({("INV-1", "PO-1"): 90.0, ("INV-1", "PO-2"): 66.0}),
    )

    assert proposals[0].po_id == "PO-1"
    assert proposals[0].routing == "suggested"


def test_a_proposal_never_carries_a_score_that_could_auto_promote():
    """A proposal is evidence for a human, never a promotion. The link score is
    reported as it was measured and no margin is ever folded into it — the
    resolution layer's own rule, and the reason F is passed through untouched."""
    proposals = lp.propose_links(
        _invoices("INV-1"), _pos("PO-1"), min_score=65.0,
        scorer=_scorer({("INV-1", "PO-1"): 74.0}),
    )

    assert proposals[0].F == 74.0
    assert proposals[0].margin > 0


# ---------------------------------------------------------------------------
# The answer does not depend on how the rows arrived
# ---------------------------------------------------------------------------
def test_shuffling_the_input_does_not_change_the_proposals():
    table = {("INV-1", "PO-1"): 80.0, ("INV-1", "PO-2"): 80.0,
             ("INV-2", "PO-1"): 80.0, ("INV-2", "PO-2"): 80.0}
    forward = lp.propose_links(_invoices("INV-1", "INV-2"), _pos("PO-1", "PO-2"),
                               min_score=65.0, scorer=_scorer(table))
    reverse = lp.propose_links(_invoices("INV-2", "INV-1"), _pos("PO-2", "PO-1"),
                               min_score=65.0, scorer=_scorer(table))

    assert _by_doc(forward) == _by_doc(reverse)


def test_each_invoice_is_proposed_at_most_one_order():
    proposals = lp.propose_links(
        _invoices("INV-1"), _pos("PO-1", "PO-2", "PO-3"), min_score=65.0,
        scorer=_scorer({("INV-1", "PO-1"): 90.0, ("INV-1", "PO-2"): 80.0,
                        ("INV-1", "PO-3"): 70.0}),
    )

    assert len(proposals) == 1


def test_only_the_strongest_candidates_reach_the_solver():
    """The model grows with the number of candidate pairs, so each document
    competes for its best few orders rather than all of a supplier's. The best
    candidate is never dropped, so a document whose orders do not compete is
    unaffected."""
    pos = _pos(*[f"PO-{i}" for i in range(1, 12)])
    table = {("INV-1", f"PO-{i}"): 90.0 - i for i in range(1, 12)}

    proposals = lp.propose_links(_invoices("INV-1"), pos, min_score=65.0,
                                 scorer=_scorer(table),
                                 max_candidates=3)

    assert proposals[0].po_id == "PO-1"
    assert len(proposals[0].alternatives) == 2


# ---------------------------------------------------------------------------
# Which documents are asked about at all
# ---------------------------------------------------------------------------
class _Cursor:
    """Answers the reads the database wrapper makes.

    Three of them, told apart the way they differ in the code: the orders read
    names the purchase-order table, the unreferenced-document read asks for a
    null reference, and the already-billed read asks for documents that carry
    one.
    """

    _INV_COLS = ("invoice_id", "supplier_id", "po_id", "converted_amount_usd",
                 "invoice_amount", "currency")

    def __init__(self, invoices, pos, billed_rows=()):
        self._invoices, self._pos = invoices, pos
        self._billed_rows = billed_rows
        self.description = None
        self._rows = []

    def _invoice_rows(self, rows):
        self.description = [(c,) for c in self._INV_COLS]
        self._rows = [tuple(r.get(c) for c in self._INV_COLS) for r in rows]

    def execute(self, sql, params=()):
        low = sql.lower()
        if "purchase_order" in low:
            cols = ("po_id", "supplier_id", "converted_amount_usd", "total_amount",
                    "currency")
            self.description = [(c,) for c in cols]
            self._rows = [tuple(p.get(c) for c in cols) for p in self._pos]
        elif "line_items" in low:
            self.description = [("po_id",)]
            self._rows = []
        elif "is null" in low:
            self._invoice_rows(self._invoices)
        else:
            self._invoice_rows(self._billed_rows)

    def fetchall(self):
        return self._rows


def test_a_document_that_names_its_order_is_never_collected():
    """A declared reference is a fact, not a candidate. Proposing an alternative
    parent for a document that already names one would be second-guessing the
    document itself, so a referenced document never reaches the scorer at all.

    A reference of blank space is not a reference. That is not hypothetical:
    it is how an unread field comes back from extraction.
    """
    cur = _Cursor(
        invoices=[{"invoice_id": "INV-1", "supplier_id": "SUP-A", "po_id": None},
                  {"invoice_id": "INV-2", "supplier_id": "SUP-A", "po_id": "  "},
                  {"invoice_id": "INV-3", "supplier_id": "SUP-A", "po_id": "PO-1"}],
        pos=[],
    )

    assert [d["invoice_id"] for d in lp.unparented_documents(cur, "invoice")] == [
        "INV-1", "INV-2"]


def test_what_is_already_billed_is_taken_off_the_order():
    """An order that has absorbed most of its value has little left to offer, and
    an order billed to its limit has none."""
    cur = _Cursor(
        invoices=[],
        pos=[{"po_id": "PO-1", "supplier_id": "SUP-A", "total_amount": 1000.0,
              "currency": "GBP"},
             {"po_id": "PO-2", "supplier_id": "SUP-A", "total_amount": 1000.0,
              "currency": "GBP"}],
        billed_rows=[{"invoice_id": "INV-A", "po_id": "PO-1", "invoice_amount": 900.0,
                      "currency": "GBP"},
                     {"invoice_id": "INV-B", "po_id": "PO-2", "invoice_amount": 1000.0,
                      "currency": "GBP"}],
    )

    orders = {p["po_id"]: p for p in lp.candidate_orders(cur, "SUP-A")}

    assert orders["PO-1"]["remaining_capacity"] == 100.0
    assert orders["PO-2"]["remaining_capacity"] == 0.0


def test_the_same_document_in_both_tiers_is_billed_once():
    """A document staged and promoted is one document. Counting it twice would
    report an order as exhausted when half of it is free."""
    billed = {"invoice_id": "INV-A", "po_id": "PO-1", "invoice_amount": 600.0,
              "currency": "GBP"}
    cur = _Cursor(
        invoices=[],
        pos=[{"po_id": "PO-1", "supplier_id": "SUP-A", "total_amount": 1000.0,
              "currency": "GBP"}],
        billed_rows=[billed, dict(billed)],
    )

    orders = {p["po_id"]: p for p in lp.candidate_orders(cur, "SUP-A")}

    assert orders["PO-1"]["remaining_capacity"] == 400.0


def test_billing_in_another_currency_is_not_taken_off_the_order():
    """Subtracting a EUR invoice from a GBP order would mean inventing a rate.
    The order reports the value it can actually account for, and says it is not
    the whole story."""
    cur = _Cursor(
        invoices=[],
        pos=[{"po_id": "PO-1", "supplier_id": "SUP-A", "total_amount": 1000.0,
              "currency": "GBP"}],
        billed_rows=[{"invoice_id": "INV-A", "po_id": "PO-1", "invoice_amount": 900.0,
                      "currency": "EUR"}],
    )

    orders = {p["po_id"]: p for p in lp.candidate_orders(cur, "SUP-A")}

    assert orders["PO-1"]["remaining_capacity"] == 1000.0
    assert orders["PO-1"]["billing_not_counted"] == 1


# ---------------------------------------------------------------------------
# What a person can do about a proposal
# ---------------------------------------------------------------------------
class _ConfirmCursor:
    """A cursor that records what was written, and answers the reads confirm makes."""

    def __init__(self, docs, proposals_for):
        self._docs, self._proposals_for = docs, proposals_for
        self.writes = []
        self.description = None
        self._rows = []

    def execute(self, sql, params=()):
        low = sql.lower()
        if low.startswith("update"):
            self.writes.append((sql, params))
            self.description, self._rows = None, []
            return
        self.description = [("invoice_id",), ("po_id",), ("supplier_id",)]
        self._rows = [(d["invoice_id"], d.get("po_id"), d.get("supplier_id"))
                      for d in self._docs]

    def fetchall(self):
        return self._rows


class _ConfirmConn:
    def __init__(self, cur):
        self._cur = cur

    def cursor(self):
        return self._cur


def _confirmable(monkeypatch, proposals):
    monkeypatch.setattr(lp, "_proposals_for_document", lambda cur, dt, doc: proposals)
    recorded = []
    monkeypatch.setattr(lp, "record_action", lambda **kw: recorded.append(kw))
    return recorded


def _proposal(po_id, alternatives=(), routing="suggested", f=74.0, margin=9.0):
    return lp.LinkProposal(doc_type="invoice", doc_pk="INV-1", po_id=po_id, F=f,
                           margin=margin, margin_normalised=1.0, routing=routing,
                           alternatives=tuple(alternatives))


def test_confirming_a_proposed_order_writes_the_reference(monkeypatch):
    cur = _ConfirmCursor([{"invoice_id": "INV-1", "po_id": None, "supplier_id": "SUP-A"}],
                         None)
    recorded = _confirmable(monkeypatch, [_proposal("PO-1")])

    result = lp.confirm_parent_link("invoice", "INV-1", "PO-1", reviewer="ana",
                                    conn=_ConfirmConn(cur))

    assert result["status"] == "linked"
    assert len(cur.writes) == 2          # both tiers
    assert all("PO-1" in w[1] for w in cur.writes)


def test_the_confirmation_names_who_made_it_and_what_it_rested_on(monkeypatch):
    """A link a person asserted must never become indistinguishable from one the
    document itself stated. The action row is where that difference lives."""
    cur = _ConfirmCursor([{"invoice_id": "INV-1", "po_id": None, "supplier_id": "SUP-A"}],
                         None)
    recorded = _confirmable(monkeypatch, [_proposal("PO-1", f=73.5, margin=9.0)])

    lp.confirm_parent_link("invoice", "INV-1", "PO-1", reviewer="ana",
                           note="checked the delivery note", conn=_ConfirmConn(cur))

    assert len(recorded) == 1
    details = recorded[0]["details"]
    assert details["confirmed_by"] == "ana"
    assert details["po_id"] == "PO-1"
    assert details["F"] == 73.5
    assert details["margin"] == 9.0
    assert details["note"] == "checked the delivery note"


def test_an_order_that_was_never_proposed_cannot_be_confirmed(monkeypatch):
    """This is the door's lock. Without it the endpoint is an unguarded way to
    write any reference onto any document, which is precisely the fabrication
    the proposals exist to avoid."""
    cur = _ConfirmCursor([{"invoice_id": "INV-1", "po_id": None, "supplier_id": "SUP-A"}],
                         None)
    _confirmable(monkeypatch, [_proposal("PO-1")])

    result = lp.confirm_parent_link("invoice", "INV-1", "PO-999",
                                    conn=_ConfirmConn(cur))

    assert result["status"] == "refused"
    assert cur.writes == []


def test_a_runner_up_may_be_confirmed_over_the_winner(monkeypatch):
    """On a contested proposal the person is choosing between near-equals, and
    the engine has no business insisting on its own tie-break. Every candidate it
    scored above the floor is confirmable; nothing else is."""
    cur = _ConfirmCursor([{"invoice_id": "INV-1", "po_id": None, "supplier_id": "SUP-A"}],
                         None)
    _confirmable(monkeypatch, [_proposal("PO-1", alternatives=("PO-2",),
                                         routing="contested")])

    result = lp.confirm_parent_link("invoice", "INV-1", "PO-2", conn=_ConfirmConn(cur))

    assert result["status"] == "linked"


def test_a_document_that_already_names_an_order_is_not_relinked(monkeypatch):
    """Nothing here overrules the document. A reference it carries is a fact."""
    cur = _ConfirmCursor([{"invoice_id": "INV-1", "po_id": "PO-7", "supplier_id": "SUP-A"}],
                         None)
    _confirmable(monkeypatch, [_proposal("PO-1")])

    result = lp.confirm_parent_link("invoice", "INV-1", "PO-1", conn=_ConfirmConn(cur))

    assert result["status"] == "refused"
    assert cur.writes == []


def test_confirming_an_unknown_document_is_not_found(monkeypatch):
    cur = _ConfirmCursor([], None)
    _confirmable(monkeypatch, [])

    result = lp.confirm_parent_link("invoice", "NOPE", "PO-1", conn=_ConfirmConn(cur))

    assert result["status"] == "not_found"
    assert cur.writes == []


# ---------------------------------------------------------------------------
# Where the floor came from
# ---------------------------------------------------------------------------
# Measured on bp_testdb: 400 invoices that DO name their order, re-scored against
# their supplier's whole order set with the reference blanked. The true order
# ranked first for all 320 whose order was in that set, so the floor decides
# whether to speak at all, not which order to name.
_TRUE_PAIR_MIN = 22.1
_TRUE_PAIR_P05 = 49.0
_TRUE_PAIR_MAX = 60.6
_WRONG_PAIR_MAX = 21.4


def test_the_floor_sits_in_the_gap_between_true_and_wrong_pairs():
    """The constant is not a guess and not a borrowed one. It fails if either
    population is ever measured onto it."""
    assert _WRONG_PAIR_MAX < _TRUE_PAIR_MIN          # the populations do not overlap
    assert lp.PROPOSAL_MIN_SCORE > _WRONG_PAIR_MAX   # admits no wrong pair
    assert lp.PROPOSAL_MIN_SCORE < _TRUE_PAIR_P05    # keeps all but the weakest true ones


def test_the_promotion_review_floor_would_have_proposed_nothing():
    """The reason this module does not reuse REVIEW_MIN, pinned so nobody
    'tidies' it back.

    A document with no reference is missing the profile's heaviest signal, so it
    is scored on a different scale from a document being promoted against the
    order it cites. Borrowing 65 puts the floor above BOTH populations — every
    pass returns an empty list, which reads as 'nothing to review' rather than
    'the bar is unreachable'."""
    from src.services.linking_engine import MIN_LINK_SCORE, REVIEW_MIN

    assert REVIEW_MIN > _TRUE_PAIR_MAX
    assert MIN_LINK_SCORE > _TRUE_PAIR_MAX   # and a proposal can never auto-promote


# ---------------------------------------------------------------------------
# The unit an order's capacity is stated in
# ---------------------------------------------------------------------------
# converted_amount_usd is populated on 4 of 5,041 purchase orders and 10 of
# 12,408 invoices in bp_testdb, while the native amount and the currency are on
# every row of both. Keyed on the converted column alone, the capacity half of
# this module is dead on the data it exists for.
def _native(doc_id, amount, currency):
    return {"invoice_id": doc_id, "invoice_amount": amount, "currency": currency}


def _native_po(po_id, amount, currency):
    return {"po_id": po_id, "total_amount": amount, "currency": currency}


def test_an_order_measures_claims_stated_in_its_own_currency():
    """No converted amount anywhere, both sides in GBP: the comparison is still
    made, because two figures in the same currency can be compared without
    inventing a rate."""
    proposals = lp.propose_links(
        [_native("INV-1", 1200.0, "GBP"), _native("INV-2", 600.0, "GBP")],
        [_native_po("PO-1", 1000.0, "GBP")], min_score=40.0,
        scorer=_scorer({("INV-1", "PO-1"): 90.0, ("INV-2", "PO-1"): 85.0}),
    )

    assert [(p.doc_pk, p.claim, p.order_remaining, p.within_order_value)
            for p in proposals] == [("INV-1", 1200.0, 1000.0, False),
                                    ("INV-2", 600.0, 1000.0, True)]


def test_an_invoice_in_another_currency_claims_nothing():
    """Converting would mean inventing a rate, and a bound nobody can verify is
    worse than no bound. The document is not disqualified for it — it simply
    draws nothing from an order it cannot be compared against."""
    proposals = lp.propose_links(
        [_native("INV-1", 600.0, "EUR"), _native("INV-2", 600.0, "GBP")],
        [_native_po("PO-1", 700.0, "GBP")], min_score=40.0,
        scorer=_scorer({("INV-1", "PO-1"): 90.0, ("INV-2", "PO-1"): 85.0}),
    )

    assert _by_doc(proposals) == {"INV-1": "PO-1", "INV-2": "PO-1"}


def test_an_invoice_in_another_currency_reports_no_value_verdict_at_all():
    """Not False, which would read as over-claiming. There is no comparison to be
    had without inventing a rate, and saying nothing is the honest answer."""
    proposals = lp.propose_links(
        [_native("INV-1", 5000.0, "EUR")], [_native_po("PO-1", 700.0, "GBP")],
        min_score=40.0, scorer=_scorer({("INV-1", "PO-1"): 90.0}),
    )

    assert proposals[0].claim is None
    assert proposals[0].within_order_value is None


def test_a_common_unit_is_preferred_to_a_matching_currency():
    """Where both sides carry a converted figure, that is the comparison — it is
    the one unit every document in a mixed-currency set can be stated in.

    The order holds 500 converted, or 1,000 in its own currency; the invoice
    claims 600 converted, or 100 natively. Read in the converted unit it does not
    fit; read natively it easily would."""
    inv = {"invoice_id": "INV-1", "converted_amount_usd": 600.0,
           "invoice_amount": 100.0, "currency": "GBP"}
    po = {"po_id": "PO-1", "converted_amount_usd": 500.0,
          "total_amount": 1000.0, "currency": "GBP"}

    proposals = lp.propose_links([inv], [po], min_score=40.0,
                                 scorer=_scorer({("INV-1", "PO-1"): 90.0}))

    assert (proposals[0].claim, proposals[0].order_remaining) == (600.0, 500.0)
    assert proposals[0].within_order_value is False


def test_a_hair_over_the_order_is_not_called_over_claiming():
    """The band is the one three_way_match allows before it raises over-billing.
    Without it, a rounding difference between an order and its invoice would be
    reported to a person as a document claiming more than was authorised."""
    proposals = lp.propose_links(
        [_native("INV-1", 1004.0, "GBP"), _native("INV-2", 1006.0, "GBP")],
        [_native_po("PO-1", 1000.0, "GBP")], min_score=40.0,
        scorer=_scorer({("INV-1", "PO-1"): 90.0, ("INV-2", "PO-1"): 90.0}),
    )

    # 0.5% of 1,000 is 5.00: 1,004 is inside the band, 1,006 is outside it.
    assert [(p.doc_pk, p.within_order_value) for p in proposals] == [
        ("INV-1", True), ("INV-2", False)]
