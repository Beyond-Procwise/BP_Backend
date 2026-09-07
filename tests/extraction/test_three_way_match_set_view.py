"""Over-billing that only the set can see.

Comparing one line, or one document, at a time misses the arithmetic that
actually costs money: two invoice lines that both land on the same PO line each
pass a check against that line's full authorised total, and two invoices each
citing the same PO at 60% of its value each pass the header check. Neither is
individually over-billing. Only the sum says so.
"""
from src.services.extraction import three_way_match as twm


def _po(total="1000", currency="GBP", lines=None):
    return (
        {"po_id": "PO-1", "total_amount": total, "currency": currency},
        lines or [{"line_number": 1, "item_description": "Widgets",
                   "quantity": 10, "unit_price": 100, "line_total": 1000}],
    )


def _check(monkeypatch, columns, line_items, po=None, siblings=None):
    monkeypatch.setattr(twm, "_load_po", lambda po_id: po or _po())
    monkeypatch.setattr(
        twm, "_billed_by_other_invoices",
        lambda po_id, exclude, currency: siblings if siblings else (0.0, [], True),
    )
    return twm.check_against_po("invoice", {"po_id": "PO-1", **columns}, line_items)


def _types(findings):
    return sorted(f.issue_type for f in findings)


# ---------------------------------------------------------------------------
# The assignment itself
# ---------------------------------------------------------------------------
def test_a_line_does_not_take_the_po_line_another_line_needs_more(monkeypatch):
    """The greedy case, live: a bundled description contains both PO lines'
    wording, so matching it on its own takes whichever came first out of the
    database — the same PO line the specific invoice line matches exactly. The
    document then bills one PO line twice and the other is reported as never
    billed, from nothing but line order.
    """
    po = _po(lines=[
        {"line_number": 1, "item_description": "Platform licence",
         "quantity": 1, "unit_price": 600, "line_total": 600},
        {"line_number": 2, "item_description": "Support retainer",
         "quantity": 1, "unit_price": 400, "line_total": 400},
    ])
    lines = [
        {"item_description": "Platform licence", "line_amount": 600},
        {"item_description": "Platform licence and support retainer", "line_amount": 400},
    ]

    # Matching each line on its own puts both on "Platform licence".
    on_its_own = [twm._match_po_line(li["item_description"], po[1]) for li in lines]
    assert [l["line_number"] for l in on_its_own] == [1, 1]

    # Deciding the document as a set gives each line a PO line of its own.
    assigned = twm._assign_lines(lines, po[1], "PO-1")
    assert [assigned[0]["line_number"], assigned[1]["line_number"]] == [1, 2]

    # And so neither the phantom double-claim nor the phantom unbilled line is raised.
    assert _types(_check(monkeypatch, {"invoice_amount": "1000"}, lines, po=po)) == []


def test_a_line_with_no_po_line_of_its_own_still_matches(monkeypatch):
    """Preferring a PO line each is not requiring one. Split billing is ordinary,
    and a line the one-to-one pass cannot place falls back to its own best match
    rather than being accused of not being on the purchase order."""
    lines = [
        {"item_description": "Widgets", "line_amount": 400},
        {"item_description": "Widgets", "line_amount": 400},
    ]
    assigned = twm._assign_lines(lines, _po()[1], "PO-1")

    assert set(assigned) == {0, 1}
    assert assigned[0] is assigned[1]
    assert "line_not_on_po" not in _types(
        _check(monkeypatch, {"invoice_amount": "800"}, lines)
    )


def test_a_line_on_nothing_is_still_reported(monkeypatch):
    """The set view must not soften a real finding: a charge with no PO line at
    all is still an unauthorised charge."""
    out = _check(monkeypatch, {"invoice_amount": "100"},
                 [{"item_description": "Executive dinner", "line_amount": 100}])
    assert "line_not_on_po" in _types(out)


# ---------------------------------------------------------------------------
# One PO line, billed twice
# ---------------------------------------------------------------------------
def test_two_lines_on_one_po_line_report_what_they_bill_together(monkeypatch):
    out = _check(monkeypatch, {"invoice_amount": "1200"}, [
        {"item_description": "Widgets", "line_amount": 600},
        {"item_description": "Widgets", "line_amount": 600},
    ])
    over = [f for f in out if f.issue_type == "po_line_over_consumed"]

    assert len(over) == 1
    assert over[0].raw_value == "1200.00"
    assert over[0].expected_value == "1000.00"
    assert over[0].computed_value == "+200.00"
    assert over[0].severity == "critical"
    assert over[0].blocks_promotion is False
    assert "billed 2 times" in over[0].notes
    # Neither line is individually over-billing, so the per-line check is silent.
    assert "line_amount_over_po" not in _types(out)


def test_two_lines_that_together_fit_are_not_flagged(monkeypatch):
    out = _check(monkeypatch, {"invoice_amount": "800"}, [
        {"item_description": "Widgets", "line_amount": 400},
        {"item_description": "Widgets", "line_amount": 400},
    ])
    assert "po_line_over_consumed" not in _types(out)


def test_a_single_line_over_billing_is_still_reported_per_line(monkeypatch):
    out = _check(monkeypatch, {"invoice_amount": "1500"},
                 [{"item_description": "Widgets", "line_amount": 1500}])
    assert "line_amount_over_po" in _types(out)
    assert "po_line_over_consumed" not in _types(out)


# ---------------------------------------------------------------------------
# One PO, billed twice across documents
# ---------------------------------------------------------------------------
def test_two_invoices_that_together_exceed_the_po_are_reported(monkeypatch):
    out = _check(
        monkeypatch,
        {"invoice_amount": "600", "currency": "GBP", "invoice_id": "INV-2"},
        [{"item_description": "Widgets", "line_amount": 600}],
        siblings=(600.0, ["INV-1"], True),
    )
    over = [f for f in out if f.issue_type == "po_over_consumed"]

    assert len(over) == 1
    assert over[0].raw_value == "1200.00"
    assert over[0].expected_value == "1000.00"
    assert over[0].blocks_promotion is False
    assert "INV-1, INV-2" in over[0].notes
    # This invoice alone is well inside the PO, so the header check says nothing.
    assert "amount_over_po" not in _types(out)


def test_the_set_is_silent_when_the_invoices_together_fit(monkeypatch):
    out = _check(
        monkeypatch,
        {"invoice_amount": "400", "currency": "GBP", "invoice_id": "INV-2"},
        [{"item_description": "Widgets", "line_amount": 400}],
        siblings=(400.0, ["INV-1"], True),
    )
    assert "po_over_consumed" not in _types(out)


def test_a_total_that_excluded_an_invoice_says_so(monkeypatch):
    out = _check(
        monkeypatch,
        {"invoice_amount": "600", "currency": "GBP", "invoice_id": "INV-2"},
        [{"item_description": "Widgets", "line_amount": 600}],
        siblings=(600.0, ["INV-1"], False),
    )
    over = [f for f in out if f.issue_type == "po_over_consumed"]

    assert len(over) == 1
    assert "another currency are excluded" in over[0].notes


def test_a_document_in_another_currency_is_never_added_to_the_po(monkeypatch):
    """Converting here would mean inventing a rate. A total nobody can verify is
    worse than no finding at all."""
    out = _check(
        monkeypatch,
        {"invoice_amount": "600", "currency": "USD", "invoice_id": "INV-2"},
        [{"item_description": "Widgets", "line_amount": 600}],
        siblings=(600.0, ["INV-1"], True),
    )
    assert "po_over_consumed" not in _types(out)


def test_a_sole_invoice_raises_nothing_about_the_set(monkeypatch):
    out = _check(
        monkeypatch,
        {"invoice_amount": "600", "currency": "GBP", "invoice_id": "INV-1"},
        [{"item_description": "Widgets", "line_amount": 600}],
    )
    assert "po_over_consumed" not in _types(out)


def test_a_line_competes_for_only_its_best_few_po_lines():
    """The model grows with the product of the two line counts unless each line's
    candidate list is bounded. What must never be dropped is the best one — that
    is what keeps an uncontested document matching exactly as it always did."""
    # Every description overlaps every other well past the eligibility floor, so
    # without a bound this one line would compete for all twenty PO lines.
    po_lines = [{"line_number": i, "item_description": f"widget assembly kit {i}",
                 "quantity": 1, "unit_price": 100, "line_total": 100}
                for i in range(20)]

    candidates = twm._candidate_po_lines("widget assembly kit 7", po_lines)

    assert len(candidates) == twm._MAX_CANDIDATES_PER_LINE
    # Best first, and the exact match — the one a bound must never drop — is top.
    assert candidates[0] == (7, twm._EXACT_SCORE)
    assert [score for _, score in candidates] == sorted(
        (score for _, score in candidates), reverse=True
    )
    # Equal scores keep PO line order, so the bound is not the database's to decide.
    assert [i for i, _ in candidates[1:]] == [0, 1, 2]
    assert twm._match_po_line("widget assembly kit 7", po_lines)["line_number"] == 7
