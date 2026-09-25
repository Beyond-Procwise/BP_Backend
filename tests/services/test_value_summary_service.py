"""Unit tests for the Value Found aggregation core. Pure functions on dicts —
no DB. Row shapes mirror proc.bp_extraction_discrepancy / proc.bp_opportunity."""
from datetime import date, datetime, timezone, timedelta

from src.services import value_summary_service as vss
from src.services.value_summary_service import (
    parse_amount, discrepancy_delta, classify_discrepancy,
    classify_opportunity, dedupe, summarise,
)


def _disc(**kw):
    row = {
        "discrepancy_id": 1, "issue_type": "amount_over_po", "status": "open",
        "raw_value": "10950.00", "expected_value": "10000.00", "computed_value": "+950.00",
        "resolution_outcome": None, "recovered_amount": None, "query_sent_at": None,
        "doc_type": "invoice", "doc_pk_candidate": "INV-1042", "deal_id": "D-1",
        "supplier_name": "Techworld", "currency": "GBP", "notes": "",
        "created_at": datetime.now(timezone.utc) - timedelta(days=45),
    }
    row.update(kw)
    return row


def test_parse_amount_strict():
    assert parse_amount("£1,234.50") == 1234.5
    assert parse_amount("+950.00") == 950.0
    assert parse_amount("a" * 64) is None          # SHA-256-like → None, never a number
    assert parse_amount("1042 refund") is None     # partial numbers don't count


def test_delta_prefers_signed_computed_value():
    assert discrepancy_delta(_disc()) == 950.0
    # convention 2: no computed_value → observed − expected
    assert discrepancy_delta(_disc(computed_value=None)) == 950.0
    assert discrepancy_delta(_disc(computed_value="garbage", raw_value="x")) is None


def test_discrepancy_tiering():
    f = classify_discrepancy(_disc())
    assert f["tier"] == "verified" and f["amount_gbp"] == 950.0 and f["queryable"] is True
    assert f["age_days"] == 45
    # ignored (= dismissed false positive) and superseded never count
    assert classify_discrepancy(_disc(status="ignored")) is None
    assert classify_discrepancy(_disc(status="superseded")) is None
    # historical resolved: stays in found; recovered now comes only from the ledger
    # (apply_ledger), never from the legacy resolution_outcome/recovered_amount columns.
    f = classify_discrepancy(_disc(status="resolved"))
    assert f["tier"] == "verified" and f["status"] == "resolved" and f["recovered_gbp"] is None
    f = classify_discrepancy(_disc(status="resolved", resolution_outcome="recovered",
                                   recovered_amount=500))
    assert f["recovered_gbp"] is None
    # non-value issue types produce no finding
    assert classify_discrepancy(_disc(issue_type="po_not_found")) is None


def test_opportunity_tiering():
    def _opp(**kw):
        row = {"opportunity_id": "O-1", "stage": "identified", "financial_impact_gbp": 1200,
               "realised_savings_gbp": None, "supplier_name": "Acme", "deal_id": "D-2",
               "po_id": None, "quote_id": None, "item_description": "notebooks",
               "created_at": datetime.now(timezone.utc)}
        row.update(kw)
        return row
    assert classify_opportunity(_opp())["tier"] == "potential"
    assert classify_opportunity(_opp(stage="negotiation"))["tier"] == "verified"
    assert classify_opportunity(_opp(stage="agreed"))["tier"] == "verified"
    # recovered now comes only from the ledger, never from realised_savings_gbp
    f = classify_opportunity(_opp(stage="realised", realised_savings_gbp=800))
    assert f["tier"] == "verified" and f["recovered_gbp"] is None
    assert classify_opportunity(_opp(stage="rejected")) is None
    assert classify_opportunity(_opp(stage="closed")) is None
    # unknown stage must raise, never be silently dropped (spec rule)
    try:
        classify_opportunity(_opp(stage="mystery"))
        assert False, "unknown stage must raise"
    except ValueError:
        pass
    # finding-dict contract: no "currency" key on any finding leaving this
    # module (build_value_summary strips it from discrepancy findings after
    # FX conversion; opportunity findings must never have carried one)
    assert "currency" not in classify_opportunity(_opp())


def test_dedupe_precedence_and_supersede_flag():
    d = classify_discrepancy(_disc(deal_id="D-9", doc_pk_candidate="INV-9"))
    o = classify_opportunity({"opportunity_id": "O-9", "stage": "agreed",
        "financial_impact_gbp": 950, "realised_savings_gbp": None,
        "supplier_name": "Techworld", "deal_id": "D-9", "po_id": None, "quote_id": None,
        "item_description": None, "created_at": datetime.now(timezone.utc),
        "doc_pk": "INV-9"})
    out = dedupe([o, d])
    kept = [f for f in out if f["superseded_by"] is None]
    supp = [f for f in out if f["superseded_by"] is not None]
    assert len(kept) == 1 and kept[0]["source"] == "discrepancy"
    assert len(supp) == 1 and supp[0]["superseded_by"] == kept[0]["id"]


def test_summarise_totals():
    fs = [
        {"tier": "verified", "amount_gbp": 950.0, "recovered_gbp": 950.0,
         "supplier_name": "Techworld", "superseded_by": None},
        {"tier": "verified", "amount_gbp": 100.0, "recovered_gbp": None,
         "supplier_name": "Techworld", "superseded_by": None},
        {"tier": "potential", "amount_gbp": 500.0, "recovered_gbp": None,
         "supplier_name": "Acme", "superseded_by": None},
        {"tier": "verified", "amount_gbp": 999.0, "recovered_gbp": None,
         "supplier_name": "X", "superseded_by": "other"},   # suppressed: excluded from sums
    ]
    t = summarise(fs)
    assert t["verified_found_gbp"] == 1050.0
    assert "recovered_gbp" not in t     # the ledger owns recovered_gbp now, not summarise()
    assert t["potential_gbp"] == 500.0
    assert t["finding_count"] == 3
    assert t["by_supplier"][0] == {"supplier_name": "Techworld",
                                   "verified_found_gbp": 1050.0, "finding_count": 2}


def test_unconvertible_currency_is_excluded_not_zeroed():
    """A foreign-currency discrepancy with no available FX rate must report an
    honest 'unknown' amount_gbp (None), never a silent £0.00 -- and summarise()
    must not crash and must not count it towards any GBP total."""
    import src.services.value_summary_service as vss

    f = classify_discrepancy(_disc(currency="JPY"))
    assert f["amount_gbp"] == 950.0    # native, pre-conversion

    out = vss._apply_discrepancy_fx(f, rates=None)   # FX unavailable
    assert out["amount_gbp"] is None
    assert out["converted_from"] == {"currency": "JPY", "amount": 950.0, "rate_date": None}
    assert "currency" not in out       # internal staging key must be stripped

    # Mixed with a normal GBP finding: totals must reflect only the known amount.
    gbp_finding = classify_discrepancy(_disc(discrepancy_id=2, doc_pk_candidate="INV-2"))
    t = summarise([out, gbp_finding])
    assert t["verified_found_gbp"] == 950.0    # only the GBP finding counted
    assert t["finding_count"] == 2             # the unconvertible one still counts as found
    supplier_row = t["by_supplier"][0]
    assert supplier_row["supplier_name"] == "Techworld"
    assert supplier_row["finding_count"] == 2
    assert supplier_row["verified_found_gbp"] == 950.0


class _FakeCursor:
    def execute(self, sql, params=()):
        self.description = []

    def fetchall(self):
        return []

    def close(self):
        pass


class FakeConn:
    def cursor(self):
        return _FakeCursor()


def test_source_failure_isolation(monkeypatch):
    import src.services.value_summary_service as vss
    monkeypatch.setattr(vss, "_load_discrepancies", lambda cur: (_ for _ in ()).throw(RuntimeError("db")))
    out = vss.build_value_summary(conn=FakeConn())   # FakeConn: cursor() returns a stub whose execute raises for opportunity SQL too if needed
    assert out["sources"]["discrepancies"] == "unavailable"
    assert out["verified_found_gbp"] == 0.0


def test_a_title_names_the_currency_it_is_quoting():
    # The title quotes the amount as BILLED; amount_gbp is the converted figure. Without a
    # currency the two read as two different numbers — live, a USD duplicate showed
    # "by 147,783.11" on a row labelled "£110,043.74".
    usd = classify_discrepancy(_disc(currency="USD", computed_value="+147783.11"))
    assert "147,783.11 USD" in usd["title"]
    gbp = classify_discrepancy(_disc())
    assert "950.00 GBP" in gbp["title"]
    # A document that never stated its currency keeps a bare amount, never a guessed one.
    bare = classify_discrepancy(_disc(currency=None))
    assert bare["title"].endswith("950.00")


# --------------------------------------------------------------------------
# The value ledger: recovered/avoided/realised/claimed figures, and triage money
# --------------------------------------------------------------------------

def _led(source_id, outcome_type, amount_gbp, *, oid, supersedes=None, source_type="finding",
         valid_from=date(2026, 9, 10), recorded_at=None, amount=None, currency="GBP"):
    return {"outcome_id": oid, "source_type": source_type, "source_id": str(source_id),
            "outcome_type": outcome_type, "amount": amount if amount is not None else amount_gbp,
            "currency": currency, "amount_gbp": amount_gbp, "supersedes_id": supersedes,
            "valid_from": valid_from,
            "recorded_at": recorded_at or datetime(2026, 9, 10, oid, tzinfo=timezone.utc)}


def _disc_finding(did, amount, status="resolved", issue_type="duplicate_invoice", deal="D1", doc=None):
    return {"id": f"disc:{did}", "source": "discrepancy", "tier": "verified",
            "amount_gbp": amount, "status": status, "deal_id": deal,
            "doc_pk": doc or f"INV{did}", "issue_type": issue_type, "superseded_by": None,
            "recovered_gbp": None}


def test_partial_recovery_counts_the_recovered_figure():
    rows = [_led(7, "claimed", 500, oid=1), _led(7, "recovered", 320, oid=2)]
    t = vss.ledger_totals(rows)
    assert t["recovered_gbp"] == 320.0 and t["claimed_open_gbp"] == 0.0
    f = vss.apply_ledger([_disc_finding(7, 500)], rows)[0]
    assert f["ledger_state"] == "recovered" and f["claim"] is None


def test_open_claim_is_in_progress_not_saved():
    t = vss.ledger_totals([_led(8, "claimed", 200, oid=1)])
    assert t["claimed_open_gbp"] == 200.0 and t["saved_gbp"] == 0.0


def test_saved_is_avoided_plus_recovered_plus_realised_and_split_by_month():
    rows = [_led(1, "avoided", 100, oid=1, valid_from=date(2026, 8, 3)),
            _led(2, "claimed", 50, oid=2), _led(2, "recovered", 50, oid=3),
            _led("OPP-1", "realised_saving", 25, oid=4, source_type="opportunity")]
    t = vss.ledger_totals(rows)
    assert (t["avoided_gbp"], t["recovered_gbp"], t["realised_gbp"], t["saved_gbp"]) == \
        (100.0, 50.0, 25.0, 175.0)
    assert t["by_month"] == [
        {"month": "2026-08", "avoided_gbp": 100.0, "recovered_gbp": 0.0, "realised_gbp": 0.0},
        {"month": "2026-09", "avoided_gbp": 0.0, "recovered_gbp": 50.0, "realised_gbp": 25.0}]


def test_a_correction_replaces_the_figure_it_supersedes():
    rows = [_led(1, "avoided", 100, oid=1), _led(1, "avoided", 90, oid=2, supersedes=1)]
    assert vss.ledger_totals(rows)["avoided_gbp"] == 90.0


def test_unconverted_outcome_counts_in_no_gbp_total():
    rows = [_led(1, "avoided", None, oid=1, amount=100, currency="NZD")]
    assert vss.ledger_totals(rows)["avoided_gbp"] == 0.0


def test_in_play_excludes_settled_findings():
    findings = vss.apply_ledger(
        [_disc_finding(1, 100, status="open"), _disc_finding(2, 200), _disc_finding(3, 300)],
        [_led(2, "claimed", 200, oid=1), _led(3, "avoided", 300, oid=2)])
    assert vss.in_play_gbp(findings) == 300.0   # open 100 + claimed 200; avoided 300 is settled


def test_line_findings_under_an_overbilled_po_are_superseded():
    po = _disc_finding(10, 20000, status="open", issue_type="invoices_exceed_po_total", doc="PO-1")
    po["po_id"] = "PO-1"
    line = _disc_finding(11, 700, status="open", issue_type="quantity_invoiced_above_po", doc="INV-9")
    line["po_id"] = "PO-1"
    other = _disc_finding(12, 50, status="open", issue_type="unit_price_differs_from_po", doc="INV-8")
    other["po_id"] = "PO-2"
    out = vss.supersede_lines_under_overbilled_po([po, line, other])
    assert line["superseded_by"] == "disc:10"
    assert other["superseded_by"] is None and po["superseded_by"] is None
    assert len(out) == 3                      # the drawer explains, never omits


# --------------------------------------------------------------------------
# R6: a PO-level finding is superseded by a live duplicate-invoice finding on an
# invoice against that PO -- the duplicate is the stronger, whole-invoice explanation
# of the overage, so the two do not both count.
# --------------------------------------------------------------------------

def test_po_level_finding_is_superseded_by_a_duplicate_on_the_same_po():
    po = _disc_finding(20, 467.67, status="open", issue_type="invoices_exceed_po_total", doc="PO-7")
    po["po_id"] = "PO-7"
    dup = _disc_finding(21, 3385.72, status="open", doc="INV-77")   # default issue_type=duplicate_invoice
    dup["po_id"] = "PO-7"
    out = vss.supersede_po_level_by_duplicate([po, dup])
    assert po["superseded_by"] == "disc:21"
    assert dup["superseded_by"] is None                             # the duplicate itself stays live
    assert len(out) == 2


def test_po_level_and_duplicate_on_a_different_po_both_stay_live():
    po = _disc_finding(22, 467.67, status="open", issue_type="invoices_exceed_po_total", doc="PO-7")
    po["po_id"] = "PO-7"
    dup = _disc_finding(23, 3385.72, status="open", doc="INV-88")
    dup["po_id"] = "PO-8"
    vss.supersede_po_level_by_duplicate([po, dup])
    assert po["superseded_by"] is None and dup["superseded_by"] is None


def test_a_line_finding_stays_live_under_a_po_level_finding_superseded_by_a_duplicate():
    po = _disc_finding(24, 467.67, status="open", issue_type="invoices_exceed_po_total", doc="PO-7")
    po["po_id"] = "PO-7"
    dup = _disc_finding(25, 3385.72, status="open", doc="INV-77")
    dup["po_id"] = "PO-7"
    line = _disc_finding(26, 700, status="open", issue_type="quantity_invoiced_above_po", doc="INV-9")
    line["po_id"] = "PO-7"
    # R6 must run before the line pass, exactly as build_value_summary now wires it.
    findings = vss.supersede_po_level_by_duplicate([po, dup, line])
    findings = vss.supersede_lines_under_overbilled_po(findings)
    assert po["superseded_by"] == "disc:25"
    assert line["superseded_by"] is None       # not the duplicate's money; stays live
    assert dup["superseded_by"] is None


def test_po_level_superseded_by_the_largest_duplicate_ties_break_on_lowest_id():
    po = _disc_finding(30, 100, status="open", issue_type="invoices_exceed_po_total", doc="PO-9")
    po["po_id"] = "PO-9"
    small = _disc_finding(31, 200, status="open", doc="INV-1")
    small["po_id"] = "PO-9"
    tie_hi_id = _disc_finding(33, 500, status="open", doc="INV-2")
    tie_hi_id["po_id"] = "PO-9"
    tie_lo_id = _disc_finding(32, 500, status="open", doc="INV-3")
    tie_lo_id["po_id"] = "PO-9"
    vss.supersede_po_level_by_duplicate([po, small, tie_hi_id, tie_lo_id])
    assert po["superseded_by"] == "disc:32"    # largest amount (500); tie -> lowest id (32 < 33)


def test_triage_row_takes_its_amount_from_exposure():
    # R16: the figure is the leading £ figure of the finding's own bp_detection_finding
    # .delta (what the Action Centre shows) -- never a bp_triage_result line's own
    # exposure_gbp, which the LATERAL-join approach this replaced could pick arbitrarily.
    row = {"discrepancy_id": 5, "issue_type": "quantity_invoiced_above_po", "status": "open",
           "raw_value": "340.17", "expected_value": "113.39", "computed_value": None,
           "triage_delta": "£226.78 (301.23 USD)", "currency": "USD", "doc_type": "invoice",
           "doc_pk_candidate": "INV5", "created_at": None, "resolved_at": None,
           "query_sent_at": None}
    f = vss.classify_discrepancy(row)
    assert f["amount_gbp"] == 226.78
    assert f["currency"] == "GBP"            # exposure is already sterling: no second FX


def test_triage_row_with_no_fx_rate_at_triage_time_is_excluded_not_zeroed():
    # money() (triage/model.py) writes a delta with no leading £ at all when no FX rate
    # was available when the finding was triaged -- parse_gbp_delta must not invent a
    # figure for it, and classify_discrepancy must drop the row rather than count £0.
    row = {"discrepancy_id": 6, "issue_type": "quantity_invoiced_above_po", "status": "open",
           "raw_value": "340.17", "expected_value": "113.39", "computed_value": None,
           "triage_delta": "315.21 USD (no FX rate)", "currency": "USD", "doc_type": "invoice",
           "doc_pk_candidate": "INV6", "created_at": None, "resolved_at": None,
           "query_sent_at": None}
    assert vss.classify_discrepancy(row) is None


def test_parse_gbp_delta():
    assert vss.parse_gbp_delta("£226.78") == 226.78
    assert vss.parse_gbp_delta("£1,687.57 (2,230.94 USD)") == 1687.57
    assert vss.parse_gbp_delta("") is None
    assert vss.parse_gbp_delta(None) is None
    assert vss.parse_gbp_delta("abc") is None
    assert vss.parse_gbp_delta("315.21 USD (no FX rate)") is None


# --------------------------------------------------------------------------
# Final-review fixes: R18 window, R20(b) unpriced claims, ledger order
# --------------------------------------------------------------------------

def test_ledger_totals_windows_settled_money_by_valid_from():
    """R18(2): the digest's "saved this week" is ledger_totals over a window -- the same
    function, the same rules. The window is inclusive and reads the CURRENT state's
    valid_from; an open claim is what is open now and is never windowed."""
    rows = [_led(1, "avoided", 100, oid=1, valid_from=date(2026, 9, 18)),
            _led(2, "claimed", 50, oid=2, valid_from=date(2026, 9, 1)),
            _led(2, "recovered", 40, oid=3, valid_from=date(2026, 9, 25)),
            _led(3, "avoided", 70, oid=4, valid_from=date(2026, 9, 17)),     # before
            _led(4, "avoided", 30, oid=5, valid_from=date(2026, 9, 26)),     # after
            _led(5, "claimed", 999, oid=6, valid_from=date(2026, 1, 1)),     # open claim
            _led(1, "avoided", 90, oid=7, supersedes=1, valid_from=date(2026, 9, 18))]
    t = vss.ledger_totals(rows, since=date(2026, 9, 18), until="2026-09-25")
    assert (t["avoided_gbp"], t["recovered_gbp"], t["saved_gbp"]) == (90.0, 40.0, 130.0)
    assert t["claimed_open_gbp"] == 999.0
    assert vss.ledger_totals(rows)["saved_gbp"] == 230.0     # no window: everything


def _triage_row(did, status="resolved", delta=None):
    return {"discrepancy_id": did, "issue_type": "quantity_invoiced_above_po",
            "status": status, "raw_value": "12", "expected_value": "10",
            "computed_value": None, "triage_delta": delta, "currency": "USD",
            "doc_type": "invoice", "doc_pk_candidate": f"INV{did}", "deal_id": f"D{did}",
            "supplier_name": "Acme", "created_at": None, "resolved_at": None,
            "query_sent_at": None, "po_id": None}


def test_an_unpriced_claimed_finding_stays_listed_but_counts_in_no_total(monkeypatch):
    """R20(b): a claim with no readable figure must still reach "Being claimed" so it can
    be settled; it never counts in a found total, and is never valued at zero. An
    unpriced finding nobody claimed is still dropped."""
    monkeypatch.setattr(vss, "_load_discrepancies",
                        lambda cur: [_triage_row(41), _triage_row(42, status="open")])
    monkeypatch.setattr(vss, "_load_opportunities", lambda cur: [])
    monkeypatch.setattr(vss, "_load_ledger", lambda cur: [_led(41, "claimed", 75, oid=1)])
    monkeypatch.setattr(vss, "_get_rates", lambda: None)
    out = vss.build_value_summary(conn=FakeConn())
    ids = [f["id"] for f in out["findings"]]
    assert ids == ["disc:41"]
    f = out["findings"][0]
    assert f["amount_gbp"] is None and f["ledger_state"] == "claimed"
    assert f["claim"]["amount_gbp"] == 75.0
    assert f["title"].endswith("bills over its purchase order")     # no invented figure
    assert out["verified_found_gbp"] == 0.0 and out["in_play_gbp"] == 0.0
    assert out["claimed_open_gbp"] == 75.0


def test_the_ledger_is_read_in_the_order_it_was_written():
    # claimed_at is the FIRST claimed row; with no ORDER BY a corrected claim could
    # report the correction's time instead.
    assert "ORDER BY recorded_at, outcome_id" in vss._LEDGER_SQL


def test_an_unpriced_claimed_po_finding_never_hides_the_priced_lines_under_it(monkeypatch):
    # Listed for settling only: it takes no part in the supersede passes.
    po = _triage_row(51) | {"issue_type": "invoices_exceed_po_total",
                            "doc_type": "purchase_order", "doc_pk_candidate": "PO-5",
                            "po_id": "PO-5"}
    line = _triage_row(52, status="open", delta="£40.00") | {"po_id": "PO-5"}
    monkeypatch.setattr(vss, "_load_discrepancies", lambda cur: [po, line])
    monkeypatch.setattr(vss, "_load_opportunities", lambda cur: [])
    monkeypatch.setattr(vss, "_load_ledger", lambda cur: [_led(51, "claimed", 75, oid=1)])
    monkeypatch.setattr(vss, "_get_rates", lambda: None)
    out = vss.build_value_summary(conn=FakeConn())
    by_id = {f["id"]: f for f in out["findings"]}
    assert by_id["disc:52"]["superseded_by"] is None
    assert out["verified_found_gbp"] == 40.0
