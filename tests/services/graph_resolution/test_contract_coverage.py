from src.services.graph_resolution.profiles import contract_coverage as cc

INV = {"invoice_id": "INV-1", "supplier_id": "SUP-Acme",
       "invoice_date": "2025-06-15", "invoice_total_incl_tax": 5000.0,
       "currency": "GBP", "_same_entity_p": 0.97}
CON = {"contract_id": "C-1", "supplier_id": "S9251",
       "contract_start_date": "2025-01-01", "contract_end_date": "2025-12-31",
       "total_contract_value": 100000.0, "currency": "GBP",
       "spend_category": None}


def test_in_term_with_resolved_supplier_is_covered():
    """Separation, not an absolute band: this profile ships DECLARED
    UNMEASURED (spec section 9), so an absolute threshold here would assert a
    calibration that deliberately does not exist yet."""
    out_of_term = {**INV, "invoice_date": "2026-06-15"}
    assert cc.score(INV, CON)["F"] > cc.score(out_of_term, CON)["F"]


def test_outside_every_term_window_is_not_covered():
    out = {**INV, "invoice_date": "2026-06-15"}
    r = cc.score(out, CON)
    temporal = [s for s in r["signals"] if s["id"] == "date_in_term"][0]
    assert temporal["status"] == "CONFLICT"


def test_missing_category_contributes_nothing_rather_than_penalising():
    r = cc.score(INV, CON)
    cat = [s for s in r["signals"] if s["id"] == "category"][0]
    assert cat["status"] == "MISSING"
    assert cat["c"] == 0.0, "an unevaluable signal must contribute exactly zero"


def test_coverage_never_reaches_auto_link_while_uncalibrated():
    from src.services.graph_resolution.edge_writer import UNCALIBRATED_PROFILES
    assert cc.PROFILE in UNCALIBRATED_PROFILES


def test_uncovered_reason_names_the_missing_evidence():
    out = {**INV, "invoice_date": "2026-06-15", "_same_entity_p": None}
    reason = cc.uncovered_reason(cc.score(out, CON))
    assert "supplier" in reason.lower() or "term" in reason.lower()
