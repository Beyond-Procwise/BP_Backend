"""Landed cost: every checklist item gets a status, and nothing unmentioned is zero."""
import pytest

from src.services.cost.landed_cost import landed_cost, parse_incoterm

BASE = dict(
    lines=[{"unit_price": 20.0, "quantity": 1000, "currency": "GBP"}], buyer_currency="GBP",
    cross_border=True, incoterm="FOB Shanghai",
    components=[{"key": "freight", "amount": 1200.0}, {"key": "brokerage", "amount": 150.0}],
    duty_rate=0.04, customs_basis="CIF", default_insurance_rate=0.003, vat_recoverable=True,
)
RATES = {"GBP": 0.75, "USD": 1.0, "EUR": 0.9}


def run(**over):
    return landed_cost(**{**BASE, **over})


def row(result, name):
    return next(r for r in result["components"] if r["name"] == name)


def test_the_specification_case():
    r = run()
    assert r["landed_total"] == pytest.approx(22270.76, abs=0.005)
    assert r["landed_status"] == "incomplete"
    assert r["missing"] == ["terminal_handling", "delivery_order_fee", "on_carriage", "unloading"]
    assert row(r, "cargo_insurance")["amount"] == pytest.approx(69.96)
    assert row(r, "cargo_insurance")["status"] == "estimated"
    assert row(r, "import_duty")["amount"] == pytest.approx(850.7984)
    assert row(r, "import_duty")["status"] == "derived"
    assert r["unit_landed"] == pytest.approx(r["landed_total"] / 1000)


def test_a_total_with_nothing_missing_is_complete_with_estimates():
    r = run(components=BASE["components"] + [
        {"key": k, "amount": 10.0} for k in
        ("terminal_handling", "delivery_order_fee", "on_carriage", "unloading")])
    assert r["missing"] == []
    assert r["landed_status"] == "complete_with_estimates"


def test_complete_when_nothing_is_estimated_or_missing():
    r = run(incoterm="DDP", components=[])
    assert r["landed_status"] == "complete"
    assert r["landed_total"] == pytest.approx(20000.0)


def test_no_incoterm_makes_every_unstated_item_missing_but_keeps_stated_ones():
    r = run(incoterm=None)
    assert row(r, "freight")["status"] == "stated"
    assert row(r, "brokerage")["status"] == "stated"
    assert row(r, "terminal_handling")["status"] == "missing"
    assert row(r, "supplier_surcharges")["status"] == "missing"
    assert row(r, "cargo_insurance")["status"] == "missing"
    assert r["landed_total"] == pytest.approx(21350.0)
    assert r["landed_status"] == "incomplete"


@pytest.mark.parametrize("term,seller_bears,buyer_bears", [
    ("EXW", [], ["inland_haulage_origin", "export_clearance", "freight", "cargo_insurance",
                 "import_duty", "terminal_handling", "unloading"]),
    ("FCA", ["inland_haulage_origin", "export_clearance"], ["freight", "terminal_handling"]),
    ("FOB", ["origin_handling"], ["freight", "cargo_insurance"]),
    ("CFR", ["freight", "inland_haulage_origin"], ["cargo_insurance", "import_duty"]),
    ("CIF", ["freight", "cargo_insurance"], ["import_duty", "on_carriage"]),
    ("DAP", ["freight", "cargo_insurance", "terminal_handling", "on_carriage"],
     ["import_duty", "unloading"]),
    ("DPU", ["freight", "on_carriage", "unloading"], ["import_duty", "brokerage"]),
    ("DDP", ["freight", "import_duty", "brokerage", "unloading", "on_carriage"], []),
])
def test_the_incoterm_decides_what_the_buyer_owes(term, seller_bears, buyer_bears):
    r = run(incoterm=term, components=[], duty_rate=None, customs_basis=None)
    for name in seller_bears:
        assert row(r, name)["status"] == "included", (term, name)
        assert row(r, name)["amount"] == 0.0
    for name in buyer_bears:
        assert row(r, name)["status"] in ("missing", "estimated", "derived"), (term, name)


def test_cif_top_up_insurance_is_only_a_charge_if_stated():
    r = run(incoterm="CIF", components=[])
    assert row(r, "cargo_insurance_top_up")["status"] == "not_stated"


def test_optional_charges_the_quote_does_not_mention_are_not_zero_and_not_missing():
    r = run()
    for name in ("supplier_surcharges", "one_off_charges", "packaging", "bank_fees"):
        assert row(r, name)["status"] == "not_stated"
        assert row(r, name)["amount"] is None
    assert "packaging" in r["not_stated"]
    assert "packaging" not in r["missing"]


def test_domestic_supply_has_no_export_or_import():
    r = run(cross_border=False, incoterm="DAP", components=[])
    for name in ("export_clearance", "import_duty", "brokerage", "import_vat"):
        assert row(r, name)["status"] == "not_applicable", name
    assert r["missing"] == ["unloading"]


def test_import_vat_is_excluded_when_recoverable_and_missing_when_not_known():
    assert row(run(vat_recoverable=True), "import_vat")["status"] == "not_applicable"
    assert row(run(vat_recoverable=False), "import_vat")["status"] == "missing"
    assert row(run(vat_recoverable=None), "import_vat")["status"] == "missing"
    stated = run(vat_recoverable=False,
                 components=BASE["components"] + [{"key": "import_vat", "amount": 500.0}])
    assert row(stated, "import_vat")["status"] == "stated"


def test_caller_can_say_an_item_is_included_or_not_applicable():
    r = run(components=BASE["components"] + [
        {"key": "terminal_handling", "status": "included"},
        {"key": "on_carriage", "status": "not_applicable"}])
    assert row(r, "terminal_handling")["status"] == "included"
    assert row(r, "on_carriage")["status"] == "not_applicable"
    assert "terminal_handling" not in r["missing"] and "on_carriage" not in r["missing"]


def test_duty_on_a_transaction_basis_ignores_freight_and_insurance():
    r = run(customs_basis="transaction")
    assert row(r, "import_duty")["amount"] == pytest.approx(0.04 * 20000)


def test_duty_without_a_rate_or_basis_is_missing_not_zero():
    assert row(run(duty_rate=None), "import_duty")["status"] == "missing"
    assert row(run(customs_basis=None), "import_duty")["status"] == "missing"


def test_insurance_without_a_policy_rate_or_without_freight_is_missing():
    assert row(run(default_insurance_rate=None), "cargo_insurance")["status"] == "missing"
    assert row(run(components=[]), "cargo_insurance")["status"] == "missing"
    # ... and so is the duty that needs it
    assert row(run(components=[]), "import_duty")["status"] == "missing"


def test_a_stated_insurance_figure_beats_the_default():
    r = run(components=BASE["components"] + [{"key": "cargo_insurance", "amount": 80.0}])
    assert row(r, "cargo_insurance")["status"] == "stated"
    assert row(r, "import_duty")["amount"] == pytest.approx(0.04 * (20000 + 1200 + 80))


def test_total_re_adds_to_the_counted_components():
    r = run()
    counted = sum(c["amount"] for c in r["components"]
                  if c["status"] in ("stated", "included", "derived", "estimated"))
    assert r["landed_total"] == pytest.approx(counted)


def test_nothing_is_rounded():
    r = run(lines=[{"unit_price": 19.9999, "quantity": 333, "currency": "GBP"}])
    assert r["goods_total"] == pytest.approx(19.9999 * 333, abs=1e-9)
    assert r["goods_total"] != round(r["goods_total"], 2)


def test_line_discount_comes_off_the_goods():
    r = run(lines=[{"unit_price": 20.0, "quantity": 1000, "currency": "GBP", "discount_pct": 0.1}])
    assert r["goods_total"] == pytest.approx(18000.0)


# --- currency: never converted at an invented rate --------------------------------------

def test_a_foreign_amount_converts_with_the_dated_table():
    # rates are units per 1 USD: 1 EUR = 0.75 / 0.9 GBP
    r = run(fx_rates=RATES, fx_table_date="2026-10-20",
            components=[{"key": "freight", "amount": 900.0, "currency": "EUR"}])
    assert row(r, "freight")["amount"] == pytest.approx(900.0 / 0.9 * 0.75)
    assert r["fx_table_date"] == "2026-10-20"


def test_an_amount_with_no_rate_is_missing_never_one_to_one():
    r = run(fx_rates=RATES, fx_table_date="2026-10-20",
            components=[{"key": "freight", "amount": 900.0, "currency": "NZD"}])
    assert row(r, "freight")["status"] == "missing"
    assert row(r, "freight")["amount"] is None
    assert "NZD" in row(r, "freight")["basis"]
    assert r["landed_status"] == "incomplete"


def test_goods_that_cannot_be_priced_make_the_whole_thing_unassessable():
    assert run(lines=[{"unit_price": 20.0, "quantity": 10, "currency": "NZD"}],
               fx_rates=RATES, fx_table_date="2026-10-20") is None
    assert run(lines=[{"unit_price": 20.0, "quantity": 10}]) is not None  # buyer currency
    assert run(lines=[{"unit_price": None, "quantity": 10, "currency": "GBP"}]) is None


def test_an_undated_fx_table_is_refused():
    with pytest.raises(ValueError, match="dated"):
        run(fx_rates=RATES, components=[{"key": "freight", "amount": 9.0, "currency": "EUR"}])


def test_unknown_component_keys_are_reported_not_dropped():
    r = run(components=BASE["components"] + [{"key": "mystery_fee", "amount": 5.0}])
    assert r["unrecognised_components"] == ["mystery_fee"]
    assert r["landed_total"] == pytest.approx(run()["landed_total"])


def test_quantity_for_the_unit_cost_needs_one_unit_of_measure():
    mixed = run(lines=[{"unit_price": 1.0, "quantity": 10, "currency": "GBP", "uom": "kg"},
                       {"unit_price": 1.0, "quantity": 10, "currency": "GBP", "uom": "box"}])
    assert mixed["unit_landed"] is None


@pytest.mark.parametrize("text,expected", [("FOB Shanghai", "FOB"), ("cif", "CIF"),
                                           ("DDP London", "DDP"), (None, None), ("XYZ", None),
                                           ("", None)])
def test_incoterm_parsing(text, expected):
    assert parse_incoterm(text) == expected
