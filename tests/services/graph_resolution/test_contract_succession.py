from src.services.graph_resolution.profiles import contract_succession as cs

PREV = {"contract_id": "C00002", "supplier_id": "S1",
        "contract_start_date": "2024-01-01", "contract_end_date": "2024-12-31",
        "total_contract_value": 100000.0, "currency": "GBP",
        "contract_title": "Service Desk 24x7", "spend_category": "IT Services",
        "_same_entity_p": 0.99}
NEXT = {"contract_id": "C00055", "supplier_id": "S1",
        "contract_start_date": "2025-01-01", "contract_end_date": "2025-12-31",
        "total_contract_value": 112000.0, "currency": "GBP",
        "contract_title": "Service Desk 24x7", "spend_category": "IT Services",
        "_same_entity_p": 0.99}


def test_adjacent_terms_same_supplier_are_a_succession():
    assert cs.score(PREV, NEXT)["F"] >= 65.0


def test_overlapping_unrelated_contract_is_not_a_succession():
    other = {**NEXT, "contract_start_date": "2024-06-01",
             "contract_title": "Office cleaning", "spend_category": "Facilities"}
    assert cs.score(PREV, other)["F"] < 65.0


def test_uplift_is_reported_in_a_single_currency():
    u = cs.uplift(PREV, NEXT)
    assert u["pct"] == 12.0
    assert u["currency"] == "GBP"


def test_uplift_is_unavailable_across_currencies_rather_than_converted():
    eur = {**NEXT, "currency": "EUR"}
    assert cs.uplift(PREV, eur) is None, (
        "converting would fabricate an FX rate; 10 of 12,408 invoices carry one"
    )


def test_succession_never_reaches_auto_link_while_uncalibrated():
    from src.services.graph_resolution.edge_writer import UNCALIBRATED_PROFILES
    assert cs.PROFILE in UNCALIBRATED_PROFILES
