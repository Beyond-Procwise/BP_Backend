from scripts.graph_resolution.calibrate import build_labelled_pairs, best

ROWS = [
    {"supplier_id": "A", "vat_number": "GB1", "supplier_name": "Acme Ltd",
     "registration_number": "R1", "duns_number": "D1", "country": "GB", "postal_code": "P1"},
    {"supplier_id": "B", "vat_number": "GB1", "supplier_name": "Acme Limited",
     "registration_number": "R1", "duns_number": "D1", "country": "GB", "postal_code": "P1"},
    {"supplier_id": "C", "vat_number": "GB2", "supplier_name": "Globex plc",
     "registration_number": "R2", "duns_number": "D2", "country": "GB", "postal_code": "P2"},
]


def test_pairs_are_labelled_by_shared_vat():
    pairs = build_labelled_pairs(ROWS)
    labels = {(a["supplier_id"], b["supplier_id"]): same for a, b, same in pairs}
    assert labels[("A", "B")] is True
    assert labels[("A", "C")] is False


def test_vat_is_withheld_from_the_scored_records():
    pairs = build_labelled_pairs(ROWS)
    for a, b, _ in pairs:
        assert "vat_number" not in a, "VAT is the label; scoring on it is circular"
        assert "vat_number" not in b


def test_best_prefers_the_higher_separation():
    results = [
        {"p0": 0.02, "alpha": 0.30, "separation": 12.0, "false_auto_links": 0},
        {"p0": 0.03, "alpha": 0.55, "separation": 31.5, "false_auto_links": 0},
    ]
    assert best(results)["alpha"] == 0.55


def test_best_refuses_a_setting_that_auto_links_a_false_pair():
    results = [
        {"p0": 0.02, "alpha": 0.30, "separation": 12.0, "false_auto_links": 0},
        {"p0": 0.09, "alpha": 0.90, "separation": 44.0, "false_auto_links": 3},
    ]
    assert best(results)["alpha"] == 0.30, "a false auto_link disqualifies a setting"
