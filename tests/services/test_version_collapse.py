from src.services import version_collapse as vc

def test_base_reference_strips_version_suffix():
    assert vc.base_reference("CPS-Q-3380") == "CPS-Q-3380"
    assert vc.base_reference("CPS-Q-3380 (V2)") == "CPS-Q-3380"
    assert vc.base_reference("CPS-Q-3380 (V3 (BAFO))") == "CPS-Q-3380"

def test_version_ordinal_defaults_to_one():
    assert vc.version_ordinal("CPS-Q-3380") == 1
    assert vc.version_ordinal("CPS-Q-3380 (V2)") == 2
    assert vc.version_ordinal("CPS-Q-3380 (V3 (BAFO))") == 3

def test_collapse_keeps_highest_version_and_records_rounds():
    quotes = [
        {"quote_id": "CPS-Q-3380", "supplier_id": "SUP-ClearPath", "total_amount": 100},
        {"quote_id": "CPS-Q-3380 (V2)", "supplier_id": "SUP-ClearPath", "total_amount": 95},
        {"quote_id": "CPS-Q-3380 (V3 (BAFO))", "supplier_id": "SUP-ClearPath", "total_amount": 90},
    ]
    bids = vc.collapse_versions(quotes)
    assert len(bids) == 1
    bid = bids[0]
    assert bid["quote_id"] == "CPS-Q-3380 (V3 (BAFO))"   # current offer
    assert bid["base_reference"] == "CPS-Q-3380"
    assert bid["version"] == 3
    assert bid["total_amount"] == 90
    assert bid["rounds"] == ["CPS-Q-3380", "CPS-Q-3380 (V2)", "CPS-Q-3380 (V3 (BAFO))"]

def test_different_base_refs_stay_separate():
    quotes = [
        {"quote_id": "CPS-Q-3380", "supplier_id": "SUP-ClearPath"},
        {"quote_id": "MFS-Q-3391", "supplier_id": "SUP-MeridianFreight"},
    ]
    assert len(vc.collapse_versions(quotes)) == 2
