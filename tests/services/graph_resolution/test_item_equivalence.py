from src.services import linking_engine as _le
from src.services.graph_resolution.profiles import item_equivalence as ie

L1 = {"invoice_line_id": "L1", "item_id": "ITM001",
      "item_description": "Dell Latitude 5540 laptop", "unit_of_measure": "each",
      "unit_price": 780.0, "_same_entity_p": None}
L2 = {"invoice_line_id": "L2", "item_id": "ITM001",
      "item_description": "Dell Latitude 5540 Laptop", "unit_of_measure": "each",
      "unit_price": 782.0, "_same_entity_p": None}


def test_identical_item_id_outscores_an_unrelated_pair():
    """Separation, not an absolute band: p0/alpha are unmeasured starting
    values (see module docstring) -- calibration was attempted and found no
    separation once the description-embeds-item_id leak was removed. A full
    match must still clearly outscore an unrelated pair regardless."""
    unrelated = {**L2, "item_id": "ITM999",
                 "item_description": "Office chair, mesh back", "unit_price": 120.0}
    assert ie.score(L1, L2)["F"] > ie.score(L1, unrelated)["F"]


def test_auto_link_is_structurally_unreachable_for_this_profile():
    """The maximal case: identical item_id, description, uom and price --
    the most evidence this profile can ever see with supplier_same MISSING
    (no SAME_ENTITY edge exists anywhere in this corpus). supplier_same
    keeps its full weight=3 (see item_equivalence.SIGNALS), which caps
    coverage C regardless of alpha, so F cannot exceed ~91.56 no matter how
    much evidence-weight (alpha) is applied. To test that ceiling as a
    structural fact independent of whichever alpha happens to be registered
    (currently an unmeasured starting value, not this), this pushes alpha
    far past saturation and confirms the ceiling still holds -- this
    documents a genuine consequence of the signal being unobservable on
    this corpus, not a defect to be tuned away, the same discipline
    test_supplier_identity.py applies to its own structural ceiling.
    """
    original = dict(_le.PROFILES[ie.PROFILE])
    try:
        _le.PROFILES[ie.PROFILE] = {**original, "alpha": 50.0}
        r = ie.score(L1, L2)
    finally:
        _le.PROFILES[ie.PROFILE] = original
    assert r["decision"] == "auto_link_with_warning", r["F"]
    assert r["F"] < _le._BAND_AUTO


def test_descriptive_drift_still_resolves_without_item_id():
    """Separation, not an absolute band: p0/alpha are starting values, not
    measured -- calibration was attempted and found no separation once the
    label leak was removed (see module docstring)."""
    a = {**L1, "item_id": None}
    b = {**L2, "item_id": None}
    unrelated = {**L2, "item_id": None,
                 "item_description": "Office chair, mesh back",
                 "unit_price": 120.0}
    assert ie.score(a, b)["F"] > ie.score(a, unrelated)["F"]


def test_different_products_do_not_link():
    b = {**L2, "item_id": "ITM999",
         "item_description": "Office chair, mesh back", "unit_price": 120.0}
    assert ie.score(L1, b)["decision"] in ("weak_relation", "block_or_exception")


def test_item_key_is_deterministic_regardless_of_member_order():
    members = [{"item_id": "ITM009"}, {"item_id": "ITM002"}]
    assert ie.item_key(members) == ie.item_key(list(reversed(members)))


def test_item_key_uses_the_lowest_item_id():
    assert ie.item_key([{"item_id": "ITM009"}, {"item_id": "ITM002"}]) == \
           ie.item_key([{"item_id": "ITM002"}])


def test_item_key_falls_back_to_description_when_no_id_exists():
    k = ie.item_key([{"item_id": None, "item_description": "Widget  A"}])
    assert k and isinstance(k, str)


def test_uom_is_recorded_never_converted():
    a = {**L1, "unit_of_measure": "box of 10"}
    r = ie.score(a, L2)
    uom = [s for s in r["signals"] if s["id"] == "uom"][0]
    assert uom["status"] in ("CONFLICT", "WEAK"), (
        "a pack of 10 and 10 each are related, not equal"
    )


def test_derived_supplier_signal_contributes_nothing_when_absent():
    r = ie.score(L1, L2)
    sup = [s for s in r["signals"] if s["id"] == "supplier_same"][0]
    assert sup["status"] == "MISSING"
