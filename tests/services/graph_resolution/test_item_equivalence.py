from src.services import linking_engine as _le
from src.services.graph_resolution.profiles import item_equivalence as ie

L1 = {"invoice_line_id": "L1", "item_id": "ITM001",
      "item_description": "Dell Latitude 5540 laptop", "unit_of_measure": "each",
      "unit_price": 780.0, "_same_entity_p": None}
L2 = {"invoice_line_id": "L2", "item_id": "ITM001",
      "item_description": "Dell Latitude 5540 Laptop", "unit_of_measure": "each",
      "unit_price": 782.0, "_same_entity_p": None}


def test_identical_item_id_reaches_the_strongest_band_this_profile_can_produce():
    """Full agreement on item_id, description, uom and price -- everything
    this profile can ever observe except supplier_same (see
    test_auto_link_is_structurally_unreachable_for_this_profile) -- lands in
    auto_link_with_warning, the ceiling for this corpus, not block/weak/review."""
    assert ie.score(L1, L2)["decision"] == "auto_link_with_warning"


def test_auto_link_is_structurally_unreachable_for_this_profile():
    """The maximal case: identical item_id, description, uom and price --
    the most evidence this profile can ever see with supplier_same MISSING
    (no SAME_ENTITY edge exists anywhere in this corpus). Measured ceiling:
    F=91.5625 (auto_link_with_warning), never auto_link (_BAND_AUTO=92).

    supplier_same keeps its full weight=3 (see item_equivalence.SIGNALS):
    this documents a genuine consequence of the signal being unobservable
    on this corpus, not a defect to be tuned away by downweighting it --
    the same discipline test_supplier_identity.py applies to its own
    structural ceiling.
    """
    r = ie.score(L1, L2)
    assert r["decision"] == "auto_link_with_warning", r["F"]
    assert r["F"] < _le._BAND_AUTO


def test_descriptive_drift_still_resolves_without_item_id():
    """Separation, not an absolute band: alpha is calibrated in Step 5."""
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
