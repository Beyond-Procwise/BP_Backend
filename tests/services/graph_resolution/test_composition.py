import pytest

from src.services import linking_engine as _le
from src.services.graph_resolution.composition import remap_clusters
from src.services.graph_resolution.profiles import supplier_identity  # noqa: F401  registers the profile

SPECS = [
    {"id": "vat", "cluster": "registration"},
    {"id": "reg_no", "cluster": "registration"},
    {"id": "name", "cluster": "identity"},
]


def test_disjoint_signals_keep_their_declared_clusters():
    obs = {
        "vat": frozenset({("D1", "vat_number")}),
        "reg_no": frozenset({("D1", "registration_number")}),
        "name": frozenset({("D1", "supplier_name")}),
    }
    assert remap_clusters(SPECS, obs) == {
        "vat": "registration", "reg_no": "registration", "name": "identity",
    }


def test_signals_sharing_an_observation_are_merged_into_one_cluster():
    # `name` secretly read the same field `vat` did: it must not count twice.
    obs = {
        "vat": frozenset({("D1", "vat_number")}),
        "reg_no": frozenset({("D1", "registration_number")}),
        "name": frozenset({("D1", "vat_number")}),
    }
    out = remap_clusters(SPECS, obs)
    assert out["name"] == out["vat"], "correlated signals must share a cluster"
    assert out["reg_no"] == "registration"


def test_merged_cluster_scores_no_higher_than_the_correlated_truth():
    """The double-counting regression test (spec section 10.3)."""
    src = {"supplier_id": "SUP-A", "vat_number": "GB1", "registration_number": "R1"}
    tgt = {"supplier_id": "SUP-A", "vat_number": "GB1", "registration_number": "R1"}
    honest = _le.score_link(src, tgt, "supplier_identity",
                            cluster_overrides={"vat": "registration",
                                               "reg_no": "registration",
                                               "name": "registration"})
    inflated = _le.score_link(src, tgt, "supplier_identity",
                              cluster_overrides={"vat": "c1", "reg_no": "c2",
                                                 "name": "c3"})
    assert honest["F"] <= inflated["F"], (
        "splitting correlated evidence into separate clusters must not be the "
        "cheaper path to a higher score"
    )


def test_score_link_without_overrides_is_unchanged():
    src = {"po_id": "PO-1", "supplier_id": "S1", "currency": "GBP",
           "converted_amount_usd": 100.0, "invoice_date": "2025-01-05",
           "country": "GB", "region": "London"}
    tgt = {"po_id": "PO-1", "supplier_id": "S1", "currency": "GBP",
           "converted_amount_usd": 100.0, "order_date": "2025-01-01",
           "ship_to_country": "GB", "delivery_region": "London"}
    a = _le.score_link(src, tgt, "invoice_po")
    b = _le.score_link(src, tgt, "invoice_po", cluster_overrides=None)
    assert a == b
