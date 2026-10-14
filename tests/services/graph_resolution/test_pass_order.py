import pytest
from src.services.graph_resolution.pass_runner import PASS_ORDER, assert_dag_safe


def test_identity_resolves_before_anything_reads_it():
    assert PASS_ORDER.index("supplier_identity") < PASS_ORDER.index("item_equivalence")
    assert PASS_ORDER.index("supplier_identity") < PASS_ORDER.index("contract_coverage")
    assert PASS_ORDER.index("supplier_identity") < PASS_ORDER.index("contract_succession")


def test_items_resolve_before_contract_coverage_reads_them():
    assert PASS_ORDER.index("item_equivalence") < PASS_ORDER.index("contract_coverage")


def test_a_profile_reading_a_downstream_edge_is_refused():
    with pytest.raises(ValueError, match="downstream"):
        assert_dag_safe("supplier_identity", reads=["UNDER_CONTRACT"])


def test_reading_an_upstream_edge_is_allowed():
    assert_dag_safe("contract_coverage", reads=["SAME_ENTITY", "OF_ITEM"]) is None
