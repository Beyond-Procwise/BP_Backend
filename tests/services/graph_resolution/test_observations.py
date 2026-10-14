from src.services.graph_resolution.observations import (
    observation_digest, intersects,
)


def test_digest_is_order_independent():
    a = [("DOC-1", "supplier_id"), ("DOC-2", "vat_number")]
    b = [("DOC-2", "vat_number"), ("DOC-1", "supplier_id")]
    assert observation_digest(a) == observation_digest(b)


def test_digest_distinguishes_different_fields_on_same_doc():
    a = [("DOC-1", "supplier_id")]
    b = [("DOC-1", "vat_number")]
    assert observation_digest(a) != observation_digest(b)


def test_empty_digest_is_stable_and_not_empty_string():
    assert observation_digest([]) == observation_digest(())
    assert observation_digest([]) != ""


def test_intersects_detects_a_shared_observation():
    a = frozenset({("DOC-1", "supplier_id"), ("DOC-1", "vat_number")})
    b = frozenset({("DOC-1", "vat_number")})
    assert intersects(a, b) is True


def test_intersects_is_false_for_disjoint_sets():
    a = frozenset({("DOC-1", "supplier_id")})
    b = frozenset({("DOC-2", "supplier_id")})
    assert intersects(a, b) is False
