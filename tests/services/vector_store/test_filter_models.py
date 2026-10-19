"""The neutral filter objects callers build (they replace qdrant_client.models for filters, so the dependency can be removed)."""
import pytest

from src.services.vector_store.filter_models import FieldCondition, Filter, MatchAny, MatchValue, models
from src.services.vector_store.filters import to_store_filter


def test_the_models_namespace_exposes_exactly_the_filter_constructs_the_application_uses():
    assert (models.Filter, models.FieldCondition, models.MatchValue, models.MatchAny) == (Filter, FieldCondition, MatchValue, MatchAny)


def test_filters_are_built_and_inspected_the_same_way_as_before():
    f = models.Filter(must=[models.FieldCondition(key="document_type", match=models.MatchValue(value="invoice"))],
                      should=[models.FieldCondition(key="supplier_id", match=models.MatchAny(any=["S1", "S2"]))])
    assert isinstance(f.must[0], models.FieldCondition) and isinstance(f.must[0].match, models.MatchValue)
    assert f.must[0].key == "document_type" and f.must[0].match.value == "invoice" and f.should[0].match.any == ["S1", "S2"]
    assert f.must_not is None


def test_a_filter_with_nothing_in_it_is_empty():
    assert to_store_filter(None) == {} and to_store_filter(Filter()) == {} and to_store_filter(Filter(must=[])) == {}


def test_conversion_is_unchanged_by_the_new_classes():
    f = Filter(must=[FieldCondition(key="document_type", match=MatchValue(value="quote")),
                     FieldCondition(key="supplier_name", match=MatchAny(any=["Acme"]))],
               should=[FieldCondition(key="supplier_id", match=MatchValue(value="S1")), FieldCondition(key="supplier_id", match=MatchValue(value="S2"))])
    assert to_store_filter(f) == {"document_type": "quote", "metadata_in": {"supplier_name": ["Acme"]},
                                  "any_of": [{"supplier_id": "S1"}, {"supplier_id": "S2"}]}


def test_must_not_and_tenant_conditions_are_still_refused():
    with pytest.raises(ValueError):
        to_store_filter(Filter(must_not=[FieldCondition(key="document_type", match=MatchValue(value="x"))]))
    with pytest.raises(ValueError):
        to_store_filter(Filter(must=[FieldCondition(key="tenant_id", match=MatchValue(value="other"))]))
