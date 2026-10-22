"""Corpus facts say when their figures are corpus-wide, so a question about one item is not answered with
the whole corpus's totals."""
from types import SimpleNamespace

from src.services import corpus_facts as cf


def _nick():
    class _Cur:
        def __enter__(self): return self
        def __exit__(self, *a): return False
    class _Conn:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def cursor(self): return _Cur()
    return SimpleNamespace(get_db_connection=lambda: _Conn(), embedding_model=None)


def test_corpus_wide_intents_carry_the_scope_note(monkeypatch):
    monkeypatch.setattr(cf, "_fetch", lambda cur, intent, query="", embed=None: {"totals": {"purchase_orders_total": 5042}})
    monkeypatch.setattr(cf, "fetch_graph", lambda query, limit=6: [])
    out = cf.fetch_facts(_nick(), "how many purchase orders for laptop computers")
    assert out["intent"] == "purchase_orders" and out["totals"]["purchase_orders_total"] == 5042
    assert "NOT applied as a filter" in out["scope"] and "whole corpus" in out["scope"]


def test_policies_and_graph_answers_carry_no_scope_note(monkeypatch):
    monkeypatch.setattr(cf, "_fetch", lambda cur, intent, query="", embed=None: {"policies_in_force": [{"policy_name": "x"}]})
    monkeypatch.setattr(cf, "fetch_graph", lambda query, limit=6: [{"node": "x"}])
    out = cf.fetch_facts(_nick(), "what is our approval policy")
    assert out["intent"] == "policies" and "scope" not in out
    monkeypatch.setattr(cf, "detect_intent", lambda q: None)
    out = cf.fetch_facts(_nick(), "tell me about widgets")
    assert out["intent"] == "graph" and "scope" not in out


def test_every_counting_intent_is_on_the_corpus_wide_list():
    names = {name for name, _p in cf._INTENTS}
    assert names - {"policies"} == set(cf.CORPUS_WIDE_INTENTS), names
