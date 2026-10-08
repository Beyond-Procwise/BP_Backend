"""The ranking agent narrows to suppliers that sold the requested item."""
from types import SimpleNamespace

import pytest

from agents.supplier_ranking_agent import SupplierRankingAgent
from src.services import supplier_shortlist as sl

_REQ = {"title": "Business Laptops (14-inch)"}


def _agent(monkeypatch, sellers=None, boom=False):
    a = object.__new__(SupplierRankingAgent)

    class _Conn:
        def close(self): pass
    a.agent_nick = SimpleNamespace(get_db_connection=lambda: _Conn())

    def fake(conn, term):
        if boom:
            raise RuntimeError("db down")
        fake.seen = term
        return set(sellers or ())
    monkeypatch.setattr(sl, "suppliers_selling", fake)
    a._fake = fake
    return a


def test_a_requirement_narrows_the_candidates_to_suppliers_who_sold_it(monkeypatch):
    a = _agent(monkeypatch, {"S2", "S1"})
    data = {"requirement": _REQ}
    assert a._apply_requirement_shortlist(data) is None
    assert data["supplier_candidates"] == ["S1", "S2"]
    assert a._fake.seen == "laptop"


def test_nobody_selling_it_is_reported_not_papered_over(monkeypatch):
    a = _agent(monkeypatch, set())
    data = {"requirement": _REQ}
    err = a._apply_requirement_shortlist(data)
    assert "laptop" in err and "No supplier" in err
    assert "supplier_candidates" not in data


def test_a_lookup_failure_does_not_fall_back_to_ranking_everyone(monkeypatch):
    a = _agent(monkeypatch, boom=True)
    data = {"requirement": _REQ}
    err = a._apply_requirement_shortlist(data)
    assert err and "supplier_candidates" not in data


@pytest.mark.parametrize("data", [
    {},                                                         # no requirement
    {"requirement": {"title": ""}},                             # nothing to search
    {"requirement": _REQ, "supplier_candidates": ["X"]},        # caller chose already
    {"requirement": _REQ, "supplier_data": [{"supplier_id": "X"}]},  # caller supplied the data
])
def test_cases_the_caller_already_decided_are_untouched(monkeypatch, data):
    a = _agent(monkeypatch, {"S1"})
    before = dict(data)
    assert a._apply_requirement_shortlist(data) is None
    assert data == before
