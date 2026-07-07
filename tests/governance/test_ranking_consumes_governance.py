"""supplier_ranking consumes the injected governance envelope (with fallback)."""
import pytest

from src.agents.supplier_ranking_agent import SupplierRankingAgent
from src.services.governance_tools.envelope import resolve_governance


class _Ctx:
    def __init__(self, governed):
        self.input_data = {"governed": governed} if governed is not None else {}


def _agent():
    return SupplierRankingAgent.__new__(SupplierRankingAgent)


def test_governed_weights_extracted_from_dict_details():
    gov = {"policies": [{"policy_type": "supplier_ranking",
                         "details": {"rules": {"default_weights": {"price": 0.4, "risk": 0.2, "delivery": 0.3, "payment_terms": 0.1}}}}]}
    w = _agent()._governed_default_weights(_Ctx(gov))
    assert w == {"price": 0.4, "risk": 0.2, "delivery": 0.3, "payment_terms": 0.1}


def test_governed_weights_details_as_repr_string():
    gov = {"policies": [{"policy_type": "supplier_ranking",
                         "details": "{'rules': {'default_weights': {'price': 0.5}}}"}]}
    assert _agent()._governed_default_weights(_Ctx(gov)) == {"price": 0.5}


def test_no_governed_returns_empty_fallback():
    assert _agent()._governed_default_weights(_Ctx(None)) == {}
    assert _agent()._governed_default_weights(_Ctx({"policies": []})) == {}


def test_resolve_governance_includes_weight_details():
    from src.services.db import get_conn
    try:
        with get_conn() as c, c.cursor() as cur:
            cur.execute("select 1")
    except Exception:
        pytest.skip("no DB")
    env = resolve_governance("supplier_ranking")
    assert any("default_weights" in str(p.get("details")) for p in env["policies"]), \
        "primary supplier_ranking policy must carry weight details"
