import pytest

from src.services.calculation_tools import _formulas, list_calculations, run_calculation

IN = {
    "criteria": [
        {"criterion_id": "quality", "parent_id": None, "weight": "0.6", "type": "scored"},
        {"criterion_id": "price", "parent_id": None, "weight": "0.4", "type": "price", "direction": "lower_better"},
    ],
    "scale_min": 0, "scale_max": 5,
    "scores": [{"supplier_id": "A", "criterion_id": "quality", "evaluator_id": "e1", "score": 4},
               {"supplier_id": "B", "criterion_id": "quality", "evaluator_id": "e1", "score": 3}],
    "metrics": [{"supplier_id": "A", "criterion_id": "price", "value": 100},
                {"supplier_id": "B", "criterion_id": "price", "value": 120}],
}


def test_registered_and_discoverable():
    assert "scoring.weighted_total" in _formulas().REGISTRY
    assert any(c["name"] == "scoring.weighted_total" for c in list_calculations("scoring")["calculations"])


def test_runs_through_run_calculation():
    out = run_calculation("scoring.weighted_total", IN)
    assert out["assessed"] is True, out
    v = out["value"] if "value" in out else out["result"]
    assert v["top_supplier"] == "A" and v["top_total"] == pytest.approx(0.88)


def test_bad_weights_come_back_unassessed_not_guessed():
    bad = dict(IN, criteria=[dict(IN["criteria"][0], weight="0.5"), IN["criteria"][1]])
    assert run_calculation("scoring.weighted_total", bad)["assessed"] is False
