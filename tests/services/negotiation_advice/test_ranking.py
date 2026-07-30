import pytest
from src.services.negotiation_advice import ranking as rk

_PB = {
    "Leverage": {
        "descriptor": "Run competitive events",
        "examples": ["bundle spend"],
        "styles": {
            "Competitive": {
                "Commercial": ["Demand tiered volume discounts", "Benchmark it"],
                "Operational": ["Require priority fulfilment"],
                "Risk": ["Full refund for non-compliant goods"],
            }
        },
    }
}


def test_rank_plays_returns_plays_for_a_valid_pair():
    out = rk.rank_plays("Leverage", "Competitive", playbook=_PB)
    assert out["plays"]
    assert out["supplier_type"] == "Leverage"
    assert out["style"] == "Competitive"
    assert set(out["lever_priorities"]) == {"Commercial", "Operational", "Risk"}


def test_every_play_carries_the_established_keys():
    play = rk.rank_plays("Leverage", "Competitive", playbook=_PB)["plays"][0]
    for key in ("supplier_type", "style", "lever", "play", "score",
                "policy_alignment", "performance_signals", "market_signals",
                "rationale", "trade_offs"):
        assert key in play, key


def test_unknown_supplier_type_yields_no_plays():
    out = rk.rank_plays("Nonsense", "Competitive", playbook=_PB)
    assert out["plays"] == []


def test_unknown_style_yields_no_plays_but_keeps_descriptor():
    out = rk.rank_plays("Leverage", "Nonsense", playbook=_PB)
    assert out["plays"] == []
    assert out["descriptor"] == "Run competitive events"


def test_base_score_follows_position_within_the_lever():
    plays = rk.rank_plays("Leverage", "Competitive", playbook=_PB)["plays"]
    commercial = [p for p in plays if p["lever"] == "Commercial"]
    first = next(p for p in commercial if p["play"] == "Demand tiered volume discounts")
    second = next(p for p in commercial if p["play"] == "Benchmark it")
    assert first["score"] == pytest.approx(1.0)
    assert second["score"] == pytest.approx(1.01)


def test_signals_change_the_ranking():
    plain = rk.rank_plays("Leverage", "Competitive", playbook=_PB)["plays"]
    boosted = rk.rank_plays(
        "Leverage", "Competitive", playbook=_PB,
        supplier_performance={"on_time_delivery": 0.72},
        market_context={"supply_risk": "high"},
    )["plays"]
    assert boosted[0]["score"] > plain[0]["score"]


def test_lever_priorities_restrict_the_levers_considered():
    out = rk.rank_plays("Leverage", "Competitive", playbook=_PB,
                        lever_priorities=["Risk"])
    assert {p["lever"] for p in out["plays"]} == {"Risk"}


def test_limit_caps_the_list():
    out = rk.rank_plays("Leverage", "Competitive", playbook=_PB, limit=2)
    assert len(out["plays"]) == 2


def test_load_playbook_reads_the_shipped_file():
    pb = rk.load_playbook()
    assert set(pb) == {"Transactional", "Leverage", "Strategic", "Bottleneck"}
    for entry in pb.values():
        assert "Competitive" in entry["styles"]
