"""The negotiation email proposes the deal's levers, not a stock menu (audit rec. 3)."""
import re

from agents import email_drafting_agent as eda
from src.services.negotiation.deal_levers import read_levers

_B = ("For the 5-year renewal we need exclusivity, CPI+2% indexation and "
      "service credits capped at 5%. Net 45 payment is fine.")

_STOCK = ("Net-15", "Net-7", "24-month", "36-month", "12 months", "12→24",
          "Guaranteed minimum order", "2% early payment")


def _play(text, state, lever="Commercial"):
    return {"play": text, "lever": lever, "state": state,
            "evidence": [{"label": "Deal value", "value": "200,000.00", "source": "deal"}]
            if state == "ready" else []}


def test_levers_are_the_ones_the_supplier_raised_whatever_the_round():
    data = {"deal_levers": read_levers(_B)}
    first = eda._negotiation_levers(data, round_no=1)
    assert first == eda._negotiation_levers(data, round_no=3)
    text = "\n".join(first)
    for label in ("Contract term", "Exclusivity", "Price indexation",
                  "Service levels / credits", "Payment terms"):
        assert label in text
    assert "CPI+2%" in text and "Net 45" in text
    assert not any(s in text for s in _STOCK)


def test_only_ready_plays_are_offered_and_at_most_two():
    plays = [_play("Groundwork play", "groundwork"),
             _play("Demand tiered volume discounts", "ready"),
             _play("Request a detailed cost breakdown", "ready"),
             _play("Consolidate spend across divisions/geographies for 12-18% savings", "ready"),
             {"play": "Ungrounded play with no state", "lever": "Risk"}]
    text = "\n".join(eda._negotiation_levers({"play_recommendations": plays}, round_no=2))
    assert "Demand tiered volume discounts" in text
    assert "Request a detailed cost breakdown" in text
    assert "Consolidate spend" not in text
    assert "Groundwork play" not in text
    assert "Ungrounded play" not in text


def test_levers_are_read_from_the_reply_when_the_caller_did_not_pass_them():
    text = "\n".join(eda._negotiation_levers({"supplier_message": _B}, round_no=1))
    assert "Exclusivity" in text and "CPI+2%" in text


def test_nothing_raised_and_nothing_evidenced_means_price_only():
    lines = eda._negotiation_levers(
        {"supplier_message": "Our rate is 945 per load. We can't move on price."}, round_no=2)
    assert len(lines) == 1
    assert "do not introduce" in lines[0].lower()
    assert not any(s in lines[0] for s in _STOCK)


def test_code_prompt_carries_no_stock_figures_or_lever_menu():
    for text in (eda.NEGOTIATION_PLAYBOOK_SYSTEM, eda.NEGOTIATION_PLAYBOOK_USER):
        for stock in ("24 months", "Net-15", "12→24", "£X", "Net-30 → Net-15"):
            assert stock not in text, stock
    tactics = "\n".join(eda._determine_recommended_tactics(r, 25.0, {}) for r in (1, 2, 3))
    assert not re.search(r"payment, volume, duration", tactics)
    assert "faster payment" not in eda.NEGOTIATION_PLAYBOOK_SYSTEM


def test_database_edit_script_matches_the_code_prompt():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "scripts" / "negotiation_playbook_prompt_levers.py"
    spec = importlib.util.spec_from_file_location("np_levers", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for old, new in mod.EDITS:
        assert old not in eda.NEGOTIATION_PLAYBOOK_SYSTEM
        assert new in eda.NEGOTIATION_PLAYBOOK_SYSTEM
