import pytest
from src.services.negotiation_advice import advisor as ad


# alternative_supplier_count must sit ABOVE the many_alternatives bar (93) for this
# deal to read as Leverage, which every test below assumes. The plan's draft used 30
# — a leftover from the abandoned 23 threshold, under which 30 counted as "many".
# Against the real per-deal median of 93 it means "few", and the deal classifies
# Strategic instead. 120 is the same "MANY" the classification tests use.
_SIGNALS = {"deal_id": "D-1", "supplier_id": "SUP-1", "supplier_name": "Orbis",
            "currency": "GBP", "deal_value": 200000.0, "invoice_total": 205000.0,
            "po_total": 198000.0, "quote_supplier_count": 1,
            "alternative_supplier_count": 120, "risk_score": 0.2,
            "is_preferred": False, "price_variance_pct": 8.4}


@pytest.fixture(autouse=True)
def _stub(monkeypatch):
    monkeypatch.setattr(ad, "gather_signals", lambda cur, deal_id:
                        dict(_SIGNALS) if deal_id == "D-1" else None)
    monkeypatch.setattr(ad, "load_thresholds", lambda conn:
                        {"high_spend": 98175.0, "many_alternatives": 93})
    monkeypatch.setattr(ad, "save_advice", lambda conn, **kw:
                        {"advice_id": "A-1", **kw})
    monkeypatch.setattr(ad, "active_facts", lambda conn, advice_id: {})
    monkeypatch.setattr(ad, "state_fact", lambda conn, **kw: None)
    monkeypatch.setattr(ad, "withdraw_fact", lambda conn, **kw: None)
    monkeypatch.setattr(ad, "load_advice", lambda conn, deal_id:
                        {"advice_id": "A-1"})


class _Conn:
    def cursor(self):
        return object()

    def commit(self):
        pass


def test_build_advice_produces_classified_grounded_plays():
    out = ad.build_advice("D-1", conn=_Conn())
    assert out["quadrant"] == "Leverage"
    assert out["quadrant_source"] == "computed"
    assert out["style"] == "Competitive"
    assert out["plays"]
    assert all(p["state"] in ("ready", "groundwork") for p in out["plays"])
    assert out["quadrant_reasons"]


def test_unknown_deal_returns_none():
    assert ad.build_advice("NOPE", conn=_Conn()) is None


def test_override_marks_the_source_as_buyer():
    out = ad.build_advice("D-1", conn=_Conn(),
                          overrides={"quadrant": "Bottleneck"})
    assert out["quadrant"] == "Bottleneck"
    assert out["quadrant_source"] == "buyer"
    assert out["style_source"] == "computed"


def test_overriding_the_quadrant_recomputes_the_style():
    # Bottleneck must not inherit Leverage's Competitive style — a thin supply
    # market calls for the opposite posture.
    out = ad.build_advice("D-1", conn=_Conn(),
                          overrides={"quadrant": "Bottleneck"})
    assert out["style"] == "Principled"
    assert out["style_source"] == "computed"
    assert out["style_reasons"]


def test_overriding_both_keeps_the_buyers_style():
    out = ad.build_advice("D-1", conn=_Conn(),
                          overrides={"quadrant": "Bottleneck",
                                     "style": "Accommodating"})
    assert out["style"] == "Accommodating"
    assert out["style_source"] == "buyer"


def test_stated_fact_changes_classification_and_is_labelled():
    out = ad.build_advice("D-1", conn=_Conn(),
                          stated={"alternative_supplier_count": 2})
    # high spend + few alternatives -> Strategic, not Leverage
    assert out["quadrant"] == "Strategic"
    assert out["stated_facts"]["alternative_supplier_count"] == 2
    # the measured value is preserved, not overwritten
    assert out["signals"]["alternative_supplier_count"] == 120


def test_indeterminate_returns_no_plays_and_does_not_guess():
    out = ad.build_advice("D-1", conn=_Conn(),
                          stated={"deal_value": None,
                                  "alternative_supplier_count": None})
    assert out["indeterminate"] is True
    assert out["quadrant"] is None
    assert out["plays"] == []
    assert out["quadrant_reasons"]


def test_turn_set_lever_restricts_the_levers():
    out = ad.apply_turn("D-1", {"action": "set_lever", "lever": "Risk"},
                        conn=_Conn())
    assert {p["lever"] for p in out["plays"]} == {"Risk"}


def test_turn_more_plays_returns_more_than_the_default():
    base = ad.build_advice("D-1", conn=_Conn())
    more = ad.apply_turn("D-1", {"action": "more_plays"}, conn=_Conn())
    assert len(more["plays"]) >= len(base["plays"])


def test_turn_compare_style_returns_both():
    out = ad.apply_turn("D-1", {"action": "compare_style",
                                "style": "Collaborative"}, conn=_Conn())
    assert out["style"] == "Competitive"
    assert out["comparison"]["style"] == "Collaborative"
    assert out["comparison"]["plays"]


def test_turn_override_flows_through():
    out = ad.apply_turn("D-1", {"action": "override", "style": "Principled"},
                        conn=_Conn())
    assert out["style"] == "Principled"
    assert out["style_source"] == "buyer"


def test_stating_a_fact_on_a_deal_with_no_advice_yet_still_records_it(monkeypatch):
    # Nothing guarantees the buyer viewed the deal before stating a fact. With no
    # advice row there was no advice_id to hang the fact off, and the turn
    # silently dropped it.
    monkeypatch.setattr(ad, "load_advice", lambda conn, deal_id: None)
    stated = []
    monkeypatch.setattr(ad, "state_fact",
                        lambda conn, **kw: stated.append(kw))
    out = ad.apply_turn("D-1", {"action": "state_fact",
                                "fact_key": "alternative_supplier_count",
                                "fact_value": "2"}, conn=_Conn())
    assert stated and stated[0]["advice_id"] == "A-1"
    assert out is not None


def test_stating_a_fact_on_an_unknown_deal_is_not_fatal(monkeypatch):
    monkeypatch.setattr(ad, "load_advice", lambda conn, deal_id: None)
    assert ad.apply_turn("NOPE", {"action": "state_fact", "fact_key": "k",
                                  "fact_value": "1"}, conn=_Conn()) is None


def test_unknown_action_is_ignored_not_fatal():
    out = ad.apply_turn("D-1", {"action": "nonsense"}, conn=_Conn())
    assert out["quadrant"] == "Leverage"
