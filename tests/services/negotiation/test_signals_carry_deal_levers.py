from agents.negotiation import analysis


def test_signals_carry_the_levers_the_supplier_raised(monkeypatch):
    monkeypatch.setattr(analysis, "LLM_ENABLED", False)
    signals = analysis.extract_negotiation_signals(
        supplier_message=("For the 5-year renewal we need exclusivity, CPI+2% indexation "
                          "and service credits capped at 5%."),
        snippets=["Net 45 payment is fine."], market={}, performance={})
    assert [lv["lever"] for lv in signals["deal_levers"]] == [
        "term", "renewal_termination", "exclusivity", "indexation",
        "service_levels", "payment_terms"]


def test_a_model_reply_cannot_overwrite_the_quoted_levers(monkeypatch):
    import json
    import services.ollama_client as oc
    monkeypatch.setattr(analysis, "LLM_ENABLED", True)
    monkeypatch.setattr(oc, "ollama_generate", lambda *a, **k: json.dumps(
        {"tone": "firm", "deal_levers": [{"lever": "volume", "figure": "1000 units"}]}))
    signals = analysis.extract_negotiation_signals(
        supplier_message="We need exclusivity.", snippets=[], market={}, performance={})
    assert [lv["lever"] for lv in signals["deal_levers"]] == ["exclusivity"]


def test_no_reply_means_no_levers(monkeypatch):
    monkeypatch.setattr(analysis, "LLM_ENABLED", False)
    signals = analysis.extract_negotiation_signals(
        supplier_message=None, snippets=[], market={}, performance={})
    assert signals["deal_levers"] == []
