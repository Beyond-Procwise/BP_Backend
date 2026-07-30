from src.services.negotiation_advice import grounding as gr


def _play(text, lever="Commercial", score=1.0):
    return {"lever": lever, "play": text, "score": score,
            "trade_offs": "some trade-off", "rationale": "because"}


_RICH = {"deal_value": 100000.0, "quote_supplier_count": 1,
         "alternative_supplier_count": 20, "price_variance_pct": 8.4,
         "invoice_total": 105000.0, "po_total": 98000.0}
_BARE = {"deal_value": None, "quote_supplier_count": 1,
         "alternative_supplier_count": None, "price_variance_pct": None,
         "invoice_total": None, "po_total": None}


def test_competitive_play_is_groundwork_with_one_quote_but_alternatives():
    out = gr.assess(_play("Leverage competitor quotes to pressure pricing"), _RICH)
    assert out["state"] == "groundwork"
    assert "20" in out["unlocked_by"]


def test_competitive_play_not_applicable_when_truly_sole_source():
    sole = dict(_RICH, alternative_supplier_count=1)
    out = gr.assess(_play("Run e-auction with 3 bidders"), sole)
    assert out["state"] == "not_applicable"


def test_overbilling_play_is_ready_when_invoice_exceeds_po():
    out = gr.assess(_play("Full refund/credit for non-compliant goods",
                          lever="Risk"), _RICH)
    assert out["state"] == "ready"
    assert any("105,000" in str(e["value"]) or "105000" in str(e["value"])
               for e in out["evidence"])


def test_overbilling_play_is_groundwork_without_the_figures():
    out = gr.assess(_play("Full refund/credit for non-compliant goods",
                          lever="Risk"), _BARE)
    assert out["state"] == "groundwork"


def test_volume_play_needs_a_known_deal_value():
    assert gr.assess(_play("Demand tiered volume discounts"), _RICH)["state"] == "ready"
    assert gr.assess(_play("Demand tiered volume discounts"), _BARE)["state"] == "groundwork"


def test_price_play_needs_known_variance():
    assert gr.assess(_play("Request a detailed cost breakdown"), _RICH)["state"] == "ready"
    assert gr.assess(_play("Request a detailed cost breakdown"), _BARE)["state"] == "groundwork"


def test_play_with_no_testable_precondition_is_ready():
    out = gr.assess(_play("Agree a quarterly governance cadence",
                          lever="Operational"), _BARE)
    assert out["state"] == "ready"


def test_evidence_carries_a_source():
    out = gr.assess(_play("Demand tiered volume discounts"), _RICH)
    assert out["evidence"]
    assert all(e.get("source") for e in out["evidence"])


def test_no_ready_play_claims_evidence_the_deal_lacks():
    for play in [_play("Leverage competitor quotes"),
                 _play("Demand tiered volume discounts"),
                 _play("Request a detailed cost breakdown")]:
        out = gr.assess(play, _BARE)
        if out["state"] == "ready":
            assert out["evidence"] == []


def test_apply_states_drops_not_applicable_and_puts_ready_first():
    sole = dict(_RICH, alternative_supplier_count=1)
    plays = [_play("Run e-auction with 3 bidders", score=9.0),
             _play("Demand tiered volume discounts", score=1.0),
             _play("Full refund for non-compliant goods", "Risk", 2.0)]
    out = gr.apply_states(plays, sole)
    assert all(p["state"] != "not_applicable" for p in out)
    assert [p["state"] for p in out] == sorted(
        [p["state"] for p in out], key=lambda s: 0 if s == "ready" else 1)
