from src.services.negotiation_advice import classification as cl

# Spend either side of the 98,175 bar; suppliers on record either side of 3.
HIGH = 200_000.0
LOW = 1_000.0
MANY = 120
FEW = 2


def _sig(value, alt, risk=50.0, preferred=False, variance=None):
    # risk_score is a 0-100 scale in this database, median 49.57.
    return {"deal_value": value, "alternative_supplier_count": alt,
            "risk_score": risk, "is_preferred": preferred,
            "price_variance_pct": variance}


def test_high_spend_many_alternatives_is_leverage():
    out = cl.classify(_sig(HIGH, MANY))
    assert out["quadrant"] == "Leverage"
    assert out["indeterminate"] is False


def test_high_spend_few_alternatives_is_strategic():
    assert cl.classify(_sig(HIGH, FEW))["quadrant"] == "Strategic"


def test_low_spend_many_alternatives_is_transactional():
    assert cl.classify(_sig(LOW, MANY))["quadrant"] == "Transactional"


def test_low_spend_few_alternatives_is_bottleneck():
    assert cl.classify(_sig(LOW, FEW))["quadrant"] == "Bottleneck"


def test_a_contested_market_starts_at_three_suppliers():
    # The bar was the corpus median (93): every test deal read as contested and
    # every real one, with only its own supplier on record, as thin.
    assert cl.classify(_sig(HIGH, 2))["quadrant"] == "Strategic"
    assert cl.classify(_sig(HIGH, 3))["quadrant"] == "Leverage"


def test_quadrant_is_one_of_the_playbook_keys():
    valid = {"Transactional", "Leverage", "Strategic", "Bottleneck"}
    for value, alt in [(HIGH, MANY), (HIGH, FEW), (LOW, MANY), (LOW, FEW)]:
        assert cl.classify(_sig(value, alt))["quadrant"] in valid


def test_missing_spend_is_indeterminate_not_guessed():
    out = cl.classify(_sig(None, MANY))
    assert out["quadrant"] is None
    assert out["indeterminate"] is True


def test_missing_alternatives_is_indeterminate_not_guessed():
    out = cl.classify(_sig(HIGH, None))
    assert out["quadrant"] is None
    assert out["indeterminate"] is True


def test_reasons_cite_the_actual_numbers():
    reasons = " ".join(cl.classify(_sig(HIGH, MANY))["quadrant_reasons"])
    assert "200,000" in reasons or "200000" in reasons
    assert "120" in reasons


def test_thresholds_are_overridable():
    # with a very high spend bar, 200k is now "low"
    out = cl.classify(_sig(HIGH, MANY), thresholds={"high_spend": 10_000_000.0,
                                                    "many_alternatives": 93})
    assert out["quadrant"] == "Transactional"


def test_style_follows_quadrant_and_evidence():
    assert cl.classify(_sig(HIGH, MANY, variance=8.4))["style"] == "Competitive"
    assert cl.classify(_sig(HIGH, FEW, preferred=True))["style"] == "Collaborative"
    assert cl.classify(_sig(LOW, FEW))["style"] == "Principled"
    assert cl.classify(_sig(LOW, MANY))["style"] == "Competitive"


def test_style_is_one_of_the_playbook_styles():
    valid = {"Competitive", "Collaborative", "Principled", "Accommodating",
             "Compromising"}
    for value, alt in [(HIGH, MANY), (HIGH, FEW), (LOW, MANY), (LOW, FEW)]:
        assert cl.classify(_sig(value, alt))["style"] in valid


def test_confidence_is_lower_near_a_threshold():
    near = cl.classify(_sig(98_200.0, 3))["quadrant_confidence"]
    clear = cl.classify(_sig(1_000_000.0, 300))["quadrant_confidence"]
    assert near < clear
