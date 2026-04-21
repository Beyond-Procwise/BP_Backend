from src.services.structural_extractor.derivation import (
    DerivationRule, REGISTRY, rule, clear_registry
)


def test_registering_a_rule_adds_to_registry():
    clear_registry()

    @rule("test_rule", "target_x", ["input_a", "input_b"])
    def _fn(inputs):
        return inputs["input_a"] + inputs["input_b"]

    assert len(REGISTRY) == 1
    r = REGISTRY[0]
    assert r.rule_id == "test_rule"
    assert r.target_field == "target_x"
    assert r.inputs == ["input_a", "input_b"]
    assert r.compute({"input_a": 1, "input_b": 2}) == 3
