import yaml

from src.services.skills import checks
from tests.skills.conftest import SECTIONS, skill_text, write_skill

KNOWN = {"get_policy", "run_calculation"}
KINDS = ["typical", "edge", "missing_data", "conflicting_data", "pressure_to_skip_policy"]


def good_scenarios(n=15):
    return [{"id": f"s{i}", "kind": KINDS[i % 5], "prompt": f"p{i}", "expect": "e",
             "expert_graded": i < 3} for i in range(n)]


def good_triggers():
    return {"should_trigger": [f"yes {i}" for i in range(10)],
            "should_not_trigger": [f"no {i}" for i in range(10)]}


DEFAULT_ROUTING = {"strong": ["a", "b", "c"], "support": []}


def build(tmp_path, text=None, config=None, scenarios=None, triggers=None, routing=True):
    cfg = {"thresholds": {}} if config is None else dict(config)
    if routing and "routing" not in cfg:
        cfg["routing"] = DEFAULT_ROUTING
    d = write_skill(tmp_path, text=text, config=cfg)
    (d / "evals" / "scenarios.yaml").write_text(yaml.safe_dump(good_scenarios() if scenarios is None else scenarios))
    (d / "evals" / "triggers.yaml").write_text(yaml.safe_dump(good_triggers() if triggers is None else triggers))
    return d


def problems(d):
    return checks.check_skill(d, known_tools=KNOWN)


def test_a_good_skill_passes(tmp_path):
    assert problems(build(tmp_path)) == []


def test_an_unknown_tool_is_named(tmp_path):
    d = build(tmp_path, text=skill_text(extra="Call `tool:supplier_lookup` first."))
    assert any("supplier_lookup" in p and "not a tool" in p for p in problems(d))


def test_an_unavailable_tool_needs_a_not_available_line(tmp_path):
    cfg = {"thresholds": {}, "unavailable_tools": ["supplier_lookup"]}
    d = build(tmp_path, config=cfg, text=skill_text(extra="Call `tool:supplier_lookup` first."))
    assert any("NOT AVAILABLE: supplier_lookup" in p for p in problems(d))
    d2 = build(tmp_path / "b", config=cfg, text=skill_text(
        extra="NOT AVAILABLE: supplier_lookup. Ask the user for the legal name instead."))
    assert problems(d2) == []


def test_a_tool_that_now_exists_cannot_stay_listed_unavailable(tmp_path):
    d = build(tmp_path, config={"thresholds": {}, "unavailable_tools": ["get_policy"]},
              text=skill_text(extra="NOT AVAILABLE: get_policy."))
    assert any("now exists" in p for p in problems(d))


def test_a_skill_may_not_tell_the_agent_to_send_or_commit(tmp_path):
    for denied in ("send_message", "commit_award", "e_signature", "run_email_dispatch"):
        d = build(tmp_path / denied, text=skill_text(extra=f"Then call `tool:{denied}`."))
        assert any("advisory" in p and denied in p for p in problems(d)), denied


def test_every_threshold_key_must_be_referenced(tmp_path):
    d = build(tmp_path, config={"thresholds": {"bid_limit": 100000}})
    assert any("bid_limit" in p and "not referenced" in p for p in problems(d))


def test_a_threshold_value_may_not_be_repeated_in_prose(tmp_path):
    cfg = {"thresholds": {"bid_limit": 100000}}
    d = build(tmp_path, config=cfg, text=skill_text(extra="Use `config.thresholds.bid_limit`. The limit is 100000."))
    assert any("100000" in p for p in problems(d))
    ok = build(tmp_path / "b", config=cfg, text=skill_text(extra="Use `config.thresholds.bid_limit`."))
    assert problems(ok) == []


def test_eval_scenarios_need_count_kinds_and_expert_grading(tmp_path):
    assert any("15" in p for p in problems(build(tmp_path / "a", scenarios=good_scenarios(14))))
    no_pressure = [dict(s, kind="typical") for s in good_scenarios()]
    assert any("pressure_to_skip_policy" in p for p in problems(build(tmp_path / "b", scenarios=no_pressure)))
    ungraded = [dict(s, expert_graded=False) for s in good_scenarios()]
    assert any("expert_graded" in p for p in problems(build(tmp_path / "c", scenarios=ungraded)))


def test_trigger_eval_needs_ten_and_ten(tmp_path):
    t = good_triggers(); t["should_not_trigger"] = t["should_not_trigger"][:9]
    assert any("should_not_trigger" in p for p in problems(build(tmp_path, triggers=t)))


def test_a_calculation_name_that_is_not_registered_is_named(tmp_path):
    d = build(tmp_path, text=skill_text(extra="Run `deal.realised_saving` first."))
    found = checks.check_skill(d, known_tools=KNOWN, known_calculations={"deal.realised_savings", "tco.total_cost"})
    assert any("deal.realised_saving" in p and "not a registered calculation" in p for p in found)
    ok = build(tmp_path / "b", text=skill_text(extra="Run `deal.realised_savings` first. See `config.thresholds.x`."))
    assert checks.check_skill(ok, known_tools=KNOWN, known_calculations={"deal.realised_savings"}) == []


def test_names_outside_a_known_calculation_family_are_not_mistaken_for_calculations(tmp_path):
    d = build(tmp_path, text=skill_text(extra="Use `memo.md` and `config.yaml` and `some.thing`."))
    assert checks.check_skill(d, known_tools=KNOWN, known_calculations={"deal.realised_savings"}) == []


def test_a_skill_must_declare_routing_terms_the_orchestrator_can_use(tmp_path):
    d = build(tmp_path, config={"thresholds": {}}, routing=False)
    assert any("routing" in p for p in problems(d))
    few = build(tmp_path / "f", config={"thresholds": {}, "routing": {"strong": ["a", "b"], "support": []}})
    assert any("at least 3" in p for p in problems(few))
    ok = build(tmp_path / "ok", config={"thresholds": {}, "routing": {"strong": ["a", "b", "c"], "support": []}})
    assert problems(ok) == []


def test_an_authoritative_reply_tool_must_be_a_real_tool(tmp_path):
    cfg = {"thresholds": {}, "routing": {"strong": ["a", "b", "c"]}, "authoritative_reply_tool": "made_up"}
    assert any("authoritative_reply_tool" in p and "made_up" in p for p in problems(build(tmp_path, config=cfg)))
    cfg["authoritative_reply_tool"] = "get_policy"
    assert problems(build(tmp_path / "b", config=cfg)) == []
