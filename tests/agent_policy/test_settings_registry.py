from services.agent_policy import registry, settings
from scripts.agent_policy import seed_registry


def test_settings_defaults_are_the_ruled_values():
    d = settings.DEFAULTS
    assert d["response_time"] == "PT4H" and d["response_time_basis"] == "clock"
    assert "live_conflict_repeat" not in d, "N is the governed agent_policy_conflicts.precedent_count"
    assert d["learning"] == {"min_decisions": 30, "min_days": 30, "min_approvers": 3,
                             "wilson_lower": 0.85, "median_seconds_floor": 30,
                             "not_yet_more": 30, "dismiss_more": 30}


def test_settings_merge_keeps_defaults_for_missing_keys():
    merged = settings.merge({"response_time": "PT2H", "learning": {"min_decisions": 40}})
    assert merged["response_time"] == "PT2H"
    assert merged["learning"]["min_decisions"] == 40
    assert merged["learning"]["min_days"] == 30


def _snap():
    return registry.snapshot_from_rows([
        {"kind": "checkpoint", "name": "tool.call.before", "checkpoint": None, "plain": "before a tool runs", "status": "live"},
        {"kind": "checkpoint", "name": "message.send.before", "checkpoint": None, "plain": "before a message is sent", "status": "planned"},
        {"kind": "action", "name": "supplier_ranking", "checkpoint": "tool.call.before", "plain": "rank suppliers", "status": "live"},
        {"kind": "input", "name": "tool.name", "checkpoint": "tool.call.before", "plain": "tool name", "value_type": "string", "source": "action", "status": "live"},
        {"kind": "input", "name": "agg.refunds_30d", "checkpoint": "tool.call.before", "plain": "refunds in 30 days", "value_type": "number", "source": "total:refunds_30d", "status": "planned"},
    ])


def test_snapshot_answers_availability():
    s = _snap()
    assert s.checkpoint_live("tool.call.before")
    assert not s.checkpoint_live("message.send.before")
    assert s.knows_action("tool.call.before", "supplier_ranking")
    assert not s.knows_action("tool.call.before", "refund.issue")
    assert s.available("tool.call.before", "tool.name")
    assert not s.available("tool.call.before", "agg.refunds_30d")   # named, not supplied
    assert s.input_row("tool.call.before", "agg.refunds_30d")["status"] == "planned"
    assert not s.available("tool.call.before", "customer.country")  # not named at all


def test_seed_matches_the_tools_the_agent_loop_really_offers():
    """Guard: the registry must name exactly the tools build_tools() hands AgentNick."""
    from agents.auto_registry import AutoRegistry
    reg = AutoRegistry.from_json()
    real = {s["function"]["name"] for s in reg.tool_schemas()} | set(seed_registry.FIXED_TOOLS)
    seeded = {r["name"] for r in seed_registry.registry_rows()
              if r["kind"] == "action" and r["checkpoint"] == "tool.call.before"}
    assert seeded == real


def test_seed_registers_only_action_inputs_as_live():
    rows = seed_registry.registry_rows()
    live_inputs = [r for r in rows if r["kind"] == "input" and r["status"] == "live"]
    assert live_inputs and all(r["source"] == "action" for r in live_inputs)
    cps = {r["name"]: r["status"] for r in rows if r["kind"] == "checkpoint"}
    assert cps == {"tool.call.before": "live", "message.send.before": "planned",
                   "data.egress.before": "planned", "record.write.before": "planned"}


def test_fixed_tools_match_what_build_tools_really_builds():
    """FIXED_TOOLS is hand-written, so check it against the live tool builders, names AND arguments.
    (The check above reads FIXED_TOOLS on both sides, so it cannot catch a wrong entry here.)"""
    from orchestration.agentnick_control import build_tools
    built = {t.name: sorted(((t.parameters or {}).get("properties") or {}).keys())
             for t in build_tools(object())}
    assert built == {k: sorted(v) for k, v in seed_registry.FIXED_TOOLS.items()}
