"""The context builder: the system prompt, assembled in labelled layers, and nothing else changes."""
from types import SimpleNamespace

import pytest

from orchestration import context_builder as cb

BASE = "You are AgentNick."
SKILL = SimpleNamespace(
    name="demo-skill",
    body="## Purpose\n\nDo the thing with `tool:check_policy` then `tool:run_calculation`. Try `tool:build_docx` for the memo.\n\nNOT AVAILABLE: build_docx. Give text.\n",
    config={"thresholds": {"limit": 5}, "unavailable_tools": ["build_docx"], "routing": {"strong": ["x"], "support": []}})


def test_with_no_skill_and_no_extra_the_prompt_is_exactly_the_base():
    ctx = cb.build(base=BASE)
    assert ctx.system == BASE and [l.name for l in ctx.layers] == ["base"] and ctx.skill is None


def test_extra_instructions_follow_the_base_exactly_as_they_always_did():
    assert cb.build(base=BASE, extra="Also be brief.").system == f"{BASE}\n\nAlso be brief."


def test_layers_come_in_a_fixed_order_and_are_labelled():
    ctx = cb.build(base=BASE, extra="E", skill=SKILL, memory="M")
    assert [l.name for l in ctx.layers] == ["base", "extra", "skill", "memory"]
    assert ctx.system.index(BASE) < ctx.system.index("E") < ctx.system.index("demo-skill") < ctx.system.rindex("M")


def test_the_skill_layer_carries_the_playbook_and_its_settings_but_not_the_routers_terms():
    layer = next(l for l in cb.build(base=BASE, skill=SKILL).layers if l.name == "skill")
    assert SKILL.body.strip() in layer.text and '"limit": 5' in layer.text
    assert "routing" not in layer.text and "strong" not in layer.text


def test_the_skill_layer_tells_the_model_not_to_spend_rounds_reading_it_again():
    text = cb.build(base=BASE, skill=SKILL).system
    assert "do not call list_skills or use_skill" in text.lower()


def test_what_the_skill_needs_is_read_from_its_playbook_minus_what_it_says_is_unavailable():
    assert cb.tools_needed(SKILL.body) == {"check_policy", "run_calculation"}


def test_a_skill_whose_tools_are_not_offered_is_not_injected_and_the_reason_names_them():
    ok, why = cb.can_inject(SKILL, offered={"run_calculation"})
    assert ok is False and "check_policy" in why
    assert cb.can_inject(SKILL, offered={"run_calculation", "check_policy", "list_skills"}) == (True, "")


def test_a_skill_that_would_overflow_the_budget_is_skipped_never_cut_short():
    big = SimpleNamespace(name="big", body="x" * 5000, config={})
    ctx = cb.build(base=BASE, skill=big, max_chars=2000)
    assert ctx.skill is None and ctx.skipped and "too long" in ctx.skipped and ctx.system == BASE


def test_the_record_of_a_context_says_what_each_layer_is_and_how_big():
    ctx = cb.build(base=BASE, skill=SKILL)
    rec = ctx.describe()
    assert rec[0] == {"layer": "base", "chars": len(BASE), "source": "orchestrator"}
    assert any(r["layer"] == "skill" and r["source"] == "skill:demo-skill" for r in rec)


def test_building_a_context_changes_nothing_it_was_given():
    import copy
    before = copy.deepcopy(SKILL.config)
    cb.build(base=BASE, skill=SKILL)
    assert SKILL.config == before


def test_the_context_builder_imports_no_model_wire_data_store_or_policy_engine():
    import ast, re
    tree = ast.parse(open(cb.__file__, encoding="utf-8").read())
    mods = {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | \
           {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not [m for m in mods if re.search(r"tool_runtime|agentnick_driver|egress|ollama|agent_policy|db\b|psycopg", m)]


def test_today_is_stated_only_when_given_and_sits_right_after_the_base():
    import datetime as dt
    from orchestration import context_builder as cb
    ctx = cb.build(base="BASE", extra="EXTRA", today=dt.date(2026, 10, 22))
    assert [l.name for l in ctx.layers] == ["base", "today", "extra"]
    assert ctx.layers[1].text.startswith("Today is Thursday 2026-10-22.") and "never guess the year" in ctx.layers[1].text
    assert ctx.system == "BASE\n\n" + ctx.layers[1].text + "\n\nEXTRA"
    assert [l.name for l in cb.build(base="BASE").layers] == ["base"]
