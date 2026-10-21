import pytest

from src.orchestration import agentnick_control as control


@pytest.fixture
def tools(make_skill, monkeypatch):
    make_skill()
    monkeypatch.setenv("SKILLS_DIR", str(make_skill.root))
    return {t.name: t for t in control._skill_tools()}


def test_both_tools_are_in_the_agents_tool_set(make_skill, monkeypatch):
    monkeypatch.setenv("SKILLS_DIR", str(make_skill.root))
    names = [t.name for t in control.build_tools(None)]
    assert "list_skills" in names and "use_skill" in names


def test_list_skills_gives_slug_and_description(tools):
    out = tools["list_skills"].handler()
    assert out["skills"] == [{"name": "demo-skill", "description": "Does a demo job. Use when demoing."}]


def test_use_skill_returns_playbook_and_config(tools):
    out = tools["use_skill"].handler(name="demo-skill")
    assert out["found"] is True and "## Purpose" in out["playbook"] and out["config"] == {"thresholds": {}}


@pytest.mark.parametrize("name", ["nope", "../etc/passwd", "demo-skill/../x", "Demo", "", None])
def test_unknown_or_path_like_names_are_refused(tools, name):
    out = tools["use_skill"].handler(name=name)
    assert out["found"] is False and out["available"] == ["demo-skill"]


def test_a_broken_skill_is_reported_not_served(make_skill, monkeypatch):
    make_skill(name="bad-skill", config=False)
    monkeypatch.setenv("SKILLS_DIR", str(make_skill.root))
    out = {t.name: t for t in control._skill_tools()}["use_skill"].handler(name="bad-skill")
    assert out["found"] is False and "config.yaml" in out["note"]


def test_description_says_a_skill_is_advice_only(tools):
    d = tools["use_skill"].description.lower()
    assert "advice" in d and "approval" in d
