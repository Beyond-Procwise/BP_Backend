import pytest

from src.services.skills import checks, loader
from tests.skills.conftest import SECTIONS, skill_text


def test_valid_skill_has_no_problems(make_skill):
    assert checks.check_structure(make_skill()) == []


def test_load_skill_returns_playbook_and_config(make_skill, monkeypatch):
    make_skill(config={"thresholds": {"x": 5}})
    monkeypatch.setenv("SKILLS_DIR", str(make_skill.root))
    s = loader.load_skill("demo-skill")
    assert s.name == "demo-skill" and "## Purpose" in s.body and s.config["thresholds"]["x"] == 5


def test_list_skills_gives_slug_and_description(make_skill, monkeypatch):
    make_skill()
    monkeypatch.setenv("SKILLS_DIR", str(make_skill.root))
    assert [(s.name, s.description) for s in loader.list_skills()] == [
        ("demo-skill", "Does a demo job. Use when demoing.")]


def test_unknown_skill_raises(make_skill, monkeypatch):
    monkeypatch.setenv("SKILLS_DIR", str(make_skill.root))
    with pytest.raises(loader.SkillNotFound):
        loader.load_skill("nope")


def test_invalid_skill_refuses_to_load(make_skill, monkeypatch):
    make_skill(config=False)
    monkeypatch.setenv("SKILLS_DIR", str(make_skill.root))
    with pytest.raises(loader.SkillInvalid):
        loader.load_skill("demo-skill")


@pytest.mark.parametrize("kw,needle", [
    (dict(text=skill_text(description="x" * 1025)), "description"),
    (dict(text=skill_text(sections=[s for s in SECTIONS if s != "Workflow"])), "Workflow"),
    (dict(text=skill_text(sections=SECTIONS[1::-1] + SECTIONS[2:])), "order"),
    (dict(text=skill_text(extra="\n" * 600)), "lines"),
    (dict(config=False), "config.yaml"),
    (dict(scenarios=False), "scenarios.yaml"),
    (dict(triggers=False), "triggers.yaml"),
])
def test_each_structural_break_is_named(make_skill, kw, needle):
    problems = checks.check_structure(make_skill(**kw))
    assert any(needle in p for p in problems), problems


def test_config_that_is_not_a_mapping_is_a_problem(make_skill):
    d = make_skill()
    (d / "config.yaml").write_text("- a\n- b\n")
    assert any("config.yaml" in p for p in checks.check_structure(d))
