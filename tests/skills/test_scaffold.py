"""The scaffold renders a definition into a skill folder that passes every checker rule."""
from pathlib import Path

import pytest
import yaml

from scripts.skills import scaffold
from src.services.skills import checks

KINDS = ["typical", "edge", "missing_data", "conflicting_data", "pressure_to_skip_policy"]
KNOWN = {"get_policy", "run_calculation"}


def definition(**over):
    d = {
        "name": "demo-skill", "description": "Does a demo job. Use when demoing.", "title": "Demo",
        "purpose": "Explain the demo.", "use_when": "Use when asked to demo.", "not_for": "Not for real work.",
        "inputs": ["The thing to demo."],
        "tools": ["get_policy: for the rule", "run_calculation: for the sums"],
        "unavailable": ["send_message: the text is drafted; a person sends it"],
        "workflow": ["Read.", "Think.", "Answer."], "heuristics": ["Keep it short."],
        "policy_hooks": "Spend is checked with `tool:get_policy`.", "outputs": ["The answer."],
        "quality": ["No invented figures."], "escalation": "Escalate to the lead.",
        "example": "Asked to demo, it demos.", "anti_patterns": ["Guessing."],
        "context": "We are demoing a skill. It's a test.",
        "thresholds": [{"key": "max_items", "value": 25, "comment": "items shown"}],
        "routing": {"strong": ["demo*", "show me the demo", "demonstrate"], "support": ["example"]},
        "scenarios": [{"kind": KINDS[i % 5], "prompt": f'Prompt {i} with a "quote" and an apostrophe\'s',
                       "expect": f"Expect {i}", "expert": i < 3} for i in range(15)],
        "triggers": {"yes": [f"Demo {i}" for i in range(10)], "no": [f"Other {i}" for i in range(10)]},
    }
    d.update(over)
    return d


@pytest.fixture
def skills_root(tmp_path, monkeypatch):
    monkeypatch.setattr(scaffold, "SKILLS", tmp_path)
    return tmp_path


def test_a_rendered_skill_passes_every_checker_rule(skills_root):
    folder = scaffold.write_skill(definition())
    assert checks.check_skill(folder, known_tools=KNOWN) == []


def test_the_threshold_key_is_named_in_the_prose_so_the_checker_rule_holds(skills_root):
    folder = scaffold.write_skill(definition())
    assert "`config.thresholds.max_items`" in (folder / "SKILL.md").read_text()
    assert yaml.safe_load((folder / "config.yaml").read_text())["thresholds"] == {"max_items": 25}


def test_only_non_typical_scenarios_carry_the_context_line(skills_root):
    folder = scaffold.write_skill(definition())
    rows = yaml.safe_load((folder / "evals" / "scenarios.yaml").read_text())
    assert all(("context" in r) == (r["kind"] != "typical") for r in rows)
    assert [r["id"] for r in rows[:5]] == ["t1", "e1", "m1", "c1", "p1"]
    assert sum(1 for r in rows if r["expert_graded"]) == 3


def test_quotes_and_apostrophes_in_prompts_survive_the_yaml_round_trip(skills_root):
    folder = scaffold.write_skill(definition())
    rows = yaml.safe_load((folder / "evals" / "scenarios.yaml").read_text())
    assert rows[0]["prompt"] == 'Prompt 0 with a "quote" and an apostrophe\'s'
    triggers = yaml.safe_load((folder / "evals" / "triggers.yaml").read_text())
    assert len(triggers["should_trigger"]) == len(triggers["should_not_trigger"]) == 10


def test_an_unavailable_tool_gets_both_its_config_entry_and_its_not_available_line(skills_root):
    folder = scaffold.write_skill(definition())
    assert yaml.safe_load((folder / "config.yaml").read_text())["unavailable_tools"] == ["send_message"]
    assert "NOT AVAILABLE: send_message. the text is drafted" in (folder / "SKILL.md").read_text()


@pytest.mark.parametrize("change, needle", [
    ({"scenarios": definition()["scenarios"][:14]}, "15 needed"),
    ({"scenarios": [s for s in definition()["scenarios"] if s["kind"] != "edge"]}, "no edge scenario"),
    ({"scenarios": [dict(s, expert=False) for s in definition()["scenarios"]]}, "fewer than 3"),
    ({"triggers": {"yes": ["one"], "no": definition()["triggers"]["no"]}}, "triggers.yes needs 10"),
    ({"routing": {"strong": ["a"]}}, "routing.strong needs 3"),
])
def test_validate_names_each_gap(change, needle):
    problems = scaffold.validate(definition(**change))
    assert any(needle in p for p in problems), problems


def test_validate_names_a_missing_required_field():
    d = definition(); del d["escalation"]
    assert any("missing escalation" in p for p in scaffold.validate(d))


def test_an_existing_folder_is_not_overwritten_without_force(skills_root):
    scaffold.write_skill(definition())
    with pytest.raises(FileExistsError):
        scaffold.write_skill(definition())
    scaffold.write_skill(definition(purpose="Changed."), force=True)
    assert "Changed." in (skills_root / "demo-skill" / "SKILL.md").read_text()


def test_main_reports_a_bad_definition_and_still_writes_the_good_ones(skills_root, tmp_path, capsys):
    spec = tmp_path / "defs.yaml"
    bad = definition(name="bad-skill"); del bad["example"]
    spec.write_text(yaml.safe_dump([definition(name="good-skill"), bad]))
    assert scaffold.main([str(spec)]) == 1
    out = capsys.readouterr().out
    assert "bad-skill: missing example" in out and (skills_root / "good-skill" / "SKILL.md").exists()
    assert not (skills_root / "bad-skill").exists()
