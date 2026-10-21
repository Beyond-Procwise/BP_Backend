"""Builds throwaway skill folders so each structural rule can be broken on its own."""
from pathlib import Path

import pytest
import yaml

SECTIONS = [
    "Purpose", "When to use and when not", "Inputs and preconditions", "Workflow",
    "Frameworks and heuristics", "Policy hooks (SC-007)", "Output specification",
    "Quality bar", "Escalation and approval", "Worked example", "Anti-patterns",
]


def skill_text(name="demo-skill", description="Does a demo job. Use when demoing.",
               sections=None, extra=""):
    sections = SECTIONS if sections is None else sections
    body = "\n\n".join(f"## {s}\n\nText for {s}." for s in sections)
    return f"---\nname: {name}\ndescription: {description}\n---\n\n# Demo\n\n{body}\n{extra}\n"


def write_skill(root: Path, name="demo-skill", text=None, config=None, scenarios=True,
                triggers=True) -> Path:
    d = root / name
    (d / "evals").mkdir(parents=True)
    (d / "SKILL.md").write_text(text if text is not None else skill_text(name=name))
    if config is not False:
        (d / "config.yaml").write_text(yaml.safe_dump(config if config is not None else {"thresholds": {}}))
    if scenarios:
        (d / "evals" / "scenarios.yaml").write_text("[]\n")
    if triggers:
        (d / "evals" / "triggers.yaml").write_text("should_trigger: []\nshould_not_trigger: []\n")
    return d


@pytest.fixture
def make_skill(tmp_path):
    def _make(**kw):
        return write_skill(tmp_path, **kw)
    _make.root = tmp_path
    return _make
