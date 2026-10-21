"""Every skill folder in the repo must pass every rule. Adding a skill adds a test."""
from pathlib import Path

import pytest

from src.orchestration import agentnick_control as control
from src.services.skills import checks

ROOT = Path(__file__).resolve().parents[2] / "skills"
FOLDERS = sorted(p for p in ROOT.iterdir() if p.is_dir()) if ROOT.is_dir() else []


def _known_tools():
    return {t.name for t in control.build_tools(None)}


@pytest.mark.parametrize("folder", FOLDERS, ids=lambda p: p.name)
def test_skill_passes_every_rule(folder):
    assert checks.check_skill(folder, known_tools=_known_tools()) == []
