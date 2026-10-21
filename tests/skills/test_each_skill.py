"""Every skill folder in the repo must pass every rule. Adding a skill adds a test."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.orchestration import agentnick_control as control
from src.services.skills import checks

ROOT = Path(__file__).resolve().parents[2] / "skills"
FOLDERS = sorted(p for p in ROOT.iterdir() if p.is_dir()) if ROOT.is_dir() else []


def _known_tools():
    # The saving-lever tools only exist when the Opportunity Miner is registered, so build the
    # set with a stand-in that has one. Handlers are never called here, only the names read.
    nick = SimpleNamespace(agents={"opportunity_miner": SimpleNamespace()})
    return {t.name for t in control.build_tools(nick)}


def _known_calculations():
    from src.services.calculation_tools import _formulas
    return set(_formulas().REGISTRY)


@pytest.mark.parametrize("folder", FOLDERS, ids=lambda p: p.name)
def test_skill_passes_every_rule(folder):
    assert checks.check_skill(folder, known_tools=_known_tools(), known_calculations=_known_calculations()) == []
