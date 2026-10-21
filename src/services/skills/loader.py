"""Read, validate and serve skill folders. Nothing here runs a skill: a skill is advice."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml

from . import checks

_REPO = Path(__file__).resolve().parents[3]


class SkillNotFound(LookupError):
    pass


class SkillInvalid(ValueError):
    pass


@dataclass
class SkillSummary:
    name: str
    description: str


@dataclass
class Skill:
    name: str
    description: str
    body: str
    config: Dict[str, Any]
    references: Dict[str, str] = field(default_factory=dict)
    eval_files: List[str] = field(default_factory=list)


def skills_dir() -> Path:
    return Path(os.getenv("SKILLS_DIR") or _REPO / "skills")


def _folders() -> List[Path]:
    root = skills_dir()
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "SKILL.md").is_file()) if root.is_dir() else []


def list_skills() -> List[SkillSummary]:
    out = []
    for p in _folders():
        meta, _ = checks.split_frontmatter((p / "SKILL.md").read_text())
        out.append(SkillSummary(name=p.name, description=str((meta or {}).get("description", ""))))
    return out


def load_skill(name: str) -> Skill:
    folder = next((p for p in _folders() if p.name == name), None)
    if folder is None:
        raise SkillNotFound(name)
    problems = checks.check_structure(folder)
    if problems:
        raise SkillInvalid(f"{name}: " + "; ".join(problems))
    meta, body = checks.split_frontmatter((folder / "SKILL.md").read_text())
    refs = {f.name: f.read_text() for f in sorted((folder / "references").glob("*")) if f.is_file()} \
        if (folder / "references").is_dir() else {}
    return Skill(
        name=folder.name, description=str(meta["description"]), body=body,
        config=yaml.safe_load((folder / "config.yaml").read_text()), references=refs,
        eval_files=sorted(f.name for f in (folder / "evals").glob("*.yaml")),
    )
