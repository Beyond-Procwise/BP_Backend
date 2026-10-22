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


DORMANT = "dormant"


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


def status_of(folder: Path) -> str:
    """A skill's status from its config: "active" unless config.yaml says ``status: dormant``. A dormant skill
    stays on disk and under the checker, but is not listed, not routed and not usable until the line is
    removed; it is how a drafted or unreviewed playbook waits without reaching the agent."""
    cfg = folder / "config.yaml"
    try:
        data = yaml.safe_load(cfg.read_text()) if cfg.is_file() else {}
    except yaml.YAMLError:
        return "active"
    return DORMANT if str((data or {}).get("status", "")).strip().lower() == DORMANT else "active"


def list_skills(*, include_dormant: bool = False) -> List[SkillSummary]:
    out = []
    for p in _folders():
        if not include_dormant and status_of(p) == DORMANT:
            continue
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


def split_parts(body: str, limit: int = 3500) -> List[str]:
    """Cut a playbook into pieces of at most ``limit`` characters that rejoin to exactly ``body``.

    Cuts fall on a section heading where one fits, else a paragraph break, else anywhere.
    A tool result longer than the runtime's cap is truncated, and a truncated playbook is
    one the model follows only halfway.
    """
    parts: List[str] = []
    rest = body
    while len(rest) > limit:
        window = rest[:limit]
        cut = window.rfind("\n## ")
        if cut <= 0:
            cut = window.rfind("\n\n")
        if cut <= 0:
            cut = limit
        else:
            cut += 1  # keep the newline that precedes the heading with the earlier part
        parts.append(rest[:cut])
        rest = rest[cut:]
    parts.append(rest)
    return parts
