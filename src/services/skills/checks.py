"""Structure rules for a skill folder (SES v1, section C)."""
from __future__ import annotations

import re
from pathlib import Path
from typing import List

import yaml

REQUIRED_SECTIONS = [
    "Purpose", "When to use and when not", "Inputs and preconditions", "Workflow",
    "Frameworks and heuristics", "Policy hooks (SC-007)", "Output specification",
    "Quality bar", "Escalation and approval", "Worked example", "Anti-patterns",
]
MAX_DESCRIPTION = 1024
MAX_LINES = 500
_FRONT = re.compile(r"\A---\n(.*?)\n---\n", re.S)


def split_frontmatter(text: str):
    m = _FRONT.match(text)
    if not m:
        return None, text
    meta = yaml.safe_load(m.group(1))
    return (meta if isinstance(meta, dict) else None), text[m.end():]


def check_structure(path: Path) -> List[str]:
    path = Path(path)
    problems: List[str] = []
    md = path / "SKILL.md"
    if not md.is_file():
        return ["SKILL.md is missing"]
    text = md.read_text()
    meta, body = split_frontmatter(text)
    if meta is None:
        problems.append("SKILL.md has no YAML frontmatter")
    else:
        if not meta.get("name"):
            problems.append("frontmatter: name is missing")
        desc = str(meta.get("description") or "")
        if not desc:
            problems.append("frontmatter: description is missing")
        elif len(desc) > MAX_DESCRIPTION:
            problems.append(f"frontmatter: description is {len(desc)} characters (max {MAX_DESCRIPTION})")
    if len(text.splitlines()) > MAX_LINES:
        problems.append(f"SKILL.md is over {MAX_LINES} lines; move long material to references/")
    headings = re.findall(r"^## (.+?)\s*$", body, re.M)
    missing = [s for s in REQUIRED_SECTIONS if s not in headings]
    for s in missing:
        problems.append(f"section missing: {s}")
    present = [h for h in headings if h in REQUIRED_SECTIONS]
    if not missing and present != REQUIRED_SECTIONS:
        problems.append("sections are out of order; required order: " + " · ".join(REQUIRED_SECTIONS))
    cfg = path / "config.yaml"
    if not cfg.is_file():
        problems.append("config.yaml is missing")
    else:
        try:
            if not isinstance(yaml.safe_load(cfg.read_text()), dict):
                problems.append("config.yaml must be a mapping")
        except yaml.YAMLError as exc:
            problems.append(f"config.yaml does not parse: {exc}")
    for f in ("scenarios.yaml", "triggers.yaml"):
        if not (path / "evals" / f).is_file():
            problems.append(f"evals/{f} is missing")
    return problems
