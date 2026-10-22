"""Render skill folders from compact definitions, so every skill has the same checked structure.

    venv/bin/python -m scripts.skills.scaffold specs/skills/*.yaml [--force]

Each definition file holds one or more skills. For each, four files are written under skills/<name>/:
SKILL.md (the eleven sections in the checker's order), config.yaml (thresholds with comments, unavailable
tools, routing), evals/scenarios.yaml (every kind, expert flags, context lines on the non-typical ones)
and evals/triggers.yaml. Existing folders are left alone unless --force is given. The output is checked
by the same rules as a hand-written skill (tests/skills/test_each_skill.py); the scaffold only removes
the chance of a missing section or a threshold nobody references.

Definition keys (all strings unless said): name, description (frontmatter, <=1024 chars), title, purpose,
use_when, not_for, inputs (list), tools (list of "tool_name: what for"), unavailable (list of
"tool_name: what the agent does instead"), workflow (list of steps), heuristics (list), policy_hooks,
outputs (list), quality (list), escalation, example, anti_patterns (list), context (one sentence),
thresholds (list of {key, value, comment}), routing {strong: [...], support: [...]},
scenarios (list of {kind, prompt, expect, expert?}), triggers {should_trigger: [...], should_not_trigger: [...]} (never yes/no, which YAML reads as booleans).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SKILLS = ROOT / "skills"
KIND_ORDER = ["typical", "edge", "missing_data", "conflicting_data", "pressure_to_skip_policy"]
KIND_PREFIX = {"typical": "t", "edge": "e", "missing_data": "m", "conflicting_data": "c", "pressure_to_skip_policy": "p"}


def _q(text: str) -> str:
    """A YAML double-quoted scalar."""
    return '"' + str(text).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _bullets(items) -> str:
    return "\n".join(f"- {i}" for i in items)


def _numbered(items) -> str:
    return "\n".join(f"{n}. {i}" for n, i in enumerate(items, 1))


def render_skill_md(d: dict) -> str:
    tool_lines = "\n".join(f"- Use `tool:{t.split(':', 1)[0].strip()}` {t.split(':', 1)[1].strip()}" for t in d.get("tools", []))
    unavailable = "\n".join(f"NOT AVAILABLE: {u.split(':', 1)[0].strip()}. {u.split(':', 1)[1].strip()}" for u in d.get("unavailable", []))
    keys = [t["key"] for t in d.get("thresholds", [])]
    settings = ("Settings this skill reads from its own configuration, named by key and never restated as numbers: "
                + ", ".join(f"`config.thresholds.{k}`" for k in keys) + ".") if keys else ""
    return f"""---
name: {d['name']}
description: {_q(d['description'])}
---

# {d['title']}

## Purpose

{d['purpose']}

## When to use and when not

{d['use_when']}

{d['not_for']}

## Inputs and preconditions

{_bullets(d.get('inputs', []))}

{tool_lines}
{unavailable}

{settings}

## Workflow

{_numbered(d.get('workflow', []))}

## Frameworks and heuristics

{_bullets(d.get('heuristics', []))}

## Policy hooks (SC-007)

{d['policy_hooks']}

## Output specification

{_numbered(d.get('outputs', []))}

## Quality bar

{_bullets(d.get('quality', []))}

## Escalation and approval

{d['escalation']}

## Worked example

{d['example']}

## Anti-patterns

{_bullets(d.get('anti_patterns', []))}
"""


def render_config(d: dict) -> str:
    lines = ["# Configurable defaults for this skill. They are starting points for review, not governed policy.", "thresholds:"]
    for t in d.get("thresholds", []):
        lines.append(f"  {t['key']}: {t['value']}    # {t.get('comment', '')}".rstrip())
    unavailable = [u.split(":", 1)[0].strip() for u in d.get("unavailable", [])]
    if unavailable:
        lines.append("unavailable_tools:")
        lines += [f"  - {u}" for u in unavailable]
    lines.append("routing:                         # how the orchestrator decides this skill covers a request (see orchestration/skill_router.py)")
    lines.append("  strong:                        # 3 points each; a trailing * matches any ending")
    lines += [f"    - {_q(s)}" for s in d["routing"]["strong"]]
    lines.append("  support:                       # 1 point each")
    lines += [f"    - {_q(s)}" for s in d["routing"].get("support", [])]
    return "\n".join(lines) + "\n"


def render_scenarios(d: dict) -> str:
    counters = {k: 0 for k in KIND_ORDER}
    out = []
    for s in d["scenarios"]:
        kind = s["kind"]
        counters[kind] += 1
        sid = f"{KIND_PREFIX[kind]}{counters[kind]}"
        expert = bool(s.get("expert"))
        out.append(f"- id: {sid}")
        out.append(f"  kind: {kind}")
        if kind != "typical" and d.get("context"):
            out.append(f"  context: {_q(d['context'])}")
        out.append(f"  prompt: {_q(s['prompt'])}")
        out.append(f"  expect: {_q(s['expect'])}")
        out.append(f"  expert_graded: {'true' if expert else 'false'}")
    return "\n".join(out) + "\n"


def render_triggers(d: dict) -> str:
    return ("should_trigger:\n" + "\n".join(f"- {_q(p)}" for p in d["triggers"]["should_trigger"])
            + "\nshould_not_trigger:\n" + "\n".join(f"- {_q(p)}" for p in d["triggers"]["should_not_trigger"]) + "\n")


def validate(d: dict) -> list:
    problems = []
    for key in ("name", "description", "title", "purpose", "use_when", "not_for", "policy_hooks", "escalation", "example", "routing", "scenarios", "triggers"):
        if key not in d:
            problems.append(f"{d.get('name', '?')}: missing {key}")
    kinds = {s["kind"] for s in d.get("scenarios", [])}
    for k in KIND_ORDER:
        if k not in kinds:
            problems.append(f"{d.get('name', '?')}: no {k} scenario")
    if len(d.get("scenarios", [])) < 15:
        problems.append(f"{d.get('name', '?')}: {len(d.get('scenarios', []))} scenarios, 15 needed")
    if sum(1 for s in d.get("scenarios", []) if s.get("expert")) < 3:
        problems.append(f"{d.get('name', '?')}: fewer than 3 scenarios marked expert: true")
    for side in ("should_trigger", "should_not_trigger"):
        if len(d.get("triggers", {}).get(side, [])) < 10:
            problems.append(f"{d.get('name', '?')}: triggers.{side} needs 10")
    if len(d.get("routing", {}).get("strong", [])) < 3:
        problems.append(f"{d.get('name', '?')}: routing.strong needs 3")
    return problems


def write_skill(d: dict, force: bool = False) -> Path:
    folder = SKILLS / d["name"]
    if folder.exists() and not force:
        raise FileExistsError(f"{folder} exists; pass --force to overwrite")
    (folder / "evals").mkdir(parents=True, exist_ok=True)
    (folder / "SKILL.md").write_text(render_skill_md(d))
    (folder / "config.yaml").write_text(render_config(d))
    (folder / "evals" / "scenarios.yaml").write_text(render_scenarios(d))
    (folder / "evals" / "triggers.yaml").write_text(render_triggers(d))
    return folder


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    bad = 0
    for spec in args.specs:
        docs = yaml.safe_load(Path(spec).read_text())
        for d in (docs if isinstance(docs, list) else [docs]):
            problems = validate(d)
            if problems:
                bad += 1
                print("\n".join(problems))
                continue
            print("wrote", write_skill(d, force=args.force))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
