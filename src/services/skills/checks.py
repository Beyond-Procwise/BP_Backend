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
    try:
        meta = yaml.safe_load(m.group(1))
    except yaml.YAMLError:
        return None, text[m.end():]
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
        problems.append("SKILL.md has no valid YAML frontmatter (a colon followed by a space inside the description breaks it)")
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


SCENARIO_KINDS = ["typical", "edge", "missing_data", "conflicting_data", "pressure_to_skip_policy"]
MIN_SCENARIOS, MIN_EXPERT, MIN_TRIGGERS = 15, 3, 10
# A skill advises; it never sends, signs, awards or commits. These are named in prose at most.
DENIED_TOOL = re.compile(r"^(commit_.*|send_message|send_esg_survey|e_signature|run_email_dispatch|"
                         r"run_e_auction|book_shipment|onboard_supplier)$")
_TOOL_REF = re.compile(r"`tool:([a-z_0-9]+)`")
_NOT_AVAILABLE = re.compile(r"NOT AVAILABLE: ([a-z_0-9]+)")


def _load_yaml(path: Path):
    try:
        return yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError):
        return None


def _tool_problems(text: str, config: dict, known: set) -> List[str]:
    out: List[str] = []
    unavailable = set(config.get("unavailable_tools") or [])
    declared_missing = set(_NOT_AVAILABLE.findall(text))
    for tool in sorted(set(_TOOL_REF.findall(text))):
        if DENIED_TOOL.match(tool):
            out.append(f"advisory only: SKILL.md tells the agent to call {tool}")
        elif tool not in known and tool not in unavailable:
            out.append(f"{tool} is not a tool here; add it to unavailable_tools or remove it")
    for tool in sorted(unavailable):
        if tool in known:
            out.append(f"{tool} now exists; remove it from unavailable_tools and use it")
        elif tool not in declared_missing:
            out.append(f"add a line 'NOT AVAILABLE: {tool}. <what the agent does instead>' to SKILL.md")
    return out


def _threshold_leaves(node, prefix=""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _threshold_leaves(v, f"{prefix}{k}.")
    else:
        yield prefix.rstrip("."), node


def _config_problems(body: str, config: dict) -> List[str]:
    out: List[str] = []
    thresholds = config.get("thresholds") or {}
    prose = re.sub(r"(?ms)^## Worked example\b.*?(?=^## |\Z)", "", body)
    for dotted, value in _threshold_leaves(thresholds):
        key = dotted.split(".")[-1]
        if key not in body:
            out.append(f"threshold {dotted} is not referenced in SKILL.md")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and (abs(value) >= 10 or value != int(value)):
            if re.search(rf"(?<![\w.]){re.escape(str(value))}(?!\w|\.\d)", prose):
                out.append(f"threshold {dotted} = {value} is repeated in prose; refer to config.thresholds.{key}")
    return out


def _routing_problems(config: dict, known: set) -> List[str]:
    out: List[str] = []
    strong = (config.get("routing") or {}).get("strong")
    if not isinstance(strong, list) or len([x for x in strong if isinstance(x, str) and x.strip()]) < 3:
        out.append("config.yaml needs routing.strong with at least 3 terms so the orchestrator can tell when the "
                   "skill covers a request")
    tool = config.get("authoritative_reply_tool")
    if tool is not None and tool not in known:
        out.append(f"authoritative_reply_tool {tool!r} is not a tool here")
    return out


def _eval_problems(path: Path) -> List[str]:
    out: List[str] = []
    sc = _load_yaml(path / "evals" / "scenarios.yaml")
    sc = sc if isinstance(sc, list) else []
    if len(sc) < MIN_SCENARIOS:
        out.append(f"evals/scenarios.yaml has {len(sc)} scenarios; at least {MIN_SCENARIOS} needed")
    kinds = {s.get("kind") for s in sc if isinstance(s, dict)}
    for kind in SCENARIO_KINDS:
        if kind not in kinds:
            out.append(f"evals/scenarios.yaml has no '{kind}' scenario")
    if sum(1 for s in sc if isinstance(s, dict) and s.get("expert_graded")) < MIN_EXPERT:
        out.append(f"evals/scenarios.yaml needs at least {MIN_EXPERT} scenarios with expert_graded: true")
    tr = _load_yaml(path / "evals" / "triggers.yaml")
    tr = tr if isinstance(tr, dict) else {}
    for key in ("should_trigger", "should_not_trigger"):
        if len(tr.get(key) or []) < MIN_TRIGGERS:
            out.append(f"evals/triggers.yaml: {key} needs {MIN_TRIGGERS} prompts")
    return out


def _calculation_problems(body: str, known_calculations) -> List[str]:
    """A `family.name` that sits in a registered family must be a registered calculation."""
    families = {n.split(".", 1)[0] for n in known_calculations}
    out = []
    for name in sorted(set(re.findall(r"`([a-z_]+\.[a-z_]+)`", body))):
        if name.split(".", 1)[0] in families and name not in known_calculations:
            out.append(f"{name} is not a registered calculation")
    return out


def check_skill(path: Path, *, known_tools, known_calculations=None) -> List[str]:
    """Everything check_structure does, plus the rules that keep a skill honest."""
    path = Path(path)
    problems = check_structure(path)
    if problems:
        return problems
    _, body = split_frontmatter((path / "SKILL.md").read_text())
    config = _load_yaml(path / "config.yaml") or {}
    calc = _calculation_problems(body, set(known_calculations)) if known_calculations is not None else []
    return (_tool_problems(body, config, set(known_tools)) + _config_problems(body, config)
            + _routing_problems(config, set(known_tools)) + calc + _eval_problems(path))
