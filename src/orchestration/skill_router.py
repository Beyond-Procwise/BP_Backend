"""The skill router: which skill, if any, covers this request. Deterministic, no model.

Each skill lists, in its own settings (``config.yaml`` under ``routing``), the terms that mark a
request as its own: strong terms worth 3 points and supporting terms worth 1. A trailing ``*`` matches
any ending. The same request always scores the same way, and a skill's owner can read why it fired.
This is how ``intent_router`` already decides, with the same shape: a minimum score and a margin.

It prefers silence. A skill is chosen only when its score reaches ``MIN_POINTS`` and beats the
runner-up by ``MARGIN``. Wrongly injecting a skill is worse than not injecting one: with no match the
model can still read a skill itself through ``list_skills`` and ``use_skill``.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml

STRONG, SUPPORT = 3, 1
MIN_POINTS = 3
MARGIN = 2


@dataclass(frozen=True)
class Entry:
    name: str
    strong: List[str] = field(default_factory=list)
    support: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class Match:
    skill: Optional[str]
    points: int = 0
    runner_up: Optional[str] = None
    margin: int = 0
    matched: Tuple[str, ...] = ()


NO_MATCH = Match(skill=None)


def tokens(text: object) -> List[str]:
    return re.findall(r"[a-z0-9]+", str(text or "").lower())


def _find(rule: str, toks: Sequence[str], used: set) -> bool:
    """True if ``rule`` (words, a trailing * on a word = any ending) occurs in ``toks`` on unused tokens."""
    words = [w for w in re.split(r"[\s\-]+", rule.lower()) if w]
    n = len(words)
    for i in range(len(toks) - n + 1):
        if any((i + j) in used for j in range(n)):
            continue
        if all((toks[i + j].startswith(w[:-1]) if w.endswith("*") else toks[i + j] == w) for j, w in enumerate(words)):
            used.update(range(i, i + n))
            return True
    return False


def load_entries(skills_dir: Optional[Path] = None) -> List[Entry]:
    root = Path(skills_dir or os.getenv("SKILLS_DIR") or Path(__file__).resolve().parents[2] / "skills")
    out: List[Entry] = []
    if not root.is_dir():
        return out
    for folder in sorted(p for p in root.iterdir() if p.is_dir() and (p / "SKILL.md").is_file()):
        cfg_path = folder / "config.yaml"
        try:
            cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.is_file() else {}
        except yaml.YAMLError:
            cfg = {}
        routing = (cfg or {}).get("routing") or {}
        strong = [str(x) for x in routing.get("strong") or []]
        if str((cfg or {}).get("status", "")).strip().lower() == "dormant":
            continue  # on disk and checked, never routed (services.skills.loader.status_of)
        if strong:
            out.append(Entry(folder.name, strong, [str(x) for x in routing.get("support") or []]))
    return out


class Router:
    def __init__(self, entries: Sequence[Entry]):
        self.entries = list(entries)

    def scores(self, request: object) -> Dict[str, Tuple[int, Tuple[str, ...]]]:
        toks = tokens(request)
        out: Dict[str, Tuple[int, Tuple[str, ...]]] = {}
        for e in self.entries:
            used: set = set()
            points, hit = 0, []
            for rules, weight in ((e.strong, STRONG), (e.support, SUPPORT)):
                for rule in rules:
                    if _find(rule, toks, used):
                        points += weight
                        hit.append(rule)
            out[e.name] = (points, tuple(hit))
        return out

    def route(self, request: object) -> Optional[Match]:
        if not tokens(request):
            return None
        ranked = sorted(self.scores(request).items(), key=lambda kv: (-kv[1][0], kv[0]))
        if not ranked or ranked[0][1][0] < MIN_POINTS:
            return None
        runner = ranked[1] if len(ranked) > 1 else (None, (0, ()))
        margin = ranked[0][1][0] - runner[1][0]
        if margin < MARGIN:
            return None
        return Match(ranked[0][0], ranked[0][1][0], runner[0], margin, ranked[0][1][1])
