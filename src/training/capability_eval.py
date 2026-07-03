"""AgentNick capability evaluation harness.

Where ``eval_gate.py`` measures *extraction* accuracy (the #1 product rule), this
module measures whether the reasoning model (``BeyondProcwise/AgentNick:unified``)
actually understands procurement end-to-end and can drive the product by spawning
the right sub-agents. It is the scorecard that gates the system-prompt uplift.

Three sections, each scored deterministically (keyword / structural match) so the
result is reproducible and is NOT circular against the pipeline's own output:

  - ``concepts``      — procurement domain knowledge (3-way match, incoterms, tax
                        direction, payment terms, deal lifecycle, maverick spend).
  - ``product``       — knowledge of THIS product (raw -> _stg -> _trgt pipeline,
                        deal_id linking, the agent roster, the dashboards).
  - ``orchestration`` — given a goal, emit a valid workflow plan naming the correct
                        agents, in the exact JSON shape the ReasoningEngine parses.

Nothing here writes to the DB or mutates a model; it only measures.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

log = logging.getLogger(__name__)

SECTIONS = ("concepts", "product", "orchestration")

DEFAULT_EVAL_PATH = "src/data/training/capability_eval.jsonl"

# Canonical agent IDs the planner is allowed to use (from AutoRegistry). Kept here
# as a static fallback so scoring does not require importing the live registry,
# but callers may pass a fresh set via ``valid_ids``.
CANONICAL_AGENT_IDS = {
    "approvals", "data_extraction", "discrepancy_detection", "email_dispatch",
    "email_drafting", "email_watcher", "negotiation", "opportunity_miner",
    "quote_comparison", "quote_evaluation", "rag", "requirements",
    "supplier_interaction", "supplier_ranking",
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass
class CapabilityItem:
    id: str
    section: str
    prompt: str
    scoring: dict


def load_items(path: str = DEFAULT_EVAL_PATH) -> list[CapabilityItem]:
    out: list[CapabilityItem] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        r = json.loads(line)
        out.append(CapabilityItem(
            id=str(r["id"]), section=str(r["section"]),
            prompt=str(r["prompt"]), scoring=dict(r.get("scoring") or {})))
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower())


def score_concept(text: str, include_groups: Iterable[Iterable[str]],
                  forbid: Optional[Iterable[str]] = None) -> float:
    """Fraction of synonym-groups present, zeroed-by-fraction for forbidden terms.

    ``include_groups`` is a list of groups; a group is satisfied if ANY of its
    synonyms appears. ``forbid`` terms (wrong/hallucinated phrasings) reduce the
    score proportionally — if every forbidden term appears the score is 0.
    """
    t = _norm(text)
    groups = [list(g) for g in include_groups]
    if not groups:
        base = 1.0
    else:
        matched = sum(1 for g in groups if any(_norm(s) in t for s in g))
        base = matched / len(groups)
    # A forbidden term is a disqualifying factual error (e.g. "tax is larger than
    # the subtotal"). Its presence hard-fails the item regardless of keyword hits.
    for f in (forbid or []):
        if _norm(f) in t:
            return 0.0
    return max(0.0, min(1.0, base))


def _extract_json_obj(text: str) -> Optional[dict]:
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _plan_agents(obj: Optional[dict]) -> list[str]:
    if not isinstance(obj, dict):
        return []
    steps = obj.get("steps")
    if not isinstance(steps, list):
        return []
    agents: list[str] = []
    for s in steps:
        if isinstance(s, dict) and isinstance(s.get("agent"), str):
            agents.append(s["agent"].strip())
    return agents


def score_orchestration(text: str, expected: set, forbidden: Optional[set] = None,
                        valid_ids: Optional[set] = None) -> dict:
    """Score a workflow-plan answer.

    score = recall(expected agents) * clean_fraction, where clean_fraction
    penalises hallucinated (not in ``valid_ids``) or ``forbidden`` agents.
    Invalid JSON or no steps -> 0.
    """
    forbidden = forbidden or set()
    valid_ids = valid_ids or CANONICAL_AGENT_IDS
    obj = _extract_json_obj(text)
    predicted = _plan_agents(obj)
    if obj is None or not predicted:
        return {"score": 0.0, "valid_json": obj is not None, "predicted": predicted,
                "hallucinated": [], "forbidden_hit": []}

    pred_set = set(predicted)
    recall = len(pred_set & expected) / len(expected) if expected else 1.0
    hallucinated = sorted(a for a in pred_set if a not in valid_ids)
    forbidden_hit = sorted(a for a in pred_set if a in forbidden)
    bad = set(hallucinated) | set(forbidden_hit)
    clean_fraction = 1.0 - len(bad) / len(pred_set)
    score = max(0.0, min(1.0, recall * clean_fraction))
    return {"score": score, "valid_json": True, "predicted": predicted,
            "hallucinated": hallucinated, "forbidden_hit": forbidden_hit}


def score_item(item: CapabilityItem, text: str, valid_ids: Optional[set] = None) -> float:
    if item.section == "orchestration":
        r = score_orchestration(
            text,
            expected=set(item.scoring.get("expected", [])),
            forbidden=set(item.scoring.get("forbidden", [])),
            valid_ids=valid_ids)
        return r["score"]
    return score_concept(text, item.scoring.get("include", []),
                         forbid=item.scoring.get("forbid"))


# ---------------------------------------------------------------------------
# Model backend
# ---------------------------------------------------------------------------
def ollama_generate_fn(model: str, timeout: int = 240) -> Callable[[str], str]:
    from src.services.ollama_client import ollama_generate

    def _gen(prompt: str) -> str:
        out = ollama_generate(prompt, model=model, temperature=0.0,
                              num_predict=1024, timeout=timeout, retries=1)
        return out or ""
    return _gen


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@dataclass
class CapabilityReport:
    model: str = ""
    n: int = 0
    overall: float = 0.0
    by_section: dict = field(default_factory=dict)
    per_item: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"model": self.model, "n": self.n, "overall": round(self.overall, 4),
                "by_section": {k: round(v, 4) for k, v in self.by_section.items()},
                "per_item": self.per_item}


def evaluate(generate_fn: Callable[[str], str], items: list[CapabilityItem],
             model_label: str = "", valid_ids: Optional[set] = None) -> CapabilityReport:
    rep = CapabilityReport(model=model_label, n=len(items))
    section_scores: dict[str, list[float]] = {s: [] for s in SECTIONS}
    all_scores: list[float] = []
    for it in items:
        ans = generate_fn(it.prompt)
        sc = score_item(it, ans, valid_ids=valid_ids)
        all_scores.append(sc)
        section_scores.setdefault(it.section, []).append(sc)
        rep.per_item.append({"id": it.id, "section": it.section,
                             "score": round(sc, 4), "answer_preview": _norm(ans)[:240]})
    rep.overall = sum(all_scores) / len(all_scores) if all_scores else 0.0
    rep.by_section = {s: (sum(v) / len(v) if v else 0.0)
                      for s, v in section_scores.items() if v}
    return rep


def main(argv: Optional[list] = None) -> int:  # pragma: no cover - CLI glue
    import argparse
    ap = argparse.ArgumentParser(description="Score an AgentNick model on the capability scorecard.")
    ap.add_argument("--model", default="BeyondProcwise/AgentNick:unified")
    ap.add_argument("--eval-path", default=DEFAULT_EVAL_PATH)
    ap.add_argument("--out", default="")
    ap.add_argument("--section", default="", help="optional: only this section")
    args = ap.parse_args(argv)

    items = load_items(args.eval_path)
    if args.section:
        items = [i for i in items if i.section == args.section]
    gen = ollama_generate_fn(args.model)
    rep = evaluate(gen, items, model_label=args.model)
    payload = rep.to_dict()
    print(json.dumps(payload, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
