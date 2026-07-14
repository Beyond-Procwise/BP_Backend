"""Phase 2a — measure. What does AgentNick actually understand about the product today?

This is the baseline. Anything proposed in 2b has to beat the number this prints, or it does
not get built.

It deliberately measures TWO conditions on the same questions:

  * ``bare``  — AgentNick with no retrieval at all. What the weights alone know.
  * ``kg``    — AgentNick with ``describe_platform``, the platform knowledge graph, exactly
                as the live support agent has it.

The gap between them is what the knowledge layer is actually worth right now. That number is
the whole point: it is the difference between "we should invest in the graph" and "the graph
is already carrying this and the problem is elsewhere", and nobody in this repo has ever
measured it.

Three metrics per condition:

  * ``coverage``  — mean fraction of the required facts an answer conveys. The headline.
  * ``leak_rate`` — fraction of answers containing internal detail. Under the output-safety
                    rule an answer that leaks is a failed answer, so a leaked answer scores
                    ZERO coverage as well as counting here. Knowing is not saying.
  * ``empty_rate``— fraction where the model produced nothing usable (timeout, refusal,
                    blank). Kept separate from "wrong" because the fixes are different.

Run:  .venv/bin/python -m src.training.product_eval
"""

from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

log = logging.getLogger(__name__)


_SYSTEM = """You are the ProcWise product assistant. A real user is asking you how the
product works.

Answer in terms of what the USER sees and does: the screen they are on, the button they
press, the status they will see, what it means for their document or their deal.

Never describe how the system works internally — no databases, tables, files, code,
endpoints, environment variables, pipelines, stages, triggers, queues, workers or models.

If you do not know, say so plainly. Do not guess: a confident wrong answer about someone's
invoice is worse than "I don't know", because they will act on it."""


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower())


@dataclass
class QuestionResult:
    id: str
    category: str
    answer: str
    covered: List[str] = field(default_factory=list)
    missed: List[str] = field(default_factory=list)
    coverage: float = 0.0
    leaked: bool = False
    leak_kinds: List[str] = field(default_factory=list)
    empty: bool = False


@dataclass
class Report:
    condition: str
    n: int = 0
    coverage: float = 0.0
    leak_rate: float = 0.0
    empty_rate: float = 0.0
    by_category: Dict[str, float] = field(default_factory=dict)
    results: List[QuestionResult] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.condition:<10} coverage={self.coverage:.3f}  "
            f"leak={self.leak_rate:.3f}  empty={self.empty_rate:.3f}  n={self.n}"
        )


def score_answer(question, answer: str) -> QuestionResult:
    from services import output_safety as osafe
    from src.training.product_questions import Question  # noqa: F401  (typing only)

    res = QuestionResult(id=question.id, category=question.category, answer=answer or "")

    if not (answer or "").strip():
        res.empty = True
        return res

    violations = osafe.inspect(answer, prose=True)
    if violations:
        # An answer that leaks is not a good answer that happens to leak. Under the rule, it
        # is a failed answer. Scoring it on content anyway would let the model buy accuracy
        # with disclosure, which is exactly the trade we are refusing to make.
        res.leaked = True
        res.leak_kinds = sorted({v.kind for v in violations})
        res.coverage = 0.0
        res.missed = [f.name for f in question.facts]
        return res

    hay = _norm(answer)
    for fact in question.facts:
        if any(_norm(a) in hay for a in fact.aliases):
            res.covered.append(fact.name)
        else:
            res.missed.append(fact.name)

    res.coverage = len(res.covered) / len(question.facts) if question.facts else 1.0
    return res


def run(answer_fn: Callable[[str], str], condition: str) -> Report:
    from src.training.product_questions import QUESTIONS

    rep = Report(condition=condition, n=len(QUESTIONS))
    cat_scores: Dict[str, List[float]] = {}

    for q in QUESTIONS:
        try:
            answer = answer_fn(q.question)
        except Exception as exc:  # noqa: BLE001
            log.warning("%s: %s failed: %s", condition, q.id, exc)
            answer = ""
        res = score_answer(q, answer)
        rep.results.append(res)
        cat_scores.setdefault(q.category, []).append(res.coverage)
        print(
            f"  [{condition}] {q.id:<28} cov={res.coverage:.2f}"
            f"{'  LEAKED:' + ','.join(res.leak_kinds) if res.leaked else ''}"
            f"{'  EMPTY' if res.empty else ''}"
        )

    rep.coverage = sum(r.coverage for r in rep.results) / rep.n if rep.n else 0.0
    rep.leak_rate = sum(1 for r in rep.results if r.leaked) / rep.n if rep.n else 0.0
    rep.empty_rate = sum(1 for r in rep.results if r.empty) / rep.n if rep.n else 0.0
    rep.by_category = {
        c: sum(v) / len(v) for c, v in sorted(cat_scores.items())
    }
    return rep


# --------------------------------------------------------------------------------------
# The two conditions.
# --------------------------------------------------------------------------------------


def bare_agentnick() -> Callable[[str], str]:
    """No retrieval. Just the model."""
    from services.tool_runtime import run_tools

    def _ask(question: str) -> str:
        return run_tools(question, [], _SYSTEM, max_rounds=2).answer

    return _ask


def kg_agentnick() -> Callable[[str], str]:
    """The model plus the platform knowledge graph — what the live support agent has."""
    from services.support_agent import _platform_tool
    from services.tool_runtime import run_tools

    def _ask(question: str) -> str:
        return run_tools(
            question,
            _platform_tool(),
            _SYSTEM,
            max_rounds=4,
            require_tool_use=True,
            nudge=(
                "You answered without looking anything up. Call describe_platform to find "
                "out how this part of ProcWise actually behaves, then answer from that."
            ),
        ).answer

    return _ask


def main() -> int:
    logging.basicConfig(level=logging.WARNING)

    print("\n=== Phase 2a baseline: product understanding ===\n")
    reports = [
        run(bare_agentnick(), "bare"),
        run(kg_agentnick(), "kg"),
    ]

    print("\n--- baseline ---")
    for r in reports:
        print("  " + r.summary())
    print("\n  per-category coverage:")
    cats = sorted({c for r in reports for c in r.by_category})
    header = "    " + f"{'category':<10}" + "".join(f"{r.condition:>9}" for r in reports)
    print(header)
    for c in cats:
        row = f"    {c:<10}" + "".join(f"{r.by_category.get(c, 0.0):>9.2f}" for r in reports)
        print(row)

    bare, kg = reports[0], reports[1]
    delta = kg.coverage - bare.coverage
    print(f"\n  what the knowledge graph is worth today: {delta:+.3f} coverage")

    out = {
        r.condition: {
            "coverage": round(r.coverage, 4),
            "leak_rate": round(r.leak_rate, 4),
            "empty_rate": round(r.empty_rate, 4),
            "by_category": {k: round(v, 4) for k, v in r.by_category.items()},
            "per_question": [
                {
                    "id": q.id,
                    "coverage": round(q.coverage, 4),
                    "leaked": q.leaked,
                    "empty": q.empty,
                    "missed": q.missed,
                    "answer": q.answer[:400],
                }
                for q in r.results
            ],
        }
        for r in reports
    }
    path = "artifacts/product_understanding_baseline.json"
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n  written: {path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
