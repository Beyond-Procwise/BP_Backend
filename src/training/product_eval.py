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
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]


_SYSTEM = """You are the ProcWise product assistant. A real user is asking you how the
product works.

Answer in terms of what the USER sees and does: the screen they are on, the button they
press, the status they will see, what it means for their document or their deal.

Never describe how the system works internally — no databases, tables, files, code,
endpoints, environment variables, pipelines, stages, triggers, queues, workers or models.

If you do not know, say so plainly. Do not guess: a confident wrong answer about someone's
invoice is worse than "I don't know", because they will act on it."""


def _norm(text: str) -> str:
    """Lowercase, collapse whitespace, and straighten quotes.

    The curly apostrophe cost a correct answer a zero. The model wrote "the system doesn't
    guess or estimate totals" — exactly the fact — and the alias said "does not guess", so it
    scored 0.00 and looked like a regression. A metric that punishes a right answer for its
    punctuation is measuring the wrong thing.
    """
    s = (text or "").lower()
    s = s.replace("’", "'").replace("‘", "'")
    s = s.replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s)


# --------------------------------------------------------------------------------------
# Fabrication.
#
# Coverage alone cannot tell a grounded answer from a confident invention, and that is not a
# theoretical worry — it is how the bare model scored 0.439. It told users to look for a
# "Pending Review" status, a "Needs Approval" status, a "discrepancy alert", a "warning icon".
# None of them exist. The real statuses are Running / Extracting / Extracted /
# Extraction_Failed. It sent people hunting for buttons that were never built, fluently, and
# scored well for it.
#
# So a control the answer POINTS THE USER AT — quoted or emboldened, the way you name a thing
# on screen — must exist in the product. The product's vocabulary is the ontology's own
# user-facing wording; nothing else counts as evidence that a button is real.
# --------------------------------------------------------------------------------------

_UI_CLAIM = re.compile(r"[\"'“”']([A-Z][\w /-]{2,40})[\"'“”']|\*\*([A-Z][\w /-]{2,40})\*\*")

_VOCAB: str = ""


def _product_vocabulary() -> str:
    """Everything the product actually calls things, from the ontology's `say` register."""
    global _VOCAB
    if _VOCAB:
        return _VOCAB

    import yaml

    from services.platform_kg import ONTOLOGY_PATH

    doc = yaml.safe_load(open(ONTOLOGY_PATH)) or {}
    parts: List[str] = []
    for group in ("processes", "agents", "screens", "known_gaps", "models"):
        for node in doc.get(group) or []:
            parts += [str(node.get(k) or "") for k in ("name", "say", "how_to", "shows")]
            for st in node.get("stages") or []:
                parts += [str(st.get(k) or "") for k in ("name", "say")]
    _VOCAB = _norm(" ".join(parts))
    return _VOCAB


def fabricated_ui(answer: str) -> List[str]:
    """Screens, statuses or controls the answer names that the product does not have."""
    vocab = _product_vocabulary()
    invented: List[str] = []
    for m in _UI_CLAIM.finditer(answer or ""):
        term = (m.group(1) or m.group(2) or "").strip()
        if len(term) < 3:
            continue
        if _norm(term) not in vocab:
            invented.append(term)
    return sorted(set(invented))


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
    invented: List[str] = field(default_factory=list)


@dataclass
class Report:
    condition: str
    n: int = 0
    coverage: float = 0.0
    leak_rate: float = 0.0
    empty_rate: float = 0.0
    fabrication_rate: float = 0.0
    by_category: Dict[str, float] = field(default_factory=dict)
    results: List[QuestionResult] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.condition:<10} coverage={self.coverage:.3f}  "
            f"fabricated={self.fabrication_rate:.3f}  "
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

    # Reported alongside coverage, not folded into it — so the before/after coverage numbers
    # stay directly comparable to the baseline that was measured without this check.
    res.invented = fabricated_ui(answer)

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
            f"{'  INVENTED: ' + ', '.join(res.invented) if res.invented else ''}"
        )

    rep.coverage = sum(r.coverage for r in rep.results) / rep.n if rep.n else 0.0
    rep.leak_rate = sum(1 for r in rep.results if r.leaked) / rep.n if rep.n else 0.0
    rep.empty_rate = sum(1 for r in rep.results if r.empty) / rep.n if rep.n else 0.0
    rep.fabrication_rate = (
        sum(1 for r in rep.results if r.invented) / rep.n if rep.n else 0.0
    )
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

    # Without this the graph is simply unreachable, and the `kg` condition silently measures
    # a dead database connection instead of the knowledge graph. It did exactly that, and the
    # failure was invisible: the tool answered "the platform description is unavailable", the
    # model dutifully relayed "I can't access the internal details of the system", and that
    # scored as if the model had nothing to say. Three runs produced byte-identical `kg`
    # answers while `bare` moved around — a deterministic result from a stochastic model is
    # not a finding, it is a bug, and it should have been the first thing I chased.
    from dotenv import load_dotenv

    load_dotenv(str(_REPO_ROOT / ".env"))

    from services.platform_kg import describe

    try:
        probe = describe("upload")
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"the knowledge graph is unreachable ({exc}) — refusing to run, because the `kg` "
            "condition would score a broken connection and call it a measurement"
        )
    if not probe:
        raise SystemExit(
            "the knowledge graph returned nothing for 'upload' — it is empty or unsynced. "
            "Run `python -m services.platform_kg` first."
        )
    print(f"\nknowledge graph reachable ({len(probe)} facts for a probe query)")

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
            "fabrication_rate": round(r.fabrication_rate, 4),
            "by_category": {k: round(v, 4) for k, v in r.by_category.items()},
            "per_question": [
                {
                    "id": q.id,
                    "coverage": round(q.coverage, 4),
                    "leaked": q.leaked,
                    "empty": q.empty,
                    "invented": q.invented,
                    "missed": q.missed,
                    # The WHOLE answer. It used to be truncated to 400 chars, which quietly
                    # made the artifact un-rescorable: re-grading it offline read only the
                    # opening of each answer, scored the facts stated later as missing, and
                    # produced a confident set of numbers that were simply wrong. If a run is
                    # worth keeping, it is worth keeping in full.
                    "answer": q.answer,
                }
                for q in r.results
            ],
        }
        for r in reports
    }
    path = os.getenv("PRODUCT_EVAL_OUT", "artifacts/product_understanding_baseline.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n  written: {path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
