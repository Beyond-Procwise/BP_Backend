"""Phase 2b — measure the DATA path. Does AgentNick convey our real numbers correctly?

`product_eval` measures whether AgentNick understands how the PRODUCT behaves. It says nothing
about the other half of the ask bar: questions about the corpus itself — "what is my spend",
"how many issues are open", "who are my suppliers". Those are answered from `corpus_facts`,
which reads the live `_trgt` tables and hands the model exact, counted figures. Nothing in
this repo ever checked whether the model then relays those figures faithfully or quietly
mangles them — states 800 findings when there are 832, sums nine currencies into one £ total,
or invents a supplier. A grounded system that garbles the numbers it was grounded in is worse
than useless, because it looks authoritative.

So this measures exactly that, on the same principle as `product_eval`: fact coverage, no LLM
judge. The gold is not written down — it is COMPUTED from the same `corpus_facts` fetch the
model is given, at run time, against the live database. It therefore cannot go stale, and it
is by construction the truth the model was handed. The only thing being measured is fidelity:
of the exact figures we put in front of it, how many does the answer state correctly, and does
it invent a figure we did not give it.

Two conditions on the same questions:

  * ``bare``    — AgentNick with no facts. What it says about our spend from the weights alone
                  (which can only be invention — it has never seen this database).
  * ``grounded``— AgentNick handed the `corpus_facts` for the question, exactly as the live ask
                  bar hands them. This is the condition that has to be right.

The gap is the whole point, and it is the mirror image of the product-understanding one: there
the graph had to beat the weights; here the counted facts have to.

Run:  PYTHONPATH=.:src .venv/bin/python -m src.training.data_eval
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]


# The system prompt mirrors the live ask persona: answer ONLY from the facts provided, never
# invent a figure, and never expose the machinery. It is deliberately close to the "Joshi"
# persona in model_selector so the measurement reflects the real surface, not a lab construct.
_SYSTEM = """You are the ProcWise spend assistant. A real buyer is asking about their own
procurement data.

You will be given AUTHORITATIVE FACTS read directly from their records. Answer ONLY from those
facts. State the figures exactly as given — do not round, re-total, or estimate. If several
currencies are present, keep them separate; never add different currencies into one total. If
the facts do not answer the question, say so plainly rather than guess. A confident wrong
number about someone's spend is worse than "I don't have that".

Never describe how the system works internally — no databases, tables, files or code."""


def _norm(text: str) -> str:
    """Lowercase, collapse whitespace, straighten quotes, and normalise number formatting.

    The model writes "17" or "seventeen" or "17 suppliers"; it also writes "1,120" for a value
    stored as "1120.00". A figure is the same figure however it is punctuated, so the score
    must not turn on a comma or a trailing ".00"."""
    s = (text or "").lower()
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = s.replace(",", "")
    # "37550.00" -> "37550" so the stored numeric and the spoken number compare equal.
    s = re.sub(r"(\d+)\.0+(\D|$)", r"\1\2", s)
    return re.sub(r"\s+", " ", s)


def _contains(hay: str, value: str) -> bool:
    """Is `value` present in `hay`, both already normalised?

    A bare integer must match on a digit boundary: gold "17" is NOT covered by "2017" or
    "170". Without this, short counts score as covered inside unrelated numbers and every
    figure in the corpus looks conveyed when it is not. Non-numeric gold (a supplier name)
    matches as a substring, which is what we want for a name that may be quoted mid-sentence.
    """
    v = _norm(value)
    if not v:
        return False
    if v.replace(".", "").isdigit():
        return re.search(rf"(?<!\d){re.escape(v)}(?!\d)", hay) is not None
    return v in hay


@dataclass(frozen=True)
class DataQuestion:
    """A question, the intent that fetches its facts, and how to read the gold from them."""

    id: str
    question: str
    intent: str
    # (label -> function mapping the fetched facts dict to the exact string that must appear).
    # Computed from the live fetch, so the gold is the truth the model was handed.
    gold: Dict[str, Callable[[Dict[str, Any]], Optional[str]]] = field(default_factory=dict)


def _first(rows: List[Dict[str, Any]], key: str) -> Optional[str]:
    if rows and key in rows[0] and rows[0][key] is not None:
        return str(rows[0][key])
    return None


QUESTIONS: List[DataQuestion] = [
    DataQuestion(
        id="suppliers_count",
        question="How many suppliers do we actually have invoices from, and how many are on record in total?",
        intent="suppliers",
        gold={
            "invoiced_from": lambda f: _first(f.get("totals", []), "suppliers_we_have_invoices_from"),
            "on_record": lambda f: _first(f.get("totals", []), "suppliers_on_record"),
        },
    ),
    DataQuestion(
        id="findings_open",
        question="How many issues or discrepancies are currently open across our documents, and how many of those are critical?",
        intent="findings",
        gold={
            "open_total": lambda f: _first(f.get("totals", []), "open_findings_total"),
            "critical": lambda f: _first(f.get("totals", []), "critical_total"),
            "docs_affected": lambda f: _first(f.get("totals", []), "documents_affected"),
        },
    ),
    DataQuestion(
        id="findings_top_type",
        question="What is the single most common type of open issue on our documents, and how many are there of it?",
        intent="findings",
        gold={
            # The by-type list is ordered by count desc, so row 0 is the most common — a fact
            # the model is handed and must not reorder.
            "top_count": lambda f: _first(f.get("open_findings_by_type", []), "findings"),
        },
    ),
    DataQuestion(
        id="spend_top_supplier",
        question="Which supplier have we invoiced the most from by amount, and how much?",
        intent="spend",
        gold={
            "supplier": lambda f: _first(f.get("invoiced_spend_by_supplier", []), "supplier_name"),
            "amount": lambda f: _first(f.get("invoiced_spend_by_supplier", []), "invoiced_amount"),
        },
    ),
    DataQuestion(
        id="invoices_count",
        question="How many invoices do we have in total, and how many of them cite a purchase order?",
        intent="invoices",
        gold={
            "total": lambda f: _first(f.get("totals", []), "invoices_total"),
            "citing_po": lambda f: _first(f.get("totals", []), "citing_a_po"),
        },
    ),
]


@dataclass
class QResult:
    id: str
    answer: str
    covered: List[str] = field(default_factory=list)
    missed: List[str] = field(default_factory=list)
    coverage: float = 0.0
    leaked: bool = False
    empty: bool = False
    gold: Dict[str, str] = field(default_factory=dict)


@dataclass
class Report:
    condition: str
    n: int = 0
    coverage: float = 0.0
    leak_rate: float = 0.0
    empty_rate: float = 0.0
    results: List[QResult] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.condition:<10} coverage={self.coverage:.3f}  "
            f"leak={self.leak_rate:.3f}  empty={self.empty_rate:.3f}  n={self.n}"
        )


def _score(q: DataQuestion, answer: str, gold: Dict[str, str]) -> QResult:
    from services import output_safety as osafe

    res = QResult(id=q.id, answer=answer or "", gold=gold)
    if not (answer or "").strip():
        res.empty = True
        res.missed = list(gold)
        return res

    if osafe.inspect(answer, prose=True):
        # Leaking internals is a failed answer under the same rule product_eval uses.
        res.leaked = True
        res.missed = list(gold)
        return res

    hay = _norm(answer)
    for label, value in gold.items():
        if value is None:
            continue
        if _contains(hay, value):
            res.covered.append(label)
        else:
            res.missed.append(label)

    checkable = [v for v in gold.values() if v is not None]
    # An all-None gold is a HARNESS failure (the facts were not fetched), not a perfect score.
    # The first version scored it 1.0 and hid a wrong-intent fetch behind a green number.
    res.coverage = len(res.covered) / len(checkable) if checkable else float("nan")
    return res


def _fetch_for(nick, q: DataQuestion) -> Dict[str, Any]:
    """The corpus_facts for this question, fetched by its PINNED intent.

    We pin the intent rather than trust `detect_intent(question)` because this eval measures
    one thing — does the model faithfully convey the facts it is handed — and pinning keeps
    the gold and the grounded context on the same, correct fact set. `detect_intent`'s routing
    is a real concern (it mis-routes "how many suppliers do we have invoices from" to the
    invoices intent, because "invoices" is scanned first), but that is a SEPARATE failure and
    conflating it here would let a routing bug masquerade as a fidelity score. `run()` records
    the routing mismatch separately so it is visible without polluting this metric.
    """
    from services import corpus_facts

    facts: Dict[str, Any] = {}
    try:
        with nick.get_db_connection() as conn:
            with conn.cursor() as cur:
                facts = corpus_facts._fetch(cur, q.intent) or {}
    except Exception:
        logger_exc("corpus_facts fetch failed for intent %s", q.intent)
    graph = corpus_facts.fetch_graph(q.question)
    if graph:
        facts["knowledge_graph"] = graph
    return facts


def logger_exc(msg: str, *args) -> None:
    log.exception(msg, *args)


def run(answer_fn: Callable[[str, Dict[str, Any]], str], condition: str, nick) -> Report:
    import math

    from services import corpus_facts

    rep = Report(condition=condition, n=len(QUESTIONS))
    for q in QUESTIONS:
        facts = _fetch_for(nick, q)
        gold = {label: fn(facts) for label, fn in q.gold.items()}
        if all(v is None for v in gold.values()):
            # Gold could not be read from the pinned intent — the harness is broken for this
            # question (renamed column, empty table). Surface it; do not score around it.
            raise SystemExit(
                f"data_eval: no gold for {q.id!r} from intent {q.intent!r} — facts fetched: "
                f"{list(facts)}. Fix the question/intent before trusting any number here."
            )
        # Routing note: would the LIVE ask bar (which uses detect_intent) even reach these
        # facts? Recorded, not scored — a mis-route is a real bug, just not a fidelity one.
        routed = corpus_facts.detect_intent(q.question)
        if routed != q.intent:
            print(f"  [routing] {q.id}: detect_intent -> {routed!r}, expected {q.intent!r}")
        try:
            answer = answer_fn(q.question, facts if condition == "grounded" else {})
        except Exception as exc:  # noqa: BLE001
            log.warning("%s: %s failed: %s", condition, q.id, exc)
            answer = ""
        res = _score(q, answer, gold)
        rep.results.append(res)
        cov = "nan" if math.isnan(res.coverage) else f"{res.coverage:.2f}"
        print(
            f"  [{condition}] {q.id:<22} cov={cov}"
            f"{'  LEAKED' if res.leaked else ''}{'  EMPTY' if res.empty else ''}"
            f"  missed={res.missed}"
        )

    scored = [r.coverage for r in rep.results if not math.isnan(r.coverage)]
    rep.coverage = sum(scored) / len(scored) if scored else 0.0
    rep.leak_rate = sum(1 for r in rep.results if r.leaked) / rep.n if rep.n else 0.0
    rep.empty_rate = sum(1 for r in rep.results if r.empty) / rep.n if rep.n else 0.0
    return rep


def bare_agentnick(nick) -> Callable[[str, Dict[str, Any]], str]:
    """No facts. The model on its own — which for this database can only be invention."""
    from services.tool_runtime import run_tools

    def _ask(question: str, _facts: Dict[str, Any]) -> str:
        return run_tools(question, [], _SYSTEM, max_rounds=2).answer

    return _ask


def grounded_agentnick(nick) -> Callable[[str, Dict[str, Any]], str]:
    """The model handed the corpus facts, exactly as the live ask bar hands them."""
    from services import corpus_facts
    from services.tool_runtime import run_tools

    def _ask(question: str, facts: Dict[str, Any]) -> str:
        rendered = corpus_facts.render_facts(facts) if facts else ""
        user = question
        if rendered:
            user = f"{question}\n\nAUTHORITATIVE FACTS from our records:\n{rendered}"
        return run_tools(user, [], _SYSTEM, max_rounds=2).answer

    return _ask


def _nick():
    """A minimal DB-capable stand-in — corpus_facts only needs get_db_connection."""
    import psycopg2

    class _N:
        def get_db_connection(self):
            return psycopg2.connect(
                host=os.environ["DB_HOST"], dbname=os.environ["DB_NAME"],
                user=os.environ["DB_USER"], password=os.environ["DB_PASSWORD"],
                port=os.environ.get("DB_PORT", "5432"), connect_timeout=8,
            )

    return _N()


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    from dotenv import load_dotenv

    load_dotenv(str(_REPO_ROOT / ".env"))

    nick = _nick()

    # Refuse to run if the data path is dark — otherwise `grounded` would score the model
    # answering from an empty fact set, which is just `bare` wearing a different label.
    from services import corpus_facts

    probe = corpus_facts.fetch_facts(nick, "what is my total spend")
    if not probe:
        raise SystemExit(
            "corpus_facts returned nothing for a spend probe — the _trgt tables are empty or "
            "unreachable. Refusing to run, because `grounded` would measure a dark data path."
        )
    print(f"\ndata path reachable (spend probe returned {len(probe)} fact groups)")

    print("\n=== Phase 2b baseline: data fidelity ===\n")
    reports = [
        run(bare_agentnick(nick), "bare", nick),
        run(grounded_agentnick(nick), "grounded", nick),
    ]

    print("\n--- baseline ---")
    for r in reports:
        print("  " + r.summary())

    bare, grounded = reports[0], reports[1]
    print(f"\n  what the corpus facts are worth: {grounded.coverage - bare.coverage:+.3f} coverage")

    out = {
        r.condition: {
            "coverage": round(r.coverage, 4),
            "leak_rate": round(r.leak_rate, 4),
            "empty_rate": round(r.empty_rate, 4),
            "per_question": [
                {
                    "id": q.id,
                    "coverage": round(q.coverage, 4),
                    "leaked": q.leaked,
                    "empty": q.empty,
                    "missed": q.missed,
                    "gold": q.gold,
                    "answer": q.answer,
                }
                for q in r.results
            ],
        }
        for r in reports
    }
    path = os.getenv("DATA_EVAL_OUT", "artifacts/data_fidelity_baseline.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\n  written: {path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
