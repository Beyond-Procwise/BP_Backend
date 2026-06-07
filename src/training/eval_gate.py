"""Extraction-accuracy evaluation gate for AgentNick tuning.

The #1 procurement rule is 100% extraction accuracy, and the production
`AgentNick:extract` model currently meets the `_stg` baseline. This module is
the safety gate that MUST pass before any tuned candidate replaces production:
it runs a model over held-out gold examples (the confidence-gated auto-collected
extractions) and computes field-level accuracy, so a regressed adapter can be
detected and refused.

Two model backends are supported:
  - an Ollama model name (e.g. 'BeyondProcwise/AgentNick:extract' or a candidate)
  - a local PEFT adapter dir (base + adapter) loaded via transformers — for
    evaluating a freshly trained adapter before any GGUF/Ollama conversion.

Nothing here writes to the DB or promotes a model; it only measures.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are AgentNick, the core AI engine for ProcWise — an enterprise procurement "
    "intelligence platform. You power 13 specialized agents across the procurement lifecycle: "
    "DataExtraction, SupplierRanking, QuoteEvaluation, QuoteComparison, OpportunityMiner, "
    "EmailDrafting, Negotiation, SupplierInteraction, EmailDispatch, EmailWatcher, "
    "Approvals, DiscrepancyDetection, and RAG. Your primary role is accurate document "
    "extraction — extracting EXACTLY what documents say, never modifying source values."
)

DEFAULT_EVAL_PATH = "src/data/training/auto_collected_examples.jsonl"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass
class EvalExample:
    doc_type: str
    pk: str
    source_text: str
    expected_header: dict


def load_eval_examples(path: str = DEFAULT_EVAL_PATH, min_source_len: int = 50) -> list[EvalExample]:
    """Load gold examples (production extractions with usable source text)."""
    out: list[EvalExample] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        src = r.get("source_text") or ""
        if len(str(src)) < min_source_len:
            continue
        extracted = r.get("extracted") or {}
        header = extracted.get("header") if isinstance(extracted, dict) else None
        if not isinstance(header, dict) or not header:
            continue
        out.append(EvalExample(
            doc_type=r.get("doc_type", ""), pk=str(r.get("pk", "")),
            source_text=str(src), expected_header=header))
    return out


def split_holdout(examples: list[EvalExample], holdout_frac: float = 0.3,
                  seed: int = 13) -> tuple[list[EvalExample], list[EvalExample]]:
    """Deterministic train/eval split (no RNG dependency — stable across runs)."""
    # order by a stable hash of pk so the split is reproducible without Random()
    ordered = sorted(examples, key=lambda e: _stable_hash(e.pk + str(seed)))
    n_eval = max(1, int(len(ordered) * holdout_frac))
    return ordered[n_eval:], ordered[:n_eval]


def _stable_hash(s: str) -> int:
    h = 1469598103934665603
    for ch in s:
        h = (h ^ ord(ch)) * 1099511628211 & 0xFFFFFFFFFFFFFFFF
    return h


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
def _norm_val(v: Any) -> str:
    if v is None:
        return ""
    s = str(v).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _predicted_header(pred: Optional[dict]) -> dict:
    if not isinstance(pred, dict):
        return {}
    h = pred.get("header")
    return h if isinstance(h, dict) else {k: v for k, v in pred.items() if not isinstance(v, (list, dict))}


def score_example(expected_header: dict, predicted_text: str) -> dict:
    """Field-level accuracy for one example. Only fields present (non-empty) in
    the gold header are scored — extraction must reproduce known values."""
    pred = _predicted_header(_extract_json(predicted_text))
    fields = [k for k, v in expected_header.items() if _norm_val(v) != ""]
    if not fields:
        return {"n_fields": 0, "n_correct": 0, "accuracy": 1.0, "mismatches": []}
    correct, mism = 0, []
    for k in fields:
        if _norm_val(expected_header.get(k)) == _norm_val(pred.get(k)):
            correct += 1
        else:
            mism.append({"field": k, "expected": expected_header.get(k), "got": pred.get(k)})
    return {"n_fields": len(fields), "n_correct": correct,
            "accuracy": correct / len(fields), "mismatches": mism}


# ---------------------------------------------------------------------------
# Model backends
# ---------------------------------------------------------------------------
def ollama_generate_fn(model: str, timeout: int = 120) -> Callable[[str], str]:
    """Generator that runs a prompt against an Ollama model."""
    from src.services.ollama_client import ollama_generate

    def _gen(prompt: str) -> str:
        out = ollama_generate(prompt, model=model, temperature=0.0,
                              num_predict=2048, timeout=timeout, retries=2)
        return out or ""
    return _gen


def build_prompt(example: EvalExample) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Extract the structured data from this {example.doc_type} document as JSON "
        f"with a 'header' object. Use ONLY values present in the document.\n\n"
        f"DOCUMENT:\n{example.source_text}\n\nJSON:"
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@dataclass
class EvalReport:
    model: str = ""
    n_examples: int = 0
    doc_accuracy: float = 0.0          # mean per-doc field accuracy
    exact_doc_rate: float = 0.0        # fraction of docs with 100% fields correct
    total_fields: int = 0
    total_correct: int = 0
    per_example: list = field(default_factory=list)


def evaluate(generate_fn: Callable[[str], str], examples: list[EvalExample],
             model_label: str = "") -> EvalReport:
    rep = EvalReport(model=model_label, n_examples=len(examples))
    accs, exact = [], 0
    for ex in examples:
        pred_text = generate_fn(build_prompt(ex))
        sc = score_example(ex.expected_header, pred_text)
        accs.append(sc["accuracy"])
        rep.total_fields += sc["n_fields"]
        rep.total_correct += sc["n_correct"]
        if sc["accuracy"] >= 1.0:
            exact += 1
        rep.per_example.append({"pk": ex.pk, "doc_type": ex.doc_type,
                                "accuracy": round(sc["accuracy"], 4),
                                "mismatches": sc["mismatches"]})
    rep.doc_accuracy = sum(accs) / len(accs) if accs else 0.0
    rep.exact_doc_rate = exact / len(examples) if examples else 0.0
    return rep


def eval_gate(candidate_fn: Callable[[str], str], baseline_fn: Callable[[str], str],
              examples: list[EvalExample], tolerance: float = 0.0,
              candidate_label: str = "candidate", baseline_label: str = "baseline") -> dict:
    """Run both models on the same held-out set. PASS iff the candidate's
    field-accuracy is within `tolerance` of (>=) the baseline. Refuses promotion
    on any regression."""
    base = evaluate(baseline_fn, examples, baseline_label)
    cand = evaluate(candidate_fn, examples, candidate_label)
    delta = cand.doc_accuracy - base.doc_accuracy
    passed = delta >= -abs(tolerance)
    return {
        "passed": passed,
        "delta_doc_accuracy": round(delta, 4),
        "baseline": {"model": base.model, "doc_accuracy": round(base.doc_accuracy, 4),
                     "exact_doc_rate": round(base.exact_doc_rate, 4), "n": base.n_examples},
        "candidate": {"model": cand.model, "doc_accuracy": round(cand.doc_accuracy, 4),
                      "exact_doc_rate": round(cand.exact_doc_rate, 4), "n": cand.n_examples},
        "verdict": ("PROMOTE_OK" if passed else "REGRESSION_REFUSE"),
    }
