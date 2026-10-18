"""Measure how the cross-encoder scores questions the store can and cannot answer (calibrates ask_grade.THRESHOLDS).

    CUDA_VISIBLE_DEVICES= python -m scripts.ask_testsheet.grade_calibration

Runs each question through RAGService.search on the local Qdrant (CPU only) and prints the best cross-encoder
score among the documents it actually scored. ``ANSWERABLE`` questions have a matching document in the store;
``ABSENT`` ones do not (a topic the corpus never mentions, or a clause that is not in the contract).
"""
from __future__ import annotations

import statistics
import sys
import warnings

ANSWERABLE = [
    "What does the competitive bidding policy require?",
    "What is our policy on changing a supplier's bank details?",
    "What does the delegation of authority say about capital expenditure?",
    "What is the onboarding process for a new supplier?",
    "Explain the three-way match process",
    "What is the preferred suppliers list?",
    "How is supplier performance measured?",
    "What conflict of interest rules apply to buyers?",
    "When does the contract with Ironbridge Managed IT Ltd expire?",
    "What are the payment terms in our contract with Meridia Cloud Platforms?",
    "Show the invoice from Ironbridge Managed IT Ltd",
    "What quote did Meridia Cloud Platforms give us?",
]
ABSENT = [
    "Does our contract with Ironbridge Managed IT Ltd have a force majeure clause?",
    "What is the recipe for chocolate cake?",
    "Who won the football world cup in 1998?",
    "What are the carbon emissions of our offshore wind farm?",
    "Which supplier supplies helium balloons for the Mars colony?",
    "What is the capital of Australia?",
    "How do I configure a Kubernetes ingress controller?",
    "What does our contract say about arbitration in Singapore?",
]


def main() -> int:
    warnings.filterwarnings("ignore")
    sys.path[:0] = [".", "src"]
    from scripts.ask_testsheet.rerank_ab import build
    svc = build(False)
    rows = []
    for label, group in (("answerable", ANSWERABLE), ("absent", ABSENT)):
        for q in group:
            hits = svc.search(q, top_k=6)
            scored = [h.rerank_score for h in hits if getattr(h, "reranked", False)]
            best = max(scored) if scored else None
            rows.append((label, best, len(hits), q))
            print(f"{label:10} best={best if best is None else round(best, 2)!s:>7}  docs={len(hits)}  {q[:70]}", flush=True)
    for label in ("answerable", "absent"):
        vals = [r[1] for r in rows if r[0] == label and r[1] is not None]
        if vals:
            print(f"{label}: min {min(vals):.2f} median {statistics.median(vals):.2f} max {max(vals):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
