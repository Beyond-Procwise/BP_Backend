"""Compare retrieval with the current reranker against the fast one (ASK_FAST_RERANK), on this host's real corpus.

    CUDA_VISIBLE_DEVICES= python -m scripts.ask_testsheet.rerank_ab [--questions file.txt]

Builds RAGService twice against the local Qdrant (same embedder, CPU only - the GPU is full), once per setting,
asks the same questions of both, and reports per question: time, whether the top document is the same, and
how many of the six returned documents overlap. It writes nothing to the corpus (an existing payload-index
check on filtered searches is the only side effect, and the live service does the same).
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
import types
import warnings

# Worded to avoid the keywords that make RAGService._rewrite_query add a hard source_type filter
# ("purchase order", "po", "invoice", "quote", "contract"): on this corpus those filters match nothing,
# so nothing would be retrieved or reranked and the comparison would measure nothing.
QUESTIONS = [
    "Who needs to sign off spending above 50,000 pounds?",
    "What does the competitive bidding policy require?",
    "What is our policy on changing a supplier's bank details?",
    "What are the payment terms we agreed with Meridia Cloud Platforms?",
    "What does the delegation of authority say about capital expenditure?",
    "Tell me about Ironbridge Managed IT Ltd",
    "Which suppliers have terms expiring this year?",
    "What is the onboarding process for a new supplier?",
    "Explain the three-way match process",
    "What discount did Meridia Cloud Platforms offer?",
    "How do we handle a supplier who is late delivering?",
    "What is the preferred suppliers list?",
    "What are the termination terms for our IT services suppliers?",
    "How is supplier performance measured?",
    "What conflict of interest rules apply to buyers?",
    "Which approvals are needed for urgent spending?",
]


def build(fast: bool):
    os.environ["ASK_FAST_RERANK"] = "1" if fast else "0"
    from sentence_transformers import SentenceTransformer
    from qdrant_client import QdrantClient
    from config.settings import settings
    from services.rag_service import RAGService
    agent = types.SimpleNamespace(settings=settings, qdrant_client=QdrantClient(url=settings.qdrant_url),
                                  embedding_model=SentenceTransformer(settings.embedding_model, device="cpu"), device="cpu")
    return RAGService(agent)


def doc_key(hit) -> str:
    p = getattr(hit, "payload", {}) or {}
    return str(getattr(hit, "id", None) or p.get("document_id") or p.get("record_id") or "")


def main(argv=None) -> int:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions")
    args = ap.parse_args(argv)
    questions = [q.strip() for q in open(args.questions)] if args.questions else QUESTIONS
    rows = []
    services = {"current": build(False), "fast": build(True)}
    print("current:", services["current"]._reranker.__class__.__name__, "limit", services["current"]._reranker_limit,
          "| fast limit", services["fast"]._reranker_limit, "chars", services["fast"]._reranker_max_chars)
    for q in questions:
        res = {}
        for name, svc in services.items():
            t = time.perf_counter()
            hits = svc.search(q, top_k=6)
            res[name] = ([doc_key(h) for h in hits], time.perf_counter() - t)
        a, b = res["current"][0], res["fast"][0]
        if not a and not b:
            print(f"  (no candidates for either setting - skipped) {q[:60]}", flush=True)
            continue
        rows.append((q, res["current"][1], res["fast"][1], bool(a and b and a[0] == b[0]), len(set(a) & set(b)), len(a)))
        print(f"{res['current'][1]:6.1f}s -> {res['fast'][1]:5.1f}s  top1 {'same' if rows[-1][3] else 'DIFF'}  overlap {rows[-1][4]}/{rows[-1][5]}  {q[:60]}", flush=True)
    print(f"\nmedian {statistics.median(r[1] for r in rows):.1f}s -> {statistics.median(r[2] for r in rows):.1f}s; "
          f"max {max(r[1] for r in rows):.1f}s -> {max(r[2] for r in rows):.1f}s; "
          f"top-1 same {sum(r[3] for r in rows)}/{len(rows)}; "
          f"mean overlap {statistics.mean(r[4] / max(r[5], 1) for r in rows):.0%}")
    return 0


if __name__ == "__main__":
    # An operator tool acts for the platform across every company; with row-level security on, a tool
    # that names no company would otherwise see no rows (D-47).
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    from src.services.tenant_context import enter_system as _enter_system
    _enter_system()
    sys.path[:0] = ["src", "."]
    raise SystemExit(main())
