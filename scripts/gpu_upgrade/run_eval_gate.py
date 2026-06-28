#!/usr/bin/env python
"""Eval-gate CLI — the model-promotion gate for AgentNick (Workstream B).

Runs a model through the REAL context_layer.synthesize step over a deterministic
30% holdout of the gold examples and reports field-level doc_accuracy. With
--candidate, also scores the candidate and prints a PROMOTE/REFUSE verdict
(candidate may replace production only if it does NOT regress vs baseline).

LLM-only (no DB writes). Runs against whatever OLLAMA_BASE_URL points at.

Usage:
    python scripts/gpu_upgrade/run_eval_gate.py --label baseline
    python scripts/gpu_upgrade/run_eval_gate.py --label cand-b2 \
        --candidate BeyondProcwise/AgentNick:cand-b2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENV = ROOT / ".env"
if ENV.exists():
    for line in ENV.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
os.environ.setdefault("PGCONNECT_TIMEOUT", "3")

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

OUT_DIR = ROOT / "artifacts" / "gpu_upgrade"
BASELINE_MODEL = os.getenv("PROCWISE_AGENTNICK_MODEL", "BeyondProcwise/AgentNick:extract")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--baseline", default=BASELINE_MODEL)
    ap.add_argument("--candidate", default=None)
    ap.add_argument("--holdout-frac", type=float, default=0.3)
    ap.add_argument("--limit", type=int, default=0, help="cap holdout size (0=all)")
    args = ap.parse_args()

    from src.training.eval_gate import (
        load_eval_examples, split_holdout, evaluate_via_context_layer,
    )

    examples = load_eval_examples()
    _, holdout = split_holdout(examples, holdout_frac=args.holdout_frac)
    if args.limit:
        holdout = holdout[: args.limit]
    print(f"holdout={len(holdout)} examples; baseline_model={args.baseline}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    base = evaluate_via_context_layer(args.baseline, holdout)
    print(f"BASELINE doc_accuracy={base.doc_accuracy:.4f} exact={base.exact_doc_rate:.4f} "
          f"n={base.n_examples} ({time.time()-t0:.0f}s)", flush=True)

    result = {
        "label": args.label,
        "holdout_n": len(holdout),
        "baseline": {"model": args.baseline, "doc_accuracy": round(base.doc_accuracy, 4),
                     "exact_doc_rate": round(base.exact_doc_rate, 4), "n": base.n_examples},
    }

    if args.candidate:
        t1 = time.time()
        cand = evaluate_via_context_layer(args.candidate, holdout)
        delta = cand.doc_accuracy - base.doc_accuracy
        passed = delta >= 0.0
        print(f"CANDIDATE doc_accuracy={cand.doc_accuracy:.4f} delta={delta:+.4f} "
              f"verdict={'PROMOTE_OK' if passed else 'REGRESSION_REFUSE'} ({time.time()-t1:.0f}s)",
              flush=True)
        result["candidate"] = {"model": args.candidate, "doc_accuracy": round(cand.doc_accuracy, 4),
                               "exact_doc_rate": round(cand.exact_doc_rate, 4), "n": cand.n_examples}
        result["delta_doc_accuracy"] = round(delta, 4)
        result["verdict"] = "PROMOTE_OK" if passed else "REGRESSION_REFUSE"

    out = OUT_DIR / f"eval_{args.label}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
