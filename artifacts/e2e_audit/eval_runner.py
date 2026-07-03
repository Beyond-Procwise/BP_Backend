"""Head-to-head extraction eval: current AgentNick:latest (Thinking 30B) vs
AgentNick:unified (Qwen3-30B-A3B-Instruct-2507). Faithful scoring through the
real context_layer over a deterministic holdout. Read-only (no DB writes)."""
from __future__ import annotations
import sys, json, time
sys.path.insert(0, "src"); sys.path.insert(0, ".")
from src.training.eval_gate import (
    load_eval_examples, split_holdout, evaluate_via_context_layer,
)

CAP = int(sys.argv[1]) if len(sys.argv) > 1 else 20
MODELS = ["BeyondProcwise/AgentNick:extract", "BeyondProcwise/AgentNick:unified"]
OUT = "artifacts/e2e_audit/eval_results.json"

ex = load_eval_examples()
_, holdout = split_holdout(ex, holdout_frac=0.3, seed=13)
holdout = holdout[:CAP]
print(f"[eval] total examples={len(ex)} | holdout(capped)={len(holdout)}", flush=True)

results = {}
for m in MODELS:
    print(f"[eval] running {m} ...", flush=True)
    t0 = time.time()
    rep = evaluate_via_context_layer(m, holdout)
    dt = time.time() - t0
    results[m] = {
        "doc_accuracy": round(rep.doc_accuracy, 4),
        "exact_doc_rate": round(rep.exact_doc_rate, 4),
        "n_scored": rep.n_examples,
        "total_fields": rep.total_fields,
        "total_correct": rep.total_correct,
        "elapsed_s": round(dt, 1),
        "worst": sorted(rep.per_example, key=lambda r: r["accuracy"])[:3],
    }
    print(f"[eval] {m}: doc_acc={results[m]['doc_accuracy']} exact={results[m]['exact_doc_rate']} "
          f"fields={rep.total_correct}/{rep.total_fields} in {dt:.0f}s", flush=True)
    json.dump(results, open(OUT, "w"), indent=2, default=str)

base = results[MODELS[0]]["doc_accuracy"]
cand = results[MODELS[1]]["doc_accuracy"]
verdict = "PROMOTE_OK" if cand >= base else "REGRESSION"
results["_verdict"] = {"baseline": base, "candidate": cand,
                       "delta": round(cand - base, 4), "verdict": verdict}
json.dump(results, open(OUT, "w"), indent=2, default=str)
print(f"[eval] VERDICT: base={base} cand={cand} delta={round(cand-base,4)} -> {verdict}", flush=True)
print("[eval] DONE", flush=True)
