import sys,json,time
sys.path.insert(0,"src"); sys.path.insert(0,".")
from src.training.eval_gate import load_eval_examples, split_holdout, evaluate_via_context_layer
ex=load_eval_examples(); _,h=split_holdout(ex,0.3,13); h=h[:16]
t0=time.time(); rep=evaluate_via_context_layer("BeyondProcwise/AgentNick:unified", h); dt=time.time()-t0
base=0.8482
out={"candidate":"AgentNick:unified","doc_accuracy":round(rep.doc_accuracy,4),
     "exact_doc_rate":round(rep.exact_doc_rate,4),"n_scored":rep.n_examples,
     "fields":f"{rep.total_correct}/{rep.total_fields}","elapsed_s":round(dt,1),
     "baseline_extract":base,"delta":round(rep.doc_accuracy-base,4),
     "verdict":"PASS" if rep.doc_accuracy>=base else "BELOW_BASELINE",
     "worst":sorted(rep.per_example,key=lambda r:r["accuracy"])[:3]}
json.dump(out,open("artifacts/e2e_audit/eval_unified_result.json","w"),indent=2,default=str)
print(f"[u] unified doc_acc={out['doc_accuracy']} vs base {base} delta={out['delta']} -> {out['verdict']} ({dt:.0f}s)")
print("[u] DONE")
