import sys,json,time
sys.path.insert(0,"src"); sys.path.insert(0,".")
from src.training.eval_gate import load_eval_examples, split_holdout, evaluate_via_context_layer
ex=load_eval_examples(); _,h=split_holdout(ex,0.3,13); h=h[:16]
out={}
for m in ["BeyondProcwise/AgentNick:extract","BeyondProcwise/AgentNick:extract-ft"]:
    t0=time.time(); rep=evaluate_via_context_layer(m,h); dt=time.time()-t0
    out[m]={"doc_accuracy":round(rep.doc_accuracy,4),"exact_doc_rate":round(rep.exact_doc_rate,4),
            "n":rep.n_examples,"fields":f"{rep.total_correct}/{rep.total_fields}","elapsed_s":round(dt,1)}
    print(f"[ft] {m}: doc_acc={out[m]['doc_accuracy']} exact={out[m]['exact_doc_rate']} {out[m]['fields']} ({dt:.0f}s)",flush=True)
    json.dump(out,open("artifacts/e2e_audit/eval_ft_result.json","w"),indent=2)
b=out["BeyondProcwise/AgentNick:extract"]["doc_accuracy"]; c=out["BeyondProcwise/AgentNick:extract-ft"]["doc_accuracy"]
out["_verdict"]={"baseline":b,"candidate":c,"delta":round(c-b,4),"verdict":"PASS" if c>=b else "BELOW_BASELINE","caveat":"holdout seen in training (train-on-test); directional only"}
json.dump(out,open("artifacts/e2e_audit/eval_ft_result.json","w"),indent=2)
print(f"[ft] VERDICT base={b} cand={c} delta={round(c-b,4)} -> {out['_verdict']['verdict']}",flush=True)
print("[ft] DONE")
