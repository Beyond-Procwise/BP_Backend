import sys,json,collections
sys.path.insert(0,"src"); sys.path.insert(0,".")
from src.training.eval_gate import load_eval_examples, split_holdout, evaluate_via_context_layer
ex=load_eval_examples(); _,h=split_holdout(ex,0.3,13); h=h[:16]
rep=evaluate_via_context_layer("BeyondProcwise/AgentNick:extract", h)
field_miss=collections.Counter()
for e in rep.per_example:
    for m in e.get("mismatches",[]):
        field_miss[m.get("field")]+=1
out={"doc_accuracy":round(rep.doc_accuracy,4),"n":rep.n_examples,
     "top_field_errors":field_miss.most_common(12)}
json.dump(out,open("artifacts/e2e_audit/field_errors.json","w"),indent=2,default=str)
print("[fe]",json.dumps(out,default=str))
print("[fe] DONE")
