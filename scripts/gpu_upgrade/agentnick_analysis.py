#!/usr/bin/env python
"""End-to-end live analysis of AgentNick's power & intelligence.

Exercises AgentNick across its real jobs and prints a capability scorecard:
  1. EXTRACTION  — accuracy on the gold holdout (via the real context_layer step)
  2. REASONING   — procurement task planning (does it decompose correctly?)
  3. NEGOTIATION — supplier-counter strategy reasoning
  4. SUMMARY     — coherent deal/document summarisation
  5. DOMAIN Q&A  — procurement domain knowledge

LLM-only (no DB writes). Uses the live Ollama models.
"""
from __future__ import annotations
import json, os, sys, time, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for line in (ROOT/".env").read_text().splitlines():
    line=line.strip()
    if line and not line.startswith("#") and "=" in line:
        k,v=line.split("=",1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
os.environ.setdefault("PGCONNECT_TIMEOUT","3")
os.environ["OLLAMA_BASE_URL"]=os.environ.get("OLLAMA_BASE_URL","http://127.0.0.1:11434")
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"src"))

from src.services.ollama_client import ollama_generate

REASON_MODEL = os.getenv("ANALYSIS_REASON_MODEL","BeyondProcwise/AgentNick:latest")
EXTRACT_MODEL = "BeyondProcwise/AgentNick:extract"


def section(t): print(f"\n{'='*70}\n{t}\n{'='*70}", flush=True)


def gen(prompt, model, **kw):
    t=time.time()
    out=ollama_generate(prompt, model=model, num_predict=kw.get("num_predict",700),
                        timeout=kw.get("timeout",180)) or ""
    return out.strip(), round(time.time()-t,1)


def assess(text, checks):
    """checks: list of (label, predicate). Returns passed count + detail."""
    res=[]
    for label,pred in checks:
        try: ok=bool(pred(text))
        except Exception: ok=False
        res.append((label,ok))
    return res


def main():
    scorecard={}

    # 1. EXTRACTION
    section("1. EXTRACTION POWER (live gold holdout via context_layer)")
    try:
        from src.training.eval_gate import load_eval_examples, split_holdout, evaluate_via_context_layer
        ex=load_eval_examples(); _,hold=split_holdout(ex); hold=hold[:12]
        rep=evaluate_via_context_layer(EXTRACT_MODEL, hold)
        print(f"doc_accuracy={rep.doc_accuracy:.3f}  exact_doc_rate={rep.exact_doc_rate:.3f}  n={rep.n_examples}")
        scorecard["extraction_doc_accuracy"]=round(rep.doc_accuracy,3)
    except Exception as e:
        print("extraction eval error:", type(e).__name__, str(e)[:150]); scorecard["extraction_doc_accuracy"]=None

    # 2. REASONING / PLANNING
    section("2. REASONING — procurement task planning")
    p=("You are AgentNick, a procurement AI. A buyer needs to source 500 ergonomic "
       "office chairs, budget GBP 40,000, delivery within 6 weeks, from 3 shortlisted "
       "suppliers. Produce a concise step-by-step procurement plan (numbered steps).")
    out,dt=gen(p, REASON_MODEL)
    print(f"[{dt}s]\n{out[:900]}")
    steps=len(re.findall(r"(?m)^\s*\d+[\.\)]", out))
    scorecard["reasoning_steps"]=steps
    print("\nASSESS:", assess(out,[("has>=4 numbered steps", lambda t: steps>=4),
        ("mentions RFQ/quote", lambda t: re.search(r"\b(rfq|quot|tender)\b",t,re.I)),
        ("mentions budget/price", lambda t: re.search(r"budget|price|cost|40,?000",t,re.I)),
        ("mentions evaluation/compare", lambda t: re.search(r"evaluat|compar|shortlist|negoti",t,re.I))]))

    # 3. NEGOTIATION strategy
    section("3. NEGOTIATION — counter-offer strategy reasoning")
    p=("You are AgentNick. A supplier quoted GBP 92/chair for 500 chairs (list 100). "
       "Target unit price is GBP 78. Their lead time is 8 weeks; we need 6. Recommend a "
       "counter-offer with reasoning on price and lead time. Be specific and concise.")
    out,dt=gen(p, REASON_MODEL)
    print(f"[{dt}s]\n{out[:800]}")
    print("\nASSESS:", assess(out,[("proposes a counter price", lambda t: re.search(r"(78|8[0-5]|GBP\s*\d)",t)),
        ("addresses lead time", lambda t: re.search(r"lead time|6 week|expedite|deliver",t,re.I)),
        ("gives reasoning", lambda t: len(t)>200)]))

    # 4. SUMMARY
    section("4. SUMMARY — coherent procurement summary")
    p=("You are AgentNick. Summarise for an executive: Supplier Assurity Ltd, invoice "
       "INV609767 for PO521031, total GBP 12,480 incl VAT, 5 line items of office "
       "furniture, due 30 days. Note any procurement risk. 3-4 sentences.")
    out,dt=gen(p, REASON_MODEL, num_predict=400)
    print(f"[{dt}s]\n{out[:700]}")
    print("\nASSESS:", assess(out,[("mentions supplier", lambda t: "assurity" in t.lower()),
        ("mentions amount", lambda t: re.search(r"12,?480",t)),
        ("coherent length", lambda t: 100<len(t)<900)]))

    # 5. DOMAIN Q&A
    section("5. DOMAIN Q&A — procurement knowledge")
    p=("You are AgentNick. In one paragraph, explain the difference between a Purchase "
       "Order and an Invoice in procurement, and why matching them (3-way match) matters.")
    out,dt=gen(p, REASON_MODEL, num_predict=350)
    print(f"[{dt}s]\n{out[:700]}")
    print("\nASSESS:", assess(out,[("mentions PO", lambda t: re.search(r"purchase order|\bpo\b",t,re.I)),
        ("mentions invoice", lambda t: "invoice" in t.lower()),
        ("mentions 3-way/match/goods receipt", lambda t: re.search(r"3-way|three-way|match|goods receipt|grn",t,re.I))]))

    section("SCORECARD")
    print(json.dumps(scorecard, indent=2))
    print("ANALYSIS_DONE")


if __name__=="__main__":
    main()
