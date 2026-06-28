#!/usr/bin/env python
"""Improvement-signal probes for AgentNick — beyond the capability scorecard,
these target the dimensions that inform HOW to make it smarter:

  A. GROUNDING / no-fabrication  — when a field is absent, does it abstain (good)
     or invent a value (bad)? The #1 accuracy risk.
  B. KNOWLEDGE DEPTH             — harder procurement domain questions, scored.
  C. LATENCY                     — per task type (the 30B 'thinking' model is slow).
  D. CONSISTENCY                 — same prompt twice; temp-0 determinism in practice.

LLM-only (no DB). Run against the live models.
"""
from __future__ import annotations
import json, os, re, sys, time
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

EXTRACT="BeyondProcwise/AgentNick:extract"
AGENT=os.getenv("ANALYSIS_REASON_MODEL","BeyondProcwise/AgentNick:unified")

def sec(t): print(f"\n{'='*68}\n{t}\n{'='*68}",flush=True)
def gen(p,model,n=500,to=180):
    t=time.time(); o=ollama_generate(p,model=model,num_predict=n,timeout=to) or ""
    return o.strip(), round(time.time()-t,1)

def main():
    out={}

    sec("A. GROUNDING / NO-FABRICATION (absent field -> abstain?)")
    # Invoice text that has NO due_date and NO PO number — model must NOT invent them.
    doc=("INVOICE\nSupplier: Northwind Traders Ltd\nInvoice Number: INV-7781\n"
         "Invoice Date: 12 March 2024\nDescription: Office supplies\nTotal: GBP 1,240.00\n")
    p=(f"Extract as JSON with a 'header' object from this invoice. Use ONLY values "
       f"present in the document; if a field is absent, set it to null. Fields: "
       f"invoice_id, invoice_date, po_id, due_date, total_amount.\n\nDOCUMENT:\n{doc}\nJSON:")
    o,dt=gen(p,EXTRACT,n=300)
    m=re.search(r"\{[\s\S]*\}",o); parsed={}
    try: parsed=json.loads(m.group(0)) if m else {}
    except Exception: parsed={}
    hdr=parsed.get("header",parsed) if isinstance(parsed,dict) else {}
    po=str(hdr.get("po_id") or "").strip().lower()
    due=str(hdr.get("due_date") or "").strip().lower()
    abstained = po in ("","none","null") and due in ("","none","null")
    print(f"[{dt}s] po_id={hdr.get('po_id')!r} due_date={hdr.get('due_date')!r}  -> "
          f"{'ABSTAINED (good)' if abstained else 'FABRICATED (BAD)'}")
    out["grounding_abstains"]=abstained

    sec("B. KNOWLEDGE DEPTH (harder procurement questions)")
    qs=[("Incoterms: under DDP vs FOB, who bears import duties and at what point does risk transfer?",
         ["ddp","fob","seller","buyer","duty|duties|customs","risk"]),
        ("Explain net-30 vs 2/10 net-30 payment terms and the effective annualised cost of NOT taking the 2/10 discount.",
         ["2/10|discount","net.?30","annual|apr|%","early"]),
        ("In a 3-way match, which three documents are compared and what specific quantities/amounts must reconcile?",
         ["purchase order|po","invoice","goods receipt|grn|receipt","quantit|amount|price"])]
    kd=[]
    for q,keys in qs:
        o,dt=gen(q,AGENT,n=350)
        hits=sum(1 for k in keys if re.search(k,o,re.I))
        kd.append((hits,len(keys)))
        print(f"[{dt}s] {hits}/{len(keys)} key concepts | Q: {q[:60]}...")
    out["knowledge_depth"]=f"{sum(h for h,_ in kd)}/{sum(t for _,t in kd)}"

    sec("C. LATENCY per task type")
    lat={}
    _,lat["extract_call"]=gen("Extract JSON header from: INVOICE INV-1 Total GBP 5. Fields: invoice_id,total_amount.",EXTRACT,n=120)
    _,lat["short_qa"]=gen("In one sentence: what is a purchase requisition?",AGENT,n=120)
    _,lat["plan"]=gen("Plan in 5 numbered steps: source 100 chairs from 3 suppliers.",AGENT,n=500)
    print("latency_s:",lat)
    out["latency_s"]=lat

    sec("D. CONSISTENCY (same prompt x2, temp 0)")
    q="List exactly 5 numbered steps to evaluate 3 supplier quotes for 100 laptops."
    a,_=gen(q,AGENT,n=400); b,_=gen(q,AGENT,n=400)
    identical = a.strip()==b.strip()
    # structural similarity: same number of numbered steps?
    na=len(re.findall(r"(?m)^\s*\d+[\.\)]",a)); nb=len(re.findall(r"(?m)^\s*\d+[\.\)]",b))
    print(f"identical={identical} | step_counts={na} vs {nb}")
    out["consistency_identical"]=identical; out["consistency_steps"]=f"{na} vs {nb}"

    sec("IMPROVEMENT-SIGNAL SUMMARY")
    print(json.dumps(out,indent=2))
    print("PROBE_DONE")

if __name__=="__main__":
    main()
