"""Read-only multi-format extraction validation on /home/muthu/Downloads/new/.
Calls the v4 engine core (run_data_extraction + to_extraction_result) directly —
NO persist, NO supplier-resolver (so zero DB writes)."""
from __future__ import annotations
import sys, json, time, os
from pathlib import Path
sys.path.insert(0, "src"); sys.path.insert(0, ".")
os.chdir("/home/muthu/PycharmProjects/BP_Backend")

NEW = Path("/home/muthu/Downloads/new")

def infer(name: str) -> str:
    n = name.lower()
    if "invoice" in n: return "invoice"
    if "_po" in n or n.startswith("po") or "purchase" in n: return "po"
    if "quote" in n or "qte" in n or "qut" in n: return "quote"
    return "invoice"

def main():
    from src.services.extraction_v3.extraction_v4 import run_data_extraction, to_extraction_result, detect_document_type
    out = []
    for f in sorted(NEW.iterdir()):
        if not f.is_file(): continue
        dt = infer(f.name)
        rec = {"file": f.name, "ext": f.suffix.lower(), "doc_type": dt}
        t0 = time.time()
        try:
            try:
                rec["detected"] = detect_document_type(str(f))
            except Exception:
                rec["detected"] = None
            raw = run_data_extraction(str(f), doc_type=dt)
            res = to_extraction_result(raw, "purchase_order" if dt == "po" else dt, source_file=str(f))
            rec["doc_pk"] = res.doc_pk
            hdr = [c for c in res.committed if not c.field_path.startswith("line_items")]
            lines = [c for c in res.committed if c.field_path.startswith("line_items")]
            rec["n_header"] = len(hdr)
            rec["n_line_fields"] = len(lines)
            rec["n_residuals"] = len(res.residuals)
            rec["header"] = {c.field_path: c.value for c in hdr}
            rec["elapsed_s"] = round(time.time() - t0, 1)
        except Exception as e:
            import traceback
            rec["error"] = str(e)[:200]
            rec["trace"] = traceback.format_exc()[-500:]
            rec["elapsed_s"] = round(time.time() - t0, 1)
        out.append(rec)
        print(f"[nd] {f.name} ({rec['ext']}) dt={dt} -> pk={rec.get('doc_pk')} hdr={rec.get('n_header')} resid={rec.get('n_residuals')} {rec.get('elapsed_s')}s {('ERR:'+rec['error']) if rec.get('error') else ''}", flush=True)
        json.dump(out, open("artifacts/e2e_audit/newdata_results.json", "w"), indent=2, default=str)
    print("[nd] DONE", flush=True)

if __name__ == "__main__":
    main()
