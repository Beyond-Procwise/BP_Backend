"""T9 live validation (hermetic): propose -> approve (via server API) -> hint live,
with cross-vendor extraction byte-identical. Cleans up all synthetic artifacts.
"""
import hashlib
import json

import requests

from src.services.db import get_conn
from src.services.extraction_feedback import proposer
from src.services.extraction_feedback.hint_store import HINT_STORE
from src.services.extraction_feedback.vendor_key import vendor_key
from src.services.extraction.parser import parse as parse_document
from src.services.extraction import context_layer

API = "http://localhost:8000"
V = "ZZLIVEVEND"
BASELINE_DOC = "documents/invoice/PO2_Apparel_Invoice_Higher _Cost.pdf"
BASELINE_SHA = "c3f7b1c18d98"


def _seed_telemetry():
    with get_conn() as c:
        with c.cursor() as cur:
            for i in range(4):
                cur.execute(
                    "INSERT INTO proc.bp_extraction_telemetry "
                    "(captured_at, doc_type, vendor_hint, doc_pk, discrepancy_types, n_discrepancies) "
                    "VALUES (now(), 'invoice', %s, %s, '{\"tax_percent_mismatch\": 1}'::jsonb, 1)",
                    (V, f"zzl{i}"),
                )
        c.commit()


def _cleanup():
    with get_conn() as c:
        with c.cursor() as cur:
            cur.execute("DELETE FROM proc.bp_extraction_telemetry WHERE vendor_hint=%s", (V,))
            cur.execute("DELETE FROM proc.bp_extraction_hint_proposal WHERE vendor_key=%s", (V,))
            cur.execute(
                "DELETE FROM proc.bp_prompt WHERE prompt_type='extraction_vendor_hint' AND prompt_name LIKE %s",
                (f"vhint::%::{V}::%",),
            )
        c.commit()


def main():
    _cleanup()
    print("== baseline: extract unrelated vendor doc (no hints) ==")
    p = parse_document(BASELINE_DOC)
    base_out = context_layer.synthesize("invoice", p.full_text, {}, file_path=BASELINE_DOC)
    base_sha = hashlib.sha256(json.dumps(base_out, sort_keys=True, default=str).encode()).hexdigest()[:12]
    print(f"   baseline sha={base_sha} (expect {BASELINE_SHA}) -> {base_sha == BASELINE_SHA}")

    print("== seed synthetic recurring failure + propose ==")
    _seed_telemetry()
    ids = proposer.propose_all(window_days=3650, min_docs=3, min_fail_rate=0.5, draft=False, vendors={V})
    print(f"   proposer created ids={ids}")
    assert len(ids) == 1
    pid = ids[0]

    print("== server API sees the pending proposal ==")
    listing = requests.get(f"{API}/extraction/proposals?status=pending", timeout=30).json()
    assert any(pp["proposal_id"] == pid for pp in listing["proposals"]), "server did not list proposal"
    print(f"   GET /extraction/proposals -> {listing['count']} pending (incl. {pid})")

    print("== approve via server API (server applies + refreshes its cache) ==")
    appr = requests.post(f"{API}/extraction/proposals/{pid}/approve",
                         json={"approver": "live-check"}, timeout=30).json()
    print(f"   approve -> {appr}")
    assert appr.get("prompt_id")

    print("== hint is live: DB row + local cache + injected in prompt ==")
    HINT_STORE.refresh()
    hints = HINT_STORE.hints_for("invoice", V)
    print(f"   hints_for(invoice,{V}) = {hints}")
    assert hints
    vh = HINT_STORE.hints_for("invoice", vendor_key(f"{V} INV1.pdf"))
    prompt = context_layer._build_prompt(
        "invoice", "some text", [("supplier_name", "text", "s")], {}, None, vh)
    assert "VENDOR-SPECIFIC HINTS" in prompt and hints[0][:20] in prompt
    print("   -> hint injected into extraction prompt for the target vendor")

    print("== server reload-governance reports the hint ==")
    rg = requests.post(f"{API}/agents/reload-governance", timeout=60)
    print(f"   reload-governance -> {rg.status_code} {rg.json().get('extraction_vendor_hints')}")

    print("== ACCURACY GUARD: unrelated vendor extraction unchanged ==")
    p2 = parse_document(BASELINE_DOC)
    after_out = context_layer.synthesize("invoice", p2.full_text, {}, file_path=BASELINE_DOC)
    after_sha = hashlib.sha256(json.dumps(after_out, sort_keys=True, default=str).encode()).hexdigest()[:12]
    print(f"   after-approval sha={after_sha} -> cross-vendor identical: {after_sha == base_sha == BASELINE_SHA}")
    assert after_sha == base_sha == BASELINE_SHA

    print("== reject path ==")
    with get_conn() as c:
        with c.cursor() as cur:
            cur.execute(
                "INSERT INTO proc.bp_extraction_hint_proposal "
                "(doc_type, vendor_key, field_name, dedup_key, evidence, proposed_hint, status) "
                "VALUES ('quote', %s, 'po_id', 'zzl-reject', '{}'::jsonb, 'x', 'pending') RETURNING proposal_id",
                (V,),
            )
            rpid = cur.fetchone()[0]
        c.commit()
    rej = requests.post(f"{API}/extraction/proposals/{rpid}/reject",
                        json={"approver": "live-check", "reason": "synthetic"}, timeout=30)
    print(f"   reject -> {rej.status_code} {rej.json()}")

    print("\nALL LIVE CHECKS PASSED")


if __name__ == "__main__":
    try:
        main()
    finally:
        _cleanup()
        HINT_STORE.refresh()
        print("cleaned up synthetic artifacts")
