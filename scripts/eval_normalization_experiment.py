"""OFFLINE experiment: how much of the 0.847 baseline gap is recoverable by
deterministic normalization (vs needing a model finetune)?

Runs the model ONCE (cached), then re-scores the held-out predictions under
progressive normalization levels and categorises the residual mismatches.
Touches NOTHING in production — pure measurement in the eval harness.
"""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from src.training.eval_gate import (
    load_eval_examples, split_holdout, context_layer_generate_fn, DOWNSTREAM_FIELDS,
    _predicted_header, _extract_json,
)

CACHE = Path("/tmp/eval_preds_holdout.json")
MODEL = "BeyondProcwise/AgentNick:extract"
ID_FIELDS = {"invoice_id", "po_id", "quote_id", "requisition_id"}
DATE_FIELDS = {"invoice_date", "due_date", "quote_date", "validity_date", "order_date",
               "expected_delivery_date", "requested_date", "invoice_paid_date"}
ADDR_FIELDS = {"supplier_address", "buyer_address"}


def n_base(v):
    if v is None:
        return ""
    return re.sub(r"\s+", " ", str(v).strip().lower())


def n_id(v):
    s = re.sub(r"[^a-z0-9]", "", n_base(v))
    return re.sub(r"^(qut|po|inv|req)", "", s)  # strip common doc-id prefixes


def n_date(v):
    s = n_base(v)
    if not s:
        return ""
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%d %b %Y", "%d %B %Y"):
        try:
            from datetime import datetime
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    return s


def get_preds(examples):
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    gen = context_layer_generate_fn(MODEL)
    preds = {}
    for ex in examples:
        out = gen(ex)
        preds[ex.pk] = _predicted_header(_extract_json(out))
    CACHE.write_text(json.dumps(preds, default=str))
    return preds


def score(examples, preds, level):
    """level: 0=raw, 1=+id-prefix, 2=+date-iso, 3=+address-swap-tolerant"""
    accs, cats = [], {}
    for ex in examples:
        pred = preds.get(ex.pk, {})
        fields = [k for k, v in ex.expected_header.items()
                  if n_base(v) != "" and k not in DOWNSTREAM_FIELDS]
        if not fields:
            continue
        correct = 0
        for k in fields:
            exp, got = ex.expected_header.get(k), pred.get(k)
            ok = n_base(exp) == n_base(got)
            if not ok and level >= 1 and k in ID_FIELDS:
                ok = n_id(exp) == n_id(got)
            if not ok and level >= 2 and k in DATE_FIELDS:
                ok = n_date(exp) == n_date(got)
            if not ok and level >= 3 and k in ADDR_FIELDS:
                # accept if the model put the right value in the other address field
                other = "buyer_address" if k == "supplier_address" else "supplier_address"
                ok = n_base(exp) == n_base(pred.get(other))
            if ok:
                correct += 1
            else:
                cat = ("id_prefix" if k in ID_FIELDS else "date" if k in DATE_FIELDS
                       else "address" if k in ADDR_FIELDS
                       else "numeric" if any(t in k for t in ("amount", "total", "price", "tax", "quantity"))
                       else "name" if "name" in k or "supplier" in k or "buyer" in k
                       else "other")
                cats[cat] = cats.get(cat, 0) + 1
        accs.append(correct / len(fields))
    return sum(accs) / len(accs), cats


def main():
    ex = load_eval_examples()
    _, hold = split_holdout(ex, holdout_frac=0.3)
    preds = get_preds(hold)
    print(f"holdout docs scored: {len(hold)}  (predictions cached at {CACHE})\n")
    prev = None
    for lvl, label in [(0, "raw baseline"), (1, "+ id-prefix norm"),
                       (2, "+ date ISO norm"), (3, "+ address-swap tolerant")]:
        acc, cats = score(hold, preds, lvl)
        delta = f"  (+{acc-prev:.3f})" if prev is not None else ""
        print(f"L{lvl} {label:26} doc_accuracy = {acc:.3f}{delta}")
        prev = acc
    print(f"\nresidual mismatch categories after full normalization (L3):")
    for c, n in sorted(score(hold, preds, 3)[1].items(), key=lambda x: -x[1]):
        print(f"  {c:12} {n}")


if __name__ == "__main__":
    main()
