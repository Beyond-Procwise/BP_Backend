"""Measure p0 and alpha for item_equivalence against real ground truth.

Two proc.bp_invoice_line_items_trgt lines are the same product iff they share
an item_id. Hold item_id out of the scored records and score on what remains
(description, unit_of_measure, unit_price, the always-MISSING supplier_same
signal): that is a labelled sample by construction, the same discipline
scripts/graph_resolution/calibrate.py used for supplier_identity, but on a
corpus that actually has repeated identity keys (4,755 distinct item_id
values appear on more than one of the 55,483 lines -- measured 2026-09-10).

This is a SEPARATE function/sweep from calibrate.py's supplier-shaped
build_labelled_pairs/sweep: Task 5's tests pin that module's exact behaviour
on vat_number, so this module does not touch it. The mutate-then-restore-in-
finally discipline on the global PROFILES dict is copied exactly.

LEAK FIX (2026-09-10, fix round 2): item_description embeds the item_id as a
literal substring on 100% of lines measured ("Heavy-Duty Excavation Kit
ITM000015"), so merely stripping item_id from the scored record left the
`desc` signal (weight 4) scoring a near-perfect textual proxy for the
withheld label -- circular, the same failure mode VAT-on-supplier would have
been. build_labelled_pairs_by_item_id now also strips the item_id substring
out of item_description on BOTH records before scoring. This is a
calibration-only transform: the production comparator in item_equivalence.py
is untouched, because real documents legitimately carry item codes in their
descriptions and the profile should keep using them at inference time.
"""
from __future__ import annotations

import itertools
import random
import re
from typing import Iterable, List, Optional

from src.services import linking_engine as _le
from src.services.graph_resolution.profiles import item_equivalence as ie


def _strip_embedded_code(description: Optional[str], item_id: Optional[str]) -> Optional[str]:
    """Remove a literal item_id substring from a description, then collapse
    whatever whitespace/punctuation that leaves behind so no double space or
    dangling separator remains. Calibration-only -- see module docstring."""
    if not description or not item_id:
        return description
    cleaned = re.sub(re.escape(str(item_id)), "", str(description), flags=re.IGNORECASE)
    cleaned = re.sub(r"[ \t]*\(\s*\)", "", cleaned)   # empty parens the code left behind
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -,")
    return cleaned or None


def build_labelled_pairs_by_item_id(rows: Iterable[dict]) -> List[tuple]:
    """Label a pair 'same' iff the two lines share a non-null item_id, then
    strip item_id -- and any occurrence of it embedded in item_description --
    from both scored records so scoring cannot see the label, directly or
    through the description's leaked copy of it."""
    rows = list(rows)
    out = []
    for a, b in itertools.combinations(rows, 2):
        same = (a.get("item_id") is not None and a.get("item_id") == b.get("item_id"))
        sa = {k: v for k, v in a.items() if k != "item_id"}
        sb = {k: v for k, v in b.items() if k != "item_id"}
        sa["item_description"] = _strip_embedded_code(a.get("item_description"), a.get("item_id"))
        sb["item_description"] = _strip_embedded_code(b.get("item_description"), b.get("item_id"))
        out.append((sa, sb, same))
    return out


def sweep(pairs: List[tuple], grid: List[tuple]) -> List[dict]:
    """Score every pair under each (p0, alpha) and report separation.

    Same mutate-the-global-then-restore-in-finally pattern as
    calibrate.sweep, applied to item_equivalence's own PROFILES entry.
    """
    results = []
    original = dict(_le.PROFILES[ie.PROFILE])
    try:
        for p0, alpha in grid:
            _le.PROFILES[ie.PROFILE] = {**original, "p0": p0, "alpha": alpha}
            same_F, diff_F, false_auto = [], [], 0
            for a, b, is_same in pairs:
                r = ie.score(a, b)
                (same_F if is_same else diff_F).append(r["F"])
                if not is_same and r["F"] >= _le._BAND_AUTO:
                    false_auto += 1
            results.append({
                "p0": p0, "alpha": alpha,
                "separation": (min(same_F) - max(diff_F)) if same_F and diff_F else 0.0,
                "false_auto_links": false_auto,
                "n_same": len(same_F), "n_diff": len(diff_F),
            })
    finally:
        _le.PROFILES[ie.PROFILE] = original
    return results


def best(results: List[dict]) -> dict:
    """Highest separation among settings that auto-link no false pair."""
    clean = [r for r in results if r["false_auto_links"] == 0]
    pool = clean or results
    return max(pool, key=lambda r: r["separation"])


def fetch_sample_rows(cur, seed: int = 20260910,
                       n_groups: int = 60, max_per_group: int = 5,
                       n_singletons: int = 150) -> List[dict]:
    """Pull a sample: lines from a random set of repeated-item_id groups
    (bounded per group so a 39-line item doesn't dominate the pair count),
    plus a batch of singleton-item_id lines for cross-item 'diff' variety."""
    cur.execute("""
        select item_id from proc.bp_invoice_line_items_trgt
        where item_id is not null
        group by item_id having count(*) > 1
    """)
    all_groups = [r[0] for r in cur.fetchall()]
    rng = random.Random(seed)
    rng.shuffle(all_groups)
    chosen_groups = all_groups[:n_groups]

    rows: List[dict] = []
    cols = ("invoice_line_id", "item_id", "item_description",
            "unit_of_measure", "unit_price")
    for item_id in chosen_groups:
        cur.execute(f"""
            select {", ".join(cols)}
            from proc.bp_invoice_line_items_trgt
            where item_id = %s
            order by invoice_line_id
            limit %s
        """, (item_id, max_per_group))
        for r in cur.fetchall():
            rows.append(dict(zip(cols, r)))

    cur.execute(f"""
        select {", ".join(cols)} from (
            select {", ".join(cols)}, count(*) over (partition by item_id) as cnt
            from proc.bp_invoice_line_items_trgt
            where item_id is not null
        ) x where cnt = 1
        order by random()
        limit %s
    """, (n_singletons,))
    for r in cur.fetchall():
        rows.append(dict(zip(cols, r)))

    for row in rows:
        row["_same_entity_p"] = None
    return rows


_CACHE_PATH = "/tmp/claude-1001/-home-muthu-PycharmProjects-BP-Backend/134e6e52-c205-4980-8644-79339c0379f5/scratchpad/item_equiv_sample_rows.json"


def main() -> None:
    import json
    import os
    import sys

    if os.path.exists(_CACHE_PATH):
        with open(_CACHE_PATH) as f:
            rows = json.load(f)
        print(f"loaded {len(rows)} cached lines from {_CACHE_PATH}")
    else:
        from src.services.db import get_conn
        with get_conn() as conn:
            cur = conn.cursor()
            rows = fetch_sample_rows(cur)
        with open(_CACHE_PATH, "w") as f:
            json.dump(rows, f, default=str)
        print(f"sampled {len(rows)} lines (cached to {_CACHE_PATH})")

    pairs = build_labelled_pairs_by_item_id(rows)
    n_same = sum(1 for _, _, s in pairs if s)
    n_diff = len(pairs) - n_same
    print(f"pairs={len(pairs)} n_same={n_same} n_diff={n_diff}")

    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "refine2":
        grid = [(p0, alpha)
                for p0 in (0.003, 0.005, 0.007, 0.01)
                for alpha in (2.2, 2.5, 2.7, 3.0, 3.3, 3.5)]
    elif mode == "refine":
        grid = [(0.01, a) for a in (2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0)] + \
               [(0.005, a) for a in (2.0, 3.0, 4.0, 6.0, 10.0)]
    else:
        grid = [(p0, alpha)
                for p0 in (0.01, 0.02, 0.05, 0.08)
                for alpha in (0.35, 0.45, 0.55, 0.60, 0.65, 0.70, 0.80,
                              0.90, 1.00, 1.20, 1.50, 2.00)]
    results = sweep(pairs, grid)
    for r in sorted(results, key=lambda r: -r["separation"]):
        print(r)

    chosen = best(results)
    print("BEST:", chosen)


if __name__ == "__main__":
    main()
