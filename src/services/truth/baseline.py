"""Accuracy over what could be checked, and how much that was.

Accuracy alone can be driven to 1.0 by making more fields unverifiable, so it is
never returned or printed without its coverage. Where nothing was verifiable the
accuracy is ``None`` -- not 1.0, and not 0.0, both of which are claims this data
cannot support.
"""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Optional


def _acc(verified: int, unsupported: int) -> Optional[float]:
    checked = verified + unsupported
    return (verified / checked) if checked else None


def score(labelled_path: str) -> dict:
    tally = {"verified": 0, "unsupported": 0, "unverifiable": 0}
    per_type: dict[str, dict] = defaultdict(
        lambda: {"verified": 0, "unsupported": 0, "unverifiable": 0})

    with open(labelled_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            counts = row.get("counts") or {}
            doc_type = row.get("doc_type") or "unknown"
            for key in tally:
                tally[key] += int(counts.get(key, 0))
                per_type[doc_type][key] += int(counts.get(key, 0))

    total = sum(tally.values())
    return {
        "accuracy": _acc(tally["verified"], tally["unsupported"]),
        "coverage": ((tally["verified"] + tally["unsupported"]) / total) if total else 0.0,
        **tally,
        "by_doc_type": {
            doc_type: {
                "accuracy": _acc(c["verified"], c["unsupported"]),
                "coverage": ((c["verified"] + c["unsupported"]) / sum(c.values()))
                            if sum(c.values()) else 0.0,
                **c,
            }
            for doc_type, c in sorted(per_type.items())
        },
    }


def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def format_report(result: dict) -> str:
    lines = [
        f"accuracy {_pct(result['accuracy'])}  coverage {_pct(result['coverage'])}"
        f"   (verified {result['verified']}, unsupported {result['unsupported']},"
        f" unverifiable {result['unverifiable']})",
        "",
        f"{'document type':20} {'accuracy':>10} {'coverage':>10}",
    ]
    for doc_type, c in result["by_doc_type"].items():
        lines.append(f"{doc_type:20} {_pct(c['accuracy']):>10} {_pct(c['coverage']):>10}")
    return "\n".join(lines)
