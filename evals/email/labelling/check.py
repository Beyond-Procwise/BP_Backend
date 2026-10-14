"""Check a filled-in sheet before anyone relies on it.

    python -m evals.email.labelling.check classifier  sheets/classifier_requests.csv
    python -m evals.email.labelling.check judge       sheets/judge_free_prompt.csv  free_prompt

Returns problems as text; an empty list means the sheet is complete and in range. A blank is a problem, not a skip:
a half-filled sheet makes every statistic computed from it wrong in a way nobody sees.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import List

from .build import LABELS, rubrics


def read(path: Path | str) -> List[dict]:
    with Path(path).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _score(value: str):
    try:
        v = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return v if 1 <= v <= 5 else None


def check_classifier(rows: List[dict]) -> List[str]:
    problems, seen = [], set()
    for r in rows:
        rid = r.get("id")
        if rid in seen:
            problems.append(f"{rid}: duplicate id")
        seen.add(rid)
        if (r.get("label") or "").strip() not in LABELS:
            problems.append(f"{rid}: label must be one of {list(LABELS)}, got {r.get('label')!r}")
        if _score(r.get("how_sure_1_to_5", "")) is None:
            problems.append(f"{rid}: how_sure_1_to_5 must be a whole number 1-5, got {r.get('how_sure_1_to_5')!r}")
    return problems


def check_judge(rows: List[dict], family: str) -> List[str]:
    cols = [c + "_1_to_5" for c in rubrics()[family]] + ["overall_1_to_5"]
    problems, seen = [], set()
    for r in rows:
        rid = r.get("id")
        if rid in seen:
            problems.append(f"{rid}: duplicate id")
        seen.add(rid)
        for c in cols:
            if c not in r:
                problems.append(f"{rid}: column {c} is missing")
            elif _score(r[c]) is None:
                problems.append(f"{rid}: {c} must be a whole number 1-5, got {r[c]!r}")
    return problems


if __name__ == "__main__":
    kind, path = sys.argv[1], sys.argv[2]
    rows = read(path)
    found = check_classifier(rows) if kind == "classifier" else check_judge(rows, sys.argv[3])
    print("\n".join(found) if found else f"OK: {len(rows)} rows complete")
    sys.exit(1 if found else 0)
