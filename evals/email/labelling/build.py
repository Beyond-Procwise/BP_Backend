"""Build the labelling sheets (what people fill in) and the keys (what they must NOT see).

    python -m evals.email.labelling.build [out_dir [key_dir]]   # default: labelling/sheets and labelling/key

Sheets go to the labellers. The key folder stays with whoever scores the labellers' work and the model: it holds my
intended answers and which drafts were deliberately flawed, and a labeller who sees it is no longer independent.
Order and ids are shuffled deterministically so an id or a position reveals nothing.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

from . import seed

HERE = Path(__file__).resolve().parent
SQL = HERE.parents[2] / "deploy" / "sql"
LABELS = ("negotiation_counter", "free_prompt", "unclear", "neither")


def rubrics() -> Dict[str, List[str]]:
    """The criteria the judge scores, read from the migration so a sheet can never drift from the config."""

    text = (SQL / "2026-10-08_email_family_v2.sql").read_text()
    found = re.findall(r'"rubric":\s*(\[[^\]]*\])', text)
    return {"negotiation_counter": json.loads(found[0]), "free_prompt": json.loads(found[1])}


def _order(items: List, salt: str) -> List:
    return sorted(items, key=lambda it: hashlib.sha256(f"{salt}:{it[0]}".encode()).hexdigest())


def build(out_dir: Path | str | None = None, key_dir: Path | str | None = None) -> Dict[str, Path]:
    """Sheets to ``out_dir`` (hand THIS folder out); keys to ``key_dir`` (a sibling; never hand it out)."""

    out = Path(out_dir) if out_dir else HERE / "sheets"
    keys = Path(key_dir) if key_dir else (HERE / "key" if not out_dir else out.parent / (out.name + "_key"))
    out.mkdir(parents=True, exist_ok=True)
    keys.mkdir(parents=True, exist_ok=True)
    rub = rubrics()
    written: Dict[str, Path] = {}

    # --- classifier ------------------------------------------------------------------------------------------------
    reqs = _order([(r[0], r) for r in seed.REQUESTS], "classifier")
    rows, key = [], {}
    for n, (_, (text, intended, kind, ids)) in enumerate(reqs, 1):
        rid = f"R-{n:03d}"
        rows.append({"id": rid, "request": text, "label": "", "how_sure_1_to_5": "", "lookup_keys_in_the_text": "", "notes": ""})
        key[rid] = {"intended": intended, "kind": kind, "lookup_keys": ids}
    written["classifier"] = _csv(out / "classifier_requests.csv", rows)
    (keys / "classifier_key.json").write_text(json.dumps(key, indent=1, sort_keys=True))

    # --- judge: one sheet per family -----------------------------------------------------------------------------------
    jkey = {}
    for family, data in (("negotiation_counter", seed.COUNTER), ("free_prompt", seed.FREE)):
        items = _order([(d[0], d) for d in data], f"judge:{family}")
        rows = []
        for n, (_, d) in enumerate(items, 1):
            jid = f"J-{'C' if family == 'negotiation_counter' else 'F'}-{n:03d}"
            if family == "negotiation_counter":
                text, facts, contact, kind, flaw, low = d
                row = {"id": jid, "this_is_contact_number": contact, "facts_you_can_rely_on": facts, "email": text}
            else:
                text, request, kind, flaw, low = d
                row = {"id": jid, "the_request": request, "email": text}
            for c in rub[family]:
                row[c + "_1_to_5"] = ""
            row.update({"overall_1_to_5": "", "notes": ""})
            rows.append(row)
            jkey[jid] = {"family": family, "kind": kind, "flaw": flaw, "expected_low": low}
        written[f"judge_{family}"] = _csv(out / f"judge_{family}.csv", rows)
    (keys / "judge_key.json").write_text(json.dumps(jkey, indent=1, sort_keys=True))
    (out / "FOR_LABELLERS.md").write_text((HERE / "FOR_LABELLERS.md").read_text())
    return written


def _csv(path: Path, rows: List[dict]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")        # LF, so git and CI see the same bytes
        w.writeheader()
        w.writerows(rows)
    return path


if __name__ == "__main__":
    for name, p in build(*(sys.argv[1:3])).items():
        print(f"{name:28s} {p}")
