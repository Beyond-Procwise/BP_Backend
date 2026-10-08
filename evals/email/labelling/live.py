"""Run the REAL classifier and judge stages over a labelled set and report. Needs a model; built and tested against a fake.

    python -m evals.email.labelling.live classifier --labels a.csv b.csv --out report.json
    python -m evals.email.labelling.live judge --family free_prompt --scores a.csv b.csv --out report.json

It uses the same stage code, the same governed prompt text (read from the migration, so it tests exactly what would be
installed) and, with ``--live``, the same ``ask`` the drafting agent uses in production. Without ``--live`` it refuses
rather than quietly using a stand-in. Results from a fake model say nothing about model quality and are labelled so.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.services.draft_assurance import stages

from . import metrics
from .build import SQL, rubrics
from .check import check_classifier, check_judge, read

Ask = Callable[[str, str], str]


def prompts_from_migration() -> Dict[str, str]:
    text = (SQL / "2026-10-08_email_assurance_prompts.sql").read_text()
    out = {}
    for name, body in re.findall(r"prompt_name = '([a-z_]+)'.*?\$p\$(\{.*?\})\$p\$", text, re.S) or []:
        out[name] = json.loads(body)["prompt_template"]
    if len(out) < 3:                                             # the insert comes before its NOT EXISTS guard in the file
        for body in re.findall(r"\$p\$(\{.*?\})\$p\$", text, re.S):
            tpl = json.loads(body)["prompt_template"]
            out["email_family_classify" if "what kind of procurement email" in tpl else
                "email_brief_plan" if "plan procurement emails" in tpl else "email_draft_judge"] = tpl
    return out


def _family_rows() -> Dict[str, Dict[str, Any]]:
    out = {}
    for path in sorted(SQL.glob("2026-10-0*_email_family_*.sql")):
        if path.name.endswith("_rollback.sql") or "family_v2" in path.name:
            continue
        text = path.read_text()
        if "$json$" not in text:
            continue
        rules = json.loads(text.split("$json$")[1])["rules"]
        if rules.get("classifiable") is False:
            continue
        m = re.search(r"'email_family',\s*'((?:[^']|'')*)'", text)
        out[rules["family_id"]] = {"description": rules.get("request_description") or (m.group(1).replace("''", "'") if m else rules["family_id"]),
                                   "label": rules.get("request_label")}
    return out


def classifiable_families() -> Dict[str, str]:
    """{family_id: description} for the families the classifier is shown, read from the family migrations."""

    return {k: v["description"] for k, v in _family_rows().items()}


def family_labels() -> Dict[str, str]:
    return {k: v["label"] for k, v in _family_rows().items() if v["label"]}


def run_classifier(ask: Ask, requests: Dict[str, str], *, template: Optional[str] = None,
                   families: Optional[Dict[str, str]] = None, clock: Callable[[], float] = time.monotonic) -> Dict[str, Dict[str, Any]]:
    template = template or prompts_from_migration()["email_family_classify"]
    families = families or classifiable_families()
    labels = family_labels()
    out = {}
    for rid, text in requests.items():
        t0 = clock()
        res = stages.classify_request(ask, template, text, families, labels=labels)
        dt = round(clock() - t0, 3)
        if res["status"] != "captured":
            out[rid] = {"status": res["status"], "reason": res.get("reason"), "latency_s": dt}
            continue
        c = res["classification"]
        out[rid] = {"status": "captured", "family": c["family_id"], "confidence": c.get("confidence"),
                    "asked": bool(res.get("clarification")), "lookup_keys": c.get("lookup_keys") or {}, "latency_s": dt}
    return out


def run_judge(ask: Ask, family: str, items: Dict[str, Dict[str, str]], *, template: Optional[str] = None,
              clock: Callable[[], float] = time.monotonic) -> Dict[str, Dict[str, Any]]:
    """``items[id]`` = {"email": ..., "facts": ...}. The judge sees the email and the facts, never a human score."""

    template = template or prompts_from_migration()["email_draft_judge"]
    out = {}
    for jid, it in items.items():
        t0 = clock()
        res = stages.judge_draft(ask, template, rubrics()[family], text=it["email"], brief=None, facts={"given": it.get("facts", "")})
        dt = round(clock() - t0, 3)
        if res.get("status") != "scored":
            out[jid] = {"status": res.get("status", "invalid"), "reason": res.get("reason"), "latency_s": dt}
        else:
            out[jid] = {"status": "scored", "scores": res["scores"], "overall": res["overall"], "latency_s": dt}
    return out


def load_classifier_labels(paths: List[str]) -> Dict[str, Dict[str, str]]:
    labels = {}
    for p in paths:
        rows = read(p)
        problems = check_classifier(rows)
        if problems:
            raise SystemExit(f"{p} is not complete:\n" + "\n".join(problems[:20]))
        labels[Path(p).stem] = {r["id"]: r["label"].strip() for r in rows}
    return labels


def load_judge_scores(paths: List[str], family: str) -> Dict[str, Dict[str, Dict[str, int]]]:
    scores = {}
    for p in paths:
        rows = read(p)
        problems = check_judge(rows, family)
        if problems:
            raise SystemExit(f"{p} is not complete:\n" + "\n".join(problems[:20]))
        per = {}
        for r in rows:
            d = {c: int(r[c + "_1_to_5"]) for c in rubrics()[family]}
            d["overall"] = int(r["overall_1_to_5"])
            per[r["id"]] = d
        scores[Path(p).stem] = per
    return scores


def _live_ask() -> Ask:
    from agents.email_drafting_agent import EmailDraftingAgent

    return EmailDraftingAgent()._assurance_env(user_id=None).ask


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("what", choices=["classifier", "judge"])
    ap.add_argument("--labels", nargs="+", help="classifier: filled sheets, one per rater")
    ap.add_argument("--scores", nargs="+", help="judge: filled sheets, one per rater")
    ap.add_argument("--family", choices=["negotiation_counter", "free_prompt"])
    ap.add_argument("--sheets", default=str(Path(__file__).resolve().parent / "sheets"), help="the blank sheets (the ids and texts)")
    ap.add_argument("--key", default=str(Path(__file__).resolve().parent / "key"), help="the key folder (keep it away from labellers)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--live", action="store_true", help="use the real model (the drafting agent's own ask)")
    a = ap.parse_args(argv)
    if not a.live:
        print("refusing to run without --live: a stand-in model would produce a number that looks like a result", file=sys.stderr)
        return 2
    ask, sheets = _live_ask(), Path(a.sheets)
    if a.what == "classifier":
        labels = load_classifier_labels(a.labels)
        cons = metrics.consensus(labels)
        rows = {r["id"]: r["request"] for r in read(sheets / "classifier_requests.csv")}
        preds = run_classifier(ask, rows)
        key = json.loads((Path(a.key) / "classifier_key.json").read_text())
        report = {"model": "live", "raters": sorted(labels), "kappa": (metrics.cohen_kappa(*[[labels[r][i] for i in sorted(rows)] for r in sorted(labels)[:2]]) if len(labels) >= 2 else None),
                  "intent_vs_team": metrics.intent_vs_labels(key, cons),
                  "model_vs_team": metrics.classifier_report({i: c["label"] for i, c in cons.items()}, preds, key), "predictions": preds}
    else:
        scores = load_judge_scores(a.scores, a.family)
        sheet = read(sheets / f"judge_{a.family}.csv")
        items = {r["id"]: {"email": r["email"], "facts": r.get("facts_you_can_rely_on") or r.get("the_request", "")} for r in sheet}
        model = run_judge(ask, a.family, items)
        key = json.loads((Path(a.key) / "judge_key.json").read_text())
        report = {"model": "live", "raters": sorted(scores), "judge_vs_team": metrics.judge_report(metrics.human_mean(scores), model, key), "scores": model}
    Path(a.out).write_text(json.dumps(report, indent=1, sort_keys=True, default=str))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
