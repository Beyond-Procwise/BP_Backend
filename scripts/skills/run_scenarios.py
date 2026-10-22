"""Run skill scenarios live through the orchestrator and write a report a person can grade.

    venv/bin/python -m scripts.skills.run_scenarios --out /path/to/dir [--skill name ...] [--kinds typical,edge]
                                                     [--per-skill 1] [--include-expert]

For each chosen scenario: the skill the router picked, the tools called with their arguments, the
rounds, the time, the verifier note if any, and the answer. The scenario's `expect` line sits beside the
answer so a reviewer can grade it; nothing here grades automatically. Reads only; the policy gate and the
verifier apply exactly as they do for a user. Needs the project's .env (model server and database) and a
loaded model; each scenario costs one orchestrator run.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def _nick():
    from src.services.db import get_conn
    return SimpleNamespace(agents={"opportunity_miner": SimpleNamespace()}, get_db_connection=get_conn)


def _scenarios(folder: Path, kinds, per_skill: int, include_expert: bool):
    rows = yaml.safe_load((folder / "evals" / "scenarios.yaml").read_text()) or []
    chosen = []
    for kind in kinds:
        for s in rows:
            if s.get("kind") != kind or (s.get("expert_graded") and not include_expert):
                continue
            chosen.append(s)
            if len([c for c in chosen if c.get("kind") == kind]) >= per_skill:
                break
    return chosen


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--skill", action="append", default=[])
    ap.add_argument("--kinds", default="typical")
    ap.add_argument("--per-skill", type=int, default=1)
    ap.add_argument("--include-expert", action="store_true")
    ap.add_argument("--replace", action="append", default=[],
                    help="OLD=NEW: substitute a fixture name in prompts with one the corpus holds, e.g. 'Acme=Falconridge Systems 15'")
    args = ap.parse_args(argv)
    swaps = [tuple(r.split("=", 1)) for r in args.replace if "=" in r]
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]

    from src.orchestration import agentnick_control as control
    nick = _nick()
    skills_dir = ROOT / "skills"
    folders = sorted(p for p in skills_dir.iterdir() if p.is_dir() and (not args.skill or p.name in args.skill))
    results = []
    for folder in folders:
        for sc in _scenarios(folder, kinds, args.per_skill, args.include_expert):
            # An edge scenario is often a mid-conversation follow-up ("Two contracts have never been read for
            # clauses."); an optional `context:` line in the scenario gives the router and the model the setting.
            if sc.get("context"):
                sc = {**sc, "prompt": f"{sc['context'].strip()} {sc['prompt']}"}
            for old_name, new_name in swaps:
                sc = {**sc, "prompt": sc["prompt"].replace(old_name, new_name)}
            t0 = time.time()
            try:
                res = control.reason(nick, sc["prompt"], user_id="scenario-runner")
                record = {
                    "skill": folder.name, "scenario": sc.get("id"), "kind": sc.get("kind"), "prompt": sc["prompt"],
                    "expect": sc.get("expect"), "routed_to": [l["source"] for l in (res.context or []) if l.get("layer") == "skill"],
                    "rounds": res.rounds, "seconds": round(time.time() - t0, 1), "error": res.error,
                    "tools": [{"name": getattr(c, "name", None), "arguments": getattr(c, "arguments", None),
                               "ok": getattr(c, "ok", None)} for c in (res.calls or [])],
                    "answer": res.answer, "unverified": "not verified" in (res.answer or ""),
                }
            except Exception as exc:  # noqa: BLE001
                record = {"skill": folder.name, "scenario": sc.get("id"), "prompt": sc["prompt"], "expect": sc.get("expect"),
                          "error": f"runner: {exc}", "seconds": round(time.time() - t0, 1)}
            results.append(record)
            (out_dir / "results.jsonl").open("a").write(json.dumps(record, default=str) + "\n")
            print(f"{folder.name:32s} {sc.get('id'):>4} {record.get('seconds'):>6}s rounds={record.get('rounds')} "
                  f"routed={'yes' if record.get('routed_to') else 'NO'} error={record.get('error')}", flush=True)

    lines = [f"# Skill scenario run {datetime.now(timezone.utc).isoformat(timespec='minutes')}", "",
             f"{len(results)} scenarios; kinds {', '.join(kinds)}; one orchestrator run each. Grade each answer against its expect line.", ""]
    for r in results:
        lines += [f"## {r['skill']} / {r.get('scenario')} ({r.get('kind')})", "",
                  f"**Prompt:** {r['prompt']}", "", f"**Expect:** {r.get('expect')}", "",
                  f"**Routed to:** {', '.join(r.get('routed_to') or []) or 'no skill'} · rounds {r.get('rounds')} · {r.get('seconds')}s"
                  + (f" · error: {r.get('error')}" if r.get('error') else "") + (" · UNVERIFIED FIGURES" if r.get('unverified') else ""), "",
                  "**Tools:** " + ("; ".join(f"{t['name']}({json.dumps(t['arguments'], default=str)[:120]})" for t in r.get("tools") or []) or "none"), "",
                  "**Answer:**", "", (r.get("answer") or "").strip() or "(none)", "", "---", ""]
    (out_dir / "report.md").write_text("\n".join(lines))
    print(f"report: {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
