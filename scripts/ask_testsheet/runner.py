"""Send the pre-retrieval test sheet through POST /workflows/ask/stream and print a result table.

    ASK_TOKEN=<buyer id token> python -m scripts.ask_testsheet.runner --sheet sheet.json \
        --base-url http://localhost:8000 [--admin-token <token>] [--json out.json]

Run it with a NON-admin buyer-role token (ASK_TOKEN). The runner does not create users or touch
Cognito; it only sends the token you give it. Output: Test ID | Result | Reason | Latency per stage.

Stage latency comes from the arrival time of the stream's stage events (routed, retrieving, grounded,
generating, answer_complete, done), measured by the client. A buyer cannot see the server trace, so
"no retrieval" is judged from the absence of a ``retrieving`` stage event (one carrying ``lookup`` is a database lookup by exact id, not a vector search). With ``--admin-token`` the
runner also sends the ``x-ask-trace`` header and reports the retrieval count from the real trace.

Sheet format (JSON list): {"id", "question" | "turns": [..], "expect": {...}} where expect may hold
  blocked: true|false          the guardrail must (not) block it       (``guard`` field in ``done``)
  category: "<guard category>" ... and in this category
  no_retrieval: true           no ``retrieving`` stage may appear
  contains / not_contains: [..] substrings of the final answer (case-insensitive)
A test with no ``expect`` is reported as UNGRADED rather than guessed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

STAGES = ("routed", "guarded", "retrieving", "grounded", "generating", "answer_complete", "done")


def parse_sse(lines: Iterable[bytes]) -> Iterator[Tuple[float, Dict[str, Any]]]:
    """(arrival time, event) for every ``data:`` line."""
    for raw in lines:
        line = raw.decode("utf-8", "replace").strip()
        if line.startswith("data:"):
            try:
                yield time.perf_counter(), json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue


def ask(base_url: str, token: str, question: str, *, trace: bool = False, session_id: Optional[str] = None,
        timeout: float = 300.0) -> Dict[str, Any]:
    body = json.dumps({"query": question, **({"session_id": session_id} if session_id else {})}).encode()
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
    if trace:
        headers["x-ask-trace"] = "1"
    req = urllib.request.Request(base_url.rstrip("/") + "/workflows/ask/stream", data=body, headers=headers, method="POST")
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return collect(started, parse_sse(resp))


def collect(started: float, events: Iterable[Tuple[float, Dict[str, Any]]]) -> Dict[str, Any]:
    stage_at: Dict[str, float] = {}
    out: Dict[str, Any] = {"answer": "", "stages": [], "done": None, "error": None, "vector_retrieval": False}
    for at, ev in events:
        kind = ev.get("type")
        if kind == "stage":
            name = str(ev.get("stage"))
            out["stages"].append(name)
            # A "retrieving" stage with a ``lookup`` key is a database lookup by exact id, not a vector search.
            if name == "retrieving" and not ev.get("lookup"):
                out["vector_retrieval"] = True
            stage_at.setdefault(name, at - started)
        elif kind == "done":
            out["done"] = ev
            out["answer"] = ev.get("answer") or ""
            stage_at.setdefault("done", at - started)
        elif kind == "error":
            out["error"] = ev.get("message")
    out["stage_at"] = stage_at
    out["total"] = max(stage_at.values()) if stage_at else 0.0
    return out


def stage_latency(stage_at: Dict[str, float]) -> str:
    """Seconds spent in each stage = gap to the next stage event."""
    seen = sorted(((t, n) for n, t in stage_at.items()))
    parts = [f"{name}={(seen[i + 1][0] - t):.1f}s" for i, (t, name) in enumerate(seen[:-1])]
    return " ".join(parts) or "-"


def grade(expect: Optional[Dict[str, Any]], run: Dict[str, Any]) -> Tuple[str, str]:
    if run.get("error"):
        return "FAIL", f"stream error: {run['error'][:80]}"
    if not expect:
        return "UNGRADED", "no expectation in the sheet"
    done = run.get("done") or {}
    guard = done.get("guard") or {}
    trace = done.get("trace") or {}
    answer = (run.get("answer") or "").lower()
    problems: List[str] = []
    if "blocked" in expect and bool(guard) != bool(expect["blocked"]):
        problems.append("blocked" if guard else "not blocked")
    if expect.get("category") and guard.get("category") != expect["category"]:
        problems.append(f"category {guard.get('category')!r}")
    if expect.get("no_retrieval"):
        retrieved = trace.get("retrieval_count") if trace else (1 if run.get("vector_retrieval") else 0)
        if retrieved:
            problems.append("retrieval ran")
    for needle in expect.get("contains", []):
        if needle.lower() not in answer:
            problems.append(f"missing {needle!r}")
    for needle in expect.get("not_contains", []):
        if needle.lower() in answer:
            problems.append(f"contains {needle!r}")
    return ("FAIL", "; ".join(problems)) if problems else ("PASS", "as expected")


def run_sheet(sheet: List[Dict[str, Any]], base_url: str, token: str, admin_token: Optional[str]) -> List[Dict[str, Any]]:
    rows = []
    for case in sheet:
        turns = case.get("turns") or [case["question"]]
        session = f"sheet-{case['id']}-{int(time.time())}"
        run: Dict[str, Any] = {}
        for turn in turns:
            run = ask(base_url, admin_token or token, turn, trace=bool(admin_token), session_id=session)
        result, reason = grade(case.get("expect"), run)
        rows.append({"id": case["id"], "result": result, "reason": reason,
                     "latency": stage_latency(run.get("stage_at", {})), "total_s": round(run.get("total", 0.0), 1)})
    return rows


def render(rows: List[Dict[str, Any]]) -> str:
    out = [f"{'Test':<6}| {'Result':<9}| {'Reason':<44}| Latency per stage"]
    out.append("-" * 110)
    for r in rows:
        out.append(f"{r['id']:<6}| {r['result']:<9}| {r['reason'][:43]:<44}| {r['latency']} (total {r['total_s']}s)")
    tally = {k: sum(1 for r in rows if r["result"] == k) for k in ("PASS", "FAIL", "UNGRADED")}
    out.append(f"\n{tally['PASS']} PASS, {tally['FAIL']} FAIL, {tally['UNGRADED']} UNGRADED of {len(rows)}")
    return "\n".join(out)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sheet", required=True)
    ap.add_argument("--base-url", default=os.environ.get("ASK_BASE_URL", "http://localhost:8000"))
    ap.add_argument("--admin-token", default=os.environ.get("ASK_ADMIN_TOKEN"))
    ap.add_argument("--only", help="comma-separated test ids")
    ap.add_argument("--json", help="also write the rows here")
    args = ap.parse_args(argv)
    token = os.environ.get("ASK_TOKEN")
    if not token:
        print("Set ASK_TOKEN to a buyer-role (non-admin) token. The runner does not create users.", file=sys.stderr)
        return 2
    sheet = json.load(open(args.sheet))
    if args.only:
        wanted = set(args.only.split(","))
        sheet = [c for c in sheet if c["id"] in wanted]
    rows = run_sheet(sheet, args.base_url, token, args.admin_token)
    print(render(rows))
    if args.json:
        json.dump(rows, open(args.json, "w"), indent=2)
    return 1 if any(r["result"] == "FAIL" for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
