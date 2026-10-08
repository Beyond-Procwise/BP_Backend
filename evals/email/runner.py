"""Run the golden cases for every email family against the throwaway database.

    python -m evals.email.runner [--family negotiation_counter] [--report evals/email/report.json]

Each case seeds rows, scripts a FAKE model (good output and bad), runs the real drafting path against
the real tables and the real policy rows, optionally has a reviewer answer assumptions, moves a fact
before the send, and runs the real send guard. Then every expectation is checked.

What this does and does not prove: it proves the plumbing and the validators against realistic data.
It says nothing about how well a real model classifies, plans or judges
(specs/2026-10-08-email-assurance-pending-live-verification.md).
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).parent
REPO = HERE.parents[1]
for p in (str(REPO), str(REPO / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

TABLES = ["supplier_response", "bp_supplier", "workflow_email_tracking", "bp_approval"]


# --- expectations ---------------------------------------------------------------------------------

def resolve(root: Any, path: str) -> Any:
    """``a.b[*].c`` -> the value, or a list of values when a ``[*]`` is crossed. Missing -> None."""

    cur = [root]
    fan = False
    for part in path.split("."):
        star = part.endswith("[*]")
        key = part[:-3] if star else part
        nxt = []
        for node in cur:
            val = node.get(key) if isinstance(node, dict) else None
            if star:
                fan = True
                nxt.extend(val if isinstance(val, list) else [])
            else:
                nxt.append(val)
        cur = nxt
    return cur if fan else cur[0]


def check(value: Any, want: Any) -> Tuple[bool, str]:
    """One expectation. A bare value means equals; a dict names an operator."""

    if not isinstance(want, dict):
        return value == want, f"expected {want!r}, got {value!r}"
    (op, arg), = want.items()
    if op == "has":
        return isinstance(value, list) and arg in value, f"expected {arg!r} among {value!r}"
    if op == "lacks":
        return not (isinstance(value, list) and arg in value), f"expected {arg!r} NOT among {value!r}"
    if op == "contains":
        return isinstance(value, str) and arg in value, f"expected {arg!r} in {value!r}"
    if op == "not_contains":
        return not (isinstance(value, str) and arg in value), f"expected {arg!r} NOT in {value!r}"
    if op == "len":
        return isinstance(value, (list, dict, str)) and len(value) == arg, f"expected length {arg}, got {value!r}"
    if op == "null":
        return (value is None) == arg, f"expected null={arg}, got {value!r}"
    if op == "in":
        return value in arg, f"expected one of {arg!r}, got {value!r}"
    if op == "gte":
        return value is not None and value >= arg, f"expected >= {arg!r}, got {value!r}"
    raise ValueError(f"unknown operator {op!r}")


def evaluate(expect: Dict[str, Any], ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for path, want in expect.items():
        path = path.split("#")[0].strip()          # "x.y #2" repeats a key; the marker is not part of the path
        ok, detail = check(resolve(ctx, path), want)
        out.append({"check": path, "ok": ok, **({} if ok else {"detail": detail})})
    return out


# --- the world a case runs in --------------------------------------------------------------------------

def _merge(a: Dict[str, List[dict]], b: Dict[str, List[dict]]) -> Dict[str, List[dict]]:
    out = {k: list(v) for k, v in a.items()}
    for k, v in (b or {}).items():
        out.setdefault(k, []).extend(v)
    return out


def snapshot(conn) -> None:
    """Remember the policy and prompt rows exactly as the migrations left them."""
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS public._pristine_policy, public._pristine_prompt")
        cur.execute("CREATE TABLE public._pristine_policy AS SELECT * FROM proc.bp_policy")
        cur.execute("CREATE TABLE public._pristine_prompt AS SELECT * FROM proc.bp_prompt")


def restore(conn) -> None:
    """Undo whatever the previous case did to a policy or prompt row (modes, orderings, deletions)."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM proc.bp_policy")
        cur.execute("INSERT INTO proc.bp_policy OVERRIDING SYSTEM VALUE SELECT * FROM public._pristine_policy")
        cur.execute("DELETE FROM proc.bp_prompt")
        cur.execute("INSERT INTO proc.bp_prompt OVERRIDING SYSTEM VALUE SELECT * FROM public._pristine_prompt")


def seed(conn, rows: Dict[str, List[dict]]) -> None:
    with conn.cursor() as cur:
        cur.execute("TRUNCATE " + ", ".join(f"proc.{t}" for t in TABLES) +
                    ", email_agent.bp_draft_outcome, email_agent.bp_draft_capture RESTART IDENTITY CASCADE")
        # A standalone sequence (supplier_response.id in production) is not reset by RESTART IDENTITY, which only
        # reaches sequences a column owns. Without this, row ids differ from one case to the next.
        cur.execute("DO $$ BEGIN IF to_regclass('proc.supplier_response_id_seq') IS NOT NULL THEN "
                    "ALTER SEQUENCE proc.supplier_response_id_seq RESTART WITH 1; END IF; END $$")
        for table, items in rows.items():
            if table not in TABLES:
                raise ValueError(f"cannot seed {table!r}")
            for item in items:
                cols = list(item)
                cur.execute(f"INSERT INTO proc.{table} ({', '.join(cols)}) VALUES ({', '.join(['%s'] * len(cols))})",
                            [json.dumps(v) if isinstance(v, (dict, list)) else v for v in item.values()])


def patch_policies(conn, patches) -> None:
    """Case-specific edits to one nested value of a policy row: [{policy, path, value}]."""
    with conn.cursor() as cur:
        for p in patches or []:
            cur.execute("UPDATE proc.bp_policy SET policy_details = jsonb_set(policy_details, %s::text[], %s::jsonb) "
                        "WHERE policy_name = %s", ("{" + ",".join(["rules", *p["path"]]) + "}", json.dumps(p["value"]), p["policy"]))
            if cur.rowcount != 1:
                raise ValueError(f"policy {p['policy']!r} not found")


def make_engine(conn):
    from engines.policy_engine import PolicyEngine

    @contextmanager
    def factory():
        yield conn
    return PolicyEngine(agent_nick=None, connection_factory=factory)


class FakeModel:
    """Scripted stage outputs. A stage the case does not script behaves like a dead model."""

    ROUTES = (("You decide what kind", "classify"), ("You plan procurement", "plan"),
              ("You score a drafted", "judge"), ("You correct a draft", "repair"))

    def __init__(self, script: Dict[str, Any]):
        self.script, self.calls = script or {}, []

    def stage_of(self, system: str) -> str:
        for prefix, name in self.ROUTES:
            if system.startswith(prefix):
                return name
        return "other"

    def __call__(self, **kw) -> Dict[str, Any]:
        system = kw["messages"][0]["content"]
        stage = self.stage_of(system)
        self.calls.append(stage)
        if stage == "other":                       # the counter path writes its email through the same client
            stage = "compose"
        if stage not in self.script:
            raise RuntimeError(f"model unavailable for stage {stage!r}")
        out = self.script[stage]
        return {"message": {"content": out if isinstance(out, str) else json.dumps(out)}}


def build_agent(conn, model: FakeModel, compose: str, prompts: bool):
    from agents import email_drafting_agent as module
    from agents.email_drafting_agent import EmailDraftingAgent

    agent = EmailDraftingAgent()

    @contextmanager
    def db():
        yield conn
    agent.agent_nick.get_db_connection = db
    agent.agent_nick.policy_engine = make_engine(conn)
    agent.call_ollama = model

    def resolve_prompt(name, **_):
        if not prompts:
            return None
        with conn.cursor() as cur:
            cur.execute("SELECT prompts_desc->>'prompt_template' FROM proc.bp_prompt WHERE prompt_name = %s", (name,))
            row = cur.fetchone()
        return row[0] if row else None
    agent.resolve_prompt = resolve_prompt
    return agent, module


def run_case(case: Dict[str, Any], conn, base: Dict[str, Any]) -> Dict[str, Any]:
    from src.services.draft_assurance import capture
    from src.services import email_dispatch_guard as guard, guardrail
    from src.services.approval_content import content_hash

    conn.autocommit = True
    restore(conn)
    seed(conn, _merge(base["seed"] if case.get("base", True) else {}, case.get("seed", {})))
    patch_policies(conn, case.get("policy_patch"))
    for name in case.get("policy_delete", []):
        with conn.cursor() as cur:
            cur.execute("DELETE FROM proc.bp_policy WHERE policy_name = %s", (name,))
    spec = case["run"]
    model = FakeModel(case.get("model", {}))
    agent, module = build_agent(conn, model, case.get("model", {}).get("compose", ""), case.get("prompts", True))
    saved_module = (module._chat, module._current_rfq_date)      # restored below: cases must not leak into each other
    module._chat = lambda m, s, u, **k: (model.calls.append("compose"), case.get("model", {}).get("compose", ""))[1]
    module._current_rfq_date = lambda: "20260101"
    stored: List[dict] = []
    agent._store_draft = lambda d: stored.append(d)
    agent._record_learning_events = lambda *a, **k: None
    try:
        return _execute(case, conn, spec, agent, model, stored)
    finally:
        module._chat, module._current_rfq_date = saved_module


def _execute(case, conn, spec, agent, model, stored) -> Dict[str, Any]:
    from src.services.draft_assurance import capture
    from src.services import email_dispatch_guard as guard, guardrail
    from src.services.approval_content import content_hash

    ctx: Dict[str, Any] = {}
    payload = copy.deepcopy(spec.get("payload", {}))
    if spec["path"] == "counter":
        from agents.base_agent import AgentContext
        c = AgentContext(workflow_id=payload.get("workflow_id", "wf-1"), agent_id="email_drafting",
                         user_id=spec.get("user_id", "AgentNick"), input_data=payload)
        agent._handle_negotiation_counter(c, payload)
        draft = stored[0]
    elif spec["path"] == "decision":
        draft = agent.from_decision(payload)
    elif spec["path"] == "prompt":
        draft = agent.from_prompt(spec["request"], context=payload)
    else:
        raise ValueError(f"unknown path {spec['path']!r}")
    rec = draft["assurance"]
    ctx.update(draft=draft, rec=rec, calls=model.calls)
    uid = draft["unique_id"]
    capture.record_draft(conn, draft)
    raw = capture.load_raw(conn, uid)
    ctx["row"] = raw
    ctx["view"] = capture.to_view(raw) if raw else None

    for step in case.get("reviewer", []):
        ctx["confirm"] = capture.confirm_assumptions(conn, uid, step["confirmations"], step.get("by", "reviewer@acme.test"))
    for m in case.get("before_send", []):
        with conn.cursor() as cur:
            sets = ", ".join(f"{k} = %s" for k in m["set"])
            wheres = " AND ".join(f"{k} = %s" for k in m["where"])
            cur.execute(f"UPDATE proc.{m['table']} SET {sets} WHERE {wheres}", [*m["set"].values(), *m["where"].values()])
    if "send" in case:
        s = case["send"]
        if s.get("mode"):
            patch_policies(conn, [{"policy": "EmailFamily_" + rec["family_id"], "path": ["mode"], "value": s["mode"]}])
        draft["assurance"] = {**rec, "mode": s.get("mode") or rec.get("mode")}
        recipients, subject, body = draft.get("recipients") or [], draft.get("subject"), draft.get("body")
        approval = {"approval_id": 1, "status": "approved", **s.get("approval", {"actioned_by": "buyer@acme.test"}),
                    "grounding": {"content_hash": content_hash({"recipients": recipients, "subject": subject,
                                                                "body": body, "attachments": None})}}
        saved = (guard.check_recipient_and_sensitivity, guardrail.authorize)
        guard.check_recipient_and_sensitivity = lambda **k: guardrail.Decision(allowed=True, reason="eval: not under test", evidence={})
        guardrail.authorize = lambda *a, **k: guardrail.Decision(allowed=True, reason="eval: not under test")
        try:
            d = guard.check_dispatch(conn=conn, draft=draft, recipients=recipients, subject=subject, body=body,
                                     attachments=None, principal=SimpleNamespace(subject=s.get("principal", "sub-nick")),
                                     policy_engine=make_engine(conn), approval_lookup=lambda **_: approval,
                                     internal_domains=[])
        finally:
            guard.check_recipient_and_sensitivity, guardrail.authorize = saved
        ctx["send"] = {"allowed": d.allowed, "reason": d.reason, "evidence": d.evidence,
                       "changed": [c["fact"] for c in (d.evidence.get("facts_recheck") or {}).get("changed", [])]}
        if d.allowed and "record_sent" in s:
            capture.record_sent(conn, uid, s["record_sent"].get("body", body),
                                reviewed_by=d.evidence.get("reviewed_by"), sent_by=s.get("principal", "sub-nick"))
            with conn.cursor() as cur:
                cur.execute("SELECT edit_class, reviewed_by, sent_by FROM email_agent.bp_draft_outcome WHERE outcome = 'sent'")
                r = cur.fetchone()
            ctx["sent"] = {"edit_class": r[0], "reviewed_by": r[1], "sent_by": r[2]} if r else None
    checks = evaluate(case["expect"], ctx)
    return {"id": case["id"], "family": case["family"], "passed": all(c["ok"] for c in checks), "checks": checks}


# --- running everything --------------------------------------------------------------------------------

def load_cases(family: Optional[str] = None) -> List[Dict[str, Any]]:
    cases = []
    for f in sorted((HERE / "cases").glob("*/*.json")):
        case = json.loads(f.read_text())
        case.setdefault("family", f.parent.name)
        case["_file"] = str(f.relative_to(REPO))
        if family in (None, case["family"]):
            cases.append(case)
    return cases


def run_all(conn, family: Optional[str] = None) -> Dict[str, Any]:
    base = json.loads((HERE / "base_seed.json").read_text())
    conn.autocommit = True
    snapshot(conn)
    results = []
    for case in load_cases(family):
        try:
            results.append(run_case(case, conn, base))
        except Exception as exc:  # noqa: BLE001 - a case that crashes FAILS, with the reason
            results.append({"id": case["id"], "family": case["family"], "passed": False,
                            "checks": [{"check": "ran without error", "ok": False,
                                        "detail": f"{type(exc).__name__}: {exc}"}]})
    fams: Dict[str, Dict[str, Any]] = {}
    for r in results:
        f = fams.setdefault(r["family"], {"cases": 0, "passed": 0, "checks": 0, "checks_passed": 0})
        f["cases"] += 1
        f["passed"] += r["passed"]
        f["checks"] += len(r["checks"])
        f["checks_passed"] += sum(c["ok"] for c in r["checks"])
    for f in fams.values():
        f["pass_rate"] = round(f["passed"] / f["cases"], 4) if f["cases"] else 0.0
    return {"families": fams, "results": results}


def compare_to_baseline(report: Dict[str, Any], baseline: Dict[str, Any]) -> List[str]:
    """Reasons this report must block a merge. Empty means it may pass."""

    problems = []
    for fam, want in baseline["families"].items():
        got = report["families"].get(fam)
        if got is None:
            problems.append(f"{fam}: no cases ran")
            continue
        if got["cases"] < want["min_cases"]:
            problems.append(f"{fam}: {got['cases']} cases, baseline requires at least {want['min_cases']} (cases were removed)")
        if got["pass_rate"] < want["min_pass_rate"]:
            problems.append(f"{fam}: pass rate {got['pass_rate']:.0%} is below the baseline {want['min_pass_rate']:.0%}")
    for fam in report["families"]:
        if fam not in baseline["families"]:
            problems.append(f"{fam}: family has cases but no baseline entry")
    return problems


def main(argv: Optional[List[str]] = None) -> int:
    import psycopg2
    from evals.email import db

    ap = argparse.ArgumentParser()
    ap.add_argument("--family")
    ap.add_argument("--report", default=str(HERE / "report.json"))
    args = ap.parse_args(argv)
    with db.database() as dsn:
        conn = psycopg2.connect(dsn)
        db.load(conn)
        report = run_all(conn, args.family)
    Path(args.report).write_text(json.dumps(report, indent=1))
    print(f"{'family':24} {'cases':>6} {'passed':>7} {'rate':>6}")
    for fam, f in sorted(report["families"].items()):
        print(f"{fam:24} {f['cases']:>6} {f['passed']:>7} {f['pass_rate']:>6.0%}")
    for r in report["results"]:
        if not r["passed"]:
            print(f"\nFAIL {r['family']}/{r['id']}")
            for c in r["checks"]:
                if not c["ok"]:
                    print(f"   - {c['check']}: {c.get('detail')}")
    problems = [] if args.family else compare_to_baseline(report, json.loads((HERE / "baseline.json").read_text()))
    for p in problems:
        print("BLOCK:", p)
    return 1 if problems or any(not r["passed"] for r in report["results"]) else 0


if __name__ == "__main__":
    sys.exit(main())
