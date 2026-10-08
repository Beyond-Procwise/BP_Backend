"""Rehearse the whole email-assurance DDL pack on a throwaway COPY of bp_sqldb's structure.

    python -m evals.email.rehearsal [--log evals/email/rehearsal-log.md]

Needs DB_HOST/DB_USER/DB_PASSWORD (and DB_PORT) for READ access to bp_sqldb, Docker, psql and pg_dump.
What it touches on bp_sqldb: nothing but a schema-only pg_dump of seven tables and a data dump of the policy and
prompt tables (read-only; a brief ACCESS SHARE lock). Everything else happens inside a disposable container.

Steps, each recorded: restore the copy -> schema fingerprint A -> apply the pack exactly as an operator would
(psql -v ON_ERROR_STOP=1 -1 -f, one file at a time) -> run the golden evals and the whole eval test suite against
the copy -> roll the pack back in reverse -> schema fingerprint B -> A must equal B -> re-apply.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import List, Tuple

from evals.email import db

REPO = db.REPO
PG16 = "/usr/lib/postgresql/16/bin"
SEQUENCES = ["supplier_response_id_seq"]      # a default that names a standalone sequence needs it in the copy
TABLES = ["supplier_response", "bp_supplier", "workflow_email_tracking", "bp_policy", "bp_prompt", "bp_approval", "bp_mailbox_binding", "bp_agent_actions"]   # bp_agent_actions: the send guard counts a user's sends there
DATA_TABLES = ["bp_policy", "bp_prompt"]


def sh(cmd: List[str], env=None, check=True, timeout=600) -> subprocess.CompletedProcess:
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
    if check and r.returncode != 0:
        raise SystemExit(f"FAILED: {' '.join(cmd[:4])}...\n{r.stderr[-1500:]}")
    return r


def fingerprint(container: str, dsn_env: dict) -> Tuple[str, str]:
    """A hash and the text of the schema (no data, no owners, no privileges) of every schema the pack may touch."""
    out = sh(["docker", "exec", container, "pg_dump", "-U", "postgres", "-d", "postgres", "-s", "--no-owner", "--no-privileges",
              "-n", "proc", "-n", "email_agent"], check=False).stdout
    text = "\n".join(l for l in out.splitlines() if not l.startswith(("--", "SET ", "SELECT pg_catalog", "\\restrict", "\\unrestrict")) and l.strip())     # pg_dump stamps each dump with a random \restrict token
    return hashlib.sha256(text.encode()).hexdigest()[:16], text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(REPO / "evals/email/rehearsal-log.md"))
    ap.add_argument("--source-db", default="bp_sqldb")
    ap.add_argument("--keep", action="store_true", help="stop after applying the pack and leave the container running, for debugging")
    args = ap.parse_args(argv)
    log: List[str] = []

    def say(line=""):
        print(line, flush=True)
        log.append(line)

    host, user, pw = os.environ["DB_HOST"], os.environ["DB_USER"], os.environ["DB_PASSWORD"]
    port = os.environ.get("DB_PORT", "5432")
    penv = {**os.environ, "PGPASSWORD": pw}
    name = "bp-rehearsal-" + uuid.uuid4().hex[:6]
    say(f"# Rehearsal of the email-assurance DDL pack on a copy of `{args.source_db}`'s structure\n")
    say(f"- run at {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}; container `postgres:16-alpine`")
    sh(["docker", "run", "-d", "--rm", "--name", name, "-e", "POSTGRES_PASSWORD=eval", "-p", "127.0.0.1::5432", "postgres:16-alpine"])
    try:
        hostport = sh(["docker", "port", name, "5432/tcp"]).stdout.strip().splitlines()[0].rsplit(":", 1)[1]
        dsn = f"postgresql://postgres:eval@127.0.0.1:{hostport}/postgres"
        import psycopg2
        for _ in range(60):
            try:
                psycopg2.connect(dsn, connect_timeout=2).close(); break
            except Exception:  # noqa: BLE001
                time.sleep(0.5)
        # ---- 1. the copy -----------------------------------------------------------------------------
        say("\n## 1. The copy (read-only dump of the real structure)\n")
        flags = [f"-t proc.{t}" for t in TABLES + SEQUENCES]
        dump_s = sh([f"{PG16}/pg_dump", "-h", host, "-p", port, "-U", user, "-d", args.source_db, "-s", "--no-owner", "--no-privileges",
                     *" ".join(flags).split()], env=penv).stdout
        dump_d = sh([f"{PG16}/pg_dump", "-h", host, "-p", port, "-U", user, "-d", args.source_db, "-a", "--no-owner", "--no-privileges",
                     *" ".join(f"-t proc.{t}" for t in DATA_TABLES).split()], env=penv).stdout
        sh(["docker", "exec", "-i", name, "psql", "-U", "postgres", "-d", "postgres", "-c", "CREATE SCHEMA IF NOT EXISTS proc"])
        for label, sql in (("schema", dump_s), ("policy+prompt data", dump_d)):
            r = subprocess.run(["docker", "exec", "-i", name, "psql", "-U", "postgres", "-d", "postgres"], input=sql, capture_output=True, text=True)
            errs = sorted({l for l in r.stderr.splitlines() if "ERROR" in l})
            say(f"- restored {label}: {len(sql.splitlines())} lines; errors: {len(errs)}")
            for e in errs[:8]:
                say(f"  - `{e[:160]}`  (a constraint pointing at a table that is not part of the copy)")
        n = sh(["docker", "exec", name, "psql", "-U", "postgres", "-d", "postgres", "-Atc",
                "select (select count(*) from proc.bp_policy), (select count(*) from proc.bp_prompt), "
                "(select count(*) from information_schema.tables where table_schema='proc')"]).stdout.strip()
        say(f"- copy holds: policies / prompts / proc tables = `{n}`")
        # ---- 2. fingerprint before -------------------------------------------------------------------------
        fa, text_a = fingerprint(name, {})
        say(f"\n## 2. Schema fingerprint BEFORE the pack: `{fa}`\n")
        # ---- 3. apply the pack as an operator would --------------------------------------------------------
        say("## 3. Apply the pack, one file at a time (`psql -v ON_ERROR_STOP=1 -f`; each file is its own transaction)\n")
        say("| file | result |\n|---|---|")
        for f in db.MIGRATIONS:
            t0 = time.time()
            r = subprocess.run(["docker", "exec", "-i", name, "psql", "-U", "postgres", "-d", "postgres", "-v", "ON_ERROR_STOP=1"],
                               input=(db.SQL / f).read_text(), capture_output=True, text=True)
            ok = r.returncode == 0
            say(f"| `{f}` | {'ok' if ok else 'FAILED: ' + r.stderr.strip()[-200:]} ({time.time() - t0:.1f}s) |")
            if not ok:
                return 1
        fp_applied, _ = fingerprint(name, {})
        say(f"\n- fingerprint after apply: `{fp_applied}` (differs from before: {fp_applied != fa})")
        if args.keep:
            say(f"\nKEPT container {name}: EMAIL_EVAL_DSN={dsn}  (remove with: docker rm -f {name})")
            args.keep_name = name
            return 0
        # ---- 4. tests against the copy ------------------------------------------------------------------------
        say("\n## 4. The eval suite, run against the copy\n")
        env = {**os.environ, "EMAIL_EVAL_DSN": dsn, "EMAIL_EVAL_EXISTING_SCHEMA": "1", "CUDA_VISIBLE_DEVICES": "", "OLLAMA_HOST": "http://127.0.0.1:9", "PYTHONPATH": str(REPO)}
        env.pop("PYTEST_CURRENT_TEST", None)
        gold = subprocess.run([sys.executable, "-m", "evals.email.runner", "--report", "/tmp/rehearsal-report.json"], env=env, capture_output=True, text=True, cwd=REPO)
        say("```\n" + "\n".join(l for l in gold.stdout.splitlines() if l.strip()) + "\n```")
        # the eval suite loads the migrations itself; hand it a database that does NOT have them yet
        say("- (the runner re-applies the migrations itself, which also proves they are idempotent on the copy)")
        pt = subprocess.run([sys.executable, "-m", "pytest", "tests/email_evals", "-q", "-p", "no:cacheprovider", "-W", "ignore"], env=env, capture_output=True, text=True, cwd=REPO)
        say(f"- `pytest tests/email_evals` against the copy: **{(pt.stdout.strip().splitlines() or ['(no output)'])[-1]}**")
        for l in pt.stdout.splitlines():
            if l.startswith("FAILED"):
                say(f"  - {l[:200]}")
        # ---- 5. roll back ----------------------------------------------------------------------------------------
        say("\n## 5. Roll the pack back, newest first\n")
        say("| file | result |\n|---|---|")
        for f in reversed(db.MIGRATIONS):
            r = subprocess.run(["docker", "exec", "-i", name, "psql", "-U", "postgres", "-d", "postgres", "-v", "ON_ERROR_STOP=1"],
                               input=(db.SQL / f.replace(".sql", "_rollback.sql")).read_text(), capture_output=True, text=True)
            say(f"| `{f.replace('.sql', '_rollback.sql')}` | {'ok' if r.returncode == 0 else 'FAILED: ' + r.stderr.strip()[-200:]} |")
            if r.returncode != 0:
                return 1
        fb, text_b = fingerprint(name, {})
        left = sh(["docker", "exec", name, "psql", "-U", "postgres", "-d", "postgres", "-Atc",
                   "select (select count(*) from pg_roles where rolname like 'email_agent%'), "
                   "(select count(*) from information_schema.schemata where schema_name='email_agent'), "
                   "(select count(*) from proc.bp_policy where created_by='email_assurance_migration'), "
                   "(select count(*) from proc.bp_prompt where created_by like 'deploy/sql/2026-10-08_email%')"]).stdout.strip()
        say(f"\n- fingerprint after rollback: `{fb}`  **equals the one before: {fb == fa}**")
        say(f"- left behind (roles / email_agent schema / policy rows / prompt rows): `{left}`")
        if fb != fa:
            import difflib
            for l in list(difflib.unified_diff(text_a.splitlines(), text_b.splitlines(), "before", "after", lineterm=""))[:30]:
                say("    " + l)
        # ---- 6. re-apply ------------------------------------------------------------------------------------------
        for f in db.MIGRATIONS:
            r = subprocess.run(["docker", "exec", "-i", name, "psql", "-U", "postgres", "-d", "postgres", "-v", "ON_ERROR_STOP=1"],
                               input=(db.SQL / f).read_text(), capture_output=True, text=True)
            if r.returncode != 0:
                say(f"- RE-APPLY FAILED at `{f}`: {r.stderr.strip()[-200:]}")
                return 1
        fp2, _ = fingerprint(name, {})
        say(f"\n## 6. Re-applied after rollback: fingerprint `{fp2}` (same as the first apply: {fp2 == fp_applied})")
        good = gold.returncode == 0 and pt.returncode == 0 and fb == fa and fp2 == fp_applied
        say(f"\n**Overall: {'PASS' if good else 'FAIL'}**")
        return 0 if good else 1
    finally:
        if not getattr(args, "keep_name", None):
            sh(["docker", "rm", "-f", name], check=False)
        Path(args.log).write_text("\n".join(log) + "\n")


if __name__ == "__main__":
    sys.exit(main())
