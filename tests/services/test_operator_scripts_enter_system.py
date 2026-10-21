"""Operator scripts act for the platform (D-47): with row-level security a script that names no
company sees no rows. Every runnable script that can reach the database must say so."""
import pathlib
import re

SCRIPTS = pathlib.Path(__file__).resolve().parents[2] / "scripts"
RUNS = re.compile(r"^if __name__ == .__main__.:", re.M)
TOUCHES_DB = re.compile(r"from (src|services|agents|orchestration)|import (src|services)|get_conn|psycopg2|create_engine")


def _runnable_db_scripts():
    for p in sorted(SCRIPTS.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        t = p.read_text(errors="ignore")
        if RUNS.search(t) and TOUCHES_DB.search(t):
            yield p, t


def test_every_runnable_database_script_enters_system_mode():
    missing = [str(p.relative_to(SCRIPTS)) for p, t in _runnable_db_scripts() if "enter_system" not in t]
    assert not missing, missing


def test_the_two_unguarded_audit_scripts_enter_system_mode_too():
    for name in ("rederive_findings_audit_1.py", "rederive_findings_audit_2.py"):
        assert "enter_system()" in (SCRIPTS / name).read_text()
