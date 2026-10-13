"""Render against the real registry and the manifest store."""
from __future__ import annotations

import subprocess
import sys
import uuid

import pytest

from services.prompt_engine import registry as R
from services.prompt_engine import render as RD
from services.prompt_engine import templates as T

SUPPLIER = {"name": "supplier", "type": "string"}


@pytest.fixture()
def prompt(key):
    """A prompt in use: version 1 uses a fragment; version 2 is a rewording."""
    voice = f"{key}_voice"
    R.save_version(voice, template="Be brief.", author="ada", kind="fragment")
    R.save_version(key, template="{{> %s@1}} Rank {{supplier}}." % voice, author="ada",
                   variables=[SUPPLIER],
                   settings={"tool_descriptions": {"lookup": "Look it up.", "pay": "Pay."}})
    R.promote(key, 1, approved_by="bob")
    return key


def _request(key, run=None, **over):
    base = {"prompt_id": key, "vars": {"supplier": "Acme"}, "allowed_tools": ["lookup"],
            "obligations": [],
            "trace": {"run_id": run or f"run-{uuid.uuid4().hex[:8]}", "step_id": "s1",
                      "attempt": 1, "decision_id": "d1"}}
    base.update(over)
    return base


def _count(db, table, where, params):
    with db() as conn:
        cur = conn.cursor()
        cur.execute(f"SELECT count(*) FROM proc.{table} WHERE {where}", params)
        return cur.fetchone()[0]


def test_one_call_returns_the_text_and_a_manifest_naming_every_version(prompt):
    out = RD.render(_request(prompt))

    assert out["instructions"] == "Be brief. Rank Acme."
    assert out["tool_descriptions"] == [{"name": "lookup", "description": "Look it up."}]
    assert out["manifest"]["version"] == 1
    assert out["manifest"]["fragments"] == {f"{prompt}_voice": 1}


def test_the_same_request_twice_is_one_stored_text_and_two_recorded_uses(prompt, db):
    run = f"run-{uuid.uuid4().hex[:8]}"
    first = RD.render(_request(prompt, run))
    second = RD.render(_request(prompt, run))

    assert first == second
    digest = first["manifest"]["hash"]
    assert _count(db, "bp_prompt_render", "render_hash = %s", (digest,)) == 1
    assert _count(db, "bp_prompt_render_use", "run_id = %s", (run,)) == 2


def test_a_run_can_be_replayed_from_its_run_id(prompt):
    run = f"run-{uuid.uuid4().hex[:8]}"
    RD.render(_request(prompt, run))
    retry = _request(prompt, run)
    retry["trace"]["attempt"] = 2
    RD.render(retry)

    uses = RD.renders_for_run(run)
    assert [(u["step_id"], u["attempt"], u["version"]) for u in uses] == [("s1", 1, 1), ("s1", 2, 1)]
    stored = RD.get_render(uses[0]["render_hash"])
    assert stored["rendered"]["instructions"] == "Be brief. Rank Acme."
    assert stored["manifest"]["hash"] == uses[0]["render_hash"]


def test_the_script_lists_a_runs_renders_and_shows_text_only_when_asked(prompt):
    run = f"run-{uuid.uuid4().hex[:8]}"
    RD.render(_request(prompt, run))
    script = str(R.__file__).split("/src/")[0] + "/scripts/prompt_renders_for_run.py"

    plain = subprocess.run([sys.executable, script, run], capture_output=True, text=True)
    shown = subprocess.run([sys.executable, script, run, "--text"], capture_output=True, text=True)

    assert plain.returncode == 0 and f"{prompt} v1" in plain.stdout
    assert "Rank Acme" not in plain.stdout
    assert "Be brief. Rank Acme." in shown.stdout


def test_rolling_back_is_one_call_and_the_very_next_render_sees_it(prompt):
    R.save_version(prompt, template="Order {{supplier}} by price.", author="ada",
                   variables=[SUPPLIER])
    R.promote(prompt, 2, approved_by="bob")
    assert RD.render(_request(prompt))["instructions"] == "Order Acme by price."

    R.rollback(prompt, approved_by="cat")

    out = RD.render(_request(prompt))
    assert (out["instructions"], out["manifest"]["version"]) == ("Be brief. Rank Acme.", 1)


def test_a_named_version_is_rendered_whatever_the_pointer_says(prompt):
    R.save_version(prompt, template="Order {{supplier}} by price.", author="ada",
                   variables=[SUPPLIER])
    assert RD.render(_request(prompt, version=2))["manifest"]["version"] == 2


def test_a_pack_is_included_at_the_version_in_use(prompt):
    pack = f"{prompt}_pack"
    R.save_version(pack, template="Escalate over the limit.", author="ada", kind="fragment")
    with pytest.raises(T.FragmentMissing):
        RD.render(_request(prompt, obligations=[f"include_pack:{pack}"]))   # not in use yet
    R.promote(pack, 1, approved_by="bob")

    out = RD.render(_request(prompt, obligations=[f"include_pack:{pack}"]))
    assert out["instructions"].endswith("\n\nEscalate over the limit.")
    assert out["manifest"]["fragments"][pack] == 1


@pytest.mark.parametrize("change", [
    {"obligations": ["do:anything"]},
    {"vars": {}},
    {"vars": {"supplier": 7}},
    {"vars": {"supplier": "Acme", "extra": 1}},
])
def test_a_refused_render_returns_nothing_and_stores_nothing(prompt, db, change):
    run = f"run-{uuid.uuid4().hex[:8]}"
    with pytest.raises(T.TemplateError):
        RD.render(_request(prompt, run, **change))
    assert _count(db, "bp_prompt_render_use", "run_id = %s", (run,)) == 0


def test_nothing_to_render_says_so_with_the_contract_codes(prompt, key):
    with pytest.raises(R.PointerUnresolved) as e:
        RD.render(_request(f"{key}_voice"))
    assert e.value.code == "POINTER_UNRESOLVED"
    with pytest.raises(R.VersionNotFound) as e:
        RD.render(_request(prompt, version=9))
    assert e.value.code == "VERSION_NOT_FOUND"


def test_a_redacted_value_is_in_neither_the_answer_nor_the_store(key, db):
    R.save_version(key, template="Customer {{customer}}.", author="ada",
                   variables=[{"name": "customer", "type": "object"}])
    R.promote(key, 1, approved_by="bob")
    out = RD.render(_request(key, vars={"customer": {"name": "Ada", "ssn": "123-45-6789"}},
                             allowed_tools=[], obligations=["redact:customer.ssn"]))

    assert "123-45-6789" not in RD.canonical(out)
    with db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT manifest::text || rendered::text FROM proc.bp_prompt_render "
                    "WHERE render_hash = %s", (out["manifest"]["hash"],))
        assert "123-45-6789" not in cur.fetchone()[0]


def test_the_record_of_uses_cannot_be_rewritten(prompt, db):
    run = f"run-{uuid.uuid4().hex[:8]}"
    RD.render(_request(prompt, run))
    with pytest.raises(Exception, match="append-only"):
        with db() as conn:
            conn.cursor().execute(
                "DELETE FROM proc.bp_prompt_render_use WHERE run_id = %s", (run,))


# --------------------------------------------------------------------------
# Retention
# --------------------------------------------------------------------------
@pytest.fixture()
def days(monkeypatch):
    from src.services import governed_limits

    state = {"value": 7}

    def limit(policy, rule, **kw):
        assert (policy, rule) == ("prompt_engine", "manifest_retention_days")
        if isinstance(state["value"], Exception):
            raise state["value"]
        return state["value"]

    monkeypatch.setattr(governed_limits, "limit", limit)
    return state


def _age(db, digest, days_old):
    with db() as conn:
        conn.cursor().execute(
            "UPDATE proc.bp_prompt_render SET last_used_at = now() - make_interval(days => %s) "
            "WHERE render_hash = %s", (days_old, digest))


def test_a_render_unused_past_the_window_loses_its_text_and_keeps_its_record(prompt, db, days):
    run = f"run-{uuid.uuid4().hex[:8]}"
    digest = RD.render(_request(prompt, run))["manifest"]["hash"]
    _age(db, digest, 8)

    assert RD.sweep() >= 1

    kept = RD.get_render(digest)
    assert kept["rendered"] is None and kept["text_removed_at"] is not None
    assert kept["manifest"]["version"] == 1 and kept["manifest"]["hash"] == digest
    assert kept["manifest"]["fragments"] == {f"{prompt}_voice": 1}
    assert [u["render_hash"] for u in RD.renders_for_run(run)] == [digest]
    with db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT to_jsonb(r)::text FROM proc.bp_prompt_render r "
                    "WHERE render_hash = %s", (digest,))
        assert "Acme" not in cur.fetchone()[0], "the text was meant to be gone"


def test_rendering_the_same_text_again_after_it_aged_out_stores_it_again(prompt, db, days):
    digest = RD.render(_request(prompt))["manifest"]["hash"]
    _age(db, digest, 8)
    RD.sweep()

    assert RD.render(_request(prompt))["manifest"]["hash"] == digest
    again = RD.get_render(digest)
    assert again["rendered"]["instructions"] == "Be brief. Rank Acme."
    assert again["text_removed_at"] is None


def test_a_hash_that_was_never_rendered_is_not_found(db):
    with pytest.raises(RD.RenderNotFound):
        RD.get_render("sha256:" + "0" * 64)


def test_a_render_inside_the_window_is_kept_and_using_it_again_resets_the_clock(prompt, db, days):
    digest = RD.render(_request(prompt))["manifest"]["hash"]
    _age(db, digest, 6)
    RD.sweep()
    assert RD.get_render(digest)["rendered"] is not None

    _age(db, digest, 30)
    RD.render(_request(prompt))
    RD.sweep()
    assert RD.get_render(digest)["rendered"] is not None


@pytest.mark.parametrize("stated", [0, -1, None])
def test_a_window_of_nothing_or_no_window_at_all_sweeps_nothing(prompt, db, days, stated):
    from src.services.governed_limits import LimitUnavailable

    digest = RD.render(_request(prompt))["manifest"]["hash"]
    _age(db, digest, 400)
    days["value"] = stated

    with pytest.raises(LimitUnavailable):
        RD.sweep()
    assert RD.get_render(digest)["rendered"] is not None
