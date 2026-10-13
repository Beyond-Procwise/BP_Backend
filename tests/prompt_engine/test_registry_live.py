"""The registry: versions that never change, and a pointer that moves."""
from __future__ import annotations

import pytest

from services.prompt_engine import registry as R


def _save(key, text="Rank the suppliers.", author="ada", **kw):
    return R.save_version(key, template=text, author=author, **kw)


# --------------------------------------------------------------------------
# Versions
# --------------------------------------------------------------------------
def test_the_first_save_is_version_one_and_records_who_and_why(key):
    assert _save(key, change_note="first wording") == 1

    v = R.get_version(key, 1)
    assert (v["template"], v["author"], v["change_note"], v["syntax"]) == (
        "Rank the suppliers.", "ada", "first wording", "v1")
    assert v["content_sha256"] and v["created_at"]


def test_every_save_is_a_new_version_and_the_old_one_is_still_there(key):
    _save(key, "one")
    assert _save(key, "two", author="bob") == 2

    assert R.get_version(key, 1)["template"] == "one"
    assert R.get_version(key, 2)["template"] == "two"
    assert [v["version"] for v in R.versions(key)] == [1, 2]


def test_a_saved_version_cannot_be_edited_or_deleted(key, db):
    _save(key)
    for statement in (
        "UPDATE proc.bp_prompt_version SET template = 'changed' WHERE prompt_key = %s",
        "DELETE FROM proc.bp_prompt_version WHERE prompt_key = %s",
    ):
        with pytest.raises(Exception, match="cannot be changed once saved"):
            with db() as conn:
                cur = conn.cursor()
                cur.execute(statement, (key,))
    assert R.get_version(key, 1)["template"] == "Rank the suppliers."


def test_a_prompt_cannot_be_deleted_or_change_kind(key, db):
    _save(key, kind="fragment")
    with pytest.raises(Exception, match="never deleted"):
        with db() as conn:
            conn.cursor().execute(
                "DELETE FROM proc.bp_prompt_registry WHERE prompt_key = %s", (key,))
    with pytest.raises(R.KindMismatch):
        _save(key, "as a prompt", kind="prompt")


def test_the_same_wording_has_the_same_hash_and_different_wording_does_not(key):
    _save(key, "one")
    _save(key, "one")
    _save(key, "one ")
    h = [v["content_sha256"] for v in R.versions(key)]
    assert h[0] == h[1] != h[2]


def test_a_missing_prompt_or_version_says_which(key):
    with pytest.raises(R.PromptNotFound):
        R.get_version(key, 1)
    _save(key)
    with pytest.raises(R.VersionNotFound):
        R.get_version(key, 2)


@pytest.mark.parametrize("bad", ["", "Has Spaces", "UPPER", "../x", "-leading"])
def test_a_key_that_is_not_a_plain_name_is_refused(db, bad):
    with pytest.raises(R.InvalidKey):
        _save(bad)


# --------------------------------------------------------------------------
# The pointer
# --------------------------------------------------------------------------
def test_a_saved_version_is_not_in_use_until_it_is_promoted(key):
    _save(key)
    with pytest.raises(R.PointerUnresolved):
        R.resolve(key)


def test_promotion_puts_a_version_in_use_and_records_the_approver(key):
    _save(key, author="ada")
    R.promote(key, 1, approved_by="bob", note="reviewed")

    assert R.resolve(key)["version"] == 1
    move = R.pointer_history(key)[-1]
    assert (move["from_version"], move["to_version"], move["change"],
            move["approved_by"], move["note"]) == (None, 1, "promoted", "bob", "reviewed")


def test_the_author_cannot_promote_their_own_version(key):
    _save(key, author="ada")
    with pytest.raises(R.SelfApproval):
        R.promote(key, 1, approved_by="ada")
    with pytest.raises(R.PointerUnresolved):
        R.resolve(key)


def test_promotion_needs_a_named_approver(key):
    _save(key)
    for nobody in ("", "   ", None):
        with pytest.raises(R.ApproverRequired):
            R.promote(key, 1, approved_by=nobody)


def test_a_version_that_does_not_exist_cannot_be_promoted(key):
    _save(key)
    with pytest.raises(R.VersionNotFound):
        R.promote(key, 7, approved_by="bob")


def test_rollback_is_one_call_and_the_next_resolve_sees_it(key):
    _save(key, "one", author="ada")
    _save(key, "two", author="ada")
    R.promote(key, 1, approved_by="bob")
    R.promote(key, 2, approved_by="bob")
    assert R.resolve(key)["template"] == "two"

    assert R.rollback(key, approved_by="cat", note="two misfired") == 1

    assert R.resolve(key)["template"] == "one"
    last = R.pointer_history(key)[-1]
    assert (last["from_version"], last["to_version"], last["change"]) == (2, 1, "rolled_back")


def test_the_author_may_roll_back_to_a_version_someone_else_already_approved(key):
    """Rolling back is the emergency lever. The version being returned to was
    approved by a second person when it first went live; the rule it protects
    has already been met."""
    _save(key, "one", author="ada")
    _save(key, "two", author="bob")
    R.promote(key, 1, approved_by="bob")
    R.promote(key, 2, approved_by="ada")

    assert R.rollback(key, approved_by="ada") == 1


def test_there_is_nothing_to_roll_back_to_before_a_second_promotion(key):
    _save(key)
    with pytest.raises(R.NothingToRollBackTo):
        R.rollback(key, approved_by="bob")
    R.promote(key, 1, approved_by="bob")
    with pytest.raises(R.NothingToRollBackTo):
        R.rollback(key, approved_by="bob")


def test_the_pointer_history_cannot_be_rewritten(key, db):
    _save(key)
    R.promote(key, 1, approved_by="bob")
    with pytest.raises(Exception, match="append-only"):
        with db() as conn:
            conn.cursor().execute(
                "UPDATE proc.bp_prompt_pointer_move SET approved_by = 'ada' "
                "WHERE prompt_key = %s", (key,))


def test_pointers_are_named_so_more_environments_are_data_not_schema(key):
    _save(key, "one")
    _save(key, "two")
    R.promote(key, 1, approved_by="bob")
    R.promote(key, 2, approved_by="bob", pointer="staging")

    assert R.resolve(key)["version"] == 1
    assert R.resolve(key, pointer="staging")["version"] == 2


# --------------------------------------------------------------------------
# The table the older readers still use
# --------------------------------------------------------------------------
def _legacy_row(db, name):
    with db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT prompts_desc, version, last_modified_by FROM proc.bp_prompt "
                    "WHERE prompt_name = %s AND prompts_status = 1", (name,))
        return cur.fetchone()


def test_promotion_writes_the_live_wording_back_for_the_older_readers(key, db):
    with db() as conn:
        conn.cursor().execute(
            "INSERT INTO proc.bp_prompt (prompt_name, prompt_type, prompts_desc, version) "
            "VALUES (%s, 'message', '{\"prompt_template\": \"old\"}'::jsonb, 1)", (key,))
    _save(key, "old", author="ada")
    _save(key, "new", author="ada")
    R.promote(key, 2, approved_by="bob")

    desc, version, by = _legacy_row(db, key)
    assert (desc, version, by) == ({"prompt_template": "new"}, 2, "bob")


def test_a_prompt_the_older_table_never_had_is_not_added_to_it(key, db):
    _save(key)
    R.promote(key, 1, approved_by="bob")
    assert _legacy_row(db, key) is None


# --------------------------------------------------------------------------
# Wording something else put in use, and taking a prompt out of use
# --------------------------------------------------------------------------
def test_adopted_wording_is_in_use_at_once_and_says_how_it_got_there(key):
    assert R.adopt(key, template="Assess it.", by="ada", change="agent_saved") == 1

    assert R.resolve(key)["template"] == "Assess it."
    assert R.pointer_history(key)[-1]["change"] == "agent_saved"


def test_a_retired_prompt_resolves_to_nothing_but_keeps_its_history(key):
    R.adopt(key, template="Assess it.", by="ada", change="agent_saved")
    assert R.retire(key, by="bob", note="agent deleted") is True

    with pytest.raises(R.PointerUnresolved, match="retired"):
        R.resolve(key)
    assert R.get_version(key, 1)["template"] == "Assess it."
    assert [m["change"] for m in R.pointer_history(key)] == ["agent_saved", "retired"]
    assert next(p for p in R.list_prompts() if p["prompt_key"] == key)["live_version"] is None


def test_retiring_what_is_not_in_use_does_nothing(key):
    _save(key)
    assert R.retire(key, by="bob") is False
    assert R.pointer_history(key) == []


def test_a_retired_prompt_cannot_be_rolled_back_but_can_be_promoted_again(key):
    _save(key, author="ada")
    R.promote(key, 1, approved_by="bob")
    R.retire(key, by="bob")
    with pytest.raises(R.NothingToRollBackTo, match="retired"):
        R.rollback(key, approved_by="bob")

    R.promote(key, 1, approved_by="bob")
    assert R.resolve(key)["version"] == 1


def test_an_unknown_kind_of_move_is_refused_by_the_database(key, db):
    _save(key)
    with pytest.raises(Exception, match="change_check"):
        with db() as conn:
            conn.cursor().execute(
                "INSERT INTO proc.bp_prompt_pointer_move (prompt_key, to_version, change, "
                "approved_by) VALUES (%s, 1, 'whatever', 'x')", (key,))
