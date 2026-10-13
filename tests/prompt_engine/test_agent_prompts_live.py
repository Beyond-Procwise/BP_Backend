"""An agent's instructions land in the registry when the agent is saved."""
from __future__ import annotations

from services.prompt_engine import agent_prompts as A
from services.prompt_engine import registry as R


def test_a_new_agents_instructions_are_recorded_with_who_wrote_them(key):
    assert A.record(key, "Assess the document.", by="sub-ada", linked_agent="doc_assessor") == 1

    live = R.resolve(key)
    assert (live["template"], live["author"]) == ("Assess the document.", "sub-ada")
    assert live["settings"]["legacy_linked_agents"] == "doc_assessor"


def test_an_edit_is_a_new_version_and_the_old_wording_is_kept(key):
    A.record(key, "one", by="sub-ada", linked_agent="a")
    assert A.record(key, "two", by="sub-bob", linked_agent="a") == 2

    assert R.resolve(key)["template"] == "two"
    assert [(v["template"], v["author"]) for v in R.versions(key)] == [
        ("one", "sub-ada"), ("two", "sub-bob")]


def test_saving_the_same_wording_again_adds_nothing(key):
    A.record(key, "one", by="sub-ada", linked_agent="a")
    assert A.record(key, "one", by="sub-bob", linked_agent="a") == 1
    assert len(R.versions(key)) == 1


def test_with_sign_in_off_the_author_is_the_same_word_the_older_table_uses(key):
    A.record(key, "one", by=None, linked_agent="a")
    assert R.resolve(key)["author"] == "workspace"


def test_deleting_the_agent_retires_its_instructions(key):
    A.record(key, "one", by="sub-ada", linked_agent="a")
    assert A.retire(key, by="sub-bob") is True
    assert A.retire(key, by="sub-bob") is False


def test_a_registry_failure_does_not_fail_the_agents_save(db, caplog):
    assert A.record("Not A Key", "one", by="sub-ada", linked_agent="a") is None
    assert "could not be recorded" in caplog.text


def test_a_failure_while_retiring_is_logged_not_raised(key, monkeypatch, caplog):
    def broken(*a, **k):
        raise RuntimeError("database gone")

    monkeypatch.setattr(R, "retire", broken)
    assert A.retire(key, by="sub-ada") is False
    assert "could not be retired" in caplog.text
