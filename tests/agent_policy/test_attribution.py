"""The server, not the browser, says who confirmed a policy's examples."""
import copy

from repositories.agent_policy_repo import attribute_confirmation as attr

NOW = "2026-10-08T10:00:00Z"
FORGED = {"by": "someone-else", "at": "1999-01-01T00:00:00Z"}


def _form(**kw):
    base = {"situation": "s", "outcome": "o", "owner": "Finance", "examples": [{"input": "a"}]}
    base.update(kw)
    return base


def test_fresh_confirmation_is_attributed_to_the_actor_not_the_browser():
    new = _form(checked=dict(FORGED))
    out = attr(new, None, actor="alice", now_iso=NOW)
    assert out["checked"] == {"by": "alice", "at": NOW}
    assert new["checked"] == FORGED          # input not mutated
    assert out is not new


def test_same_confirmation_carried_through_an_unrelated_edit_is_kept():
    prev = _form(checked={"by": "alice", "at": "2026-10-01T00:00:00Z"})
    new = _form(checked=dict(prev["checked"]), owner="Procurement")
    out = attr(new, prev, actor="bob", now_iso=NOW)
    assert out["checked"] == prev["checked"]


def test_no_confirmation_is_stored_as_none():
    assert attr(_form(checked=None), None, actor="a", now_iso=NOW)["checked"] is None
    assert attr(_form(), None, actor="a", now_iso=NOW)["checked"] is None


def test_situation_changed_but_old_checked_resent_is_reattributed_to_the_saver():
    # Intended: saving a confirmed form after a change re-attributes it to whoever saved it.
    prev = _form(checked={"by": "alice", "at": "2026-10-01T00:00:00Z"})
    new = copy.deepcopy(prev)
    new["situation"] = "changed"
    out = attr(new, prev, actor="bob", now_iso=NOW)
    assert out["checked"] == {"by": "bob", "at": NOW}


def test_forged_checked_differing_from_previous_is_fresh():
    prev = _form(checked={"by": "alice", "at": "2026-10-01T00:00:00Z"})
    out = attr(_form(checked=dict(FORGED)), prev, actor="bob", now_iso=NOW)
    assert out["checked"] == {"by": "bob", "at": NOW}
