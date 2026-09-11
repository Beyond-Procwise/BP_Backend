"""On-disk captures of document and agent content must age out.

Two directories accumulate content extracted from customer documents, outside
the database and outside every retention rule the product has:

  artifacts/llm_failures/  prompt head, prompt tail and the model's response,
                           written by llm_diagnostics.capture_llm_failure on
                           every failed LLM call. 4,515 files at the time this
                           was written, oldest 2026-05-04.
  data/conversations/      agent context and response, written by
                           models/context_trainer. 517 files, oldest 2026-03-25.

Nothing deleted either one. A right-to-erasure request could not reach them and
nobody could say how long they were kept, because the answer was "forever".

The deletion is the dangerous part, so the tests below spend most of their
effort on what must NOT be deleted: anything newer than the cutoff, anything
outside the configured directory, and anything reached by following a symlink
out of it.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from src.services import capture_retention as R


def _age(path: Path, days: float) -> None:
    """Backdate a file's mtime."""
    when = time.time() - days * 86400
    os.utime(path, (when, when))


@pytest.fixture
def captures(tmp_path, monkeypatch):
    """A capture directory, registered as a known root for the test.

    Registering is the point of the fixture: purge() refuses any directory not
    in CAPTURE_DIRS, so a test that forgets this gets the refusal rather than a
    delete — which is the containment guard working, and is asserted directly
    in test_it_refuses_a_directory_that_is_not_a_known_capture_root.
    """
    root = tmp_path / "llm_failures"
    root.mkdir()
    monkeypatch.setattr(R, "CAPTURE_DIRS", (root,))
    old = root / "20260504T190652Z_anon_extract.json"
    new = root / "20260807T224951Z_anon_extract.json"
    old.write_text('{"prompt_head": "Invoice ELEANOR PRICE ..."}')
    new.write_text('{"prompt_head": "Invoice NORTHGATE ..."}')
    _age(old, days=90)
    _age(new, days=1)
    return root, old, new


# --------------------------------------------------------------------------
# What it deletes
# --------------------------------------------------------------------------

def test_a_capture_past_the_cutoff_is_deleted(captures):
    root, old, new = captures
    removed = R.purge(root, max_age_days=30)
    assert removed == 1
    assert not old.exists()


def test_a_capture_inside_the_window_is_kept(captures):
    root, old, new = captures
    R.purge(root, max_age_days=30)
    assert new.exists(), "a capture inside the retention window was deleted"


def test_a_longer_window_keeps_everything(captures):
    root, old, new = captures
    assert R.purge(root, max_age_days=365) == 0
    assert old.exists() and new.exists()


# --------------------------------------------------------------------------
# What it must never touch
# --------------------------------------------------------------------------

def test_a_missing_directory_is_not_an_error(tmp_path, monkeypatch):
    """The scheduler calls this on every host, including ones that have never
    written a capture. A missing directory is normal, not a failure."""
    absent = tmp_path / "never-created"
    monkeypatch.setattr(R, "CAPTURE_DIRS", (absent,))
    assert R.purge(absent, max_age_days=30) == 0


def test_it_refuses_a_directory_that_is_not_a_known_capture_root(tmp_path):
    """The one guard that matters. This function deletes files by age, and an
    age-based delete pointed at the wrong directory is a data-loss incident,
    not a bug. Only the two known capture roots may be purged."""
    victim = tmp_path / "src"
    victim.mkdir()
    doomed = victim / "important.py"
    doomed.write_text("# a year old and still load-bearing")
    _age(doomed, days=400)

    with pytest.raises(ValueError, match="not a known capture directory"):
        R.purge(victim, max_age_days=30)
    assert doomed.exists()


def test_a_symlink_is_removed_as_a_link_and_its_target_is_untouched(
    captures, tmp_path
):
    """A symlink is cleaned up by its own presence, not by its target's age.

    The target deliberately has a RECENT mtime. Without the explicit symlink
    branch in purge(), ``entry.stat()`` follows the link, reads the target's
    recent timestamp and keeps the link forever — so a stale pointer into
    somebody else's tree accumulates in a directory we claim to age out.

    An earlier version of this test used an OLD target and asserted only that
    the target survived. That passes whether or not the branch exists, because
    ``Path.unlink()`` removes a link and can never reach what it points at — so
    it was a test that could not fail. The target assertion is kept, because the
    property is worth stating; the link assertion is what makes the test able to
    fail.
    """
    root, old, new = captures
    outside = tmp_path / "outside.txt"
    outside.write_text("not a capture")
    _age(outside, days=1)          # recent on purpose — see docstring

    link = root / "link.json"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this platform")

    R.purge(root, max_age_days=30)

    assert outside.exists(), "purge deleted a symlink's target"
    assert not link.is_symlink(), (
        "the symlink was left behind: purge followed it to a recent target "
        "instead of removing the link itself"
    )


def test_a_subdirectory_is_left_alone(captures):
    root, old, new = captures
    sub = root / "keep"
    sub.mkdir()
    _age(sub, days=400)
    R.purge(root, max_age_days=30)
    assert sub.is_dir(), "purge removed a directory, not just files"


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

def test_a_zero_or_negative_window_is_refused_rather_than_deleting_everything(
    captures,
):
    """A misconfigured retention of 0 would mean "delete everything", which is
    never what someone meant to type."""
    root, old, new = captures
    for bad in (0, -1):
        with pytest.raises(ValueError, match="max_age_days"):
            R.purge(root, max_age_days=bad)
    assert old.exists() and new.exists()


def test_the_environment_still_overrides_the_window_for_one_release(monkeypatch):
    monkeypatch.setenv("CAPTURE_RETENTION_DAYS", "14")
    assert R.retention_days() == 14


def test_an_unparseable_window_falls_back_to_policy_rather_than_zero(
    monkeypatch,
):
    # The seeded policy (tests/conftest.py) states 30. Policy-first behaviour,
    # with a value distinct from the old in-code default, is pinned in
    # tests/governance/test_p9_tail.py.
    monkeypatch.setenv("CAPTURE_RETENTION_DAYS", "not-a-number")
    assert R.retention_days() == 30


def test_purge_all_reports_each_directory(monkeypatch, captures):
    root, old, new = captures
    monkeypatch.setattr(R, "CAPTURE_DIRS", (root,))
    monkeypatch.setenv("CAPTURE_RETENTION_DAYS", "30")
    assert R.purge_all() == {str(root): 1}
