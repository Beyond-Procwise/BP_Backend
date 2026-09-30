"""The two names the playbook layer gates on, and the class they belong to."""

import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")))

from src.services import actions  # noqa: E402


@pytest.mark.parametrize("name", ["playbook.write", "playbook.approve"])
def test_the_name_is_known(name):
    assert actions.is_known(name)


@pytest.mark.parametrize("name", ["playbook.write", "playbook.approve"])
def test_configuring_a_playbook_is_a_configure(name):
    """There is no generic 'approve' class in RoleDefinitionPolicy, so these
    follow policy.write and prompt.write."""
    assert actions.action_class(name) == "configure"


def test_approving_a_proposal_introduces_no_new_action():
    """It causes a workflow to run and gates on workflow.run, which already
    exists and is already policied. A second name for the same act would be a
    second thing to keep in step."""
    assert actions.action_class("workflow.run") == "delegate"
    assert not actions.is_known("proposal.approve")
    assert not actions.is_known("playbook.run")


def test_the_names_follow_the_domain_verb_shape():
    for name in ("playbook.write", "playbook.approve"):
        domain, _, verb = name.partition(".")
        assert domain == "playbook"
        assert verb and verb.islower() and not verb.endswith("s")
