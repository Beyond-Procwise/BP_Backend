"""Every gate-visible policy names a known action and states what it does.

Two failures this closes, both of which look exactly like the rule working:

  * A policy written against a name the gate never matches governs nothing, and
    nothing says so. ``email_send`` is not ``email.send``.
  * A policy that matches but states no effect now defers rather than granting
    (see test_unresolved_outcome), so a permit that was meant to be a permit
    becomes a question nobody expected. Better than the silent grant it
    replaced, but still not what the author intended.

These run against the LIVE policy set, because a vocabulary enforced only in
fixtures is not enforced.
"""

from __future__ import annotations

import os

import pytest

from src.services import actions

pytestmark = pytest.mark.skipif(
    not os.getenv("DB_HOST"), reason="reads the live policy set"
)


def _gate_visible():
    """Active policies carrying an applies_to, as the gate would see them."""

    # A direct connection on purpose: the suite's conftest fakes services.db,
    # and a vocabulary check that reads a fake policy set checks nothing.
    import psycopg2

    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT", 5432),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        connect_timeout=8,
    )
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT policy_id, policy_name,
                   policy_details->'applies_to',
                   policy_details->'rules'->>'effect',
                   policy_details->>'required_role'
              FROM proc.bp_policy
             WHERE policy_status = 1
               AND policy_details->'applies_to' IS NOT NULL
            """
        )
        return cur.fetchall()
    finally:
        conn.close()


def test_every_gate_visible_policy_uses_a_known_action_name():
    unknown = [
        (pid, name, a)
        for pid, name, applies, _e, _r in _gate_visible()
        for a in (applies or [])
        if not actions.is_known(a)
    ]
    assert unknown == [], (
        "these policies are written against action names the gate will never "
        f"match, so they govern nothing: {unknown}"
    )


def test_every_gate_visible_policy_states_an_effect():
    """Silence is no longer a grant, but it is not an intention either."""

    silent = [
        (pid, name)
        for pid, name, _a, effect, _r in _gate_visible()
        if effect not in ("allow", "deny")
    ]
    assert silent == [], (
        "these policies apply to an action but state no effect, so they defer "
        f"every request to a human instead of deciding: {silent}"
    )


def test_every_permitting_policy_names_the_role_it_permits():
    """A permit with no role cap permits every role that passes the class check."""

    uncapped = [
        (pid, name)
        for pid, name, _a, effect, required in _gate_visible()
        if effect == "allow" and not required
    ]
    assert uncapped == [], f"permits with no required_role: {uncapped}"


def test_the_two_actions_that_can_never_be_shadowed_are_in_the_vocabulary():
    from src.services import guardrail

    for action in guardrail.NEVER_SHADOW:
        assert actions.is_known(action), f"{action} is not a known action name"


def test_action_classes_match_the_role_policy_vocabulary():
    """A class this module invents would be governed by no role grant at all."""

    import json

    import psycopg2

    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT", 5432),
        dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"), connect_timeout=8,
    )
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT policy_details->'rules' FROM proc.bp_policy "
            "WHERE policy_status = 1 "
            "AND policy_details->>'policy_identifier' = 'role_definition'"
        )
        row = cur.fetchone()
    finally:
        conn.close()
    rules = row[0] if row else {}
    known_classes = set(rules.get("irreversible_classes") or []) | set(
        rules.get("reversible_classes") or []
    )
    assert known_classes, "could not read the role definition policy"

    invented = sorted(set(actions.ACTIONS.values()) - known_classes)
    assert invented == [], f"action classes no role policy defines: {invented}"
