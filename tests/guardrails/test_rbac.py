"""Roles come from policy, never from Python.

The load-bearing cases are the unhappy ones: no principal at all, a group
nobody mapped, and a policy that failed to load. All three must land on the
least-privileged answer rather than the most convenient one.
"""

import pytest

from src.services import rbac


class FakePolicyEngine:
    """Stands in for PolicyEngine with the row shape it really returns."""

    def __init__(self, policies):
        self._policies = policies

    def get_policy(self, slug):
        return self._policies.get(slug)


class FakePrincipal:
    def __init__(self, subject, claims=None):
        self.subject = subject
        self.claims = claims or {}


ROLE_DEFINITION = {
    "policyId": "role_definition",
    "details": {
        "policy_identifier": "role_definition",
        "required_role": "Admin",
        "rules": {
            "roles": {
                "Viewer": {"rank": 1, "allow": ["read"]},
                "Buyer": {"rank": 2, "allow": ["read", "compute", "write"]},
                "Approver": {
                    "rank": 3,
                    "allow": ["read", "compute", "write", "communicate", "transact"],
                },
                "Admin": {
                    "rank": 4,
                    "allow": [
                        "read", "compute", "write", "communicate",
                        "transact", "share", "configure", "delegate",
                    ],
                },
            },
            "irreversible_classes": [
                "communicate", "transact", "share", "configure", "delegate",
            ],
            "reversible_classes": ["read", "compute", "write"],
            "on_missing_role": "deny",
        },
    },
}

ROLE_ASSIGNMENT = {
    "policyId": "role_assignment",
    "details": {
        "policy_identifier": "role_assignment",
        "required_role": "Admin",
        "rules": {
            "claim": "cognito:groups",
            "group_to_role": {
                "bp-viewers": "Viewer",
                "bp-buyers": "Buyer",
                "bp-approvers": "Approver",
                "bp-admins": "Admin",
            },
            "no_principal_role": "Viewer",
            "unmapped_group_role": "Viewer",
            "multiple_groups": "highest_rank",
        },
    },
}


@pytest.fixture
def engine():
    return FakePolicyEngine(
        {"role_definition": ROLE_DEFINITION, "role_assignment": ROLE_ASSIGNMENT}
    )


def test_no_principal_is_viewer(engine):
    assert rbac.effective_role(None, policy_engine=engine) == "Viewer"


def test_group_maps_to_role(engine):
    principal = FakePrincipal("sub-1", {"cognito:groups": ["bp-approvers"]})
    assert rbac.effective_role(principal, policy_engine=engine) == "Approver"


def test_multiple_groups_take_the_highest_rank(engine):
    principal = FakePrincipal(
        "sub-2", {"cognito:groups": ["bp-viewers", "bp-admins", "bp-buyers"]}
    )
    assert rbac.effective_role(principal, policy_engine=engine) == "Admin"


def test_unmapped_group_falls_back_to_viewer(engine):
    principal = FakePrincipal("sub-3", {"cognito:groups": ["some-other-group"]})
    assert rbac.effective_role(principal, policy_engine=engine) == "Viewer"


def test_principal_with_no_groups_claim_is_viewer(engine):
    principal = FakePrincipal("sub-4", {})
    assert rbac.effective_role(principal, policy_engine=engine) == "Viewer"


def test_viewer_may_read_but_not_communicate(engine):
    assert rbac.may("Viewer", "read", policy_engine=engine) is True
    assert rbac.may("Viewer", "communicate", policy_engine=engine) is False


def test_approver_may_communicate(engine):
    assert rbac.may("Approver", "communicate", policy_engine=engine) is True


def test_approver_may_not_configure(engine):
    assert rbac.may("Approver", "configure", policy_engine=engine) is False


def test_irreversible_classes_come_from_policy(engine):
    assert rbac.is_irreversible("communicate", policy_engine=engine) is True
    assert rbac.is_irreversible("read", policy_engine=engine) is False


def test_missing_policy_denies_everything(engine):
    """A policy that failed to load must not silently grant."""
    empty = FakePolicyEngine({})
    assert rbac.effective_role(None, policy_engine=empty) == "Viewer"
    assert rbac.may("Admin", "communicate", policy_engine=empty) is False


def test_unknown_role_may_do_nothing(engine):
    assert rbac.may("Wizard", "read", policy_engine=engine) is False


def test_unknown_action_class_is_treated_as_irreversible(engine):
    """An action class nobody classified must not slip through as safe."""
    assert rbac.is_irreversible("teleport", policy_engine=engine) is True


def test_granted_but_unlisted_action_is_still_irreversible():
    """A class dropped from irreversible_classes must not become 'safe'.

    A role must be granted a dangerous action for it to be useful, so
    inferring safety from a grant gets every real case backwards.
    """
    drifted = {
        "policyId": "role_definition",
        "details": {
            "rules": {
                "roles": {"Admin": {"rank": 4, "allow": ["read", "delegate"]}},
                "irreversible_classes": ["communicate", "transact", "share", "configure"],
                "reversible_classes": ["read", "compute", "write"],
            }
        },
    }
    engine = FakePolicyEngine({"role_definition": drifted})
    assert rbac.is_irreversible("delegate", policy_engine=engine) is True


def test_reversible_class_is_not_irreversible(engine):
    """A class in reversible_classes must not be marked irreversible."""
    assert rbac.is_irreversible("read", policy_engine=engine) is False


def test_mapping_to_undefined_role_falls_back_to_viewer(engine):
    """A group mapping to an undefined role must resolve to Viewer (ranked branch).

    This test uses the default fixture's multiple_groups="highest_rank", which
    routes to the if branch. See test_undefined_role_is_rejected_on_the_non_ranked_branch
    for coverage of the else branch.
    """
    undefined_mapping = {
        "policyId": "role_assignment",
        "details": {
            "rules": {
                "claim": "cognito:groups",
                "group_to_role": {"bp-superadmins": "SuperAdmin"},  # not defined
                "no_principal_role": "Viewer",
                "unmapped_group_role": "Viewer",
                "multiple_groups": "highest_rank",
            },
        },
    }
    role_def = {
        "policyId": "role_definition",
        "details": {
            "rules": {
                "roles": {"Viewer": {"rank": 1, "allow": ["read"]}},
                "irreversible_classes": [],
                "reversible_classes": ["read"],
            }
        },
    }
    restricted_engine = FakePolicyEngine(
        {"role_definition": role_def, "role_assignment": undefined_mapping}
    )
    principal = FakePrincipal("sub-x", {"cognito:groups": ["bp-superadmins"]})
    assert rbac.effective_role(principal, policy_engine=restricted_engine) == "Viewer"


def test_the_live_role_definition_classifies_reads_as_reversible():
    """A fixture that outruns the database hides an unapplied migration.

    The rest of this file asserts against ROLE_DEFINITION. This one reads the
    real row, so it fails when the fixture and the database disagree.
    """
    import os

    import psycopg2
    from dotenv import load_dotenv

    from src.engines.policy_engine import PolicyEngine

    load_dotenv()

    def factory():
        return psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT", 5432),
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            connect_timeout=10,
        )

    engine = PolicyEngine(connection_factory=factory)

    assert rbac.is_irreversible("read", policy_engine=engine) is False
    assert rbac.is_irreversible("compute", policy_engine=engine) is False
    assert rbac.is_irreversible("write", policy_engine=engine) is False
    assert rbac.is_irreversible("communicate", policy_engine=engine) is True
    assert rbac.is_irreversible("delegate", policy_engine=engine) is True


def test_undefined_role_is_rejected_on_the_non_ranked_branch():
    """The fallback branch must apply the same rank-0 guard as the ranked one.

    multiple_groups is deliberately NOT "highest_rank" here: that is what
    routes execution into the branch this test exists to protect.
    """
    assignment = {
        "policyId": "role_assignment",
        "details": {
            "rules": {
                "claim": "cognito:groups",
                "group_to_role": {"bp-mystery": "SuperAdmin"},   # not in roles
                "no_principal_role": "Viewer",
                "unmapped_group_role": "Viewer",
                "multiple_groups": "first_match",                # not highest_rank
            }
        },
    }
    engine = FakePolicyEngine(
        {"role_definition": ROLE_DEFINITION, "role_assignment": assignment}
    )
    principal = FakePrincipal("sub-x", {"cognito:groups": ["bp-mystery"]})
    assert rbac.effective_role(principal, policy_engine=engine) == "Viewer"


# --- C3(b): proc.bp_role_assignment is the only route to an Approver
# before Cognito groups exist, and until this wiring it was created and
# schema-tested but read by no code -- resolve_roles only read
# cognito:groups. ---------------------------------------------------------

def test_a_direct_role_assignment_grants_that_role(engine, monkeypatch):
    monkeypatch.setattr(
        rbac, "_load_role_assignments", lambda: {"sub-5": ["Approver"]}
    )
    rbac.reset_policy_cache()
    principal = FakePrincipal("sub-5", {})  # no cognito:groups at all
    assert rbac.effective_role(principal, policy_engine=engine) == "Approver"


def test_a_revoked_role_assignment_does_not_grant(engine, monkeypatch):
    """_load_role_assignments only ever returns ACTIVE rows (revoked_at IS
    NULL is in the SQL WHERE clause) -- a revoked row is simply absent from
    what it returns, so a subject with only a revoked grant resolves the
    same as a subject with none at all."""
    monkeypatch.setattr(rbac, "_load_role_assignments", lambda: {})
    rbac.reset_policy_cache()
    principal = FakePrincipal("sub-6", {})
    assert rbac.effective_role(principal, policy_engine=engine) == "Viewer"


def test_a_direct_grant_combines_with_cognito_groups_via_highest_rank(engine, monkeypatch):
    """A direct Buyer grant plus a Cognito Admin group must resolve to
    Admin: multiple_groups: highest_rank governs across BOTH sources, not
    just the Cognito ones."""
    monkeypatch.setattr(rbac, "_load_role_assignments", lambda: {"sub-7": ["Buyer"]})
    rbac.reset_policy_cache()
    principal = FakePrincipal("sub-7", {"cognito:groups": ["bp-admins"]})
    assert rbac.effective_role(principal, policy_engine=engine) == "Admin"

    # And the reverse: a direct Admin grant outranks a mere Viewer group.
    monkeypatch.setattr(rbac, "_load_role_assignments", lambda: {"sub-7": ["Admin"]})
    rbac.reset_policy_cache()
    principal = FakePrincipal("sub-7", {"cognito:groups": ["bp-viewers"]})
    assert rbac.effective_role(principal, policy_engine=engine) == "Admin"


def test_a_failed_role_assignment_lookup_never_raises_and_grants_nothing(monkeypatch):
    """An unenforceable table read must not become a raise, and must not
    grant more than Cognito groups alone would -- it degrades to "nobody has
    a direct grant", never to something more permissive. This exercises the
    REAL _load_role_assignments (not a monkeypatched stand-in), forcing its
    own get_conn() call to raise.
    """

    class _ExplodingConnCtx:
        def __enter__(self):
            raise RuntimeError("bp_role_assignment is unreachable")

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(
        "src.services.db.get_conn", lambda: _ExplodingConnCtx(), raising=False
    )
    assert rbac._load_role_assignments() == {}


def test_direct_roles_cache_does_not_hit_the_loader_every_call(monkeypatch):
    calls = []
    monkeypatch.setattr(
        rbac, "_load_role_assignments", lambda: calls.append(1) or {"sub-8": ["Buyer"]}
    )
    rbac.reset_policy_cache()
    rbac._direct_roles("sub-8")
    rbac._direct_roles("sub-8")
    assert len(calls) == 1
    rbac.reset_policy_cache()
    rbac._direct_roles("sub-8")
    assert len(calls) == 2


def test_default_path_reuses_the_cached_engine(monkeypatch):
    """Rebuilding per call re-reads every policy row; this project has been
    bitten by that pattern before.
    """
    builds = []

    class Counting:
        def __init__(self):
            builds.append(1)

        def get_policy(self, slug):
            return None

    monkeypatch.setattr(rbac, "_build_engine", lambda: Counting(), raising=False)
    rbac.reset_policy_cache()
    rbac._engine(None)
    rbac._engine(None)
    assert len(builds) == 1
    rbac.reset_policy_cache()
    rbac._engine(None)
    assert len(builds) == 2
