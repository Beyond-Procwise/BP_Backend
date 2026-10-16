"""The state machine is the fail-closed guarantee in code form: an artifact
cannot reach extraction except by passing every gate, because no transition
into READY_FOR_EXTRACTION exists from anywhere but STORED."""
import pytest

from src.services.mail_intake.states import (
    ArtifactState,
    BundleState,
    IllegalTransition,
    QuarantineReason,
    assert_transition,
)


def test_happy_path_artifact_transitions_are_legal():
    chain = [
        ArtifactState.DETACHED,
        ArtifactState.TYPED,
        ArtifactState.GATED,
        ArtifactState.STORED,
        ArtifactState.READY_FOR_EXTRACTION,
        ArtifactState.EXTRACTION_COMPLETE,
    ]
    for current, target in zip(chain, chain[1:]):
        assert_transition(current, target)


def test_cannot_skip_the_safety_gate():
    with pytest.raises(IllegalTransition):
        assert_transition(ArtifactState.TYPED, ArtifactState.STORED)


def test_cannot_reach_extraction_without_being_stored():
    for state in (ArtifactState.DETACHED, ArtifactState.TYPED, ArtifactState.GATED):
        with pytest.raises(IllegalTransition):
            assert_transition(state, ArtifactState.READY_FOR_EXTRACTION)


def test_any_live_state_may_quarantine():
    for state in (
        ArtifactState.DETACHED,
        ArtifactState.TYPED,
        ArtifactState.GATED,
        ArtifactState.STORED,
    ):
        assert_transition(state, ArtifactState.QUARANTINED)


def test_quarantine_is_terminal_for_the_pipeline():
    with pytest.raises(IllegalTransition):
        assert_transition(ArtifactState.QUARANTINED, ArtifactState.READY_FOR_EXTRACTION)


def test_bundle_happy_path():
    chain = [
        BundleState.RECEIVED,
        BundleState.PARSED,
        BundleState.AUTHENTICATED,
        BundleState.BOUND,
        BundleState.COMPLETE,
    ]
    for current, target in zip(chain, chain[1:]):
        assert_transition(current, target)


def test_bundle_cannot_complete_without_authentication():
    with pytest.raises(IllegalTransition):
        assert_transition(BundleState.PARSED, BundleState.BOUND)


def test_quarantine_reasons_cover_the_spec_gate():
    # Spec §6 stage 7 plus stages 2, 6 and the reference-attachment case.
    for name in (
        "MALFORMED_MIME",
        "UNKNOWN_TYPE",
        "ENCRYPTED",
        "MACRO",
        "ARCHIVE",
        "EXECUTABLE",
        "OVERSIZE",
        "TYPE_MISMATCH",
        "HIDDEN_TEXT",
        "UNRESOLVABLE_REFERENCE",
    ):
        assert hasattr(QuarantineReason, name)
