"""The playbook layer: human-authored strategy, proposed and never auto-run.

A playbook says *when a finding like this appears, the strategy an expert wrote
for it is that one*. Selecting a playbook queues a proposal; a person approves
it; only then does the existing workflow run path execute anything. Nothing in
this package starts a workflow.
"""

from .finding_source import (
    DETECTION_FINDING,
    OPPORTUNITY,
    Finding,
    MATCH_FIELDS,
    normalise,
    validate_trigger_match,
)

__all__ = [
    "DETECTION_FINDING",
    "OPPORTUNITY",
    "Finding",
    "MATCH_FIELDS",
    "normalise",
    "validate_trigger_match",
]
