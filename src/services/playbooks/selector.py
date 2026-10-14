"""Which playbook governs a finding. Pure, deterministic, and it refuses.

    1. Take the active playbooks whose trigger_source is the finding's source.
    2. Keep those whose every trigger_match key equals the finding's value for
       that key. Equality only -- no fuzzy matching, no aliases, no substrings.
    3. The winner is the one with the MOST match keys.
    4. If two or more tie on key count, propose nothing, and say so at ERROR
       naming both by id and name.

STEP 4 IS THE LOAD-BEARING ONE.

The detection registry this codebase removed bound policies to detectors by
accumulating aliases and letting whichever matched last win. Four of five
bindings were silently wrong for months, and price variance spent them
reporting itself as maverick spend. An ambiguous playbook match is a
configuration error, and a configuration error that resolves itself quietly is
that same failure in a new table. A tie is visible or it is nothing.

Do not add a tiebreak here. Not lowest id, not most recently approved, not
highest version. Any of them would make the ambiguity disappear from the logs
while leaving it in the data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from .finding_source import Finding, canonical
from .store import Playbook

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Selection:
    """The one playbook that governs a finding, and why it matched."""

    playbook: Playbook
    evidence: Dict[str, Any]


def _matches(finding: Finding, playbook: Playbook) -> bool:
    """Every trigger_match key equals the finding's value for that key.

    Both sides go through ``canonical`` first, so a boolean column authored in
    JSON as ``"true"`` and a severity authored as ``Critical`` still match. A
    finding value that is absent or NULL matches nothing: it is not a wildcard.
    """

    for key, wanted in playbook.trigger_match.items():
        have = finding.attrs.get(key)
        if have is None:
            return False
        if canonical(have) != canonical(wanted):
            return False
    return True


def _best(finding: Finding, playbooks: Sequence[Playbook]) -> List[Playbook]:
    """The matching playbooks tied on the most match keys. Empty if none match."""

    matching = [
        pb for pb in playbooks
        if pb.trigger_source == finding.source and _matches(finding, pb)
    ]
    if not matching:
        return []
    most = max(len(pb.trigger_match) for pb in matching)
    return [pb for pb in matching if len(pb.trigger_match) == most]


def _evidence(finding: Finding, playbook: Playbook) -> Dict[str, Any]:
    """What the finding held, canonically, at the moment it matched."""

    return {
        "matched": {
            key: canonical(finding.attrs.get(key))
            for key in playbook.trigger_match
        },
        "key_count": len(playbook.trigger_match),
        "playbook_version": playbook.version,
    }


def select(finding: Finding, playbooks: Sequence[Playbook]) -> Optional[Selection]:
    """The one playbook that governs ``finding``, or ``None``.

    ``None`` covers two different situations -- nothing matched, and more than
    one thing matched equally well. Call :func:`tied_candidates` to tell them
    apart; the sweep does, and audits the second.
    """

    best = _best(finding, playbooks)
    if not best:
        return None
    if len(best) > 1:
        logger.error(
            "ambiguous playbook match for %s finding %s: %s tie on %d match "
            "key(s), so nothing is proposed. Make one of them more specific or "
            "retire one.",
            finding.source,
            finding.finding_id,
            ", ".join(f"{pb.playbook_name!r} (id {pb.playbook_id})" for pb in best),
            len(best[0].trigger_match),
        )
        return None
    winner = best[0]
    return Selection(playbook=winner, evidence=_evidence(finding, winner))


def tied_candidates(finding: Finding, playbooks: Sequence[Playbook]) -> List[Playbook]:
    """The playbooks that tied, or ``[]`` when the match was unambiguous.

    Kept separate from :func:`select` so that both stay pure and the caller,
    not this module, decides what an ambiguity is worth recording.
    """

    best = _best(finding, playbooks)
    return best if len(best) > 1 else []
