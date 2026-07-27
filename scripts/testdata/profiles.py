"""Build profiles: the same generator, different defect density.

The dataset has three jobs, and they pull against each other. A test dataset
wants many defects and an answer key so precision and recall can be measured. A
demo estate wants to look healthy, because a prospect shown three thousand
existing findings reasonably asks why the estate is such a mess. And a demo that
adds a document to show an anomaly being caught needs that document to be the
thing that lights up, not finding number 3,201.

One generator, two profiles, both deterministic from a seed:

    test   every defect type planted, full answer key. What V10 and V11 score.
    demo   a clean estate. No planted true positives, and the lossy behaviour
           the generator produces on its own is turned down.

The demo profile keeps the negative controls. Lump-sum services lines and credit
notes are not anomalies, they are ordinary procurement, and an estate without
them looks synthetic to anyone who knows the domain.
"""
from __future__ import annotations

from dataclasses import dataclass

from scripts.testdata.defects import DEFECT_SPECS

# Share of chains that stop before a purchase order is raised. In the test
# profile this is the "maverick spend" population; in a demo it is the small
# tail any real estate has.
TEST_NO_PO_SHARE = 0.16
DEMO_NO_PO_SHARE = 0.02

NEGATIVE_CONTROL_REFS: frozenset[str] = frozenset(
    spec.ref for spec in DEFECT_SPECS if spec.kind == "negative_control"
)
TRUE_POSITIVE_REFS: frozenset[str] = frozenset(
    spec.ref for spec in DEFECT_SPECS if spec.kind == "true_positive"
)


@dataclass(frozen=True)
class Profile:
    name: str
    no_po_share: float
    plant_refs: frozenset[str]
    purpose: str

    @property
    def plants_true_positives(self) -> bool:
        return bool(self.plant_refs & TRUE_POSITIVE_REFS)


TEST = Profile(
    name="test",
    no_po_share=TEST_NO_PO_SHARE,
    plant_refs=frozenset(spec.ref for spec in DEFECT_SPECS),
    purpose=(
        "Every defect type planted with a published answer key. Use this to "
        "measure whether the detectors find what is there and leave alone what "
        "is not."
    ),
)

DEMO = Profile(
    name="demo",
    no_po_share=DEMO_NO_PO_SHARE,
    plant_refs=NEGATIVE_CONTROL_REFS,
    purpose=(
        "A healthy estate for demonstrations. Nothing is planted for a detector "
        "to find, so a document added afterwards is the only thing that raises a "
        "finding. Legitimate lump-sum services and credit notes are kept: an "
        "estate without them does not look real."
    ),
)

PROFILES: dict[str, Profile] = {profile.name: profile for profile in (TEST, DEMO)}


def get(name: str) -> Profile:
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(
            f"unknown profile {name!r}; choose from {sorted(PROFILES)}"
        ) from None
