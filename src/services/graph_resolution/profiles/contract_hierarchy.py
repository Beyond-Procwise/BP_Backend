"""Which contract does this document sit under?

Not succession (which contract REPLACED which -- contract_succession.py) and not
coverage (what spend sits under a contract -- contract_coverage.py). This is the
parent-child hierarchy the GPSS build spec is about: a SOW under its master
agreement, a call-off or order form under its framework, a variation against the
contract it changes.

THE ONE RULE WORTH READING BEFORE CHANGING ANYTHING HERE. The build spec's
principle 3 says exact identifiers link automatically. The Discovery Report
overturned it on measurement and this profile is built on the overturned version:
proc.bp_contract_master.parent_contract_id is populated on 1,561 contracts and
resolves to a real contract on ZERO of them, because the references were minted
in a different namespace (C00002 pointing at C1543, which does not exist). So a
reference that matches exactly is the strongest signal available and is still
only a signal. Scoring it heavily enough to auto-link would link 1,561 contracts
to nothing.

MEASURED HEADROOM (test fixtures, 2026-10-02): a full match (reference, kind,
supplier, term and title all agreeing) scores F=96.95 and decides auto_link,
which edge_writer then refuses because the profile is in UNCALIBRATED_PROFILES.
An exact reference with every other signal MISSING scores F=30.82 and decides
block_or_exception: far below the auto band (92) and below even review (65). So
the reference alone does not put a candidate in front of a person; it needs
corroboration. That margin is what test_an_exact_reference_alone_does_not_reach_
the_auto_band protects.

Direction is CHILD -> PARENT: score_link(source_row=child, target_row=parent).
"""
from __future__ import annotations

import re
from datetime import date, datetime
from typing import Optional

from src.services import linking_engine as _le
from src.services.concepts.vocabulary import Vocabulary, ensure_vocabulary
from ..composition import remap_clusters
from ..observations import Observation

PROFILE = "contract_hierarchy"
VERSION = "1.0.0"

#: The reference fields a child may carry its parent's identifier in, strongest
#: first. framework_ref and parent_agreement_ref are declared pointers (Task 8);
#: parent_contract_id is the amends/supplements case and is read last because a
#: document that amends its parent also sits under it.
_REFERENCE_FIELDS = ("framework_ref", "parent_agreement_ref", "parent_contract_id")

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_PLACEHOLDERS = {"", "na", "n a", "tbc", "tbd", "none", "see"}


def _norm_ref(value) -> str:
    """One normalisation for every reference comparison.

    'MSA-4417', ' msa 4417 ' and 'MSA/4417' are the same reference printed
    differently, and a comparison that called them different would discard the
    heaviest signal on a formatting difference.
    """
    return _NON_ALNUM.sub("", str(value or "").strip().lower())


def _to_date(v) -> Optional[date]:
    if v is None:
        return None
    if isinstance(v, date):
        return v
    try:
        return datetime.strptime(str(v)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def expected_parent_type(
    child_structure: Optional[str],
    *,
    vocabulary: Optional[Vocabulary] = None,
) -> Optional[str]:
    """The structure the vocabulary says this one sits under, or None.

    Read from proc.bp_document_type.default_parent_type rather than hard-coded
    here, so the hierarchy is the same single fact the classifier and the upload
    gate already read. A fifth copy would need a fifth drift test.
    """
    if not child_structure:
        return None
    vocab = vocabulary if vocabulary is not None else ensure_vocabulary()
    dt = vocab.document_types.get(child_structure)
    return dt.default_parent_type if dt else None


def _cmp_reference(src, tgt) -> tuple[float, str]:
    """Does the child name this parent's identifier?

    MISSING when the child names no parent at all -- absence is not a
    contradiction, and treating it as one would make every standalone document
    look like a wrong parent.
    """
    parent_id = _norm_ref(tgt.get("contract_id"))
    if not parent_id:
        return 0.5, "MISSING"
    claimed = [
        _norm_ref(src.get(f)) for f in _REFERENCE_FIELDS
        if _norm_ref(src.get(f)) not in _PLACEHOLDERS
    ]
    if not claimed:
        return 0.5, "MISSING"
    if parent_id in claimed:
        return 1.0, "OK"
    return 0.0, "CONFLICT"


def _cmp_expected_structure(src, tgt) -> tuple[float, str]:
    """Is this candidate the KIND of document the child's structure sits under?"""
    want = expected_parent_type(src.get("resolved_doc_type"))
    have = tgt.get("resolved_doc_type")
    if not want or not have:
        return 0.5, "MISSING"
    return (1.0, "OK") if want == have else (0.0, "CONFLICT")


def _cmp_supplier(src, tgt) -> tuple[float, str]:
    a, b = src.get("supplier_id"), tgt.get("supplier_id")
    if not a or not b:
        return 0.5, "MISSING"
    return (1.0, "OK") if str(a) == str(b) else (0.0, "CONFLICT")


def _cmp_term_containment(src, tgt) -> tuple[float, str]:
    """Does the child's term sit inside the parent's?

    A child that starts before its parent or ends after it is not impossible --
    signature lag and extensions both happen -- so this is a graded signal, not a
    gate. Only a child wholly outside the parent's term CONFLICTs.
    """
    cs, ce = _to_date(src.get("contract_start_date")), _to_date(src.get("contract_end_date"))
    ps, pe = _to_date(tgt.get("contract_start_date")), _to_date(tgt.get("contract_end_date"))
    if cs is None or ps is None:
        return 0.5, "MISSING"
    if pe is not None and cs > pe:
        return 0.0, "CONFLICT"       # starts after the parent ended
    if ce is not None and ce < ps:
        return 0.0, "CONFLICT"       # ended before the parent began
    inside_start = cs >= ps
    inside_end = pe is None or ce is None or ce <= pe
    if inside_start and inside_end:
        return 1.0, "OK"
    return 0.6, "WEAK"


_STOPWORDS = {"the", "and", "of", "for", "agreement", "contract", "services",
              "service", "statement", "work", "master", "framework", "order",
              "form", "schedule", "ltd", "limited", "plc"}


def _cmp_title_overlap(src, tgt) -> tuple[float, str]:
    """Shared DISTINCTIVE words, so 'Agreement' is not evidence.

    Without the stopword set every contract shares 'Agreement' and 'Services'
    with every other, and the signal would score the vocabulary rather than the
    documents.
    """
    def words(row):
        raw = (row.get("contract_title") or "").lower()
        return {w for w in _NON_ALNUM.sub(" ", raw).split() if w and w not in _STOPWORDS}

    a, b = words(src), words(tgt)
    if not a or not b:
        return 0.5, "MISSING"
    j = len(a & b) / len(a | b)
    if j >= 0.5:
        return 1.0, "OK"
    if j >= 0.2:
        return 0.6, "WEAK"
    return 0.0, "CONFLICT"


_le.register_signal("csh_reference", lambda s, t, sl, tl: _cmp_reference(s, t))
_le.register_signal("csh_structure", lambda s, t, sl, tl: _cmp_expected_structure(s, t))
_le.register_signal("csh_supplier", lambda s, t, sl, tl: _cmp_supplier(s, t))
_le.register_signal("csh_term", lambda s, t, sl, tl: _cmp_term_containment(s, t))
_le.register_signal("csh_title", lambda s, t, sl, tl: _cmp_title_overlap(s, t))

SIGNALS = [
    # The reference is tier 1 and weight 5 -- the heaviest available -- and its
    # conflict_cap of 0.45 is what stops it deciding alone. See the module
    # docstring: 1,561 references resolve to nothing.
    {"id": "declared_reference", "cluster": "reference", "tier": 1, "weight": 5,
     "appl": 1.0, "cap": 0.45, "kind": "csh_reference",
     "reads": ["framework_ref", "parent_agreement_ref", "parent_contract_id"]},
    {"id": "expected_structure", "cluster": "structure", "tier": 1, "weight": 5,
     "appl": 1.0, "cap": 0.45, "kind": "csh_structure",
     "reads": ["resolved_doc_type"]},
    {"id": "supplier", "cluster": "identity", "tier": 1, "weight": 5,
     "appl": 1.0, "cap": 0.45, "kind": "csh_supplier",
     "reads": ["supplier_id"]},
    {"id": "term_containment", "cluster": "temporal", "tier": 2, "weight": 3,
     "appl": 1.0, "cap": 0.70, "kind": "csh_term",
     "reads": ["contract_start_date", "contract_end_date"]},
    {"id": "title_overlap", "cluster": "description", "tier": 2, "weight": 3,
     "appl": 1.0, "cap": 0.70, "kind": "csh_title",
     "reads": ["contract_title"]},
]

# DECLARED UNMEASURED, like contract_succession and contract_coverage: no
# labelled sample of true parent links exists, because the only parent pointers
# in the corpus all dangle.
_le.register_profile(PROFILE, {
    "p0": 0.02, "alpha": 0.35, "floor": 0.55,
    "signals": SIGNALS, "date_field": "contract_start_date",
})


def observations_for(src: dict, tgt: dict) -> dict[str, frozenset]:
    sid, tid = str(src.get("contract_id")), str(tgt.get("contract_id"))
    out: dict[str, frozenset] = {}
    for spec in SIGNALS:
        obs: set[Observation] = set()
        for field in spec["reads"]:
            if src.get(field) is not None:
                obs.add((sid, field))
            if tgt.get(field) is not None:
                obs.add((tid, field))
        out[spec["id"]] = frozenset(obs)
    return out


def score(src: dict, tgt: dict) -> dict:
    """Score child -> parent, with correlated signals merged into one cluster.

    This is the entry point callers use, NOT score_link directly. It mirrors
    contract_succession.score exactly: remap_clusters does union-find over
    "shares at least one observation", so two signals reading the same field
    cannot each contribute full weight for what is really one piece of
    evidence. Today every signal here reads a distinct field, so the remap is a
    no-op -- which is precisely why it must be wired now rather than when
    someone adds a sixth signal that overlaps an existing one.
    """
    overrides = remap_clusters(SIGNALS, observations_for(src, tgt))
    return _le.score_link(src, tgt, PROFILE, cluster_overrides=overrides)
