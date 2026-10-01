"""What the page says about its own type, next to what the uploader declared.

Deterministic: alias and structural-signal matching over text L0 has already
parsed. Every piece of evidence it returns is a verbatim substring of that
text, which is the same grounding contract judge_gate holds for extracted
values — a reviewer can be shown the exact words.

Three things this module refuses to do:

  * It never refines the declared type. Declared 'contract', page says
    'framework agreement' -> both are recorded and agreement='disagreed'. A
    human confirms the refinement from the review queue. declared_linkage.py
    already holds the principle: a human's decision outranks an inference.
  * It never touches routing. Task 5's gate already decided which physical
    pipeline runs, from the declared category. Nothing here can change that.
  * It never breaks a tie. Two concepts with equal evidence give
    status='unresolved' and both candidates.

No model call. The build spec's AI-answered ruling tests belong with the
conflict register, which is deferred: there is no sector or normalised region
to scope a ruling by, and a yes/no judgement cannot be substring-grounded the
way every other model output in this pipeline is.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.services.concepts.vocabulary import Vocabulary, ensure_vocabulary, fold

log = logging.getLogger(__name__)

#: How much of the start of the document counts as the title zone. A type named
#: in the heading is far stronger evidence than one mentioned in a clause, and
#: an invoice citing a purchase order must stay an invoice.
_TITLE_CHARS = 600

_WEIGHT = {"title_alias": 5.0, "body_alias": 1.0, "structural_signal": 0.5}

#: Minimum score before the evidence names a type at all. One passing mention
#: in a clause is not a classification.
_MIN_SCORE = 1.0

#: How far ahead of the runner-up the winner must be. Below this the evidence
#: has not chosen, and saying it has would be fabrication.
_MIN_MARGIN = 1.0


@dataclass(frozen=True)
class Evidence:
    kind: str           # title_alias | body_alias | structural_signal
    text: str           # verbatim substring of the page
    start: int          # its offset, so the span can be shown
    concept_code: str


@dataclass(frozen=True)
class TypeResolution:
    declared_concept: Optional[str]
    evidence_concept: Optional[str]
    status: str         # matched | unknown | unresolved
    agreement: str      # agreed | declared_only | evidence_only | disagreed
    candidates: Tuple[str, ...]
    evidence: Tuple[Evidence, ...]


def _find_all(needle: str, haystack_lower: str) -> List[int]:
    """Offsets of every whole-word occurrence of ``needle``."""
    if not needle:
        return []
    pattern = r"(?<![a-z0-9])" + re.escape(needle) + r"(?![a-z0-9])"
    return [m.start() for m in re.finditer(pattern, haystack_lower)]


def resolve_document_type(
    *,
    declared_concept: Optional[str],
    full_text: str,
    vocabulary: Optional[Vocabulary] = None,
    title_chars: int = _TITLE_CHARS,
) -> TypeResolution:
    """Classify a document from its own text, reporting rather than deciding."""
    vocab = vocabulary if vocabulary is not None else ensure_vocabulary()
    text = full_text or ""
    # Match on a lowercase copy with EXACTLY the same length as the page, so an
    # offset found in it is an offset in the page. str.lower() alone cannot
    # promise that ("\u0130".lower() is two characters), so a character whose
    # lowercase is not one character is left as it is. fold() serves only the
    # alias side, because it collapses whitespace and would shift offsets.
    lowered = "".join(
        (lc if len(lc := ch.lower()) == 1 else ch) for ch in text
    ).replace("_", " ").replace("-", " ")

    # (concept, kind, start, length) for every alias hit, before scoring.
    hits: List[Tuple[str, str, int, int]] = []
    signals: List[Tuple[str, int, int]] = []

    # Only active types are in vocabulary.document_types, so a proposed type
    # can never become the evidence answer.
    for code, dt in vocab.document_types.items():
        if dt.status != "active":
            continue  # belt and braces: build_vocabulary already filters
        # A set: a concept's own name is also an alias, and counting the same
        # phrase twice would hand that concept a spurious double score.
        for folded in {fold(a) for a in (*dt.aliases, code.split(".", 1)[-1])}:
            if not folded:
                continue
            for start in _find_all(folded, lowered):
                kind = "title_alias" if start < title_chars else "body_alias"
                hits.append((code, kind, start, len(folded)))
        for signal in dt.structural_signals:
            folded = fold(signal)
            if not folded:
                continue
            found = _find_all(folded, lowered)
            if found:
                signals.append((code, found[0], len(folded)))

    # Longest match wins: 'agreement' inside 'framework agreement' is the same
    # words read twice, not a second type claiming the page. Identical spans
    # from different concepts are KEPT, since that is a genuine collision and
    # must surface as a tie.
    def _inside_longer(h: Tuple[str, str, int, int]) -> bool:
        _, _, s0, n0 = h
        return any(
            o[3] > n0 and o[2] <= s0 and s0 + n0 <= o[2] + o[3] for o in hits
        )

    evidence: List[Evidence] = []
    scores: Dict[str, float] = {}

    def credit(concept_code: str, kind: str, start: int, length: int) -> None:
        scores[concept_code] = scores.get(concept_code, 0.0) + _WEIGHT[kind]
        evidence.append(Evidence(
            kind=kind, text=text[start:start + length], start=start,
            concept_code=concept_code,
        ))

    for h in hits:
        if not _inside_longer(h):
            credit(*h)
    for code, start, length in signals:
        credit(code, "structural_signal", start, length)

    evidence_concept: Optional[str] = None
    candidates: Tuple[str, ...] = ()
    status = "unknown"

    if scores:
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        best_code, best_score = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        if best_score < _MIN_SCORE:
            status = "unknown"
        elif (best_score - runner_up) < _MIN_MARGIN:
            status = "unresolved"
            candidates = tuple(
                code for code, score in ranked if score >= best_score - _MIN_MARGIN
            )
        else:
            status = "matched"
            evidence_concept = best_code
            candidates = (best_code,)

    if declared_concept:
        # A declared type is a human's statement and always stands. The page
        # only ever adds a second reading.
        if evidence_concept is None:
            agreement = "declared_only"
        elif evidence_concept == declared_concept:
            agreement = "agreed"
        else:
            agreement = "disagreed"
        if status != "unresolved":
            status = "matched"
    else:
        agreement = "evidence_only" if evidence_concept else "declared_only"
        if not evidence_concept and status == "matched":
            status = "unknown"

    # Keep the evidence that bears on the answer, strongest first, and cap it:
    # this is written to a review row a person reads, not a log.
    relevant = {c for c in (declared_concept, evidence_concept, *candidates) if c}
    kept = sorted(
        (ev for ev in evidence if ev.concept_code in relevant),
        key=lambda ev: (-_WEIGHT[ev.kind], ev.start),
    )[:12]

    return TypeResolution(
        declared_concept=declared_concept,
        evidence_concept=evidence_concept,
        status=status,
        agreement=agreement,
        candidates=candidates,
        evidence=tuple(kept),
    )


__all__ = ["Evidence", "TypeResolution", "resolve_document_type"]
