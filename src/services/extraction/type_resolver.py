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

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.services.concepts.vocabulary import Vocabulary, ensure_vocabulary, fold

#: How much of the start of the document counts as the title zone. A type named
#: in the heading is far stronger evidence than one mentioned in a clause, and
#: an invoice citing a purchase order must stay an invoice.
_TITLE_CHARS = 600

#: Ordering of evidence kinds when a reviewer is shown the strongest first.
_WEIGHT = {"title_alias": 5.0, "body_alias": 1.0, "structural_signal": 0.5}

#: How the evidence is compared. NOT one summed score: scores summed in a narrow
#: numeric range let each new term outbid the last, so a passing citation in a
#: header or a list of schedules could bury the document's own heading. Instead
#: a concept lands in a TIER and tiers are compared first, ordinally:
#:   tier 1 : an alias hit on the heading (the first non-blank line, bounded by
#:            _HEADING_CHARS). A document names itself there; nothing below it
#:            can outweigh it, however much there is.
#:   tier 2 : alias hits anywhere else, title zone and body alike. The
#:            600-character title zone is only a proxy for a heading, so it earns
#:            nothing extra here.
#: Only within one tier does a bounded sub-score decide: distinct aliases, plus
#: a damped (log2, capped) occurrence count. Structural signals are
#: corroboration only: they add to a concept that already has an alias hit and
#: can break a tie, but never name a type alone.
_HEADING_CHARS = 120
_OCCURRENCE_CAP = 3.0
_SIGNAL = 0.5
_SIGNAL_CAP = 1.0

#: A tier-2 concept's alias sub-score must be STRICTLY above this to name a
#: type. One passing mention in a clause scores exactly 1.0 and is not a
#: classification. (Tier 1 needs no minimum: the heading is the claim.)
_MIN_SCORE = 1.0

#: How far ahead of the runner-up the winner must be. Below this the evidence
#: has not chosen, and saying it has would be fabrication.
_MIN_MARGIN = 0.5


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
    agreement: str      # agreed | declared_only | evidence_only | disagreed | neither
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
    hits: List[Tuple[str, str, int, int, str]] = []
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
                hits.append((code, kind, start, len(folded), folded))
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
    # must surface as a tie. One sweep: sorted by (start, -length), a hit is
    # covered when an earlier, different span already reaches its end.
    hits.sort(key=lambda h: (h[2], -h[3], h[0]))
    kept_hits: List[Tuple[str, str, int, int, str]] = []
    max_end = -1
    i = 0
    while i < len(hits):
        span = (hits[i][2], hits[i][2] + hits[i][3])
        j = i
        while j < len(hits) and (hits[j][2], hits[j][2] + hits[j][3]) == span:
            j += 1
        if span[1] > max_end:
            kept_hits.extend(hits[i:j])
        max_end = max(max_end, span[1])
        i = j

    first = re.search(r"\S", text)
    # The heading ends at the first newline, but never later than
    # _HEADING_CHARS in: parsed text with unreliable line breaks must not turn
    # the whole document into "the heading", which would hand every concept
    # tier 1 and cancel the signal.
    heading_start = first.start() if first else 0
    heading_end = heading_start
    if first:
        nl = text.find("\n", heading_start)
        heading_end = min(len(text) if nl < 0 else nl, heading_start + _HEADING_CHARS)

    region: Dict[str, List[Tuple[str, str, int, int, str]]] = {}
    tier: Dict[str, int] = {}
    for h in kept_hits:
        on_heading = first is not None and heading_start <= h[2] < heading_end
        t = 1 if on_heading else 2
        code = h[0]
        if code not in tier or t < tier[code]:
            tier[code], region[code] = t, []
        if t == tier[code]:
            region[code].append(h)

    alias_sub: Dict[str, float] = {}
    for code, rhits in region.items():
        alias_sub[code] = (
            len({h[4] for h in rhits})
            + min(_OCCURRENCE_CAP, math.log2(len(rhits)))
        )
    sig_for: Dict[str, List[Tuple[str, int, int]]] = {}
    for sig in signals:
        if sig[0] in region:  # corroboration only: needs an alias hit already
            sig_for.setdefault(sig[0], []).append(sig)
    sub = {
        code: alias_sub[code] + min(_SIGNAL_CAP, _SIGNAL * len(sig_for.get(code, ())))
        for code in region
    }
    # Eligible to be named at all: a heading hit, or more than a passing mention.
    eligible = [
        c for c in region if tier[c] == 1 or alias_sub[c] > _MIN_SCORE
    ]

    evidence_concept: Optional[str] = None
    candidates: Tuple[str, ...] = ()
    status = "unknown"

    if eligible:
        best_tier = min(tier[c] for c in eligible)
        in_tier = sorted(
            (c for c in eligible if tier[c] == best_tier),
            key=lambda c: (-sub[c], c),
        )
        best_score = sub[in_tier[0]]
        runner_up = sub[in_tier[1]] if len(in_tier) > 1 else None
        if runner_up is not None and (best_score - runner_up) < _MIN_MARGIN:
            status = "unresolved"
            candidates = tuple(
                c for c in in_tier if sub[c] > best_score - _MIN_MARGIN
            )
        else:
            status = "matched"
            evidence_concept = in_tier[0]
            candidates = (in_tier[0],)

    # Only evidence that COUNTED is returned: a human reading "why" must not be
    # shown spans that scored nothing. That is the hits in the concept's own
    # tier region, plus its corroborating signals, and only for a concept that
    # was eligible to be named.
    # ...and only for concepts that bear on the answer: the declared one, the
    # winner and any tied candidates. A runner-up that lost is not the reason.
    relevant = {c for c in (declared_concept, evidence_concept, *candidates) if c}
    evidence: List[Evidence] = []
    for code in eligible:
        if code not in relevant:
            continue
        for c, kind, start, length, _ in region[code]:
            evidence.append(Evidence(kind=kind, text=text[start:start + length],
                                     start=start, concept_code=c))
        for c, start, length in sig_for.get(code, ()):
            evidence.append(Evidence(kind="structural_signal",
                                     text=text[start:start + length],
                                     start=start, concept_code=c))

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
        agreement = "evidence_only" if evidence_concept else "neither"

    # Strongest first, capped: this is written to a review row a person reads.
    kept = sorted(
        evidence,
        key=lambda ev: (-_WEIGHT[ev.kind], ev.start, ev.concept_code),
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
