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

import bisect
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.services.concepts.vocabulary import Vocabulary, ensure_vocabulary, fold

#: ``title_chars`` only LABELS evidence: a hit before this offset is reported as
#: kind='title_alias', after it 'body_alias'. It does not influence status,
#: concept or candidates. What a document is called is decided by LINES (below),
#: not by an offset, because an offset cliff lets one character invert an answer.
_TITLE_CHARS = 600

#: Ordering of evidence kinds when a reviewer is shown the strongest first.
_WEIGHT = {"title_alias": 5.0, "body_alias": 1.0, "structural_signal": 0.5}

#: How the evidence is compared. NOT one summed score: a summed score in a narrow
#: range let each new term outbid the last. A concept lands in a TIER and tiers
#: are compared first, ordinally (so a sub-score change can never leap a tier):
#:   tier 1 : an alias hit on a HEADING-LIKE LINE, wherever in the document that
#:            line is (a letterhead above the title is normal). A line is
#:            heading-like when it is short (<= _HEADING_LINE_MAX characters
#:            after stripping, so a sentence is not a heading) and the type
#:            phrases on it make up at least _HEADING_COVERAGE of its
#:            non-whitespace characters (so 'PURCHASE ORDER', '## INVOICE' and
#:            'INVOICE / QUOTE' qualify while 'Ref: your quotation of 1 January'
#:            and 'Invoice No: INV-2026-0001' do not), and it carries at most
#:            _HEADING_MAX_PHRASES distinct type phrases and does not end like
#:            a sentence or a label. Case is NOT required.
#:   tier 2 : every other alias hit.
#: Only within one tier does a bounded sub-score decide: distinct aliases plus a
#: damped (log2, capped) occurrence count. Structural signals are corroboration:
#: they add to a concept that already has an alias hit and can break a tie, but
#: never name a type alone. A text with no line breaks has no heading-like line,
#: so volume decides there; that is the honest answer for a page with no title.
_HEADING_LINE_MAX = 80
_HEADING_COVERAGE = 0.75
#: A title names its type once, or twice ('INVOICE / QUOTE'). A short line that
#: is nothing but a list of type words ('Schedule 1, Annex A, Appendix 2,
#: Exhibit B.' or 'See our Quotation, Estimate and prior Quotes') is a list of
#: cross-references, not a heading.
_HEADING_MAX_PHRASES = 2
#: ...and never the SAME phrase twice ('purchase order PO1 and purchase order
#: PO2' is a sentence citing orders), and never ends like a sentence or a label
#: ('Please see the purchase order.', 'Bill To:').
_SENTENCE_END = (".", ",", ";", ":")
_OCCURRENCE_CAP = 3.0
_SIGNAL = 1.0
_SIGNAL_CAP = 1.0

#: A tier-2 concept's ALIAS sub-score must be STRICTLY above this to name a
#: type. One passing mention in a clause scores exactly 1.0 and is not a
#: classification. (Tier 1 needs no minimum: the heading is the claim.)
_MIN_SCORE = 1.0

#: How far ahead of the runner-up the winner must be. Below this the evidence
#: has not chosen, and saying it has would be fabrication.
_MIN_MARGIN = 1.0

#: Cap on evidence rows (a review row a person reads, not a log). Each
#: candidate of an unresolved result is guaranteed at least one of them.
_MAX_EVIDENCE = 12


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
    """Classify a document from its own text, reporting rather than deciding.

    ``title_chars`` labels evidence as title_alias / body_alias; it does not
    change the outcome.
    """
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
                for st in found:
                    # Pseudo-hit: suppresses alias hits inside the phrase ('bill'
                    # inside 'Bill-To Address'). Never scored or returned.
                    hits.append((code, "signal_span", st, len(folded), folded))

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
            kept_hits.extend(h for h in hits[i:j] if h[1] != "signal_span")
        max_end = max(max_end, span[1])
        i = j

    # Which lines are heading-like: short, and mostly made of type phrases.
    line_starts = [0] + [m.end() for m in re.finditer(r"[\n|]", text)]
    def _line_of(pos: int) -> int:
        return bisect.bisect_right(line_starts, pos) - 1
    covered: Dict[int, set] = {}
    for h in kept_hits:
        covered.setdefault(_line_of(h[2]), set()).add((h[2], h[2] + h[3]))
    heading_lines = set()
    for ln, spans in covered.items():
        lo = line_starts[ln]
        hi = line_starts[ln + 1] - 1 if ln + 1 < len(line_starts) else len(text)
        line = text[lo:hi]
        content = len(re.sub(r"[\s#*]", "", line))
        if (not content or len(line.strip()) > _HEADING_LINE_MAX
                or len(spans) > _HEADING_MAX_PHRASES
                or line.rstrip(" *").endswith(_SENTENCE_END)
                or len({fold(text[a:b]) for a, b in spans}) < len(spans)):
            continue
        said = sum(len(re.sub(r"[\s#*]", "", text[a:b])) for a, b in spans)
        if said >= _HEADING_COVERAGE * content:
            heading_lines.add(ln)

    region: Dict[str, List[Tuple[str, str, int, int, str]]] = {}
    tier: Dict[str, int] = {}
    for h in kept_hits:
        t = 1 if _line_of(h[2]) in heading_lines else 2
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
        sig_for.setdefault(sig[0], []).append(sig)
    # Only concepts in `region` (those with an alias hit) are ever scored, so a
    # signal-only concept has no score and cannot be named.
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

    # Strongest first, capped, but every candidate of an unresolved result keeps
    # at least its best span: a person asked to choose must be shown both sides.
    key = lambda ev: (-_WEIGHT[ev.kind], ev.start, ev.concept_code)
    ordered = sorted(evidence, key=key)
    chosen: List[Evidence] = []
    if status == "unresolved":
        for cand in candidates:
            best = next((ev for ev in ordered if ev.concept_code == cand), None)
            if best is not None:
                chosen.append(best)
    chosen += [ev for ev in ordered if ev not in chosen][:_MAX_EVIDENCE - len(chosen)]
    kept = sorted(chosen, key=key)

    return TypeResolution(
        declared_concept=declared_concept,
        evidence_concept=evidence_concept,
        status=status,
        agreement=agreement,
        candidates=candidates,
        evidence=tuple(kept),
    )


__all__ = ["Evidence", "TypeResolution", "resolve_document_type"]
