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
from typing import Dict, List, Mapping, Optional, Tuple

from src.services.concepts.vocabulary import Vocabulary, ensure_vocabulary, fold

#: ``title_chars`` only LABELS evidence: a hit before this offset is reported as
#: kind='title_alias', after it 'body_alias'. It does not influence status,
#: concept or candidates. What a document is called is decided by its TITLE
#: SEGMENT (below), not by an offset, because an offset cliff lets one
#: character invert an answer.
_TITLE_CHARS = 600

#: Ordering of evidence kinds when a reviewer is shown the strongest first.
_WEIGHT = {"title_alias": 5.0, "body_alias": 1.0, "structural_signal": 0.5}

# ---------------------------------------------------------------------------
# How a document's own title is found.
#
# A concept lands in a TIER, and tiers are compared first, ordinally, so no
# sub-score can ever leap a tier:
#
#   tier 1 : the concept is named by the document's TITLE SEGMENT — the FIRST
#            segment of the page that, once stripped of markup and numbering,
#            *is* a type phrase rather than merely containing one.
#   tier 2 : every other alias hit, ranked by a bounded sub-score.
#
# Three rules decide that, and all three are categorical. Earlier rounds used a
# length bound plus a "the type words cover >= 75% of the line" fraction, and a
# continuous fraction cannot make a categorical distinction: measured on real
# pages, 'Bill To' scores 0.667, 'Order No' 0.714, 'Contract sum' 0.727,
# 'Quotation to' 0.818 and 'Contract #', 'PO #', 'Invoice #' and 'Quote #' all
# score 1.000 — exactly what 'PURCHASE ORDER' scores. There is no cut point.
#
#   A. EQUALITY, not coverage. Normalise a segment (strip whitespace — which
#      also disposes of '\r' and '\t' — then markup, then surrounding
#      punctuation, then a trailing bare number or reference token) and require
#      what remains to EQUAL a matched alias. So '## INVOICE', '| INVOICE |',
#      '**PURCHASE ORDER**', 'Schedule 1' and 'tax invoice' are titles, while
#      'TAX INVOICE for services rendered in period', 'Against purchase order
#      PO1 and purchase order PO2.' and 'purchase order purchase order' are
#      not. Nothing here has a threshold to tune.
#   B. A LABEL IS NOT A TITLE. In a rendered table row with more than one
#      non-empty cell, every cell except the last has its value sitting to its
#      right, which makes it a key, not a heading. That is what separates the
#      live invoice workbook's title row (where 'INVOICE' is the last non-empty
#      cell) from '| PO # | 4412 |', '| Contract sum | 1,936,000.00 |' and
#      '| Quotation to | Smith Ltd |'.
#   C. THE FIRST ONE WINS. Among title segments the first in document order is
#      the document's title, and only its concepts reach tier 1. This is
#      ordinal: no character count decides it and inserting a word cannot
#      change which segment comes first. It is what stops a multi-schedule
#      contract being classified by its own schedule headings — 'FRAMEWORK
#      AGREEMENT' precedes 'Schedule 1', so the schedules never compete.
#      REPETITION PLAYS NO PART IN TIER 1 AT ALL.
#   D. A title naming two or more types is a tie: 'INVOICE / QUOTE' gives
#      status='unresolved' with both candidates, never a choice.
#
# Within tier 2 a bounded sub-score decides: distinct aliases plus a damped
# (log2, capped) occurrence count. Structural signals are corroboration — they
# add to a concept that already has an alias hit and can break a tier-2 tie,
# but never name a type alone. A page with no title segment is decided by
# volume, which is the honest answer for a page that does not say what it is.
# ---------------------------------------------------------------------------

#: Markup the parsers wrap a title in: markdown heading hashes and bold stars,
#: and the pipes of a rendered table row.
_MARKUP = "#*|"

#: Punctuation that can sit around a title without changing what it says.
_PUNCT = "\"'`()[]{}<>«».,;:!?-–—…/\\&+=~^%$£€@"

#: A trailing token that NUMBERS a title rather than naming it: 'Schedule 1',
#: 'Annex A', 'Appendix 2.1', 'Part IV'. Deliberately narrow — 'No' is not in
#: it, so 'Order No' and 'Contract No' stay labels.
_REF_TOKEN = re.compile(r"^(?:\d+(?:[.,/]\d+)*|[a-z]|[ivxlcdm]+)$", re.IGNORECASE)

#: The one separator a title uses to name two types at once: 'INVOICE / QUOTE'.
_TITLE_SPLIT = "/"

_OCCURRENCE_CAP = 3.0
_SIGNAL = 1.0
_SIGNAL_CAP = 1.0

#: A tier-2 concept's ALIAS sub-score must be STRICTLY above this to name a
#: type. One passing mention in a clause scores exactly 1.0 and is not a
#: classification. (Tier 1 needs no minimum: the title is the claim.)
_MIN_SCORE = 1.0

#: How far ahead of the runner-up a tier-2 winner must be. Below this the
#: evidence has not chosen, and saying it has would be fabrication.
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


def _normalise(raw: str) -> str:
    """A segment reduced to what it CALLS itself, folded for comparison.

    Whitespace (so '\\r' and '\\t' cannot make a one-character cliff), then
    markup, then surrounding punctuation, then a trailing reference token —
    repeatedly, because '## Schedule 1 ##' needs all four.
    """
    s = raw
    while True:
        t = s.strip().strip(_MARKUP).strip(_PUNCT).strip()
        words = t.split()
        if len(words) > 1 and _REF_TOKEN.match(words[-1]):
            t = " ".join(words[:-1])
        if t == s:
            return fold(s)
        s = t


def _title_owners(raw: str, owners: Mapping[str, Tuple[str, ...]]) -> Tuple[str, ...]:
    """Every concept this segment NAMES, or () if it is not a title at all.

    Equality, not coverage: each '/'-separated part must itself be an alias, so
    'INVOICE', '## INVOICE' and 'INVOICE / QUOTE' are titles while 'Invoice To',
    'PO #4412' and 'TAX INVOICE for services rendered' are not.
    """
    norm = _normalise(raw)
    if not norm:
        return ()
    named: List[str] = []
    for part in norm.split(_TITLE_SPLIT):
        key = _normalise(part)
        if not key:
            return ()
        claiming = owners.get(key)
        if not claiming:
            return ()
        for code in claiming:
            if code not in named:
                named.append(code)
    return tuple(named)


def _segments(text: str) -> List[Tuple[int, int, bool]]:
    """``(start, end, may_be_a_title)`` for every line and every table cell.

    Cells are segments because the spreadsheet parser renders a whole sheet row
    as one '| a | b | c |' line, so a line-only rule cannot see a spreadsheet's
    title at all. In a row with more than one non-empty cell only the LAST
    non-empty cell may be a title: anything with a further non-empty cell to
    its right is a key whose value that cell is (rule B).
    """
    out: List[Tuple[int, int, bool]] = []
    pos, n = 0, len(text)
    while True:
        nl = text.find("\n", pos)
        end = n if nl < 0 else nl
        line = text[pos:end]
        if "|" in line:
            cells: List[Tuple[int, int]] = []
            cs = pos
            for piece in line.split("|"):
                ce = cs + len(piece)
                if piece.strip():
                    cells.append((cs, ce))
                cs = ce + 1
            last = len(cells) - 1
            for i, (a, b) in enumerate(cells):
                out.append((a, b, i == last))
        else:
            out.append((pos, end, True))
        if nl < 0:
            return out
        pos = nl + 1


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
    # promise that ("İ".lower() is two characters), so a character whose
    # lowercase is not one character is left as it is. fold() serves only the
    # alias side, because it collapses whitespace and would shift offsets.
    lowered = "".join(
        (lc if len(lc := ch.lower()) == 1 else ch) for ch in text
    ).replace("_", " ").replace("-", " ")

    # folded alias -> the concepts claiming it. Built here rather than taken
    # from vocab.alias_index so the status rule is held by THIS module: only
    # status='active' types resolve, whatever a caller hands in.
    owners: Dict[str, Tuple[str, ...]] = {}
    # (concept, kind, start, length, folded alias) for every alias hit.
    hits: List[Tuple[str, str, int, int, str]] = []
    signals: List[Tuple[str, int, int]] = []
    # ONE loop, so the status rule is enforced in exactly one place. A second
    # copy of this `continue` downstream could not fire and so could not fail,
    # and a check that cannot fail hides which line is load-bearing.
    for code, dt in vocab.document_types.items():
        if dt.status != "active":
            continue
        # A set: a concept's own name is also an alias, and counting the same
        # phrase twice would hand that concept a spurious double score.
        folded_aliases = {
            f for f in (fold(a) for a in (*dt.aliases, code.split(".", 1)[-1])) if f
        }
        for folded in folded_aliases:
            owners[folded] = (*owners.get(folded, ()), code)
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

    # Rules A, B and C: the document's title is the FIRST segment that IS a
    # type phrase, and only the concepts it names reach tier 1.
    title_concepts: Tuple[str, ...] = ()
    title_span: Tuple[int, int] = (-1, -1)
    for a, b, may_be_a_title in _segments(text):
        if not may_be_a_title:
            continue
        named = _title_owners(text[a:b], owners)
        if named:
            title_concepts, title_span = named, (a, b)
            break

    # A concept's region is the evidence that COUNTED for it. For a tier-1
    # concept that is the title segment alone; its other mentions decided
    # nothing and must not be shown as the reason.
    region: Dict[str, List[Tuple[str, str, int, int, str]]] = {}
    for h in kept_hits:
        if h[0] in title_concepts:
            if title_span[0] <= h[2] and h[2] + h[3] <= title_span[1]:
                region.setdefault(h[0], []).append(h)
        else:
            region.setdefault(h[0], []).append(h)

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

    evidence_concept: Optional[str] = None
    candidates: Tuple[str, ...] = ()
    status = "unknown"

    if title_concepts:
        # Tier 1. The title is the claim: no score, no repetition, no margin.
        if len(title_concepts) == 1:
            status = "matched"
            evidence_concept = title_concepts[0]
            candidates = title_concepts
        else:
            status = "unresolved"
            candidates = tuple(sorted(title_concepts))
    else:
        # Tier 2. Nothing on the page says what it is, so volume decides, and
        # only above a floor and by a margin.
        ranked = sorted(
            (c for c in region if alias_sub[c] > _MIN_SCORE),
            key=lambda c: (-sub[c], c),
        )
        if ranked:
            best_score = sub[ranked[0]]
            runner_up = sub[ranked[1]] if len(ranked) > 1 else None
            if runner_up is not None and (best_score - runner_up) < _MIN_MARGIN:
                status = "unresolved"
                candidates = tuple(
                    c for c in ranked if sub[c] > best_score - _MIN_MARGIN
                )
            else:
                status = "matched"
                evidence_concept = ranked[0]
                candidates = (ranked[0],)

    # Only evidence that COUNTED is returned: a human reading "why" must not be
    # shown spans that scored nothing. That is a concept's own region plus its
    # corroborating signals, and only for a concept that could be named at all.
    # ...and only for concepts that bear on the answer: the declared one, the
    # winner and any tied candidates. A runner-up that lost is not the reason.
    shown = {c for c in region if c in title_concepts or alias_sub[c] > _MIN_SCORE}
    relevant = {c for c in (declared_concept, evidence_concept, *candidates) if c}
    evidence: List[Evidence] = []
    for code in shown:
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
