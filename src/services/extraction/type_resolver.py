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
#   B. A LABEL IS NOT A TITLE. Only the LAST non-empty cell of a rendered table
#      row may be a title: every earlier cell has its value sitting to its
#      right, which makes it a key. That separates '| PO # | 4412 |',
#      '| Contract sum | 1,936,000.00 |' and '| Quotation to | Smith Ltd |'
#      from a real title row. It must be the last-cell reading and NOT "a row
#      with two or more non-empty cells has no title": NONE of the 50
#      process_monitor documents measured has a single-cell title row (local
#      files do — 'quote_scenario_1.xlsx' titles itself '| Quotation |  |  |'),
#      so the cell-count reading throws those 50 away. They title themselves
#
#          | Ironbridge Managed IT Ltd |  |  | INVOICE |  |
#          | M | Meridia Cloud Platforms Ltd |  | INVOICE |  |
#
#      because the spreadsheet parser renders a sheet row at the full sheet
#      width. Measured over those 50 documents, the cell-count reading scores
#      7/10 invoices and calls MCP-INV-1148 an ORDER, confidently and wrongly;
#      the last-cell reading scores 10/10 invoices and 5/5 POs.
#      B's premise is "a key's value sits in the next position the layout
#      provides", and beside-it is only one such layout. VERTICALLY, the value
#      sits BELOW: a segment whose value position holds a bare reference token
#      (one token carrying a digit and naming no type) is a key too. That is
#      what stops 'Purchase Order' above 'PO-2024-0145' and the last column
#      header of '| Line | Description | Order |' above '| … | PO-1 |' from
#      being read as the page's title. Still categorical, still no threshold.
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

#: ...but NOT on the tail. A trailing ',', ';' or ':' is the one typographic
#: mark whose whole meaning is "the value follows", so stripping it would turn
#: 'Order:' and 'Order Date:' — both real parsed segments in the live corpus —
#: into confident titles. This is rule B applied consistently rather than a
#: separate label rule. The cost is that a genuine 'INVOICE:' title falls to
#: tier 2, and vertical reference blocks are far commoner than those.
_PUNCT_TAIL = "\"'`()[]{}<>«».!?-–—…/\\&+=~^%$£€@"

#: A trailing token that NUMBERS a title rather than naming it: 'Schedule 1',
#: 'Annex A', 'Appendix 2.1', 'Part IV'. Deliberately narrow — 'No' is not in
#: it, so 'Order No' and 'Contract No' stay labels.
_REF_TOKEN = re.compile(r"^(?:\d+(?:[.,/]\d+)*|[a-z]|[ivxlcdm]+)$", re.IGNORECASE)

#: A trailing '#'-prefixed reference, as one token ('INVOICE #9920') or two
#: ('Quotation  # WSG100024'). Real documents title themselves this way, and
#: without this the whole segment is not an alias and the page falls through to
#: whatever its body clauses happen to mention. '#' already means "number", so
#: the token after it is not required to carry a digit — a digit test there
#: changed no outcome any test could reach, and a condition that cannot fail
#: hides which line is load-bearing.
_HASH_REF = re.compile(r"^#\S*\d\S*$")
_HAS_DIGIT = re.compile(r"\d")

#: The one separator a title uses to name two types at once: 'INVOICE / QUOTE'.
_TITLE_SPLIT = "/"

#: A segment that is somebody's VALUE, not a name: one whitespace-separated
#: token carrying a digit and naming no type ('PO-2024-0145', '4412', 'PO-1').
#: A segment whose value position holds one of these is a key (rule B,
#: vertical form) — see _is_reference_value.
_VALUE_MAX_TOKENS = 1

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

#: Cap on DISCRETIONARY evidence rows (a review row a person reads, not a log).
#: The per-candidate reservation is a FLOOR that outranks it: every candidate of
#: an unresolved result keeps at least one span, so the true bound is
#: max(_MAX_EVIDENCE, len(candidates)) and a 15-candidate tie returns 15 rows.
#: Capping below the candidate count would hide options from the person being
#: asked to choose, which is worse than a long row.
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
    markup, then surrounding punctuation — but NOT a trailing ',', ';' or ':',
    which mean "the value follows" — then a trailing '#'-prefixed reference,
    then a trailing bare reference token. Repeatedly, because '## Schedule 1 ##'
    needs several passes.
    """
    s = raw
    while True:
        t = s.strip().strip(_MARKUP).lstrip(_PUNCT).rstrip(_PUNCT_TAIL).strip()
        words = t.split()
        if len(words) > 1 and _HASH_REF.match(words[-1]):
            t = " ".join(words[:-1])            # 'INVOICE #9920'
        elif len(words) > 2 and words[-2] == "#":
            t = " ".join(words[:-2])            # 'Quotation  # WSG100024'
        elif len(words) > 1 and _REF_TOKEN.match(words[-1]):
            t = " ".join(words[:-1])            # 'Schedule 1', 'Annex A'
        if t == s:
            return fold(s)
        s = t


def _is_reference_value(raw: str, owners: Mapping[str, Tuple[str, ...]]) -> bool:
    """Is this segment somebody's VALUE rather than a name of its own?

    One whitespace-separated token carrying a digit and naming no type:
    'PO-2024-0145', '4412', 'PO-1'. Rule B's vertical form uses it twice: the
    segment ABOVE one of these is its key, and the value itself is not a title.

    'Naming no type' is tested on the token WITH its digits intact, deliberately
    not through _normalise(): that strips trailing numbers repeatedly, so
    'PO-2024-0145' reduces through 'po 2024' to the bare alias 'po' and the
    reference would claim to name an order. A real alias carries no digits, so
    folding the token once is the whole test.
    """
    body = raw.strip()
    if not body or len(body.split()) > _VALUE_MAX_TOKENS:
        return False
    return bool(_HAS_DIGIT.search(body)) and fold(body) not in owners


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


def _segments(text: str) -> List[Tuple[int, int, bool, int, int]]:
    """``(start, end, may_be_a_title, line_index, column)`` for every line and
    every table cell, in document order. ``column`` is -1 for a plain line, else
    the cell's index among the pipe-separated pieces of its row.

    Cells are segments because the spreadsheet parser renders a whole sheet row
    as one '| a | b | c |' line, so a line-only rule cannot see a spreadsheet's
    title at all. In a row with more than one non-empty cell only the LAST
    non-empty cell may be a title: anything with a further non-empty cell to
    its right is a key whose value that cell is (rule B, horizontal form).
    """
    out: List[Tuple[int, int, bool, int, int]] = []
    pos, n, ln = 0, len(text), 0
    while True:
        nl = text.find("\n", pos)
        end = n if nl < 0 else nl
        line = text[pos:end]
        if "|" in line:
            cells: List[Tuple[int, int, int]] = []
            cs = pos
            for col, piece in enumerate(line.split("|")):
                ce = cs + len(piece)
                if piece.strip():
                    cells.append((cs, ce, col))
                cs = ce + 1
            last = len(cells) - 1
            for i, (a, b, col) in enumerate(cells):
                out.append((a, b, i == last, ln, col))
        else:
            out.append((pos, end, True, ln, -1))
        if nl < 0:
            return out
        pos = nl + 1
        ln += 1


def _value_position(
    segs: List[Tuple[int, int, bool, int, int]], i: int, text: str
) -> Optional[Tuple[int, int]]:
    """Where this segment's value would sit, if it were a key.

    Rule B's premise is that a key's value sits in the next position the layout
    provides. For a plain LINE that is the next non-empty segment in document
    order. For a table CELL it is the same COLUMN of the next row, because a
    column header's value sits below it rather than beside it — which is what
    stops the last column header of '| Line | Description | Order |' becoming
    the page's title.
    """
    a, b, _, ln, col = segs[i]
    if col < 0:
        for j in range(i + 1, len(segs)):
            if text[segs[j][0]:segs[j][1]].strip():
                return segs[j][0], segs[j][1]
        return None
    for j in range(i + 1, len(segs)):
        if segs[j][3] != ln + 1:
            if segs[j][3] > ln + 1:
                return None
            continue
        if segs[j][4] == col:
            return segs[j][0], segs[j][1]
    return None


def _title_parts(
    text: str, span: Tuple[int, int], code: str,
    owners: Mapping[str, Tuple[str, ...]],
) -> List[Tuple[int, int]]:
    """Tight spans within the title segment that name ``code``.

    The '/'-delimited part that names this concept, where one can be identified,
    otherwise the whole segment. Offsets are into the ORIGINAL text and the
    content is trimmed of surrounding whitespace, so the span stays verbatim and
    byte-exact.
    """
    lo, hi = span

    def tight(a: int, b: int) -> Tuple[int, int]:
        piece = text[a:b]
        return (a + len(piece) - len(piece.lstrip()),
                b - (len(piece) - len(piece.rstrip())))

    found: List[Tuple[int, int]] = []
    start = lo
    for raw in text[lo:hi].split(_TITLE_SPLIT):
        stop = start + len(raw)
        if code in _title_owners(raw, owners):
            a, b = tight(start, stop)
            if b > a:
                found.append((a, b))
        start = stop + 1
    if found:
        return found
    a, b = tight(lo, hi)
    return [(a, b)] if b > a else []


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
    segs = _segments(text)
    for i, (a, b, may_be_a_title, _ln, _col) in enumerate(segs):
        if not may_be_a_title:
            continue
        named = _title_owners(text[a:b], owners)
        if not named:
            continue
        # Rule B, vertical form, twice over.
        # (i) A reference is not a name. 'PO-2024-0145' would otherwise BE a
        #     title, because _normalise strips its numbers down to the alias
        #     'po' — so without this the defect merely moves one line down from
        #     the key to its own value.
        if _is_reference_value(text[a:b], owners):
            continue
        # (ii) If this segment's value position holds a bare reference token then
        #     this segment is that value's KEY, not the page's title. 'Purchase
        #     Order' above 'PO-2024-0145' is a reference block, and 'Order' as
        #     the last column header above 'PO-1' is a column name.
        value = _value_position(segs, i, text)
        if value is not None and _is_reference_value(text[value[0]:value[1]], owners):
            continue
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

    # Rule A compares a FOLDED segment against a folded alias, and fold()
    # collapses internal whitespace while the match copy keeps the page's own
    # length. So 'FRAMEWORK  AGREEMENT' names a concept that has NO alias hit
    # inside its own span, and without this a 'matched' result — or a candidate
    # a person is asked to choose between — would carry no reason at all. The
    # title segment itself is the reason: verbatim original text at a real
    # offset. Attributed per '/'-part where the title names more than one type,
    # so both sides of a tie get their own words.
    if title_concepts:
        for code in title_concepts:
            if region.get(code):
                continue
            for lo, hi in _title_parts(text, title_span, code, owners):
                region.setdefault(code, []).append((
                    code,
                    "title_alias" if lo < title_chars else "body_alias",
                    lo, hi - lo, _normalise(text[lo:hi]) or code,
                ))

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
    # max(0, ...): once the reservation alone exceeds the cap, a plain
    # subtraction is NEGATIVE and the slice then ADDS rows instead of none — a
    # 15-candidate page returned 15 rows plus whatever the negative slice let
    # through. Rule D is what makes more than 12 candidates reachable, where the
    # deleted phrase cap used to bound it at 2.
    room = max(0, _MAX_EVIDENCE - len(chosen))
    chosen += [ev for ev in ordered if ev not in chosen][:room]
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
