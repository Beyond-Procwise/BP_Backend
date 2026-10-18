"""Retrieval grading for /ask: is what came back relevant enough to answer from?

Retrieval always returns *something*, the nearest text, however far away it is. Asked about a force majeure clause
that is not in the contract, it returns the contract anyway, and the model is left to decide that the clause is
missing (test B1). It did so honestly; this makes it a decision made on a number instead of left to chance.

The number is the cross-encoder's score for the best document it scored (the reranker reads the question and the
passage together, so it is the one score that says "this passage answers this question"). Below a per-model
threshold the query is rewritten and tried again, at most twice and without a model call: the retries are
deterministic rewordings. If the best attempt is still below the threshold and nothing else (the database facts,
an exact-id record, an uploaded file) can ground an answer, the reply is an explicit "not found" and nothing is
written from the weak documents.

A reranker this module has no threshold for, or a retrieval where the reranker did not run, is *ungraded*: it is
passed through untouched rather than judged on a score that does not mean anything.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

# Per-reranker "relevant enough" thresholds, from scripts/ask_testsheet/grade_calibration.py.
#
# DELIBERATELY EMPTY, so grading is inert (every retrieval is "ungraded" and passes through untouched).
# Measured 2026-10-18 on this host's corpus with bge-reranker-large (scores are probabilities, 0..1):
#   questions the store CAN answer (12):  best score 0.0, 0.0, 0.08, 0.15, 0.0, 0.94, 0.07, 0.0, 0.02, 0.0, 0.99, 0.01
#   questions it CANNOT (8):              best score 0.01, 0.0, 0.0, 0.0, 0.0, 0.04, 0.0, 0.0 (max 0.04)
# Ten of twelve answerable questions score no higher than the unanswerable ones (a Delegation of Authority
# policy exists, yet "what does the delegation of authority say about capital expenditure?" scores 0.08), so no
# threshold separates the two: one that caught every absent question would refuse most answerable ones, and a
# false "not found" is worse than the honest "no clause found" the model gives today (test B1). Do not fill this
# in without re-running the calibration after the reranker input or the store has changed.
THRESHOLDS: Dict[str, float] = {}

MAX_RETRIES = 2
RETRY_BUDGET_S = 45.0        # no retry is started once retrieval has already taken this long

_STOP = frozenset("""a an the of to in on for with by from at as is are was were be been being do does did can could would should
will shall may might must i we you he she it they them his her its our your their this that these those there here what which who
whom whose when where why how any some all each every please tell show give me us about regarding into than then also just""".split())
_QUESTION_FILLER = frozenset("explain describe list find get summarise summarize provide include includes included say says".split())


@dataclass(frozen=True)
class Grade:
    score: Optional[float]        # best cross-encoder score among documents it scored; None if none were scored
    threshold: Optional[float]
    graded: bool                  # False -> no judgement was possible; treat as relevant
    relevant: bool

    def as_dict(self) -> Dict[str, Any]:
        return {"score": None if self.score is None else round(self.score, 3), "threshold": self.threshold,
                "graded": self.graded, "relevant": self.relevant}


def best_score(hits: Sequence[Any]) -> Optional[float]:
    scores = [float(h.rerank_score) for h in hits if getattr(h, "reranked", False) and isinstance(getattr(h, "rerank_score", None), (int, float))]
    return max(scores) if scores else None


def grade(hits: Sequence[Any], reranker_model: Optional[str]) -> Grade:
    threshold = THRESHOLDS.get(reranker_model or "")
    score = best_score(hits)
    if threshold is None or score is None:
        return Grade(score, threshold, False, True)
    return Grade(score, threshold, True, score >= threshold)


def _tokens(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9_\-/.]*", text)


def retry_queries(query: str) -> List[str]:
    """Up to MAX_RETRIES deterministic rewordings of ``query``, each different from it and from each other.

    1. the significant words only (question words and filler removed), which stops a long sentence from
       out-voting the one phrase that matters;
    2. the names and identifiers, plus the longest remaining words: the entity and the topic, nothing else.
    """
    toks = _tokens(query or "")
    # An all-capitals token is a name or acronym ("IT" in "Ironbridge Managed IT Ltd"), never the pronoun it looks like.
    significant = [t for t in toks if t.isupper() and len(t) > 1 or (t.lower() not in _STOP and t.lower() not in _QUESTION_FILLER)]
    variants: List[str] = []
    if significant:
        variants.append(" ".join(significant))
    names = [t for i, t in enumerate(toks) if (t[:1].isupper() and i > 0) or any(c.isdigit() for c in t)]
    rest = sorted({t.lower() for t in significant if t not in names}, key=lambda w: (-len(w), w))[:3]
    focus = " ".join(dict.fromkeys(names + rest))
    if focus:
        variants.append(focus)
    seen = {re.sub(r"\s+", " ", query or "").strip().lower()}
    out: List[str] = []
    for v in variants:
        key = v.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(v.strip())
    return out[:MAX_RETRIES]


NOT_FOUND = ("I couldn't find anything in your documents or records that answers this, so I haven't guessed. "
             "I searched {count} {documents} and none was relevant enough to rely on. "
             "If you tell me the supplier, document or clause you mean, I'll look again.")


def not_found_text(searched: int) -> str:
    return NOT_FOUND.format(count=searched, documents="document" if searched == 1 else "documents")
