"""Validate a classifier's answer. The model proposes; this decides whether it can be used.

Only ``from_prompt`` is classified. The negotiation paths declare their family, so for them the
only question is whether the declared family exists in config (``family_exists``).

Anything a model says that points at a record (a PO number, a supplier id) is a LOOKUP KEY: a
candidate to be confirmed in Postgres, never a fact. A key whose value is not written in the
request is rejected here, because a model that supplies an identifier nobody typed is inventing one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

MIN_CONFIDENCE = 0.70
MIN_GAP = 0.15
_KEY = re.compile(r"^[a-z][a-z0-9_]{1,40}$")


class ClassificationInvalid(ValueError):
    """The answer is unusable (malformed, unknown family, out-of-range confidence)."""


@dataclass
class Classification:
    family_id: str
    confidence: float
    candidates: List[Dict[str, Any]]
    lookup_keys: Dict[str, str]
    rejected_lookup_keys: Dict[str, str] = field(default_factory=dict)
    user_instruction: str = ""
    instruction_verbatim: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {"family_id": self.family_id, "confidence": self.confidence, "candidates": self.candidates,
                "lookup_keys": self.lookup_keys, "rejected_lookup_keys": self.rejected_lookup_keys,
                "user_instruction": self.user_instruction, "instruction_verbatim": self.instruction_verbatim}


def family_exists(family_id: Any, known: Iterable[str]) -> bool:
    return isinstance(family_id, str) and family_id in set(known)


def _confidence(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise ClassificationInvalid(f"{what} must be a number from 0 to 1")
    return float(value)


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def parse_classification(raw: Any, request: str, known_families: Iterable[str]) -> Classification:
    known = set(known_families)
    if not isinstance(raw, dict):
        raise ClassificationInvalid("the answer is not a JSON object")
    fam = raw.get("family_id")
    if not family_exists(fam, known):
        raise ClassificationInvalid(f"family_id {fam!r} is not a configured family")
    conf = _confidence(raw.get("confidence"), "confidence")
    candidates: List[Dict[str, Any]] = []
    for c in raw.get("candidates") or []:
        if not isinstance(c, dict) or not family_exists(c.get("family_id"), known):
            raise ClassificationInvalid("a candidate names a family that is not configured")
        candidates.append({"family_id": c["family_id"], "confidence": _confidence(c.get("confidence"), "candidate confidence")})
    if fam not in {c["family_id"] for c in candidates}:
        candidates.append({"family_id": fam, "confidence": conf})
    candidates.sort(key=lambda c: -c["confidence"])
    lk_in = raw.get("lookup_keys") or {}
    if not isinstance(lk_in, dict):
        raise ClassificationInvalid("lookup_keys must be an object")
    text = _squash(request)
    keys, rejected = {}, {}
    for k, v in lk_in.items():
        if not _KEY.match(str(k)) or not isinstance(v, (str, int)) or isinstance(v, bool):
            raise ClassificationInvalid(f"lookup key {k!r} is malformed")
        (keys if _squash(str(v)) and _squash(str(v)) in text else rejected)[str(k)] = str(v)
    instruction = raw.get("user_instruction")
    if not isinstance(instruction, str):
        raise ClassificationInvalid("user_instruction must be text")
    verbatim = bool(_squash(instruction)) and _squash(instruction) in text
    return Classification(family_id=fam, confidence=conf, candidates=candidates, lookup_keys=keys,
                          rejected_lookup_keys=rejected,
                          user_instruction=instruction.strip() if verbatim else (request or "").strip(),
                          instruction_verbatim=verbatim)


def clarification_for(c: Classification, labels: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
    """One question offering the top two, when confidence is low or the top two are close."""

    top = c.candidates[:2]
    # rounded: 0.70 - 0.55 is 0.1499999... in floating point, and the boundary must mean what it says
    close = len(top) == 2 and round(top[0]["confidence"] - top[1]["confidence"], 6) < MIN_GAP
    if c.confidence >= MIN_CONFIDENCE and not close:
        return None
    labels = labels or {}
    names = [labels.get(t["family_id"], t["family_id"].replace("_", " ")) for t in top]
    options = [t["family_id"] for t in top]
    question = (f"Is this {names[0]}, or {names[1]}?" if len(names) == 2
                else f"Is this {names[0]}? Say what kind of email you want.")
    return {"question": question, "options": options,
            "reason": "low confidence" if c.confidence < MIN_CONFIDENCE else "top two families are close"}
