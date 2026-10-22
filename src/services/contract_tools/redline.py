"""Compare two versions of a clause or contract text, word by word."""
from __future__ import annotations

import difflib
import re
from typing import Any, Dict, List

MAX_CHARS = 60_000
_WORD = re.compile(r"\S+")


def _words(text: str) -> List[str]:
    return _WORD.findall(text)


def compare(original: Any, revised: Any) -> Dict[str, Any]:
    """What was removed, what was added and what was replaced, in reading order.

    Wording only: this says the texts differ, never whether a change is acceptable.
    """
    if not isinstance(original, str) or not isinstance(revised, str):
        return {"compared": False, "note": "Give both texts, original and revised, as plain text."}
    if not original.strip() or not revised.strip():
        return {"compared": False, "note": "One of the two texts is empty; there is nothing to compare."}
    if len(original) > MAX_CHARS or len(revised) > MAX_CHARS:
        return {"compared": False,
                "note": f"A text is longer than {MAX_CHARS} characters. Compare it clause by clause."}
    a, b = _words(original), _words(revised)
    changes: List[Dict[str, Any]] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        before, after = " ".join(a[i1:i2]), " ".join(b[j1:j2])
        kind = {"delete": "removed", "insert": "added", "replace": "replaced"}[tag]
        changes.append({"change": kind, "original": before or None, "revised": after or None,
                        "context": " ".join(a[max(0, i1 - 6):i1])})
    return {"compared": True, "identical": not changes, "changes": changes,
            "words_original": len(a), "words_revised": len(b),
            "note": "Wording differences only. Whether a change is acceptable is a decision for the contract owner."}
