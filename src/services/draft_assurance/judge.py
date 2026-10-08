"""Validate an LLM judge's scores. The model scores; code computes the overall and refuses nonsense.

A missing criterion, a score outside 1..5 or a non-number makes the judgement ``invalid`` -- it is
never averaged over what is left, because a partial score reads as a complete one.
"""

from __future__ import annotations

from typing import Any, Dict, List


class JudgeInvalid(ValueError):
    pass


def parse_judgement(raw: Any, rubric: List[str]) -> Dict[str, Any]:
    if not rubric:
        raise JudgeInvalid("the family defines no rubric")
    if not isinstance(raw, dict) or not isinstance(raw.get("scores"), dict):
        raise JudgeInvalid("the answer carries no scores object")
    scores: Dict[str, int] = {}
    for criterion in rubric:
        v = raw["scores"].get(criterion)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v != int(v) or not 1 <= v <= 5:
            raise JudgeInvalid(f"{criterion}: score must be a whole number from 1 to 5")
        scores[criterion] = int(v)
    extra = sorted(set(raw["scores"]) - set(rubric))
    rationale = raw.get("rationale")
    return {"status": "scored", "scores": scores,
            "overall": round(sum(scores.values()) / len(scores), 2),
            "rationale": rationale.strip()[:500] if isinstance(rationale, str) else None,
            "ignored_criteria": extra}
