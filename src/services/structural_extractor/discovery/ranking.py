WEIGHTS = {
    "pattern_hit": 0.40,
    "positional_prior": 0.25,
    "arithmetic_fit": 0.20,
    "uniqueness": 0.10,
    "label_semantic_similarity": 0.05,
}


def score(*, pattern_hit: float = 0, positional_prior: float = 0,
          arithmetic_fit: float = 0, uniqueness: float = 0,
          label_semantic_similarity: float = 0) -> float:
    return (
        WEIGHTS["pattern_hit"] * pattern_hit
        + WEIGHTS["positional_prior"] * positional_prior
        + WEIGHTS["arithmetic_fit"] * arithmetic_fit
        + WEIGHTS["uniqueness"] * uniqueness
        + WEIGHTS["label_semantic_similarity"] * label_semantic_similarity
    )


def pick_best(scored_candidates: list[tuple]) -> object:
    """scored_candidates: list of (Candidate, score) tuples.
    Returns the candidate with highest score; ties broken by earliest token order."""
    if not scored_candidates:
        return None
    max_score = max(s for _, s in scored_candidates)
    top_tied = [c for c, s in scored_candidates if abs(s - max_score) < 0.05]
    # Earliest document position wins — use first token's order
    return min(top_tied, key=lambda c: c.tokens[0].order if c.tokens else 0)
