from src.services.structural_extractor.discovery.ranking import WEIGHTS, score, pick_best


def test_weights_sum_to_one():
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6


def test_score_all_signals_zero_gives_zero():
    assert score(pattern_hit=0, positional_prior=0, arithmetic_fit=0, uniqueness=0, label_semantic_similarity=0) == 0.0


def test_score_all_signals_one_gives_one():
    assert abs(score(pattern_hit=1, positional_prior=1, arithmetic_fit=1, uniqueness=1, label_semantic_similarity=1) - 1.0) < 1e-6


def test_pick_best_tie_breaker_earlier_wins():
    # Two candidates with identical scores — earlier doc position wins
    from src.services.structural_extractor.discovery.type_entities import Candidate
    from src.services.structural_extractor.parsing.model import Token, BBox
    t1 = Token(text="a", anchor=BBox(1, 0, 0, 0, 0), order=10)
    t2 = Token(text="b", anchor=BBox(1, 0, 0, 0, 0), order=20)
    c1 = Candidate(text="a", tokens=[t1], parsed_value=None)
    c2 = Candidate(text="b", tokens=[t2], parsed_value=None)
    # Both score 0.5
    scored = [(c1, 0.5), (c2, 0.5)]
    assert pick_best(scored) is c1
