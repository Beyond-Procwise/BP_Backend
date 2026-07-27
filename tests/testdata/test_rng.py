import pytest

from scripts.testdata.rng import make_rng, weighted_apportion


def test_same_seed_and_stream_give_identical_sequences():
    a = [make_rng(42, "suppliers").random() for _ in range(5)]
    b = [make_rng(42, "suppliers").random() for _ in range(5)]
    assert a == b


def test_different_streams_diverge():
    a = make_rng(42, "suppliers").random()
    b = make_rng(42, "documents").random()
    assert a != b


def test_different_seeds_diverge():
    assert make_rng(1, "suppliers").random() != make_rng(2, "suppliers").random()


def test_apportion_sums_to_exact_total():
    assert sum(weighted_apportion([1, 1, 1], 5000)) == 5000


def test_apportion_respects_relative_weights():
    result = weighted_apportion([3, 1], 100)
    assert result == [75, 25]


def test_apportion_handles_uneven_division():
    result = weighted_apportion([1, 1, 1], 10)
    assert sum(result) == 10
    assert sorted(result) == [3, 3, 4]


def test_apportion_is_deterministic():
    assert weighted_apportion([5, 3, 2, 7], 999) == weighted_apportion([5, 3, 2, 7], 999)


def test_apportion_rejects_empty_weights():
    with pytest.raises(ValueError):
        weighted_apportion([], 10)


def test_apportion_rejects_zero_total_weight():
    with pytest.raises(ValueError):
        weighted_apportion([0, 0], 10)
