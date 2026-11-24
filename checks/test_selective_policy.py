from fizgig.krea2.selective_policy import STRATEGIES, select_blocks
import random


def test_budget_and_inverse_rankings():
    scores = list(range(28))
    for strategy in STRATEGIES:
        assert len(select_blocks(strategy, scores, 14, 0)) == 14
    for high, low in [("downhill", "uphill"), ("high_gradient", "low_gradient")]:
        assert select_blocks(high, scores, 14, 0) == set(range(14, 28))
        assert select_blocks(low, scores, 14, 0) == set(range(14))


def test_random_does_not_consume_training_rng_and_rotation_covers_every_block():
    before = random.getstate()
    assert select_blocks("random", [None] * 28, 14, 3) == select_blocks("random", [None] * 28, 14, 3)
    assert random.getstate() == before
    a, b = [select_blocks("rotational", [None] * 28, 14, i) for i in range(2)]
    assert not a & b and a | b == set(range(28))


def test_unknown_and_tied_scores_are_budgeted():
    for strategy in STRATEGIES:
        assert len(select_blocks(strategy, [None, float("nan")] * 14, 14, 0)) == 14
