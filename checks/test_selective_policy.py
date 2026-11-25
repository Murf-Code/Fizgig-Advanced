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


def test_static_early_is_fixed():
    for probe in range(10):
        assert select_blocks("static_early", [None] * 28, 14, probe) == set(range(13)) | {27}


def test_walking_coverage_exact_three_phase_cycle():
    scores = list(range(28, 0, -1))
    priority = set(range(7))
    before = random.getstate()
    coverage = []
    for phase in range(6):
        chosen = select_blocks("walking_coverage", scores, 14, 0, walk_phase=phase)
        assert chosen == priority | set(range(7 + phase % 3, 28, 3))
        coverage.append(chosen - priority)
    assert set.union(*coverage[:3]) == set(range(7, 28))
    assert sum(map(len, coverage[:3])) == 21
    assert random.getstate() == before


def test_walking_coverage_handles_changed_priorities_and_small_budgets():
    scores = [i if i % 2 else -i for i in range(28)]
    priority = {15, 17, 19, 21, 23, 25, 27}
    for phase in range(3):
        chosen = select_blocks("walking_coverage", scores, 14, 2, walk_phase=phase)
        assert len(chosen) == 14 and priority <= chosen
    for n in range(1, 12):
        for budget in range(1, n + 1):
            for phase in range(4):
                assert len(select_blocks("walking_coverage", [None] * n, budget, 0, walk_phase=phase)) == budget


def test_balanced_random_covers_every_block_in_each_pair_and_preserves_rng():
    before = random.getstate()
    for pair in range(10):
        a, b = [select_blocks("balanced_random", [None] * 28, 14, 2 * pair + i) for i in range(2)]
        assert not a & b and a | b == set(range(28))
    assert random.getstate() == before


def test_static_first_half_and_hybrid_coverage():
    scores = list(range(28, 0, -1))
    counts = [0] * 28
    for probe in range(3):
        assert select_blocks("static_first_half", scores, 14, probe) == set(range(14))
        chosen = select_blocks("hybrid_coverage", scores, 14, probe, update_counts=counts)
        assert len(chosen) == 14 and set(range(7)) <= chosen
        for i in chosen:
            counts[i] += 7
    assert counts == [21] * 7 + [7] * 21
