import pytest
from flsim.metrics import compute_round_statistics


def test_compute_round_statistics():
    results = [
        {"round": 1, "metrics": {"acc": 0.7}, "balances": {1: 5.0, 2: 3.0}},
        {"round": 2, "metrics": {"acc": 0.8}, "balances": {1: 8.0, 2: 5.0}},
    ]
    detection_summary = [
        {"round": 1, "recall": 0.5},
        {"round": 2, "recall": 1.0},
    ]

    stats = compute_round_statistics(results, detection_summary)
    assert stats["accuracy_per_round"] == [(1, 0.7), (2, 0.8)]
    assert stats["detection_rate_per_round"] == [(1, 0.5), (2, 1.0)]
    assert stats["balances_per_round"] == [
        (1, {1: 5.0, 2: 3.0}),
        (2, {1: 8.0, 2: 5.0}),
    ]
