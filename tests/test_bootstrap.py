import numpy as np

from edge_opt.metrics import bootstrap_ci


def test_bootstrap_ci() -> None:
    data = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    mean, low, high = bootstrap_ci(data, statistic=np.mean, n_resamples=500, ci=0.95)
    assert 0.4 <= mean <= 0.6
    assert low <= mean <= high
