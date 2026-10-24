import numpy as np

from kma.metrics import equal_error_rate, population_stability_index


def test_equal_error_rate_known_half_error():
    eer, threshold, far, frr = equal_error_rate(
        np.array([0.1, 0.4]), np.array([0.3, 0.5])
    )
    assert eer == 0.5
    assert threshold == 0.3
    assert far == frr == 0.5


def test_psi_same_and_shifted_distributions():
    reference = np.arange(100, dtype=float)
    assert population_stability_index(reference, reference) == 0.0
    assert population_stability_index(reference, reference + 1000) > 0.2


def test_minimum_drift_window_keeps_noise_under_half_threshold():
    from kma.pipeline import minimum_drift_window, null_psi

    window = minimum_drift_window(bins=10, reference=200, threshold=0.2)
    assert window == 164
    assert null_psi(10, window, 200) <= 0.1
    assert null_psi(10, window - 1, 200) > 0.1
