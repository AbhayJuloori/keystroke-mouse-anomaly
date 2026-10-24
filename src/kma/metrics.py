"""Shared evaluation and monitoring metrics."""

from __future__ import annotations

import numpy as np


def equal_error_rate(
    genuine_scores: np.ndarray, impostor_scores: np.ndarray
) -> tuple[float, float, float, float]:
    """Return EER, threshold, FAR, and FRR for anomaly scores.

    Scores at or below the threshold are accepted. The chosen threshold is the
    observed score with the smallest absolute difference between FAR and FRR.
    """
    genuine = np.asarray(genuine_scores, dtype=float)
    impostor = np.asarray(impostor_scores, dtype=float)
    if genuine.size == 0 or impostor.size == 0:
        raise ValueError("EER requires both genuine and impostor scores")
    values = np.unique(np.concatenate([genuine, impostor]))
    thresholds = np.concatenate(([-np.inf], values, [np.inf]))
    far = np.array([(impostor <= threshold).mean() for threshold in thresholds])
    frr = np.array([(genuine > threshold).mean() for threshold in thresholds])
    index = int(np.argmin(np.abs(far - frr)))
    eer = float((far[index] + frr[index]) / 2.0)
    return eer, float(thresholds[index]), float(far[index]), float(frr[index])


def far_frr_at_threshold(
    genuine_scores: np.ndarray, impostor_scores: np.ndarray, threshold: float
) -> tuple[float, float]:
    """Compute false-accept and false-reject rates for anomaly scores."""
    genuine = np.asarray(genuine_scores, dtype=float)
    impostor = np.asarray(impostor_scores, dtype=float)
    return float((impostor <= threshold).mean()), float((genuine > threshold).mean())


def global_far_operating_point(
    genuine_scores: np.ndarray, impostor_scores: np.ndarray, target_far: float
) -> dict[str, float]:
    """Return the largest observed threshold that does not exceed target FAR."""
    genuine = np.asarray(genuine_scores, dtype=float)
    impostor = np.asarray(impostor_scores, dtype=float)
    candidates = np.sort(np.unique(impostor))
    measured_far = np.asarray([(impostor <= value).mean() for value in candidates])
    eligible = candidates[measured_far <= target_far]
    threshold = float(eligible[-1]) if eligible.size else float("-inf")
    far, frr = far_frr_at_threshold(genuine, impostor, threshold)
    return {
        "target_far": float(target_far),
        "threshold": threshold,
        "far": far,
        "frr": frr,
    }


def population_stability_index(
    reference: np.ndarray,
    recent: np.ndarray,
    bins: int = 10,
    smoothing: float = 0.5,
) -> float:
    """Compute PSI with open quantile bins and additive count smoothing."""
    reference = np.asarray(reference, dtype=float)
    recent = np.asarray(recent, dtype=float)
    reference = reference[np.isfinite(reference)]
    recent = recent[np.isfinite(recent)]
    if reference.size == 0 or recent.size == 0:
        return 0.0
    edges = np.unique(np.quantile(reference, np.linspace(0.0, 1.0, bins + 1)))
    if edges.size < 2:
        span = max(abs(float(reference[0])) * 0.01, 1e-6)
        edges = np.array([reference[0] - span, reference[0] + span])
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(reference, bins=edges)
    new_counts, _ = np.histogram(recent, bins=edges)
    bin_count = len(ref_counts)
    ref_pct = (ref_counts + smoothing) / (reference.size + smoothing * bin_count)
    new_pct = (new_counts + smoothing) / (recent.size + smoothing * bin_count)
    return float(np.sum((new_pct - ref_pct) * np.log(new_pct / ref_pct)))
