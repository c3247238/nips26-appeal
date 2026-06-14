"""Alignment analysis utilities."""

import numpy as np
from typing import Dict, List, Tuple


def compute_alignment_stats(trajectories: Dict[str, Dict[str, List[float]]]) -> Dict:
    """Compute alignment statistics from logged trajectories.

    Args:
        trajectories: Per-layer trajectory data from DiagnosticLogger.

    Returns:
        Dictionary with:
        - alpha_bar: Average alignment across layers and epochs
        - delta_max: Max absolute alignment
        - per_layer_alpha_bar: Per-layer average alignment
        - alignment_variance: Variance of alignment signal
    """
    all_alphas = []
    per_layer_stats = {}

    for layer_name, data in trajectories.items():
        alphas = np.array(data.get('alpha_t', []))
        if len(alphas) == 0:
            continue

        all_alphas.extend(alphas.tolist())
        per_layer_stats[layer_name] = {
            'alpha_bar': float(np.mean(alphas)),
            'alpha_std': float(np.std(alphas)),
            'delta_max': float(np.max(np.abs(alphas))),
        }

    all_alphas = np.array(all_alphas)

    return {
        'alpha_bar': float(np.mean(all_alphas)) if len(all_alphas) > 0 else 0.0,
        'delta_max': float(np.max(np.abs(all_alphas))) if len(all_alphas) > 0 else 0.0,
        'alignment_variance': float(np.var(all_alphas)) if len(all_alphas) > 0 else 0.0,
        'per_layer': per_layer_stats,
    }


def compute_snr(alignment_values: List[float]) -> float:
    """Compute alignment Signal-to-Noise Ratio.

    SNR = E[alpha]^2 / Var[alpha]

    Args:
        alignment_values: List of alignment cosine values.
    """
    arr = np.array(alignment_values)
    if len(arr) < 2:
        return 0.0
    mean_sq = np.mean(arr) ** 2
    var = np.var(arr)
    if var < 1e-12:
        return float('inf') if mean_sq > 0 else 0.0
    return mean_sq / var


def fit_polynomial_alignment(alpha_trajectory: List[float], degree: int = 4) -> Tuple[float, np.ndarray]:
    """Fit polynomial to alignment trajectory and return R-squared.

    Used for H7 temporal predictability gate.

    Args:
        alpha_trajectory: Alignment values over epochs.
        degree: Polynomial degree (default 4).

    Returns:
        (r_squared, coefficients)
    """
    n = len(alpha_trajectory)
    if n <= degree + 1:
        return 0.0, np.zeros(degree + 1)

    x = np.arange(n, dtype=float)
    y = np.array(alpha_trajectory)

    coeffs = np.polyfit(x, y, degree)
    y_pred = np.polyval(coeffs, x)

    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    if ss_tot < 1e-12:
        r_squared = 1.0  # Constant signal
    else:
        r_squared = 1.0 - ss_res / ss_tot

    return float(r_squared), coeffs
