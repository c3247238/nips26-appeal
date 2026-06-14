"""Standardized evaluation metrics: BEM, CSI, AIS."""

import numpy as np
from typing import Dict, List, Optional


def compute_bem(accuracy_improvement: float, total_wd_budget: float,
                epsilon: float = 1e-8) -> float:
    """Compute Budget Equivalence Metric.

    BEM = accuracy_improvement / total_WD_budget

    Higher BEM means more efficient use of WD budget.

    Args:
        accuracy_improvement: Accuracy gain over NoWD baseline.
        total_wd_budget: Sum of lambda_t * eta_t * ||w_t||^2 over training.
        epsilon: Small constant for numerical stability.
    """
    return accuracy_improvement / (total_wd_budget + epsilon)


def compute_csi(effective_lr_trajectory: List[float]) -> float:
    """Compute Coupling Stability Index.

    CSI = 1 - Var(effective_LR) / Mean(effective_LR)^2

    CSI close to 1 means stable coupling (good).
    CSI close to 0 or negative means unstable coupling.

    Args:
        effective_lr_trajectory: List of effective LR values over training
                                (LR * (1 - lambda * LR)).
    """
    arr = np.array(effective_lr_trajectory)
    if len(arr) == 0:
        return 0.0
    mean_lr = np.mean(arr)
    var_lr = np.var(arr)
    if mean_lr == 0:
        return 0.0
    return 1.0 - var_lr / (mean_lr ** 2)


def compute_ais(alignment_signal: List[float],
                generalization_gaps: List[float],
                time_steps: Optional[List[float]] = None) -> float:
    """Compute Alignment Informativeness Score (simplified).

    AIS = correlation(alignment_residual, gen_gap_residual)

    where residuals are after removing time trend.

    A high AIS means alignment carries genuine geometric information
    beyond what time-scheduling provides.

    Args:
        alignment_signal: Per-epoch alignment values.
        generalization_gaps: Per-epoch (train_acc - test_acc).
        time_steps: Optional time indices; defaults to range.
    """
    n = min(len(alignment_signal), len(generalization_gaps))
    if n < 3:
        return 0.0

    alpha = np.array(alignment_signal[:n])
    gaps = np.array(generalization_gaps[:n])

    if time_steps is None:
        t = np.arange(n, dtype=float)
    else:
        t = np.array(time_steps[:n], dtype=float)

    # Remove linear time trend
    alpha_detrended = _detrend(alpha, t)
    gaps_detrended = _detrend(gaps, t)

    # Pearson correlation of residuals
    if np.std(alpha_detrended) < 1e-10 or np.std(gaps_detrended) < 1e-10:
        return 0.0

    corr = np.corrcoef(alpha_detrended, gaps_detrended)[0, 1]
    return float(abs(corr))  # AIS is magnitude of correlation


def _detrend(signal: np.ndarray, time: np.ndarray) -> np.ndarray:
    """Remove linear trend from signal."""
    coeffs = np.polyfit(time, signal, 1)
    trend = np.polyval(coeffs, time)
    return signal - trend
