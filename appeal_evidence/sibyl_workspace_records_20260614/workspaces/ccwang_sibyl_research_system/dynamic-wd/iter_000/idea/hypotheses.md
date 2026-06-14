# Testable Hypotheses with Expected Outcomes

## H1: Convergence Preservation Under Time-Varying Decay

**Statement:** When lambda_t satisfies (C1) lambda_t <= c_1 * gamma_t^2 and (C2) sum lambda_t <= Lambda < infinity, time-varying SGDW achieves:

(1/T) sum_{t=0}^{T-1} E[||nabla f_S(w_t)||^2] <= C_1/sqrt(T) + C_2 * Lambda / T

preserving the O(T^{-1/2}) convergence rate of standard SGD.

**Expected Outcome:** Training loss curves of AADWD and fixed SGDW converge at the same rate (within statistical noise). The augmented potential Phi_t = f(w_t) + beta_t ||w_t||^2 decreases monotonically in expectation.

**Experimental Validation:**
- Plot train loss vs. epoch for fixed WD and AADWD on ResNet20/CIFAR-10
- Compute empirical Phi_t and verify approximate monotonic decrease
- Compare convergence speed: AADWD should be no more than 5% slower in reaching a given loss threshold

**Falsification Criterion:** If AADWD train loss is consistently >10% worse than best fixed WD at the same epoch count, H1 is weakened (though the theory allows for constant-factor differences).

---

## H2: Tighter Generalization via Cumulative Contraction

**Statement:** Define the effective alignment measure:
Delta_bar_T = sum_{t} lambda_t * delta_t / sum_{t} lambda_t

The stability bound under AADWD scales as exp(-sum lambda_t * (1 - Delta_bar_T)), which is strictly tighter than the fixed-WD bound exp(-T * lambda * (1 - delta_max)) whenever Var(delta_t) > 0.

**Expected Outcome:**
- delta_t varies significantly across training (std > 0.1 across phases)
- Delta_bar_T < delta_max (typically by 0.1-0.3)
- The implied stability bound ratio (dynamic/fixed) < 0.8

**Experimental Validation:**
- Track delta_hat_t throughout training, compute Delta_bar_T and delta_max
- Compute the implied cumulative contraction product Pi_T = prod(1 - lambda_t + L*gamma_t) for both dynamic and fixed WD
- Compare train-test accuracy gap (generalization gap): AADWD should have smaller or equal gap

**Falsification Criterion:** If delta_t is approximately constant throughout training (std < 0.03), then Delta_bar_T ~ delta_max and H2 provides negligible improvement. This would redirect the work toward Alternative A (empirical characterization to find settings where delta_t does vary).

---

## H3: Mini-Batch Alignment Proxy Reliability

**Statement:** For batch size B, the EMA-smoothed alignment proxy satisfies:
E[(hat{delta}_t^{EMA} - delta_t)^2] <= C_delta * sigma^2 / (B * (1-beta) * ||nabla f||^2 * ||w||^2)

With B=128 and beta=0.99, this error is small enough that the practical algorithm inherits the oracle stability bound up to O(1/sqrt(B)) additive degradation.

**Expected Outcome:**
- Pearson correlation between mini-batch hat{delta}_t (EMA-smoothed) and large-batch delta_t exceeds 0.85
- The EMA trajectory shows clear phase-dependent structure (not white noise)
- hat{delta}_t has std < 0.15 (after EMA smoothing with beta=0.99)

**Experimental Validation (Tier 0 diagnostic):**
- At 9 checkpoints (3 per phase: early/mid/late), compute delta_hat_t using B=128 and B=8192
- Scatter plot: x-axis = large-batch delta, y-axis = small-batch delta; compute Pearson r
- Time series: plot raw delta_hat_t and EMA(delta_hat_t) for beta in {0.9, 0.95, 0.99}
- Histogram: delta_hat_t distribution per training phase

**Falsification Criterion:** If Pearson r < 0.6 even after EMA smoothing, the mini-batch proxy is unreliable. Fallback: use epoch-level full-batch alignment, or increase batch size, or adopt Kalman filtering (from interdisciplinary perspective).

---

## H4: Practical Performance of AADWD

**Statement:** The conservative AADWD rule achieves:
- test accuracy >= best fixed WD baseline (with matched or lower hyperparameter tuning budget)
- wider optimal hyperparameter plateau: the range of c values achieving >99% of best accuracy is at least 4x wider than the analogous range for fixed lambda

**Expected Outcome:**
- On CIFAR-10/ResNet20: test accuracy 92.5% +/- 0.2% (baseline: ~92.2%)
- On CIFAR-100/ResNet20: test accuracy 69.5% +/- 0.3% (baseline: ~68.8%)
- Hyperparameter sensitivity: c in [0.5, 2.0] all achieve within 0.3% of best; for fixed WD, the analogous range is ~[3e-4, 7e-4] (a factor of 2.3x)

**Experimental Validation (Tier 1):**
- 7-way comparison: No-WD, Fixed-WD, Stagewise, CWD, Conservative, Aggressive, Square
- 3 random seeds per setting, report mean +/- std
- Hyperparameter sensitivity plot: accuracy vs. c (for dynamic) and accuracy vs. lambda (for fixed)

**Falsification Criterion:** If AADWD underperforms best fixed WD by >0.5% consistently across seeds, H4 is falsified for accuracy. The robustness claim requires the plateau analysis. If both fail, pivot to Alternative C (theory-only contribution).

---

## H5: Alignment-Awareness vs. Mere Time-Variation

**Statement:** The alignment signal in AADWD provides information beyond mere temporal variation of weight decay. Specifically, AADWD outperforms a "random dynamic WD" baseline that uses the same temporal statistics (mean, variance) of lambda_t but with shuffled/random delta_hat_t values.

**Expected Outcome:**
- AADWD outperforms random dynamic WD by >= 0.3% in test accuracy
- AADWD produces more structured lambda_t trajectories (higher autocorrelation)
- Correlation between delta_hat_t and subsequent loss change DeltaL_t is negative (confirming the alignment signal has predictive value)

**Experimental Validation (Tier 2 ablation):**
- Replace delta_hat_t with uniform random in [0, 1], keep all other hyperparameters identical
- Compare test accuracy, generalization gap, and weight norm trajectories

**Falsification Criterion:** If random dynamic WD matches AADWD (within 0.1%), then the alignment signal adds no value beyond temporal variation, undermining the core thesis. This would redirect toward studying the value of time-varying WD per se (Alternative C).

---

## Summary of Hypothesis Priority and Dependencies

| Hypothesis | Priority | Dependency | Phase |
|-----------|----------|------------|-------|
| H3 (proxy reliability) | Highest | None (gate for all others) | Tier 0 |
| H1 (convergence) | High | Theory proof | Tier 1 |
| H4 (practical performance) | High | H3 passes | Tier 1 |
| H2 (tighter bound) | Medium | Theory proof + H3 | Tier 1-2 |
| H5 (alignment vs. randomness) | Medium | H4 shows improvement | Tier 2 |

**Critical path:** H3 (Tier 0) -> H4 (Tier 1) -> H5 (Tier 2). If H3 fails, reconsider entire approach. If H4 fails on accuracy, reframe around robustness. H1 and H2 are theoretical and can proceed in parallel with experiments.
