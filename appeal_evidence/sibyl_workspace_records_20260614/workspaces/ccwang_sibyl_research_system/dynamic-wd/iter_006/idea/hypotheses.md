# Testable Hypotheses

## H1: Unified Lyapunov Certificate

**Statement**: For V_t = f(w_t) + mu_t * ||w_t||^2 with mu_t satisfying a backward recursion, there exist computable bounds [lambda_min(t), lambda_max(t)] such that any WD schedule lambda(t) in this band guarantees convergence (E[V_T] <= V_0 - Omega(T)).

**Expected outcome**: The certified band is non-trivial (lambda_max - lambda_min > 0.1 * lambda_max at all training phases) and widens during early training (when gradients are large, more WD is tolerated) and narrows during fine-tuning (when stability is critical).

**Falsification**: If the certified band is so narrow that only lambda(t) approximately constant satisfies it, the certificate is too conservative. This outcome is still informative -- it theoretically explains why constant WD is hard to beat (supported by iter_003 data showing 0.25% total spread on CIFAR-10).

**Measurement**: Compute lambda_min(t), lambda_max(t) from L, sigma^2, ||w_t||, ||g_t||, delta_t at each epoch. Overlay actual lambda(t) for each method.

---

## H2: Cumulative Alignment Predicts Generalization Better Than Worst-Case

**Statement**: The cumulative average alignment bar_delta_T = (1/T) sum_t delta_t correlates more strongly with generalization gap (train_acc - test_acc) than worst-case alignment sup_t delta_t.

**Expected outcome**: Spearman |rho(bar_delta_T, gen_gap)| > |rho(sup_t delta_t, gen_gap)| by at least 0.1 across the WD strength x schedule grid.

**Falsification**: If the correlation difference is < 0.05, or if both |rho| < 0.3, the cumulative alignment improvement is not empirically supported.

**Measurement**: Grid of 6 WD strengths x 4 schedules x 2 architectures x 2 datasets = 96 configs x 3 seeds. Compute full-batch delta_t every 10 epochs. Bootstrap 95% CI for correlation difference. Partial correlation controlling for WD strength.

---

## H3: Subsumption of Existing Methods

**Statement**: Constant WD, CWD, cosine schedule, SWD, and PMP-WD all lie within the certified band [lambda_min(t), lambda_max(t)] under their standard hyperparameter ranges.

**Expected outcome**: For each method, lambda(t) lies within the certified band for >= 95% of training steps. Occasional violations (< 5% of steps) may occur during phase transitions (LR drops, BN statistics shifts).

**Falsification**: If a known-convergent method (constant, CWD) violates the band for > 20% of training steps, the certificate needs relaxation.

**Measurement**: For each method on each seed, compute the fraction of training steps where lambda(t) in [lambda_min(t), lambda_max(t)].

---

## H4: PMP-WD Optimality and Bang-Bang Structure

**Statement**: Among all WD schedules in the certified band, PMP-WD achieves the tightest (lowest) final Lyapunov function value V_T. The optimal schedule exhibits bang-bang behavior: lambda* = Lambda_max when costate-weight alignment <p(t), w(t)> > 0, and lambda* = 0 otherwise.

**Expected outcome**: V_T(PMP-WD) < V_T(constant) < V_T(cosine) < V_T(CWD). The bang-bang structure should be observable in the lambda(t) trajectory of PMP-WD.

**Falsification**: If another method achieves lower V_T than PMP-WD across all seeds, H4 is falsified. If PMP-WD does not exhibit clear bang-bang behavior, the continuous-time ODE approximation is invalid.

**Measurement**: Track V_t for all methods across training. Report V_T (mean +/- std across seeds) and test accuracy. Visualize PMP-WD's lambda(t) trajectory for bang-bang structure.

**Connection to CWD**: CWD is a specific bang-bang controller (switching on sign-alignment). PMP-WD should reveal whether CWD's switching criterion (sign alignment) is close to the optimal switching function (<p, w>). If it is, CWD's binary mask is a good approximation of the optimal control. If it isn't, CWD switches at the wrong boundary, explaining its underperformance on CIFAR (iter_003 data).

---

## H5: Alignment Informativeness Depends on Architecture

**Statement**: The alignment signal delta_hat_t is informative for WD decisions in non-BN architectures but uninformative in BN architectures where WD primarily controls effective LR.

**Expected outcome**:
- ResNet-20 with BN: alignment variance < 0.1, all WD methods perform within 0.5% (as observed in iter_003: 0.25% spread)
- ResNet-20 without BN: alignment variance > 0.2, alignment-aware methods differentiate by > 1%
- VGG-16-BN: similar to ResNet-20 with BN (alignment uninformative)

**Falsification**: If alignment-aware WD helps significantly even with BN (>0.5% improvement, p<0.05, N>=5 seeds), the effective-LR interpretation is incomplete. If alignment is uninformative even without BN, the alignment framework has deeper problems.

**Measurement**: AIS (mutual information between delta_hat_t and next-epoch test accuracy change). BN vs no-BN head-to-head comparison with same LR schedule and WD base rate.

---

## H6: Minibatch Alignment Proxy Quality

**Statement**: Minibatch alignment delta_hat_t concentrates around population alignment delta_t with deviation O(sigma / (||grad f|| * sqrt(B))).

**Expected outcome**: For batch size B=128 on CIFAR-10/ResNet-20, |delta_hat_t - delta_t| < 0.05 on average. Concentration improves with larger batches and larger gradient norms (early training).

**Falsification**: If average deviation > 0.2, the minibatch proxy is too noisy for practical alignment-aware WD. This would support the "alignment mirage" backup.

**Measurement**: Compute both full-batch (full training set) and minibatch delta_t every 10 epochs. Report mean and std of |delta_hat_t - delta_t|.

---

## H7: Budget-Matched Alignment Test

**Statement**: Alignment-aware WD achieves statistically significant test accuracy improvement over budget-matched time-only scheduling (same total sum(lambda_t * ||w_t||^2)).

**Expected outcome**: On CIFAR-10/ResNet-20 with BN: no significant difference (p > 0.1). On CIFAR-10/ResNet-20 without BN: significant improvement (p < 0.05, effect > 0.3%).

**Falsification**: If alignment-aware WD shows no benefit even without BN, alignment is not an actionable signal for WD decisions.

**Measurement**: Paired t-test across 5 seeds. Bootstrap 95% CI. Random redistribution control.

**Evidence from iter_003**: On CIFAR-10 with BN, CWD (90.06%) does not beat constant (90.13%), consistent with the prediction.

---

## H8: Contraction Spectrum Equivalence

**Statement**: Two WD methods with matched expected contraction spectra E[eig(I - Lambda_t)] achieve the same generalization bound and similar final test accuracy.

**Expected outcome**: A continuous alignment-adaptive rule calibrated to match CWD's average contraction achieves test accuracy within the statistical error bars of CWD.

**Falsification**: If methods with matched spectra achieve significantly different accuracy (> 0.5%, p < 0.05), the contraction spectrum is an insufficient characterization.

**Measurement**: Calibrate continuous alignment-adaptive lambda_t to match CWD's average eig(I - Lambda_t). Head-to-head with 3 seeds.

---

## Hypothesis Priority Ranking

| Priority | Hypothesis | Rationale |
|----------|-----------|-----------|
| 1 | H1 (Certificate) | Core theoretical contribution; if this fails, need pivot |
| 2 | H5 (BN confound) | Central empirical question; supported by iter_003 data |
| 3 | H3 (Subsumption) | Key unification claim; depends on H1 |
| 4 | H4 (PMP optimality) | Novel prediction; bang-bang explains CWD |
| 5 | H2 (Cumulative alignment) | Incremental theory; lower novelty per novelty report |
| 6 | H7 (Budget-matched) | Important control experiment |
| 7 | H6 (Minibatch concentration) | Supporting proposition |
| 8 | H8 (Contraction spectrum) | Advanced test; lower priority |
