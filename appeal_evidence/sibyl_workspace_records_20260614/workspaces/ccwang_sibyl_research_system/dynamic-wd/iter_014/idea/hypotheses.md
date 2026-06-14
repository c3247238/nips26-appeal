# Testable Hypotheses

## H1: Unified Control Law Subsumption

**Statement**: All effective dynamic WD methods (CWD, SWD, CPR, Defazio corrective) can be expressed as special cases of the UDWDC control law parameterized by (rho*, K_p, K_i, K_d, alignment signal).

**Test**: For each existing method, compute its effective lambda_t trajectory on ResNet-20/CIFAR-10 over 200 epochs. Find the UDWDC gain settings (K_p, K_i, K_d) that minimize ||lambda_t^UDWDC - lambda_t^method||_2 over the trajectory. Report relative error.

**Expected outcome**: Relative error < 10% for CWD, SWD, Defazio corrective, and CPR.

**Falsification**: If more than 2 methods have relative error > 20%, the unified parameterization is too coarse. The methods are genuinely independent rather than special cases.

**Priority**: HIGH (foundational claim)

---

## H2: Prescriptive Target Trajectory

**Statement**: Setting rho*(t) = eta_t / tau with tau derived from initial gradient statistics (tau = 1 / (lambda_0 * eta_0)) produces WD coefficients that match or exceed the best fixed-lambda baseline without hyperparameter search.

**Test**:
- CIFAR-10/ResNet-20: Train with UDWDC (K_p only, rho* from theory) vs grid-searched fixed WD (lambda in {1e-4, 3e-4, 5e-4, 1e-3, 3e-3, 5e-3}). Compare test accuracy.
- CIFAR-100/VGG-16-BN: Same comparison.
- ImageNet/ResNet-50: Same comparison with lambda in {1e-3, 5e-3, 1e-2, 3e-2, 5e-2}.

**Expected outcome**: UDWDC with theory-derived rho* within 0.3% of best grid-searched fixed WD on all three benchmarks.

**Falsification**: UDWDC underperforms the best fixed-WD baseline by > 0.5% on 2+ benchmarks.

**Priority**: HIGH (practical value)

---

## H3: Batch-Size-Dependent Alignment Signal Quality

**Statement**: The continuous alignment signal delta_hat_t = cos(g_t, w_t) has a noise-to-signal ratio that increases with parameter dimensionality d and decreases with batch size b. At b <= 256, binary masking (CWD) is preferred over continuous modulation.

**Test**: ResNet-20/CIFAR-100, batch sizes in {64, 128, 256, 512, 1024}. For each batch size, compare:
- Fixed WD
- CWD (binary sign-alignment)
- UDWDC with continuous alignment (K_d > 0)
- UDWDC with EMA-smoothed continuous alignment
Track effective SNR = E[delta_hat_t]^2 / Var[delta_hat_t] per batch size.

**Expected outcome**:
- At b=64-128: CWD >= continuous UDWDC >= fixed WD
- At b=512-1024: continuous UDWDC >= CWD >= fixed WD
- SNR increases monotonically with batch size
- Crossover point at approximately b=256-512

**Falsification**: Continuous alignment outperforms CWD at b=64, contradicting the noise hypothesis.

**Priority**: HIGH (addresses major contrarian concern)

---

## H4: Budget-Efficient Generalization

**Statement**: Alignment-modulated WD achieves higher marginal generalization improvement per unit WD budget than fixed WD, with the efficiency gain proportional to alignment variance Var_t[delta_t].

**Test**: CIFAR-100/ResNet-20, WD budget sweep. Compare:
- Fixed WD at total budgets B in {0.5x, 1x, 2x, 4x} of baseline
- Alignment-modulated WD (UDWDC) at same total budgets
Measure generalization gap at each budget level.

**Expected outcome**: dAcc/dB is steeper for alignment-modulated WD than fixed WD at low B, with slope ratio correlating with Var_t[delta_t].

**Falsification**: Fixed WD at matched total budget equals or exceeds alignment-modulated WD at all budget levels.

**Priority**: MEDIUM (theoretical claim, secondary to empirical performance)

---

## H5: Per-Layer Fixed-Point Differentiation

**Statement**: Under alignment-modulated WD on networks with normalized layers, per-layer gradient-to-weight ratios r*_l converge to layer-specific values that anti-correlate with per-layer steady-state alignment delta*_l.

**Test**:
- ResNet-50/ImageNet, 90 epochs with fixed WD and UDWDC.
- Track per-layer rho_t^l and delta_t^l averaged over last 10 epochs.
- Under fixed WD: compute CV(r*_l) across layers (predicted: low).
- Under UDWDC: compute Spearman(r*_l, delta*_l) across layers.

**Expected outcome**:
- Fixed WD: CV(r*_l) < 0.15
- UDWDC: Spearman(r*_l, delta*_l) < -0.4

**Falsification**: Anti-correlation not significant (Spearman rho > -0.3).

**Priority**: MEDIUM (extends Defazio's theory)

---

## H6: Average Alignment Predicts Generalization Better Than Worst-Case

**Statement**: Average alignment alpha_bar = (1/T) sum_t delta_hat_t has strictly higher predictive validity for generalization gap than worst-case alignment delta_max = sup_t delta_hat_t, across diverse WD conditions.

**Test**: 54 training runs on CIFAR-100/ResNet-20 (WD grid x LR grid x 3 seeds). Compute both alpha_bar and delta_max. LOO-CV R-squared comparison.

**Expected outcome**: LOO-CV R-squared(alpha_bar) > LOO-CV R-squared(delta_max) by at least 0.05.

**Falsification**: R-squared difference < 0.05 or delta_max is a better predictor.

**Priority**: MEDIUM (theoretical validation)

---

## H7: Temporal Predictability Gate (Pre-Registered Control)

**Statement**: The alignment trajectory delta_hat_t is NOT fully predictable from training time alone, carrying genuine geometric information beyond the training clock.

**Test**: For each training run, fit a degree-4 polynomial to delta_hat_t as a function of epoch. Compute R-squared.

**Expected outcome**: R-squared < 0.85 in at least 30% of training runs.

**Falsification**: R-squared > 0.85 in >= 70% of runs. In this case, alignment-conditioned WD reduces to time-scheduled WD and the "geometric information" claim is wrong.

**Decision rule**: If this gate fails, all subsequent alignment comparisons use residual alignment (delta_hat_t minus polynomial fit) instead of raw alignment.

**Priority**: HIGH (must run early; gates subsequent experiments)
