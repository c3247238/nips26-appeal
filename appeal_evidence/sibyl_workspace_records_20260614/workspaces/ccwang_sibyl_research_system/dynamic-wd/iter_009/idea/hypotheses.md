# Testable Hypotheses

## H1: Phi-Modulator Taxonomy — Unified Steady-State Formula

**Statement**: Every existing WD method (constant, CWD, SWD, AdamWN, AdamO, CPR) corresponds to a specific modulation function Phi_t, and the steady-state weight norm satisfies:

    r*_Phi = gamma * E[||g|| * cos(theta)] / (lambda * E[Phi_t])

**Expected outcome**: For ResNet-20/CIFAR-10 with 3 methods (constant, CWD, cosine_schedule), the formula predicts the observed end-of-training ||w|| to within 15% relative error.

**Falsification**: If the formula prediction error exceeds 30% for any method, the steady-state analysis is invalid and the taxonomy is merely descriptive, not predictive.

**Experiment**: Run 3 methods with full per-layer tracking of ||g||, ||w||, cos(theta). Compute E[Phi_t] for each method. Compare predicted r* vs observed ||w|| at convergence.

---

## H2: Lyapunov Alignment Cross-Term Elimination

**Statement**: Under the augmented potential Phi_t = f_S(w_t) + beta*||w_t||^2, the one-step Lyapunov decrease contains an alignment cross-term 2*beta*gamma_t*||g_t||*||w_t||*delta_t. Setting lambda_t proportional to (1 - delta_hat_t) eliminates this cross-term contribution, achieving lower CSI (Coupling Stability Index) than fixed WD.

**Expected outcome**: Alignment-aware WD (lambda_t proportional to (1 - delta_hat_t)) achieves CSI at least 20% lower than fixed WD with equal budget, on ResNet-20/CIFAR-10 with 3 seeds.

**Falsification**: If CSI under alignment-aware WD is NOT measurably lower than fixed WD (reduction < 10%), the cross-term elimination does not manifest in practice and the Lyapunov analysis is too crude.

**Experiment**: Budget-matched comparison (same sum(lambda_t)). Track ||g||/||w|| per layer throughout training. Compute CSI = Var(R_t) across layers and time.

---

## H3: Budget Equivalence under Low AIS

**Statement**: When AIS = mean(delta_hat_t) is below 0.5 throughout training across all layers, any WD schedule {lambda_t} with total budget B = sum(lambda_t) achieves statistically equivalent generalization to constant WD with lambda = B/T, up to O(epsilon) correction where epsilon = Var(AIS).

**Expected outcome**: On CIFAR-10/100 with ResNet-20 and VGG-16-BN (where existing data shows AIS ~ 0.35-0.45), no dynamic WD method achieves > 0.5% accuracy improvement over BEM-matched constant WD (p < 0.05, n=9 seeds).

**Falsification**: If any dynamic WD method achieves > 0.5% improvement at p < 0.05 with n=9 seeds despite AIS < 0.5, the budget equivalence theorem is wrong.

**Experiment**: 7 WD methods across 3 architectures, 9 seeds each. Paired t-tests with Bonferroni correction. TOST equivalence tests at +/- 0.5%.

---

## H4: PMP Derivation Recovers Alignment-Aware WD

**Statement**: The PMP stationarity condition dH/d(lambda) = 0, applied to the gradient-flow optimal control problem with lambda(t) as the control variable, yields the optimal WD schedule:

    lambda_t* proportional to gamma_t * (1 - delta_t) * tau / ||w_t||

where tau is the target norm from the terminal boundary condition.

**Expected outcome**: The PMP-derived schedule (with norm feedback: lambda proportional to gamma*(1-delta)/||w||) outperforms the heuristic alignment-aware schedule (lambda proportional to gamma*(1-delta)) because the norm term captures target-norm convergence. Specifically: correlation between lambda_t^optimal (oracle) and gamma_t*(1-delta_t)/||w_t|| is significantly higher than correlation with gamma_t*(1-delta_t) alone.

**Falsification**: If the norm-feedback term adds no predictive value (correlation improvement < 0.05), the PMP derivation reduces to the heuristic and does not provide additional insight.

**Experiment**: Train with oracle WD (grid search over lambda per epoch block). Measure correlation of oracle lambda_t with both heuristic and PMP-derived formulas.

---

## H5: AIS Architecture Dependence

**Statement**: The information content of the alignment signal delta_hat_t is architecture-dependent: in BN/LN networks, delta_hat_t is dominated by effective-LR dynamics (scale invariance), while in non-BN networks, delta_hat_t contains genuine alignment information.

**Expected outcome**: (a) AIS is higher (> 0.5) for non-BN networks than for BN networks (< 0.5) on the same task; (b) alignment-aware WD shows measurable improvement over constant WD on non-BN networks but not on BN networks.

**Falsification**: If AIS is approximately equal for BN and non-BN networks, or if alignment-aware WD fails on both, the scale-invariance explanation for the null result is incorrect.

**Experiment**: ResNet-20 with BN vs ResNet-20 without BN (bias-only replacement), same training setup, 3 seeds. Compare AIS trajectories and accuracy gaps.

---

## H6: SWD Instability Explained by CSI

**Statement**: SWD's poor performance on CIFAR-10/100 is caused by gradient-norm-driven schedule changes that increase CSI above 1.0 (coupling instability), creating oscillatory WD dynamics that oppose stable convergence.

**Expected outcome**: SWD has CSI > 1.0 (confirmed by iter_003 data: CSI ~ 1.15), while constant WD has CSI ~ 0.83. This CSI gap correlates with the accuracy gap (SWD is -0.46% on CIFAR-10).

**Falsification**: If SWD's CSI is not significantly higher than constant WD, or if high-CSI methods sometimes outperform low-CSI methods, CSI is not a reliable quality indicator for WD methods.

**Experiment**: Already partially validated from iter_003 data. Extend to VGG-16-BN and ImageNet.

---

## H7: Hill Function Cooperativity (n > 1) Improves Noise Robustness

**Statement**: The Hill-function form lambda_t = lambda_max * (1 - delta_t)^n / (delta*^n + (1-delta_t)^n) with n = 2-4 outperforms linear (n=1) and binary (n=infinity, i.e., CWD) alignment-aware WD, because the sigmoidal response filters alignment noise while preserving threshold sensitivity.

**Expected outcome**: Optimal n is intermediate (2-4), not extreme (1 or infinity). The optimal n correlates with batch-size-dependent gradient noise: smaller batch -> larger optimal n.

**Falsification**: If n=1 (linear) is optimal or indistinguishable from n=2-4, the Hill function adds no value and the simpler linear form is preferred. If n=infinity (CWD) is optimal, binary masking is sufficient.

**Experiment**: Train with n in {0.5, 1, 2, 4, 8, infinity} on CIFAR-10/ResNet-20, 3 seeds. Also test at batch sizes {64, 128, 256, 512}.
