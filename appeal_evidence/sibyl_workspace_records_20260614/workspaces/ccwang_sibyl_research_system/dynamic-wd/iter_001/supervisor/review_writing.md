# Area Chair Review: Revision Assessment
## "On the Sufficiency of Constant Weight Decay: Alignment Dynamics, Learning Rate Coupling, and Budget Equivalence in Nonconvex SGD"

**Iteration:** 1 (revision from score 7/10)
**Recommended Score:** 7.8 / 10
**Verdict:** PASS (accept with minor revisions)

---

## Summary

This paper investigates whether gradient-parameter alignment information can guide dynamic weight decay scheduling in nonconvex SGD. Through 39 systematic experiments on CIFAR-10/100 with ResNet-20 and VGG-16-BN, it establishes three negative but mechanistically informative results: (1) budget equivalence -- the time-averaged weight decay magnitude, not its temporal distribution, determines generalization; (2) LR-WD coupling necessity -- removing the learning rate multiplier from adaptive schedules causes catastrophic collapse; and (3) alignment signal inapplicability -- the alignment proxy remains near-constant at O(10^-3) throughout training, rendering it equivalent to random noise as a scheduling signal.

The revision addresses all four issues raised in the previous round. Budget equivalence is now grounded in a formal argument quantifying the error due to trajectory-dependent terms. The CIFAR-100 asymmetry receives a mechanistic explanation. The two concurrent references (chen2025correction and malladi2025weight) are substantively integrated into the related work. The single-seed limitation is honestly disclosed with multi-seed runs in progress. These improvements collectively lift the paper's quality, though a numerical inconsistency in the CIFAR-100 results and the limited scope of the budget equivalence verification require attention.

---

## Assessment of Previous Issues

**1. Single seed (addressed: disclosure + multi-seed in progress)**
The revision handles this appropriately. The limitation is disclosed prominently in the training protocol section and enumerated in the limitations section with calibrated language: differences below 0.3% should be interpreted cautiously. The commitment to multi-seed validation (seeds 123, 456) for camera-ready is reasonable given the resource constraints. The paper correctly notes that large-effect-size results (the 82% collapse, the 0.01% random-vs-alignment equivalence, the exact metric match for budget equivalence) are robust to single-seed concerns, while small differences among AADWD variants and fixed WD are appropriately qualified. This is a satisfactory response to the previous feedback.

**2. Missing recent references (now added: [2512.08217] and [2510.19093])**
The integration of chen2025correction and malladi2025weight is substantive. Chen et al. (the WD^2/LR correction paper) is discussed in the context of the Square variant satisfying the O(gamma_t^2) condition, creating a direct theoretical connection. Malladi et al. is cited as independent large-scale empirical confirmation that LR-WD interaction dominates generalization, corroborating the coupling necessity finding. Both additions are placed in the body of the related work section rather than appended as afterthoughts. This is well done.

**3. CIFAR-100 asymmetry unexplained (now explained in analysis)**
Section 5.3 now provides a mechanistic explanation: the inverse formula of AADWD-Aggressive amplifies budget variance under near-zero alignment (denominator fluctuations produce large lambda_t spikes), and CIFAR-100's tighter generalization optimum (100 classes tolerates less over-regularization) exposes this instability more severely. The falsifying test is also provided: if alignment were informative on CIFAR-100, both AADWD variants should outperform fixed WD, but neither does. This is a genuine analytical contribution that strengthens the paper. One numerical inconsistency remains (see issues below), but the mechanistic argument is sound.

**4. Budget equivalence unclear (now strengthened with quantitative conditions)**
The formal argument in Section 5.1 is substantially improved. The key steps -- (1) the product approximation exp(-2 sum lambda_t) for small lambda_t, (2) the trajectory-dependent error bounded by O(delta_t * gamma_t * G * |w_t|) per step, (3) the quantitative bound that this error is small given delta_t <= 0.005 and gamma_t <= 0.1 -- are now spelled out explicitly. Proposition 1 in the theory section is also updated to state the error as O(lambda_max * delta_max) per step with explicit numerical context (lambda ~ 10^-4 to 10^-3, delta_t ~ 10^-3). The empirical confirmation (exact metric matching for all four reported quantities) closes the argument. This is a meaningful improvement.

---

## Strengths

**Experimental design.** The controlled experimental framework is the paper's main methodological contribution. The budget equivalence experiment is particularly clean: run AADWD, compute the time-average of its lambda_t trajectory, run a constant at that average, compare. The decoupling ablation is similarly well-designed: a single change (remove gamma_t) produces a catastrophic and mechanistically interpretable failure. The random WD ablation is the decisive test for alignment inapplicability. Each experiment targets exactly one claim.

**Effect sizes.** The core results are not marginal. A 92.05% to 10.00% collapse upon decoupling is unambiguous. The 0.01% difference between random and alignment-based dynamic WD is equally decisive in the opposite direction. Exact metric matching (92.54% = 92.54%, weight norm 23.49 = 23.49) for budget equivalence is striking. These effect sizes support the claims even under single-seed conditions.

**Theoretical framework.** The extension of Xie et al.'s fixed-WD analysis to time-varying WD (Theorem 1) and the alignment-weighted bound (Theorem 2) provide the appropriate theoretical context. The theorems correctly identify the quantity that matters (cumulative alignment bar{delta}_T rather than sup delta_t) and the paper correctly shows that this quantity provides no leverage in practice (only 1.5x variation in a term that is O(10^-3)). The theoretical framework motivates the experiments without overclaiming.

**Practical contributions.** The CWD instability finding (4.84%-12.57% best-to-final degradation across all settings) is a useful secondary contribution. The four practical guidelines in Section 6.1 are concrete and actionable. The budget-matching diagnostic (compare any new WD schedule against a constant with matched time-average) is a simple and useful recommendation.

**Scope awareness.** The paper is clear about what it does and does not show. The Discussion explicitly limits the conclusions to SGD with momentum, milestone LR schedules, CIFAR-scale tasks, and moderate WD values. The conjectured conditions under which alignment might become actionable (adversarial training, large WD near instability boundary, highly nonconvex early training) are reasonable and provide a roadmap for follow-up work.

---

## Weaknesses and Issues

**Major: Numerical inconsistency in CIFAR-100 results.** Table 2 (cross-architecture results) reports AADWD Conservative at 68.24% and AADWD Aggressive at 61.34% for CIFAR-100/ResNet-20. Section 5.3 (CIFAR-100 asymmetry analysis) references Conservative at 62.12% and Aggressive at 61.44% for the same setting. The 6.12% discrepancy for Conservative is large enough to affect the analysis. This must be resolved before final submission. If Table 2 is correct, the asymmetry analysis numbers should be updated; if Section 5.3 is correct, Table 2 needs correction.

**Major: Budget equivalence verification is limited to a single coincidental case.** The AADWD-Aggressive time-average lambda coincidentally equals 5e-4, which is also the optimal fixed WD for this setting. The exact match in performance is compelling, but it is impossible to distinguish between two explanations: (a) budget equivalence holds generally, or (b) any two well-configured WD schedules near the optimum lambda happen to perform similarly. To establish the principle more convincingly, the experiment should be repeated for at least one other c value (e.g., c=1.0, where the time-average would be approximately 2e-4, below the fixed WD optimum) to confirm that the constant WD at that suboptimal budget also matches the dynamic schedule -- not just that two runs at the globally optimal budget agree.

**Minor: Figure captions describe 20-epoch pilot runs, not 200-epoch full runs.** The quantitative tables report 200-epoch results, but Figures 1-5 all show 20-epoch pilots with captions explicitly noting this. For a reader trying to connect the figures to the tables, this creates confusion. The figures could be updated to show 200-epoch dynamics, or a clear statement should be added in the experimental setup that figures are illustrative pilots and all quantitative comparisons are from the 200-epoch full runs.

**Minor: Per-phase alignment statistics are incomplete.** Table 4 shows mean alignment for Early, Mid, and Late phases but reports standard deviation as '---' for the first two phases (only the Overall row shows std = 0.000753). Without per-phase variance, the claim of a "monotonic decrease from 0.0045 to 0.0028" cannot be assessed for statistical significance. Given that the mean decreases from 0.0045 to 0.0028 while the overall std is 0.000753, the phase-to-phase differences may be meaningful, but this should be confirmed.

**Minor: Proxy fidelity is below the stated diagnostic threshold.** The Pearson correlation of r = 0.849 falls below the r = 0.85 threshold mentioned as the diagnostic criterion. The authors correctly argue that this does not affect the core conclusions (the underlying signal is too weak regardless of measurement quality), but the mismatch between the diagnostic threshold and the measured value should be addressed more directly -- either by revising the threshold or by showing that r = 0.849 is sufficient for the specific use case.

**Minor: Inconsistency between Proposition 1 error characterization and Section 5.1 formal argument.** Proposition 1 states the per-step error as O(lambda_max * delta_max) while Section 5.1 derives O(delta_t * gamma_t * G * |w_t|). These should be reconciled, with one referencing the other explicitly, and the relationship between the two error bounds should be stated.

---

## Overall Assessment

This is a well-executed negative-result paper that makes a genuine contribution to understanding why constant weight decay remains competitive. The experimental methodology is sound, the claims are appropriate for the scope, and the revision has meaningfully addressed all four previously identified issues. The main remaining concerns are a numerical inconsistency that needs correction, a gap in the budget equivalence verification, and the pending multi-seed validation. None of these are fatal: the numerical inconsistency can be fixed in revision, the budget equivalence argument is strengthened by the formal analysis even if the single experiment is not definitive, and the large effect sizes support the core conclusions under single-seed conditions.

The paper is appropriate for NeurIPS as a negative-result contribution that provides mechanistic insight into a commonly used technique. The score of 7.8 reflects the improvements from the previous round and the remaining minor issues. The verdict is PASS.

**Recommended actions before camera-ready:**
1. Resolve the CIFAR-100 numerical inconsistency between Table 2 and Section 5.3.
2. Add at least one additional budget equivalence verification at a non-optimal lambda budget.
3. Include multi-seed results with confidence intervals for primary CIFAR-10/ResNet-20 comparisons.
4. Update Figure captions or add a statement clarifying that all quantitative claims rest on 200-epoch full runs.
5. Add per-phase standard deviations to Table 4 alignment statistics.
