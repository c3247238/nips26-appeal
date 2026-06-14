# Supervisor Review: Gradient-to-Weight Ratio Homeostasis

## Overall Score: 7.0 / 10 (Weak Accept)

**Verdict: CONTINUE** -- The paper has a solid conceptual contribution but needs another iteration to address incomplete experiments and unresolved confounds.

---

## Executive Summary

This paper proposes a PID control framework that unifies four weight decay sub-traditions by identifying the per-layer gradient-to-weight ratio (rho_t^l) as their shared control variable. The taxonomy maps FixedWD, CWD, SWD, CPR, and DefazioCorrective to specific PID gain configurations. The framework is supplemented by three evaluation metrics (BEM, CSI, AIS) and a proposed proportional controller (UDWDC).

The conceptual contribution -- the PID taxonomy itself -- is genuine and useful. The honest reporting of negative results (UDWDC's failure, 2/5 fitting failures, CWD confound) is commendable and above the field norm. However, multiple significant gaps prevent an Accept-level score.

---

## Dimension Scores

### Novelty & Significance: 7.5 / 10

**Strengths:**
- The PID control lens on WD methods is genuinely novel. No prior work maps CWD/SWD/CPR to (K_p, K_i, K_d) gain configurations. The distinction from PIDAO (which controls the optimizer step, not the WD coefficient) is clearly drawn.
- The three-way family taxonomy (alignment/scheduling/constraint) with the control-theoretic interpretation is a conceptual advance that clarifies why CPR and CWD achieve different results.
- The standardized metrics (BEM, CSI, AIS) address a real gap in the literature -- current WD papers use incomparable evaluation protocols.

**Weaknesses:**
- UDWDC underperforms FixedWD on every benchmark. This is honestly reported, but a paper proposing a new method that never beats the trivial baseline has limited algorithmic impact.
- The unification is partial: 2/5 methods (SWD, DefazioCorrective) do not fit the framework (45.8% and 37.6% error). The paper handles this honestly by delineating scope, but this limits the framework's generality.
- The CWD fitting success (4.7% error) is driven by the extended model's rho_target_scale=0.50 parameter, not by PID gains (all ~1e-7). This means CWD is well-modeled by "FixedWD at half magnitude" rather than by the PID control law. This weakens the unification claim for CWD specifically.

### Technical Soundness: 7 / 10

**Strengths:**
- The control-theoretic formulation is mathematically clean and well-specified.
- The mapping from existing methods to PID gains is convincing for CPR (integral control via augmented Lagrangian penalty accumulation).
- Cross-validation of paper numbers against raw data shows all reported values are accurate -- Tables 3, 4, and 5 match the source JSON files exactly.

**Weaknesses:**
- Theorem 1 and Propositions 2-3 are stated without proofs. The paper provides "intuition" and "statements" but no formal derivations. For a paper listing "Theoretical analysis" as a contribution, this is a significant gap.
- The extended fitting model (offset + target-scale beyond basic K_p, K_i, K_d) has 5 free parameters per method and 72 data traces per method. With this flexibility, low fitting error for CWD (4.7%) is less impressive -- it may reflect overfitting the magnitude-reduction effect rather than capturing genuine PID structure.
- The claim that "all four traditions implicitly manipulate rho_t^l" is partially falsified by the fitting results themselves: scheduling methods do not manipulate rho_t^l through the per-layer feedback mechanism the paper formalizes.

### Experimental Rigor: 6.5 / 10

**Strengths:**
- CIFAR-10 experiments are thorough: 8 methods, 3 seeds, 200 epochs, full diagnostic tracking. All numbers cross-validate against raw data.
- The PID gain ablation on CIFAR-100 (Table 4) with 7 variants is well-designed and isolates individual gain contributions.
- Honest reporting of negative results (UDWDC, fitting failures, CWD confound) is exemplary.

**Weaknesses:**
- **CRITICAL: Incomplete ImageNet results.** Only 4 of 7 methods were run. CWD has 1 seed. SWD, DefazioCorrective, NoWD are missing. The paper claims "comprehensive experiments" but Table 5 is a partial comparison.
- **CWD vs. halved-lambda ablation missing.** This was explicitly planned in the proposal but not executed. Without it, the CWD analysis remains inconclusive -- acknowledged by the paper itself.
- **No standard-augmentation ImageNet experiments.** FixedWD at 71.72% (vs. standard 76-77%) means all conclusions are restricted to minimal-augmentation regimes. This is a significant external validity concern.
- **No ViT experiments.** Proposed but not completed. Proposition 3's BN-only scope cannot be evaluated.
- **CSI from pilot data.** Table 6 uses 10-epoch pilot CSI values despite 200-epoch data being available. Early and late training dynamics differ substantially.
- **AIS's negative LOO CV R-squared (-0.18) is not discussed.** This suggests AIS has zero predictive power out-of-sample, undermining its utility as a metric.

### Reproducibility: 7.5 / 10

**Strengths:**
- Algorithm 1 (UDWDC) is precisely specified and straightforward to implement.
- Hyperparameters are fully documented: lr=0.1, momentum=0.9, WD=1e-4, cosine annealing, seeds={42,123,456}, augmentation details.
- The diagnostic logging (per-layer rho, alpha, effective WD, weight norms per epoch) is unusually thorough.

**Weaknesses:**
- Code is not explicitly referenced for availability.
- The extended fitting model parameters are not specified in enough detail to reproduce the unification fitting.

---

## Cross-Validation Summary

All paper numbers were verified against raw experimental data:
- **Table 3 (CIFAR-10)**: All 8 method means and stds match source JSON to 2 decimal places. VERIFIED.
- **Table 4 (CIFAR-100 ablation)**: All 7 variant means match source JSON exactly. VERIFIED.
- **Table 5 (ImageNet)**: All 4 method means match source JSON. FixedWD and UDWDC stds use population stdev (0.36, 0.19) vs. sample stdev (0.40, 0.24) -- this is a minor discrepancy that should be documented. VERIFIED with caveat.
- **Table 2 (Fitting results)**: CWD 4.71%, CPR 9.57%, SWD 45.81%, DefazioCorrective 37.56% all match fitting_results.json. VERIFIED.
- **Table 6 (CSI)**: All values match metrics_results.json refined CSI section. VERIFIED.

**No data-paper inconsistencies found.** This is a significant improvement over previous iterations where multiple contradictions were flagged.

---

## Key Concerns for Next Iteration

### Priority 1 (Required for score >= 8.0):
1. **Complete ImageNet experiments**: Run SWD and DefazioCorrective on ImageNet with 3 seeds. Add 2 more CWD seeds. This is the single most impactful action.
2. **CWD vs. halved-lambda ablation**: Run FixedWD at lambda=5e-5 on CIFAR-10 (3 seeds). Minimal compute, high diagnostic value.
3. **Resolve theoretical status**: Either provide proofs for Theorem 1 / Propositions 2-3 or demote them from "contributions."

### Priority 2 (Recommended):
4. **Reframe UDWDC**: Position it as a diagnostic probe, not a proposed method. The paper's value is the taxonomy + metrics.
5. **Report CWD extended model honestly**: The 4.7% fit error comes from rho_target_scale=0.50, not PID gains. Discuss this.
6. **Compute 200-epoch CSI**: Replace pilot CSI values in Table 6 with full-run values.
7. **Acknowledge AIS limitations**: The negative LOO CV R-squared needs discussion.

### Priority 3 (Nice-to-have):
8. **Standard augmentation ImageNet experiment** (even just FixedWD + CPR) to verify ranking stability.
9. **ViT pilot** to test Proposition 3 scope.
10. **Specify stdev formula** (population vs. sample) consistently.

---

## What Would Raise the Score

**From 7.0 to 8.0**: Complete ImageNet experiments (all 7 methods, 3+ seeds), run the CWD vs. halved-lambda ablation, and either prove or demote the theoretical claims. These three changes address the critical issue and two major issues.

**From 8.0 to 8.5+**: Additionally validate metrics externally, run standard-augmentation ImageNet experiments, and propose a PI-controller variant that actually beats FixedWD.
