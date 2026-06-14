# Supervisor Review -- Iteration 15

**Score: 6.5 / 10** | **Verdict: CONTINUE**

---

## Executive Summary

The paper proposes a PID-style taxonomy for dynamic weight decay methods -- a genuinely novel conceptual contribution. No prior work maps the four WD sub-traditions (scheduling, alignment-aware, decoupled, norm-matched) to a single parameterized control law. The experimental infrastructure is impressive: 24 CIFAR-10 runs, 45 batch-size sweep runs, 12 ImageNet runs, all with per-layer diagnostic tracking. Cross-validation confirms all reported accuracy numbers match source data exactly.

However, the score drops from the prior review's 7.0 to 6.5 because multiple data integrity and internal consistency issues -- flagged in the iter_014 review -- remain unfixed in the paper. These are not writing quality issues; they are soundness problems where the paper makes specific quantitative claims that are contradicted by its own experimental data.

---

## Dimension Scores

### Novelty & Significance: 7.5 / 10

**Strengths:**
- The PID control-law mapping is a fresh perspective that connects four independent research communities
- The three evaluation metrics (BEM, CSI, AIS) address a real gap -- accuracy-only evaluation masks important efficiency and stability differences
- Honest reporting of negative results (UDWDC below NoWD, fitting failures for 2/5 methods) is rare and valuable

**Weaknesses:**
- The "unified" claim is overstated: only CPR (9.57% error) is genuinely captured by PID gains. CWD's fit (4.71%) is explained by scale=0.5, not K_d. SWD and DefazioCorrective fail fitting
- UDWDC provides zero practical value -- it underperforms FixedWD and even NoWD. The algorithmic contribution is effectively null
- The paper's most practically useful finding (CPR's integral control is effective, proportional-only control is not) could be stated in 2 sentences without the full framework

### Technical Soundness: 5.5 / 10

**Critical issues (3):**

1. **CWD K_d claim contradicted by own data.** Table 1 states CWD uses "K_d > 0, Derivative only." The unifying_table.md shows the fitted K_d = 0.0000. The 4.71% fit comes from scale=0.5 (halving the WD magnitude). This means Table 1 -- the paper's core mapping table -- contains an empirically falsified claim for CWD.

2. **AIS=0.566 is statistically meaningless.** Computed from 6 data points, the 95% CI for Spearman rho at n=6 is approximately +/-0.71, so the CI comfortably includes zero. The LOO-CV R^2 is -0.18 (negative = worse than predicting the mean). The paper presents this as validated evidence that "the alignment signal carries information beyond time-polynomial trends."

3. **CSI formula is internally inconsistent.** Four contradictory definitions appear across Section 4.2, the Table 6 caption, and the normalization paragraph. None of the stated formulas can produce the reported -5.75 value for UDWDC. The paper's central instability finding rests on an unverifiable computation.

**Major issues:**
- Theorem 1, Propositions 2-3 stated without proofs (6+ iterations unfixed)
- CPR's 3.02 pp advantage confounded by 2.3x WD budget without budget-matched controls

### Experimental Rigor: 6 / 10

**Strengths:**
- 3-seed runs on CIFAR with full per-layer diagnostics
- H3 falsification correctly reported (CWD collapses at large batch sizes)
- Comprehensive batch-size sweep (5 batch sizes x 3 methods x 3 seeds = 45 runs)

**Weaknesses:**
- ImageNet 12/40 complete; BEM (the paper's primary metric) uncomputable at ImageNet scale
- CWD halved-lambda ablation flagged as P0 for 3+ iterations, never executed (~1 GPU-hour)
- UDWDC-v2 BN bug produces 205,000x WD budget, contaminating all v2 BEM values
- Budget-matched FixedWD controls absent at ImageNet scale
- CPR hyperparameters (kappa, mu) not disclosed in experimental setup

### Reproducibility: 6 / 10

**Strengths:**
- Training protocol clearly stated (optimizer, LR schedule, augmentation, seeds)
- Per-layer diagnostic tracking enables deep analysis
- Minimal augmentation protocol isolates WD effects

**Weaknesses:**
- CSI computation formula undisclosed (no stated formula produces the reported values)
- CPR hyperparameters not specified
- Batch size 432 and LR scaling rule not explained
- Fraction of training at clamp boundaries not reported for UDWDC
- Proposition 3 anti-correlation "verified on ResNet-50" but no verification figure exists

---

## Cross-Validation Results

### Numbers that MATCH source data:
- CIFAR-10 Table 3: All 8 methods' accuracy, gen gap, and WD budget match phase1_diagnostic/summary.json exactly
- ImageNet Table 5: CPR 74.742+/-0.042 (paper: 74.74+/-0.05, acceptable rounding), FixedWD 71.722+/-0.360 (paper: 71.72+/-0.36, match), UDWDC 69.933+/-0.199 (paper: 69.93+/-0.19, match), CWD 70.66 (match)
- CIFAR-100 Table 4: All 7 ablation variant accuracies match phase2_ablation/summary.md

### Numbers that DO NOT match or are unverifiable:
- **AIS=0.566**: The value exists in metrics_results.json (v1) as AIS_global=0.566491, but with LOO-CV R^2=-0.18 (negative), meaning zero predictive power. The paper omits the negative LOO-CV R^2
- **CWD K_d > 0**: Contradicted by K_d=0.000 in unifying_table.md
- **CSI_temporal = -5.75 for UDWDC**: No stated formula can produce this value
- **r_alpha_gengap = 0.698**: Appears in metrics_results.json as correlation_alpha_gengap=0.698442, but this is an in-sample correlation from 6 data points (no cross-validation reported alongside it)

---

## Priority Actions (Ordered by Impact per GPU-Hour)

1. **Fix CWD K_d claim** (0 GPU-hours, 30 min text edit): Acknowledge scale=0.5 explains the fit, revise Table 1
2. **Fix CSI formula** (0 GPU-hours, 1 hour text + recompute): Standardize to one formula, verify against 200-epoch data
3. **Report AIS with proper caveats** (0 GPU-hours, 15 min): Add n=6, CI, negative LOO-CV R^2
4. **Demote theorems to conjectures** (0 GPU-hours, 30 min text edit)
5. **Run halved-lambda ablation** (~1 GPU-hour): FixedWD at lambda=5e-5, CIFAR-10, 3 seeds
6. **Fix UDWDC-v2 BN bug** (~3 GPU-hours): Exclude BN/bias, rerun CIFAR-10
7. **Complete ImageNet NoWD** (~24 GPU-hours): 3 seeds, 90 epochs -- enables BEM
8. **Budget-matched FixedWD for ImageNet** (~18 GPU-hours): lambda=2e-4, 3e-4, 3 seeds each

Items 1-4 are zero-cost text fixes that would raise the score from 6.5 to 7.0-7.5. Items 5-6 are cheap experiments that would address the most prominent unresolved confounds. Items 7-8 are the path to 8.0.

---

## Score Justification

The score of 6.5 reflects genuine novelty in the taxonomy (7.5) dragged down by critical soundness issues (5.5). The paper makes three specific quantitative claims (CWD K_d>0, AIS=0.566, CSI=-2.41) that do not survive cross-validation against its own experimental data. These are not presentation issues -- they are claims about what the experiments show that contradict what the experiments actually show.

The score is below the prior review's 7.0 because the critical issues flagged in iter_014 (CWD K_d, AIS, CSI formula) were not addressed in the current paper. Unfixed critical issues across iterations should not receive the same or better score.

A score of 8.0 (accept threshold) requires: (a) all four zero-cost text fixes, (b) the halved-lambda ablation, (c) UDWDC-v2 BN fix, and (d) ImageNet NoWD baseline. The taxonomy contribution merits 8.0 if the supporting evidence is honest and complete.
