# Supervisor Review: Iteration 5

## Score: 6.0 / 10 (Borderline Reject)

**Verdict: REVISE**

**Dimension Scores:**
- Novelty: 7/10
- Technical Soundness: 5/10
- Experimental Rigor: 6/10
- Reproducibility: 5/10

---

## Executive Summary

This paper attempts three contributions to the SAE feature absorption literature: (1) confound resolution via epidemiological causal inference methods on 48 SAEs, (2) cross-domain absorption measurement on knowledge hierarchies using GPT-2 Small, and (3) a scaling surface analysis across 420 SAEs. Contributions 1 and 3 contain genuinely novel and interesting findings -- the SP-F1 suppression effect and the highly significant width-L0 interaction are both real and informative. However, the paper's credibility is undermined by three critical soundness issues: causal overclaiming from observational data with a devastating within-width matching null, mixed coefficient types in the core mediation table, and a misclassified hypothesis verdict for the cross-domain experiment. There are also multiple data integrity failures in the canonical summary JSON, a missing architecture confound in the scaling surface, and no multiple comparison correction despite 50+ tests. The underlying science is solid enough to merit revision, but the current presentation would not survive top-venue peer review.

---

## What Works

1. **Methodological novelty of applying epidemiological methods to SAE evaluation.** Partial correlation with confound control, Baron-Kenny mediation, and Rosenbaum sensitivity analysis have never been applied to SAE evaluation. The methodological transplant is appropriate and well-executed for SP-F1 and SCR.

2. **The SP-F1 suppression effect is a genuine surprise.** Controlling for L0 *strengthened* the partial correlation from -0.664 to -0.746. This is the opposite of what the confound hypothesis predicted and is the most informative single finding of the paper.

3. **The 420-SAE scaling surface is statistically powerful.** N=420 with interaction p=3.1e-15 is a robust finding. The practical implication (target L0 > 14 to avoid absorption) is immediately actionable.

4. **Honest reporting of negative results.** The unlearning null (no association), the within-width matching null (Gamma=1.0), the 0% cosine-calibrated cross-domain rate, and the model fallback are all reported transparently. This builds reviewer trust.

5. **The taxonomy correction (92.3% to 19.2%) is a valuable audit.** Identifying that the original Type II rate was an artifact of feature specificity, not absorption, is a genuine contribution to the measurement literature.

---

## Critical Issues (Would Cause Rejection)

### C1: Causal Overclaiming from Observational Data

The paper repeatedly claims "causal mediation" (Section 5.1), "causal chain" (throughout), and "causal status of absorption" (Section 5.1 heading). Baron-Kenny mediation from 48 cross-sectional observations does not establish causation. The within-width matching null (Table 4) is the smoking gun: under exact-width matching (4 pairs), within-width median (23 pairs), and tertile matching (16 pairs), Gamma = 1.0 for ALL metrics. The causal evidence vanishes when width is held constant by design.

An alternative common-cause explanation (better-trained SAEs have both lower absorption and higher quality) is consistent with all data and indistinguishable from mediation in this design. The paper acknowledges this in Discussion 5.4 but buries it under the causal narrative.

**Verdict**: This would be a reject-on-its-own issue at NeurIPS if a causal inference reviewer is assigned. The fix (language downgrade + structural repositioning) is straightforward.

### C2: Table 3 Coefficient Confusion

Paths a and b are labeled "(std)" and match standardized values. Direct effect c' = -0.029 matches the standardized value for SCR, but the abstract uses c' = -0.003 (unstandardized). The indirect effect ab = 0.025 matches the unstandardized value, not standardized (0.248). The table therefore has paths a, b in standardized units but c' and ab in different units. Any reviewer who checks a*b = ab will find the arithmetic fails.

### C3: H2 Verdict Misclassification

By the paper's own pre-registered falsification criterion, H2 is falsified: the shuffled baseline is 100%, so no real hierarchy can exceed 3x. The "PARTIALLY_SUPPORTED" label in the abstract, JSON, and summary misleads readers about the Phase 2 outcome. The paper body (Section 4.2) correctly describes the result, creating an internal contradiction.

---

## Major Issues (Significantly Weaken the Paper)

### M1: Phase 2 on Wrong Model
GPT-2 Small with 98% dead features is categorically incapable of testing cross-domain absorption. The 0% cosine-calibrated rate is an inevitable model limitation, not evidence. The abstract's Contribution 2 framing still implies the experiment answers the cross-domain question.

### M2: Architecture Confound in Scaling Surface
360 L1 + 54 JumpReLU SAEs span different width/L0 ranges without architecture as a GAM covariate. The interaction could capture architecture differences. A 30-minute zero-GPU fix resolves this.

### M3: SP-F1 Suppression Interpretation Fragility
The SP-F1 total effect is non-significant (p=0.45). Baron-Kenny step 3 fails. The "indirect-only mediation" interpretation per Zhao et al. is technically valid but the alternative (confounding by a latent feature-quality variable) is equally plausible and not discussed.

### M4: No Multiple Comparison Correction
50+ tests on n=48. TPP results (partial r p=0.022, Sobel p=0.037) are borderline and may not survive FDR correction. SP-F1 (p=1.2e-9) and SCR (p=6.6e-5) will survive.

### M5: Data Pipeline Integrity
final_results.json has wrong taxonomy rate (92.3% vs 19.2%), wrong TPP Sobel z (2.63 vs 2.08), wrong n_full_mediations (0 vs 2), and inconsistent phase_boundary_detected. The paper body is correct, but the canonical JSON is corrupted.

### M6: Rosenbaum Gamma Inflation
Gamma=2.65 (Mahalanobis) vs Gamma=1.0 (within-width) is not adequately problematized. The Mahalanobis matches allow cross-width pairs, inflating the apparent robustness.

---

## Minor Issues

- Figure numbering jumps from 3 to 5; Sections 4.3 and 4.4 lack visual support
- "Super-absorber" undefined on first use
- 54 pp gap between Chanin (73.1%) and corrected (19.2%) taxonomy rates not reconciled
- Phase 1/3 absorption score measurement protocol consistency not confirmed
- Gaussian GAM on [0,1] bounded response; no specification details reported

---

## Cross-Validation: Paper Claims vs Raw Data

| Claim in Paper | Source Data | Match? |
|---|---|---|
| SP-F1 partial r = -0.746 (p=1.2e-9) | P1_synthesis: -0.7461 (p=1.16e-9) | Yes |
| SCR partial r = -0.570 (p=6.6e-5) | P1_synthesis: -0.5702 (p=6.57e-5) | Yes |
| TPP partial r = -0.331 (p=0.022) | P1_synthesis: -0.3309 (p=0.022) | Yes |
| SCR Sobel z = 3.62 (p=2.9e-4) | P1_mediation: 3.621 (p=2.94e-4) | Yes |
| TPP Sobel z = 2.08 (p=0.037) | P1_mediation: 2.083 (p=0.037) | Yes |
| SCR c' = -0.003 (abstract) | P1_mediation unstd: -0.00290 | Yes (unstd) |
| SCR c' = -0.029 (Table 3) | P1_mediation std: -0.0291 | Yes (std) |
| SCR ab = 0.025 (Table 3) | P1_mediation unstd: 0.02469 | Yes (unstd) |
| Interaction GAM R2 = 0.693 | P3_scaling: 0.6927 | Yes |
| Interaction p = 3.1e-15 | P3_scaling: 3.11e-15 | Yes |
| Corrected taxonomy = 19.2% | P4_taxonomy: 0.1923 | Yes |
| Phase boundary detected | P3_scaling: true; final_results: false | MISMATCH |
| n_full_mediations | P1_mediation: SCR+TPP=2; final_results: 0 | MISMATCH |
| TPP Sobel z in final_results.json | P1_mediation: 2.08; final_results: 2.63 | MISMATCH |
| H5 corrected_rate in final_results.json | P4_taxonomy: 0.192; final_results: 0.923 | MISMATCH |

Four confirmed data pipeline mismatches in final_results.json. Paper body numbers match source data.

---

## What Would Raise the Score

**To 7.0 (Weak Accept):** Fix all critical issues (downgrade causal language, fix Table 3 coefficients, reclassify H2). Fix data pipeline integrity. Add architecture covariate to GAM. Apply FDR correction.

**To 7.5 (Borderline Accept):** Additionally add competing causal explanation in Discussion, power analysis for within-width tests, GAM specification details, and missing figures.

**To 8.0 (Accept):** Would require either (a) cross-domain replication on Gemma 2B showing positive results, or (b) an interventional study demonstrating that reducing absorption (e.g., via OrtSAE) improves quality at matched width/L0. Neither is achievable without additional compute.

---

## Recommendation

**REVISE.** The core findings (SP-F1 suppression effect, SCR full mediation, 420-SAE scaling interaction, taxonomy correction) are genuine contributions that no prior work in the SAE literature has achieved. However, the causal overclaiming, coefficient confusion, and H2 misclassification are each independently sufficient to sink the paper at a top venue. All three are fixable with writing changes and zero additional compute. The architecture GAM ablation requires 30 minutes of CPU time. After these fixes, the paper would be in the 6.5-7.0 range -- a genuine workshop contribution with clear novelty, positioned for a main-conference submission after Gemma 2B replication of Phase 2.
