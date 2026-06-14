# Supervisor Review: Feature Absorption as Competitive Suppression

## Overall Assessment

**Score: 5.0 — Reject (Below top 40%)**

The paper presents a theoretically interesting LCA-SAE connection that provides a plausible mechanistic explanation for feature absorption. The structural correspondence between decoder correlation matrix G = W_dec^T W_dec and the LCA inhibition matrix is mathematically sound and may be a valuable contribution to the field. However, the empirical validation is critically weak: zero hypotheses survive multiple comparison correction, statistical power is severely inadequate, the primary predictive hypothesis (H6) is falsified, and the "optimal compression" framing is post-hoc without direct empirical support. The one robust finding (precision-recall asymmetry) is real but insufficient to carry the paper.

**Verdict: revise**

---

## Dimension Scores

| Dimension | Score | Assessment |
|-----------|-------|------------|
| Novelty | 6 | LCA-SAE connection is novel in the SAE literature, but the specific claim (decoder correlations predict absorption) is falsified. The theoretical contribution is real but partial. |
| Soundness | 6 | The LCA structural correspondence is mathematically correct. However, the "optimal compression" framing is post-hoc and not directly tested. The paper derives theoretical predictions from a falsified hypothesis. |
| Experiments | 4 | Critically underpowered (n=26, ~20% power). Zero significant results after Bonferroni correction (12 tests). H6 falsified (precision@20=0.0). H8 not supported (r=0.12, p=0.55). MCC failure in random baseline comparison. Multiple methodological issues. |
| Reproducibility | 5 | Core experiments described adequately, but analysis code not clearly linked. Steering prompts documented. Probe quality confound not addressed. Post-hoc power analysis is not informative. |

---

## Critical Issues

### 1. Zero Significant Results After MCP
The paper presents 12 hypothesis tests with zero survivors. The abstract, introduction, and conclusion all present H1b (layer 8 delta-corrected steering correlation, r=-0.431, p=0.028) as evidence of a real effect, but this p-value does not survive Bonferroni correction (alpha=0.00417 for 12 tests, corrected p=0.334) or Benjamini-Hochberg FDR (q=0.107). Presenting uncorrected p-values as evidence of effects is methodologically misleading.

**Fix:** Remove all claims about H1b representing a real effect. Restate: "We did not find statistically significant evidence that absorption degrades downstream tasks after rigorous multiple comparison correction."

### 2. Severe Underpowering
With n=26 features, the study has approximately 20% power to detect a medium effect size (r=0.5). This means the study is 4x more likely to miss a real effect than to find it. The null results are uninterpretable: they may reflect true absence of effect OR insufficient sample size.

**Fix:** Add a prominent limitation section stating that non-significant results at this sample size are inconclusive. Use confidence intervals instead of p-values to present results.

### 3. Post-Hoc "Optimal Compression" Framing
H6 (decoder graph predicts absorption) is falsified (precision@20=0.0) and H8 (graph statistics predict at-risk features) is not supported (r=0.12, p=0.55). The "optimal compression" framing is introduced to explain why absorption is "benign", but this was not pre-registered and is not directly tested. The paper uses a falsified hypothesis as theoretical motivation.

**Fix:** Either remove the "optimal compression" claim or run an experiment that directly tests it.

---

## Major Issues

### 4. Probe Quality Confound
Absorption rate correlates with probe F1 at rho=-0.67 (p<0.001). The CMI-absorption analysis uses data from L0 where mean probe F1=0.817. Low-CMI letters may be harder to probe, causing both low estimated CMI and artificially high absorption rates. The paper never computes partial correlation controlling for probe F1.

**Fix:** Compute partial correlation of CMI vs. absorption controlling for probe F1. If the correlation disappears, acknowledge the confound.

### 5. CMI Dimension Instability
The CMI-absorption correlation at d'=10 (rho=-0.383, p=0.059 uncorrected) **reverses sign** at d'>=20 (d'=20: rho=+0.048, d'=30: rho=+0.299, d'=50: rho=+0.197). Bonferroni-corrected p=0.236. This qualitative sign reversal across dimensions is a red flag suggesting the phenomenon is not stable.

**Fix:** Acknowledge the dimension instability. Do not present the d'=10 correlation as a robust finding.

### 6. MCC Failure in Random Baseline Comparison
The random SAE baseline comparison (H7) uses Hungarian matching with MCC~0.21 across ALL variants including Random control. This suggests chance-level recovery regardless of training. The paper acknowledges MCC failure but does not address the implication.

**Fix:** Either use a different metric for the random baseline comparison or acknowledge that the comparison may not be meaningful.

### 7. Post-Hoc Power Analysis
Section 3.6 claims "approximately 20% power to detect a medium effect size." This is a post-hoc power analysis, which is methodologically questionable—power should be computed before the experiment.

**Fix:** Remove post-hoc power analysis. Use confidence intervals instead.

### 8. Unmatched OrtSAE Ablation
The OrtSAE ablation comparison is at unmatched L0: without penalty L0~920 vs. with penalty L0~550. The paper uses this to conclude "the orthogonality penalty does not appear to reduce absorption", but this is confounded by dictionary size.

**Fix:** Match L0 between conditions or explicitly acknowledge the confound.

---

## What Would Raise the Score

**To reach 6 (Borderline Reject):**
1. Use larger feature set to achieve adequate power (>80% for medium effects)
2. Pre-register analysis plan
3. Report all results with confidence intervals
4. Remove or significantly weaken "optimal compression" claim
5. Address MCC failure in random baseline comparison

**To reach 7.5 (Borderline Accept):**
1. Additionally validate LCA-SAE connection on a different SAE architecture
2. Test homeostatic rebalancing empirically
3. Demonstrate cross-model generalizability

---

## Evidence Cross-Validation

**H6 (inhibition graph predicts absorption):** Verified falsified. Precision@20=0.0 across all 26 features. Fisher exact test p=1.0. The raw data in `h6_inhibition_graph.json` confirms zero hits out of 520 predictions (26 features × 20 neighbors).

**H7 (precision-recall asymmetry):** Verified supported. Layer 8 at k=20: precision std = 0.016, recall std = 0.167 (10x difference). 25 of 26 features achieve precision=1.0. This is consistent with competitive suppression mechanism.

**H1 (steering vs. absorption):** The correlation at layer 8 (r=-0.431, p=0.028 uncorrected) does NOT survive Bonferroni (p=0.334) or FDR (q=0.107). The paper's presentation of this as evidence of effect is misleading.

---

## Summary

The mechanistic framework (LCA-SAE connection) is theoretically sound and may be a valuable contribution. However, the empirical validation is too weak to support publication at a top venue. The study is underpowered, the primary hypothesis is falsified, and the "optimal compression" framing is post-hoc. With honest reporting of limitations and effect sizes, this could be a credible negative-result paper—but the current presentation claims more than the evidence supports.