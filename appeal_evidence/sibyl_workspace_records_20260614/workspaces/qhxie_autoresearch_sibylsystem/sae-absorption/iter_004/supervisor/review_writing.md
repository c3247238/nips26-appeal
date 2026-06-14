# Supervisor Review: Iteration 4

## Score: 6.5 / 10 (Borderline Reject -- approaching Weak Accept)

**Verdict: CONTINUE** -- significant improvement needed on confound controls before the paper's strongest result is defensible.

---

## Executive Summary

Iteration 4 represents a genuine step forward over the stagnant iterations 1-3 (all scored 5.5). The project pivoted from the OMP/EncNorm approach -- which was stuck in pilot-scale with proxy-label problems -- to a clean LV competitive exclusion framework. All core experiments now run at FULL scale. The paper honestly reports negative results (LV detector, PMI predictor, DAS width paradox) alongside one strong positive finding (downstream correlation).

The downstream correlation result (H3) is the paper's centerpiece and is genuinely novel: no prior work systematically quantifies the absorption-performance link. The effect sizes are large (r=-0.595, partial r=-0.661 for sparse probing), survive Bonferroni correction, and the matched-pair Cohen's d=2.13 is substantial. If this result survives confound controls, it anchors a publishable paper.

However, three problems prevent a higher score.

---

## Dimension Scores

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| **Novelty** | 7 | Three-tier taxonomy, LV competitive exclusion applied to SAE absorption, and first systematic downstream correlation are all novel. The LV application is novel even though it fails; honest negative results on a novel approach have value. |
| **Soundness** | 6 | The H3 downstream analysis is well-designed (Bonferroni, partial correlations, matched pairs) but has a critical L0/width confound gap. The taxonomy has a known measurement artifact. The LV framework fails but is correctly analyzed. |
| **Experiments** | 6 | Major improvement: all experiments run at full scale (C1B: 1,900 pairs; C2B: 806 observations; C3A: 54 SAEs). However, the safety probe is underpowered, the DAS analysis lacks significance tests, and the regression needs clustered SEs. |
| **Reproducibility** | 6 | GPT-2 Small is fully open and deterministic. SAEBench data is publicly available. Code uses standard SAELens/TransformerLens. The cross-model gap (GPT-2 vs. Gemma) limits generalizability but not reproducibility per se. |

---

## What Works

1. **H3 downstream correlation is the paper's genuine contribution.** Pearson r=-0.595 (sparse probing), r=-0.454 (RAVEL), r=-0.431 (SCR) across 54 Gemma Scope SAEs. Three of four survive Bonferroni. Partial correlations strengthen after controlling width/layer/architecture. Matched-pair Cohen's d=2.13. This is novel, well-executed, and addresses a critical open question in the SAE literature.

2. **Honest negative result reporting.** The LV detector (H1: F1=0.128 vs. 0.35 target), PMI predictor (H2: partial R^2=0.0006 vs. 0.10 target), and DAS width prediction (H4: 42.3% vs. 80% target) are reported transparently with pre-registered success criteria. The sharpness test and cross-architecture failure strengthen the negative findings. This is exemplary scientific practice.

3. **Full-scale experiments resolve the previous systemic failure.** All core result JSONs now carry mode='FULL'. The 30-SAE absorption survey (806 data points, 31 configurations) is a substantial data collection effort. The C1B validation uses proper train/test letter splits.

4. **Well-structured paper.** The writing is clear, claims are appropriately hedged, caveats are disclosed prominently (Type II inflation, safety probe limitations, model scope). The overall paper structure (LV framework -> taxonomy -> downstream impact -> ablations -> limitations) is logical.

---

## Critical Issues

### Issue 1: Width/L0 Confound in H3 (Critical)

The paper's strongest finding is at risk. Cross-validating the C3A raw data:

- **All 5 high-absorption SAEs in C3B** are 1M-width with L0 range 16-58
- **All 5 low-absorption SAEs in C3B** are 16k/65k with L0 range 137-297

The partial correlations control for log(width), layer, and arch_class -- but NOT for L0, which varies by 15x (9 to 445). Low-L0 SAEs are known to have both higher absorption and potentially worse downstream performance. Until L0 is included as a covariate, and within-width correlations are computed, the causal interpretation is unclear.

The SCR partial r inflation from -0.431 to -0.677 (a 57% increase) is unusually large for partial correlations and could indicate suppressor variable artifacts. The paper mentions this possibility but does not investigate.

**Required action**: Add L0 as covariate; compute within-width correlations (all zero-GPU analysis).

### Issue 2: Type II Taxonomy Inflation (Critical)

The C2D taxonomy JSON confirms n_comparison_tokens=0 for almost all letters. The magnitude ratio baseline therefore reflects global activation patterns rather than letter-specific expected behavior. The 88.5% Type II rate is an artifact of this fallback, not evidence of partial absorption. The paper's caveat is clear in Section 5.3 and 7.4, but the abstract headlines "92.3% comprehensive rate" without qualification.

**Required action**: Reframe 92.3% as "upper bound" in abstract and conclusion; lead with validated rates.

---

## Major Issues

### Issue 3: LV Pre-filter Dooms Success Criterion

The decoder cosine > 0.15 filter passes only 34% of absorbed pairs. With maximum recall = 0.34, the F1 > 0.35 criterion is structurally unachievable (F1 <= 2 * 0.34 * 1.0 / (0.34 + 1.0) = 0.507 at perfect precision, but perfect precision requires very aggressive thresholding that would further reduce recall). The paper should analyze alpha_ij discrimination within the filtered set to separate pre-filter failure from coefficient failure.

### Issue 4: Safety Probe Contributes Noise

The C3C experiment with n=3 SAEs differing in layer, width, and hook type is confounded beyond interpretation. The non-monotone result (highest absorption = smallest probe gap) should not appear alongside the SAEBench results.

### Issue 5: Cross-Model Gap

No single hypothesis is tested on both models. The paper frames this as a unified study but is actually two disconnected analyses on different models. The connection is assumed, not demonstrated.

### Issue 6: DAS Lacks Statistical Tests

H4 "partial support" should be "insufficient evidence" without significance tests. The per-letter variance is enormous (letter X goes from DAS=0.0 to 1.0) and 26 letters at 3 widths is underpowered for detecting trends.

### Issue 7: PMI Regression Clustering

Clustered SEs at the letter level would be more conservative. The L0 coefficient (p=0.012) is marginal and likely fails under clustering, leaving only layer as a robust predictor.

---

## Score Trajectory

| Iteration | Score | Key Change |
|-----------|-------|------------|
| 1 | 5.5 | EDA/OMP approach, pilot scale only |
| 2 | 5.5 | Same approach, still pilot scale, proxy label issues |
| 3 | 5.5 | Same approach, 3rd iteration stuck at pilot scale |
| **4** | **6.5** | **Complete pivot to LV framework, full-scale experiments, strong H3 result** |

The 1.0-point increase reflects genuine improvements: full-scale data, novel research questions, honest negative results, and a strong positive finding (H3). The remaining gap to 7.0+ is addressable with analysis-only work (no GPU required).

---

## Path to Higher Scores

**To reach 7.0 (Weak Accept):** Add L0 as covariate in partial correlations + within-width correlations. Reframe 92.3% as upper bound throughout.

**To reach 7.5 (Borderline Accept):** Additionally: clustered PMI regression SEs, alpha_ij AUROC within filtered set, formal DAS significance tests, move safety probe to supplementary.

**To reach 8.0 (Accept):** Additionally: demonstrate at least one finding replicates on a second model (e.g., Gemma taxonomy rates from SAEBench, or Pythia LV coefficient).

The core H3 result is strong enough to anchor a publishable paper IF the confound controls survive. The investment required is modest (analysis-only tasks, zero GPU) and the expected payoff is high.
