# Supervisor Review: L0-Matched or Misleading?

**Reviewer**: Independent third-party supervisor (NeurIPS-calibrated)
**Date**: 2026-04-26
**Overall Score**: 5.5 / 10 (Borderline Reject)
**Verdict**: REVISE

---

## Executive Summary

This paper makes an important methodological observation---that L0 (sparsity level) is the dominant driver of absorption rate, confounding cross-architecture comparisons. However, the execution is undermined by **three critical issues** that fundamentally compromise the credibility of the main claims:

1. **MCC metric is at chance level** (~0.22) for ALL variants including an untrained Random control, making the central "null causal result" uninterpretable
2. **Table 1 reports 0% dead latents** for all variants, but raw data shows TopK has 82% dead latents and Matryoshka has 56%
3. **Explained variance shows an implausible 1.8-unit gap** (-0.88 for Baseline vs. +0.994 for OrtSAE), suggesting a computational bug

A prior review (critic/findings.json) incorrectly claimed that pilot data shows lambda=0.02 achieves L0=50. Cross-validation of the actual pilot file (pilot_rq1_l0_match_lambda_0.02.json) shows L0=963.26, NOT 50. The paper's claim that Baseline L1 cannot achieve L0=50 is therefore **consistent with available data**, though the claim would be strengthened by reporting the complete lambda sweep and testing even higher values.

These issues are not minor presentation problems. They strike at the core validity of the paper's evidence. A NeurIPS reviewer would likely reject this submission and demand a major revision with corrected data and additional experiments.

---

## Dimension Scores

### 1. Novelty & Significance: 6/10

**What works**: The L0 confound observation is genuinely important. If true, it redirects community effort away from architectural "mitigations" and toward understanding when absorption matters. The contrarian framing (absorption may be a feature, not a bug) is provocative and well-motivated.

**What doesn't**: The novelty is overstated. SAEBench (Karvonen et al., 2025) already compares architectures and notes sparsity differences. The contribution is quantifying the magnitude of the confound, not discovering it. The paper does not adequately acknowledge this prior observation. The dose-response causality design is novel in the SAE literature, but the execution (single metric, synthetic data only) limits its impact.

**Cross-validation**: The proposal's novelty assessment (Section 7) correctly identifies "Medium" risk for the L0-matched comparison because "SAEBench may release similar analysis." This risk has materialized---SAEBench does note sparsity differences, and the paper's contribution is incremental quantification rather than discovery.

### 2. Technical Soundness: 4/10

**Critical issues**:

**MCC invalidity**: Feature recovery MCC is ~0.216-0.222 across ALL variants including Random (0.222 +/- 0.0004). Reconstruction MSE shows 100x variance (0.01 for Baseline vs. 0.67 for Random), proving the SAEs are genuinely different. But MCC cannot discriminate trained from random. Using a metric at chance level to "falsify" a causal hypothesis is methodologically invalid. The paper's central claim---"absorption does not causally predict downstream interpretability"---rests on an invalid instrument.

**Explained variance bug**: Baseline (-0.88), Gated (-0.49), TopK (-0.39), Matryoshka (-0.28) all have NEGATIVE explained variance, while OrtSAE has POSITIVE (+0.994). This 1.8-unit gap is impossible for variants trained on identical data without a computational error. Negative explained variance means the model is worse than predicting the mean. This bug may affect other derived metrics.

**Dead latent data error**: Table 1 reports 0% dead latents for all variants. Raw data contradicts this:
- TopK: 1677/2048 = 81.9% dead (mean across 5 seeds: 1660-1700)
- Matryoshka: 1151/2048 = 56.2% dead (mean: 1013-1229)
- OrtSAE: 11/2048 = 0.5% dead
- Baseline: 0/2048 = 0% dead (correct)
- Gated: 0/2048 = 0% dead (correct)
- Random: 0/2048 = 0% dead (correct)

This is a clear data reporting error. The paper's Table 1 is factually incorrect for 3 of 6 variants.

**L0-matching claim**: The paper claims Baseline L1 cannot achieve L0=50. The pilot lambda sweep tested up to lambda=0.02, achieving L0=963---still 19x higher than the target. The claim is consistent with available data, but the paper should report the complete sweep (including lambda=0.02) and consider testing even higher values to definitively establish the L1 sparsity floor.

**Duplicate variant**: Matryoshka and MultiScale have identical data (byte-for-byte identical seed-42 values: absorption_rate=0.056937857978690644, dead_latents=1214, etc.). They are reported as distinct variants but are the same experiment.

### 3. Experimental Rigor: 5/10

**Missing statistical tests**: The methodology promises Welch's t-test, Cohen's d, and Bonferroni correction. None appear in the results. Claims like "falsifies the hypothesis," "statistically indistinguishable," and "overlapping substantially" have no statistical basis.

**OrtSAE ablation is self-contradictory**: The paper criticizes unmatched L0 comparisons, then makes an unmatched L0 comparison in the OrtSAE ablation (without penalty at L0~920 vs. with penalty at L0~550). This undermines the paper's own methodological standard.

**Missing downstream metrics**: The methodology (Section 4.3 of proposal) lists three downstream metrics: sparse probing F1, steering efficacy, and circuit-tracing precision. Only MCC is reported. Two of the three promised metrics are absent.

**Missing Manipulation A**: The dose-response design promises two independent manipulations (A: architectural, B: sparsity-induced). Only Manipulation B is reported.

**Training time concerns**: 2.8 seconds per seed for 2M tokens is implausibly fast. No loss curves are provided to verify convergence.

**What works**: The honest reporting of negative results (mutual coherence, semantic generalization) is commendable. The 5-seed replication is appropriate. The synthetic ground-truth approach eliminates probe-based confounds.

### 4. Reproducibility: 4/10

**Data integrity pipeline not evidenced**: Section 3.4 promises five validation checks but no output is provided.

**No code availability**: No hyperparameter configuration files or training scripts are provided.

**No convergence diagnostics**: Training loss curves are promised but absent.

**Explained variance discrepancy**: The 1.8-unit gap suggests a bug that would prevent reproduction.

**Random control seeds are meaningless**: The Random control is untrained; "5 seeds" for Random refer only to random initialization, not training replication.

---

## Issue Classification Summary

| Severity | Count | Categories |
|----------|-------|------------|
| Critical | 3 | MCC invalidity, dead latent error, explained variance bug |
| Major | 6 | Missing stats, OrtSAE ablation confound, duplicate variant, training time, reproducibility gaps, novelty overclaim, L0-matching underpowered |
| Minor | 4 | Missing Manipulation A, mutual coherence correlation, scope reduction, true_l0 explanation |

---

## What Would Raise the Score

**To 6.5 (+1 point)**: Fix Table 1 dead latent data, debug explained variance, add statistical tests with p-values for key comparisons, report complete pilot lambda sweep, remove duplicate MultiScale variant.

**To 7.5 (+2 points)**: Additionally demonstrate MCC validity (or replace with reconstruction MSE which shows 100x variance), provide training loss curves, test higher lambda values to definitively establish L1 sparsity floor.

**To 8.0 (pass threshold)**: Additionally validate on real LLM SAEs (GPT-2 small), add a second downstream metric that shows variance across conditions, address OrtSAE ablation L0 confound by matching L0 before comparing.

---

## Bottom Line

The paper's central methodological observation---that L0 confounds architecture comparisons---is important and potentially impactful. However, the evidence is compromised by critical data errors (dead latents, duplicate variant), an invalid central metric (MCC at chance level), a likely computational bug (explained variance), and overstated causal claims ("falsifies" without statistical support). The L0-matching claim is consistent with data but underpowered. A NeurIPS reviewer would reject this submission and demand a major revision with corrected data, valid metrics, and proper statistical testing. The score of 5.5 reflects that the core idea has merit but the execution is not yet publication-ready.
