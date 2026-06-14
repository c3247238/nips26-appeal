# Experiment Critique: SAE Feature Absorption Experiments

## Overall Assessment

The experiments are technically competent but suffer from severe confounds, underpowered designs, and training failures that undermine the validity of most results. The honest reporting of null results is commendable, but the experimental design does not support the paper's key claims.

---

## Critical Issues

### 1. Dead Feature Ratio of 89-99% Invalidates Trained-SAE Results

**Evidence** (f2_causal_results.json):
- k=10: 99.4% dead features
- k=25: 98.7% dead features
- k=50: 97.3% dead features
- k=100: 94.1% dead features
- k=200: 89.1% dead features

**Analysis**: With 99.4% dead features at k=10, the SAE effectively has ~100 active features out of 16,384. This is not a functional SAE---it is a severely broken one. The collision rate of 23.1% at k=10 means 6 out of 26 letters share features, but with only ~100 active features total, this measurement is statistically fragile and likely dominated by training artifacts.

**Why this matters**: All E2, E3 results depend on these trained SAEs. If the SAEs are not training properly, the collision rates, probing accuracies, and correlations are measuring training failure, not architectural properties.

**Root cause**: Likely insufficient training tokens (1M), inappropriate learning rate (1e-3), or SAELens configuration issues.

**Fix**: Retrain with 10M+ tokens, lower learning rate, monitor dead feature ratio during training. Target < 50% dead features.

---

### 2. E1 Architecture Comparison is Confounded

**Evidence** (f1_caab_results.json):

| Factor | JumpReLU | TopK |
|--------|----------|------|
| Source | GemmaScope pretrained | Trained from scratch |
| Training data | Gemma (unspecified) | OpenWebText |
| d_SAE | 24,576 | 16,384 |
| L0 sparsity | 52.4 | 50.0 |
| Recon MSE | 0.93 | 6.58 |
| Dead features | 94.4% | 96.0% |

**Analysis**: The 4x collision rate difference (15.4% vs 3.8%) cannot be attributed to architecture because at least 3 variables differ: source (pretrained vs trained), training data (Gemma vs OpenWebText), and dictionary size (24K vs 16K). The paper acknowledges this in Section 4.2 but still presents the 4x difference as the headline result.

**Fix**: Either (a) retrain JumpReLU on OpenWebText with d_SAE=16K, or (b) demote the architecture comparison to an observation and reframe the paper.

---

### 3. E2 "Causal" Design is Not Causal

**Evidence** (f2_causal_results.json): 5 k-values (10, 25, 50, 100, 200) with collision rates [23.1%, 15.4%, 0%, 23.1%, 19.2%] and probe accuracies [15%, 27.5%, 45%, 77.5%, 72.5%].

**Analysis**: 
- k directly controls both sparsity AND reconstruction quality (MSE ranges from 914 to 10)
- The "causal" claim requires varying only absorption while holding reconstruction constant
- Instead, k varies both simultaneously, creating a structural confound
- The near-perfect correlation between reconstruction MSE and accuracy (rho_S = -1.0) is expected because k directly determines both

**Fix**: Reframe as "sparsity sweep" not "causal impact." Alternatively, train SAEs to a target reconstruction quality rather than fixed k.

---

### 4. E3 Sparsity Sweep Shows Noise, Not Signal

**Evidence** (f3_sparsity_results.json): Collision rates [23.1%, 11.5%, 0%, 15.4%, 19.2%] across k=[10, 25, 50, 100, 200].

**Analysis**: The "non-monotonic" pattern (k=50 has 0% collision) is likely noise. With n=5 and dead feature ratios of 89-99%, collision rate measurements are unstable. The pilot showed 30.8% at k=25, while the full experiment shows 11.5% at k=25---an unexplained 3x discrepancy.

**Fix**: Run multiple seeds and report error bars. The pattern may disappear with replication.

---

### 5. E4 Layer Analysis is Underpowered

**Evidence** (f4_layer_results.json): 6 layers with collision rates [7.7%, 19.2%, 3.8%, 3.8%, 15.4%, 15.4%]. Spearman r = 0.09, p = 0.87.

**Analysis**: With n=6, the study has ~20% power to detect a medium effect. The null result is consistent with both "no depth effect" and "insufficient power to detect an effect."

**Fix**: Add more layers (e.g., all 12 layers of GPT-2 Small) or report as inconclusive rather than "no clear trend."

---

### 6. UAD Validation is Homogeneous and Tiny

**Evidence** (f5_uad_results.json): 25 true collisions, all from first-letter features. 46 same-cluster pairs total.

**Analysis**: UAD is validated on a single, homogeneous concept set (first letters). Perfect recall (100%) is achieved by flagging all same-cluster pairs, but 54.3% precision means nearly half are false positives. The validation does not test generalization to other concept domains.

**Fix**: Validate on at least 2-3 concept domains. Report precision-recall curves across threshold values, not just a single threshold.

---

### 7. DFDA is Evaluated on Only 4 Pairs, All Sharing One Feature

**Evidence** (f6_dfda_results.json): 4 pairs, all with shared_feature=18486 (letters c, i, o, p, u). Improvements: [41.8%, 6.2%, 18.0%, -21.4%].

**Analysis**: 
- All pairs share the SAME feature, so DFDA is not tested across diverse absorption patterns
- One pair degrades by 21.4%, indicating overfitting
- The improvement is on per-pair residual MSE at 10^-6 scale, not overall reconstruction
- 388 total parameters is trivial; the question is whether it generalizes

**Fix**: Test on at least 10+ distinct shared features. Report generalization to unseen pairs.

---

### 8. Pilot-Full Discrepancy is Unexplained

**Evidence**:
- Pilot P1 (TopK, k=25, d_SAE=3072): 30.8% collision rate
- Full E1 (TopK, k=50, d_SAE=16384): 3.8% collision rate
- Full E3 (TopK, k=25, d_SAE=16384): 11.5% collision rate

**Analysis**: The pilot and full experiments show dramatically different collision rates even for the same architecture. The pilot used a smaller SAE (3K vs 16K features) and different k, but the 8x difference (30.8% vs 3.8%) is never explained. This raises concerns about measurement stability.

**Fix**: Explain the discrepancy or rerun the pilot with the same configuration as the full experiment.

---

## Strengths

1. **All experiments completed successfully** with no runtime errors
2. **Data is well-structured** in JSON format with clear schemas
3. **Honest null result reporting** for H2-H4
4. **Source data is verifiable** against paper claims
5. **Fixed seed (42)** ensures reproducibility of the exact run

---

## Summary Table

| Experiment | Status | Key Issue | Severity |
|-----------|--------|-----------|----------|
| E1 (CAAB) | Completed | Confounded comparison | Critical |
| E2 (Causal) | Completed | Not causal; structural confound | Major |
| E3 (Sparsity) | Completed | Likely noise; pilot-full discrepancy | Major |
| E4 (Layer) | Completed | Underpowered (n=6) | Minor |
| E5 (UAD) | Completed | Homogeneous validation set | Major |
| E6 (DFDA) | Completed | Only 4 pairs, 1 feature | Major |
