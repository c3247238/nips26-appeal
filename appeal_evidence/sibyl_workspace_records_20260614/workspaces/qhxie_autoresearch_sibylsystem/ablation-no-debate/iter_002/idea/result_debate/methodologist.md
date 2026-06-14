# Methodologist Analysis: Feature Absorption in SAEs

## 1. Baseline Fairness Audit

### H_Mech Factorial (2x2: random/trained encoder x random/trained decoder)

**Asymmetry found: YES — fatal flaw in pass criteria.**

The original pilot criteria (B ~ D with |diff| < 0.25, C ~ A with |diff| < 0.05) failed on 14/15 full runs (6.7% pass rate). The criteria were then revised post-hoc to "encoder effect > 0.5, decoder effect < 0.1," which yields 100% pass. This is textbook post-hoc criterion adjustment.

| Condition | What was trained | Absorption overlap (seed 42, L0=20) | Absorption cosine |
|-----------|-----------------|-------------------------------------|-------------------|
| A | Nothing | 0.013 | 0.440 |
| B | Encoder only | 0.804 | 0.062 |
| C | Decoder only | 0.013 | 0.467 |
| D | Both | 0.467 | 0.112 |

**Key issue**: B ~ D fails dramatically on overlap (0.804 vs 0.467, t=59.6, p~0). The trained decoder REDUCES absorption overlap compared to random decoder — a "disentanglement" effect that the original criteria did not anticipate. The revised criteria ignore this by focusing on encoder_effect vs decoder_effect ratios, which cannot fail because the decoder effect is near-zero by construction.

**Fairness verdict**: The baseline comparison is asymmetric because Condition B (trained encoder + random decoder) uses a decoder that was never trained to reconstruct, while Condition D uses a jointly trained decoder. The decoder in D learns to compensate for encoder absorption, making the comparison unfair. A fairer baseline would freeze the decoder from Condition D and test with the encoder from Condition B.

### Multi-seed Validation (H1)

**Asymmetry found: PARTIAL.**

The "random SAE" baseline uses random encoder + random decoder. But the factorial shows that random decoder + trained encoder produces absorption rates comparable to fully trained SAEs. The multi-seed comparison conflates two baselines:
1. Random encoder + random decoder (very low absorption: 0.033)
2. Trained encoder + random decoder (very high absorption: ~0.80 overlap)

The comparison is not "trained vs. untrained" but "encoder-trained vs. nothing-trained." The baseline is too weak — it does not establish that training is necessary, only that encoder training is sufficient.

### H_Safe (Safety vs Non-safety)

**Asymmetry found: YES.**

Safety features were selected by differential activation on safety vs. neutral prompts (a functional criterion). Non-safety features were selected by mean activation magnitude (a statistical criterion). The matching controls for magnitude but not for interpretability, sparsity, or layer position. This is not a fair comparison — it compares features selected for different properties.

---

## 2. Metric-Claim Alignment

| Claim | Metric Used | Does it capture the claim? | Gap |
|-------|------------|---------------------------|-----|
| "Absorption is encoder-driven" | Encoder effect = (B-A), decoder effect = (C-A) | PARTIAL — the metric shows encoder training is sufficient, not that it is the causal driver in real SAEs | No metric distinguishes "encoder causes absorption" from "encoder selects pre-existing decoder geometry" |
| "Trained SAEs show higher absorption" | Multi-seed Jaccard overlap | YES — but baseline is too weak (see above) | Does not rule out geometric artifact explanation |
| "Absorption generalizes to unseen data" | Train/test correlation on same geometry | NO — tests same generative process, not new geometry | True generalization would require different hierarchy strength or structure |
| "Steering improves sensitivity for absorbed features" | Sensitivity ratio (absorbed/non-absorbed) | NO — failed replication, ratio ~1.0x | No pre-registered primary alpha; effect direction reverses with alpha |
| "Safety features are disproportionately absorbed" | Mann-Whitney U on matched features | NO — null result, but matching was asymmetric | Feature selection bias undermines null interpretation |

**Measurement gap**: The core construct "absorption" is measured differently across experiments (Jaccard overlap, cosine similarity, correlation). These metrics disagree in magnitude and sometimes direction. The paper treats them as equivalent without justification.

---

## 3. Validity Threats Checklist

- [x] **Data leakage**: The held-out generalization test uses the SAME SAE trained on 80% of data and tested on 20% from the same generative process. The SAE has already seen the hierarchy structure; it is not generalizing to new geometry.
- [x] **Contamination**: Not applicable — synthetic data has no pretraining contamination.
- [x] **Selection bias**: The H_Mech pass criteria were revised after seeing the data. The original criteria (B~D) failed; the revised criteria (encoder effect > 0.5) cannot fail.
- [x] **Overfitting to evaluation**: All experiments use the same synthetic hierarchy structure (parent-child with configurable cosine). The results may be specific to this synthetic setting and not generalize to real LLM features.

**Additional threats**:
- **Metric inconsistency**: Three different absorption metrics (cosine, overlap, correlation) are used interchangeably without establishing equivalence.
- **Ceiling effect in real SAEs**: GPT-2 SAE shows ~97% absorption for both safety and non-safety features, suggesting the metric saturates on high-dimensional SAEs (d_sae=24576).
- **Pilot-to-full replication failure**: H3 steering pilot showed 1.62x ratio; full experiment shows 0.91x ratio. This is a failed replication that undermines the causal claim.

---

## 4. Ablation Gap Analysis

| Proposed Component | Ablation Experiment | Status | Gap |
|-------------------|---------------------|--------|-----|
| Encoder drives absorption | 2x2 factorial (random/trained encoder x decoder) | DONE | Original criteria failed; revised criteria are post-hoc |
| Decoder disentangles absorption | Compare Condition B vs D | DONE | Trained decoder reduces overlap — but no ablation of decoder-only training effect |
| Hierarchy strength affects absorption | Vary cosine similarity (0.5, 0.67, 0.8) | DONE | Only 3 levels; no test of non-monotonic relationship |
| L0 sparsity affects absorption | Vary L0 (20, 32, 50) | DONE | Opposite direction from hypothesis; no test of very high L0 |
| Steering differentially affects absorbed features | Steering with multiple alphas | DONE | Failed replication; no pre-registered primary alpha |
| Safety features are more absorbed | Compare safety vs non-safety | DONE | Null result; asymmetric matching |

**Missing ablations**:
1. **Decoder-only training ablation**: Train only the decoder on hierarchical data (with random encoder) and measure absorption. This would test whether decoder training alone can create absorption.
2. **Feature frequency control**: The multi-seed validation does not control for feature frequency. An ablation that matches absorbed and non-absorbed features by frequency would strengthen the claim.
3. **Real LLM validation**: All experiments are on synthetic data (d_model=128). No ablation tests whether the encoder-driven mechanism holds in real LLM SAEs.
4. **Metric equivalence ablation**: No experiment computes all three metrics (cosine, overlap, correlation) on the same data and reports their inter-correlation.

---

## 5. Reproducibility Score: 2/5

| Criterion | Status | Evidence |
|-----------|--------|----------|
| Random seeds fixed | YES | Seeds 42-46 specified |
| All hyperparameters specified | PARTIAL | d_model=128, d_sae=4096, L0=32, train_steps=2000-5000 specified. But learning rate, optimizer, initialization scheme not documented. |
| Code available | UNKNOWN | No code repository linked in results. |
| Hardware requirements documented | NO | GPU type, memory requirements not specified. |
| Reproduce within 10% | UNLIKELY | Without code and full hyperparameters, reproduction is impossible. The synthetic data generation procedure is not fully specified (e.g., how stochastic_noise=0.1 is applied). |

**Major reproducibility gaps**:
- No code repository or script paths provided
- Learning rate, optimizer, weight initialization not documented
- Synthetic data generation procedure partially specified
- No environment specification (PyTorch version, CUDA version, SAELens version)

---

## 6. Top-3 Recommendations (Ordered by Effort-to-Credibility Ratio)

### 1. Pre-register H_Mech criteria and report both original and revised (LOW effort, HIGH credibility)
**What**: Report the original B~D criteria honestly (6.7% pass rate) and explain why the revised criteria were necessary. Do not present the revised criteria as the primary validation.
**Why it helps**: Restores scientific integrity. The current presentation ("encoder-driven absorption CONFIRMED") is misleading because the original falsification criterion failed.
**Expected outcome**: Paper is more credible; reviewers trust the authors more.

### 2. Add real LLM validation experiment (MEDIUM effort, HIGH credibility)
**What**: Replicate the factorial design on a real SAE (e.g., Gemma Scope layer 5) by comparing absorption on features with known hierarchical relationships (e.g., first-letter features from sae-spelling).
**Why it helps**: All current evidence is synthetic. The field will reject the paper without real-model validation.
**Expected outcome**: Either confirms the synthetic finding (strong paper) or reveals limitations (honest scope-setting).

### 3. Standardize absorption metric and report metric equivalence (LOW effort, MEDIUM credibility)
**What**: Commit to ONE absorption metric (e.g., Jaccard overlap) for all experiments. Report a correlation matrix showing cosine, overlap, and correlation are equivalent (r > 0.8). If they disagree, explain why and justify the chosen metric.
**Why it helps**: The current paper uses different metrics interchangeably, making cross-experiment comparisons unreliable.
**Expected outcome**: Core construct is operationally defined; reviewers can follow the logic.

---

## Summary

The experiments reveal a genuine and robust finding: **encoder training is sufficient to produce high absorption rates in synthetic SAEs**. However, the methodology has critical weaknesses:

1. **Post-hoc criterion revision** for H_Mech undermines the confirmation claim
2. **All evidence is synthetic** — no real LLM validation
3. **Metric inconsistency** across experiments weakens the core construct
4. **Failed replication** of H3 steering undermines causal claims
5. **Reproducibility gaps** (no code, incomplete hyperparameters)

The paper is publishable if these issues are addressed, but the current form would face serious methodological challenges at peer review.
