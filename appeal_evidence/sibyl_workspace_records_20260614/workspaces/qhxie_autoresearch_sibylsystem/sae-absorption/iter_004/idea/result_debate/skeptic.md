# Skeptic Analysis: Competitive Geometry of Feature Absorption in SAEs

**Agent**: skeptic  
**Timestamp**: 2026-04-14  
**Evidence base**: Full experimental results from P0, C1A-D, C2A-D, C3A-C, plus ablations

---

## 1. Statistical Risk Inventory

### Risk 1: The LV detector (H1) has catastrophically failed, yet the paper proceeds as if it contributed something

The core theoretical contribution -- the Lotka-Volterra competition coefficient alpha_ij as an unsupervised absorption detector -- achieves **test F1 = 0.128** at the best threshold (tau=0.5), with **precision = 8.5% and recall = 25.6%**. The ROC-AUC is **0.148**, which is *below random chance* (0.5). This is not merely a failed hypothesis -- the detector is **anti-predictive**. Features with high alpha_ij are *less* likely to be absorbed than chance would predict.

Furthermore, the LV detector is **outperformed by the naive cosine-similarity baseline** (test F1 = 0.165, AUC = 0.201) -- the exact baseline the proposal dismissed as "performing poorly" based on a LessWrong post. The F1 delta vs. cosine is **-0.037**, meaning the LV theory's additional complexity (frequency ratio, co-activation normalization) actively *hurts* detection.

The sharpness test also fails: the AIC comparison finds **linear > sigmoid** (-61.05 vs -60.95), directly falsifying the LV theory's prediction of a sharp phase transition at alpha ~= 1.0. The threshold sensitivity ablation confirms the F1 peak is at tau=0.4, **not within {0.75, 1.25}** as the LV theory predicts.

**Severity: FATAL FLAW.** The entire LV theoretical framework -- the paper's claimed primary contribution -- is empirically falsified. An AUC < 0.5 means the metric has negative informational value.

### Risk 2: The C3A SAEBench correlation (H3) is confounded by width as a common cause

The correlations look impressive: absorption vs. sparse_probing_f1 Pearson r = -0.595, vs. SCR r = -0.431, vs. RAVEL_TPP r = -0.454. However, the matched comparison (C3B) selected the 5 highest-absorption SAEs that are **all 1M-width** SAEs (absorption scores: 0.81-0.90) and the 5 lowest-absorption SAEs that are **16k or 65k width** (absorption scores: 0.003-0.038). The width difference is ~60x between groups.

Wider SAEs (1M) are known to have worse downstream performance for reasons entirely independent of absorption: sparser activations, lower reconstruction fidelity per-feature, and more dead features. The partial correlations controlling for log_width, layer, and arch_class still show r = -0.66 for sparse_probing, but "arch_class" is coded categorically as {gemma_scope_2b, gemma_scope_2b_canonical} -- only 2 levels for 54 SAEs. This does not adequately control for the continuous effect of width.

The C3B paired t-test (t=4.27, p=0.006) compares groups that differ systematically on width (all high-absorption = 1M, all low-absorption = 16k/65k). This is not a matched comparison -- it is a width comparison with absorption as a label.

**Severity: SERIOUS CONCERN.** The causal claim "absorption degrades downstream utility" cannot be distinguished from "wider SAEs perform worse downstream" without matched-width comparisons.

### Risk 3: The safety probe (C3C) contradicts the paper's narrative

C3C was designed to show that higher absorption = worse safety probe performance. The actual results: the **highest-absorption SAE** (resid_pre_L5_24k, absorption=0.119) has the **smallest probe gap** (0.051, 1-sparse AUC=0.947), while the **lowest-absorption SAE** (resid_mid_L10_32k, absorption=0.0) has a **larger probe gap** (0.118, 1-sparse AUC=0.882). The correlation between absorption and probe gap is **r = -0.759** -- the *opposite* of what H3 predicts.

The three SAEs are at different layers (5, 8, 10) and different hook points (resid_pre, resid_pre, resid_mid), making the comparison uninterpretable even for directional evidence.

**Severity: SERIOUS CONCERN.** C3C directly contradicts the downstream-impact narrative. The paper cannot cite C3A's correlation while ignoring C3C's reversal without cherry-picking.

---

## 2. Alternative Explanations

### For C3A correlations (absorption predicts downstream)
**Alternative**: L0 (average number of active features) is the common cause. Low-L0 SAEs are sparser, have higher absorption (because fewer features compete, leading to winner-take-all dynamics), and also have worse downstream performance (because fewer features = less information retained). The raw data shows the 1M-width SAEs with L0 ~16-26 have absorption scores of 0.88-0.90, while 16k-width SAEs with L0 ~137-279 have absorption scores of 0.003-0.027. L0 was not controlled in the partial correlations. This single confound could explain the entire C3A result.

### For the width paradox (C2B: absorption rises with width)
**Alternative**: The width trend at layer 8 (768: 0.047, 1536: 0.020, 3072: 0.009, 6144: 0.025, 12288: 0.048, 24576: 0.086, 49152: 0.092, 98304: 0.104) shows a U-shape, not monotonic increase. The very smallest SAEs (768) have absorption = 0.047, comparable to the 12k SAE. This pattern is more consistent with a "sweet spot" effect where moderate-width SAEs have enough features to specialize but not enough to fully split, rather than the LV competitive exclusion narrative.

### For the taxonomy (C2D: 92.3% comprehensive absorption)
**Alternative**: The 88.5% Type II rate is acknowledged in the results as likely inflated due to parent-feature misidentification. The "expected magnitude" is computed from global mean-when-active, and n_comparison_tokens is 0 for most letters. This means Type II reduces to "the feature fires weakly on letter tokens relative to its global average," which has nothing to do with absorption per se -- it is a property of polysemantic features that fire at different magnitudes in different contexts. The same metric would flag Type II for features that have nothing to do with absorption.

---

## 3. Proxy Metric Audit

### "Absorption rate" (Chanin metric) is the wrong unit of analysis for C3A
The Chanin absorption metric measures letter-feature absorption on a highly specific task (first-letter identification). The downstream tasks (sparse probing, SCR, RAVEL) measure completely different capabilities. The paper's logic assumes absorption on letter features generalizes to absorption on *all* features, but this is not established. SAEs with high letter-absorption may have perfectly functional features for other domains.

### DAS(k=3) does not measure what the paper claims
DAS(k=3) is described as measuring "distributed competitive exclusion." In practice, it is computed via multi-label logistic regression of parent feature activation on the top-3 children's activations. This measures **predictability**, not **causation**. High DAS(k=3) means children's activations correlate with parent activation -- which is expected for any hierarchically related features, whether or not absorption occurs.

The H4 evaluation confirms: only 42.3% of letters show positive DAS(k=3) slope with width (predicted: 80%). Mean DAS(k=3) actually *decreases* from 24k (0.320) to 49k (0.227) to 98k (0.260), contradicting the monotonic increase prediction. The overall assessment is "PARTIAL" but this is generous -- 42% vs 80% predicted is a clear failure.

### PMI as absorption predictor (H2) measures the wrong thing
The PMI is computed at the token level (token co-occurrence in OpenWebText), but absorption operates at the feature level. The median of top-10 tokens per letter is a crude proxy. The regression confirms this: partial R^2 for PMI = 0.0006 (essentially zero), the coefficient is negative (contrary to H2), and sign consistency across layers is False. H2 is comprehensively falsified.

---

## 4. Severity Classification

| Issue | Category | Rating |
|-------|----------|--------|
| LV detector AUC < random (0.148) | Invalidates H1, the paper's primary theoretical contribution | **FATAL FLAW** |
| LV detector underperforms cosine baseline | Shows the competition coefficient adds no value | **FATAL FLAW** |
| LV sharpness test: linear > sigmoid | Falsifies the phase-transition prediction | **FATAL FLAW** |
| PMI partial R^2 = 0.0006, wrong sign | Comprehensively falsifies H2 | **FATAL FLAW** |
| C3A correlation confounded by width/L0 | Undermines H3 causal claim | **SERIOUS CONCERN** |
| C3B matched comparison not actually matched (all high-abs = 1M width) | Invalidates the paired t-test interpretation | **SERIOUS CONCERN** |
| C3C safety probe shows opposite direction | Directly contradicts H3 | **SERIOUS CONCERN** |
| DAS(k=3) does not increase with width (42% vs 80% predicted) | Largely falsifies H4 | **SERIOUS CONCERN** |
| Type II taxonomy rate likely inflated (88.5%) | Weakens taxonomy contribution | **SERIOUS CONCERN** |
| Model substitution: GPT-2 Small instead of Gemma 2 2B | All results may not transfer to target model | **SERIOUS CONCERN** |
| C1C cross-arch: v5-128k LV F1 = 0.0 at all thresholds | Complete failure of generalization | **SERIOUS CONCERN** |
| Small sample sizes (n=3 for safety probe, n=5 per group for C3B) | Insufficient power for reliable inference | **MINOR CAVEAT** |
| P0 alpha gate FAIL (no letter-feature pair in top-10 alpha pairs) | Early warning sign that was overridden | **MINOR CAVEAT** |

---

## 5. Concrete Remediation

### For Fatal Flaw 1 (LV detector failure)
**Experiment**: Compute alpha_ij on the *correct* model (Gemma 2 2B, if access can be obtained) and validate against sae-spelling ground truth. If still failing, abandon the LV theoretical framework entirely. The paper could pivot to an empirical taxonomy + downstream analysis paper without the LV theory.

**Expected outcome**: If AUC remains < 0.5 on Gemma, the LV framework is definitively dead. If AUC > 0.65 on Gemma, the failure was model-specific (GPT-2 Small has d_model=768, much smaller representation space than Gemma 2B's d_model=2304).

### For Fatal Flaw 2 (PMI falsification)
**Experiment**: Rather than token-level PMI, compute feature-level co-activation PMI: PMI(feature_i, feature_j) from SAE activation statistics. This is a more direct measure of the mechanism the paper proposes. Also try computing PMI on a per-position basis (first token position vs. mid-word positions).

**Expected outcome**: Feature-level PMI might show r > 0.3 with absorption rates even if token-level PMI does not, because feature co-activation captures the SAE's learned representation rather than surface statistics.

### For Serious Concern (C3A width confound)
**Experiment**: Re-run C3A controlling for L0. Specifically: within each width class (16k, 65k, 1M), compute Pearson r between absorption and downstream metrics. If the correlation holds *within* width (especially within 16k and 65k where L0 variation is more natural), the confound objection is weakened. Report width-stratified correlations in a table.

Additionally, for C3B: select top-5 and bottom-5 absorption SAEs *within the same width* (e.g., among 65k SAEs only, select by L0). This would be a properly matched comparison.

**Expected outcome**: Within-width correlations should be weaker (r ~ -0.2 to -0.3) but may still be meaningful. If within-width r is near zero, absorption's downstream impact is entirely mediated by width.

### For Serious Concern (C3C reversal)
**Experiment**: Re-run C3C with SAEs from the same layer and hook point, varying only in absorption level (possible by selecting from the C2B survey at layer 8 with different widths but same hook). Use at least n=10 SAEs per group for adequate power.

**Expected outcome**: If the layer/hook confound is removed, the direction may reverse. But if it persists, the paper must report the C3C null result honestly.

### For Serious Concern (model substitution)
**Experiment**: Obtain HuggingFace gated access to Gemma 2 2B and replicate the P0 pipeline + C1B LV validation on Gemma Scope SAEs. The proposal was designed for Gemma; all GPT-2 results are out-of-distribution for the theoretical claims.

**Expected outcome**: Gemma 2 2B may show different absorption patterns (higher rates at more layers, more pronounced width effects) due to its larger representation space. The LV detector's failure on GPT-2 may be artifact of the small d_model.

---

## Summary Assessment

This experiment iteration has produced **two catastrophic falsifications** (H1: LV detector, H2: PMI predictor) and **one promising but confounded result** (H3: downstream correlation). The theoretical framework that was supposed to unify the paper (LV competitive exclusion) has no empirical support: the detector fails, the sharpness prediction fails, the width paradox prediction partially fails, and the cross-architecture generalization produces literal zeros.

The salvageable contributions are:
1. The C3A downstream correlation analysis, *if* the width/L0 confound can be addressed
2. The absorption taxonomy, *if* Type II can be properly operationalized (currently inflated)
3. The observation that absorption peaks at early-to-mid layers and declines to zero at late layers (C2B)

The paper in its current form cannot claim any of its four stated contributions (probe-free detector, corpus predictor, downstream causal chain, theoretical unification). It should be substantially restructured around the downstream correlation finding as the primary contribution, with the LV theory demoted to a negative result section and the PMI analysis reported as a null finding.
