# Methodologist Audit: SAE Absorption-Mitigation Experiments

## Scope
This audit examines the methodology of three experiments (E1, E2, E3) conducted for the project "The Hidden Cost of Fixing Feature Absorption." The experiments are training-free evaluations using publicly released SAE checkpoints.

---

## 1. Baseline Fairness Audit

### E1 (Multi-Objective Pareto Evaluation)
- **Baseline:** Standard SAEs (res-jb, resid-post-v5, resid-mid-v5, etc.).
- **Treatment:** Feature-splitting SAEs (res-jb-fs) and various TopK/MLP/Attention variants.
- **Fairness Issue — Hook-point mismatch:** The comparison mixes `resid_pre`, `resid_post`, `resid_mid`, `hook_mlp_out`, `hook_attn_out`, and `hook_z` SAEs. The methodology.md states "Use identical hook points (`resid_pre` or `resid_post`) within each comparison," but the actual E1 full GPT-2 run violates this. For example, `mlp-out-v5-32k-l8` and `attn-out-v5-32k-l8` are evaluated alongside residual-stream SAEs. This is a **major asymmetry** because reconstruction fidelity, dead-neuron rates, and absorption are all hook-point dependent.
- **Fairness Issue — Dictionary width mismatch:** Checkpoints range from d_sae=768 (feature_splitting) to d_sae=131,072 (resid-post-v5-128k). The methodology claims to "control for dictionary width," but the actual checkpoint selection does not enforce width matching.
- **Fairness Issue — Model fallback:** The planned Gemma-2-2B evaluation was abandoned due to gated-model access issues and replaced with GPT-2 Small. This means the E1 "full" results come from a much smaller model than proposed, limiting external validity.

### E2 (Downstream Meta-Analysis)
- **Baseline fairness: GOOD.** The regression includes architecture-family dummy variables, and the dataset is the same (SAEBench) for all architectures. The predictor (absorption) and controls (L0, CE loss recovered, width) are measured on the same underlying checkpoints.
- **Caveat:** The pilot note in `e2_meta_regression_results.json` states: "Sparse probing and RAVEL metrics were unavailable from HF due to rate limits. Synthetic proxies were generated for pipeline validation in this pilot." If the reported E2 summary is based on synthetic proxies, the entire E2 result is **methodologically invalid** for publication. The summary.md does not repeat this caveat, creating a risk of presenting synthetic data as real.

### E3 (Task-Agnostic Metric Validation)
- **Baseline fairness: POOR.** The task-agnostic metric (geography-hierarchy probe) is compared against a "simplified first-letter proxy" that the pilot summary itself calls "too coarse" and "degenerate on most checkpoints (0.0)." Comparing a novel, carefully designed metric against a broken baseline is not a fair validation.

---

## 2. Metric–Claim Alignment

| Claim | Metric | Alignment Assessment |
|-------|--------|---------------------|
| "Absorption-mitigation methods do not dominate standard SAEs on a multi-objective Pareto front" | Absorption score, hedging score, explained variance, CE loss recovered, L0, dead-neuron rate | **MISALIGNED.** The E1 full run only evaluated 27 GPT-2 checkpoints, and the key mitigation architectures (OrtSAE, Matryoshka, JumpReLU, masked regularization) are **completely absent**. Only "standard" and "feature_splitting" families appear. The claim about mitigation methods cannot be tested with this checkpoint set. |
| "Higher absorption correlates negatively with downstream interpretability utility" | Partial correlation of absorption_mean vs. sparse_probing_f1 / RAVEL Cause / RAVEL Isolation | **ALIGNED, IF DATA ARE REAL.** The regression specification is appropriate (OLS with architecture dummies and cluster-robust SEs). However, the synthetic-proxy caveat threatens validity. |
| "Task-agnostic absorption metric correlates with first-letter benchmark" | Pearson r between task-agnostic score and first-letter score | **MISALIGNED.** The first-letter proxy used in E3 is acknowledged to be degenerate. A failed correlation with a broken benchmark does not falsify the construct validity of the new metric. |

---

## 3. Validity Threats Checklist

- [x] **Data leakage / test-set contamination:** Not applicable for training-free checkpoint evaluation, but the SAEBench meta-analysis could inherit any leakage present in the original SAEBench training/evaluation pipeline. The audit cannot verify this without inspecting SAEBench source code.
- [x] **Selection bias:** The E1 checkpoint selection is convenience-based (what loads without errors on GPT-2 Small) rather than systematic. High-MSE checkpoints (e.g., `res-sce-ajt-l6` with MSE=47,186) were included without exclusion criteria, potentially skewing family averages.
- [x] **Overfitting to evaluation:** The E1 results are specific to GPT-2 Small and a single C4 subset. The pilot summary explicitly warns: "No cross-model validation."
- [x] **Measurement construct validity:** The absorption metric in E1/E3 is a simplified proxy, not the canonical Chanin et al. spelling-task pipeline. The methodology.md admits this is "too coarse."
- [x] **Confounding by hook point:** As noted above, mixing `resid_pre`, `resid_post`, `mlp_out`, and `attn_out` SAEs without stratification confounds architecture comparisons with hook-point differences.

---

## 4. Ablation Gap Analysis

The project is framed as evaluating existing pretrained checkpoints, so traditional ablations (removing components from a single model) are not expected. However, from a methodological standpoint:

- **Missing:** A controlled comparison that isolates the effect of **dictionary width** from architecture family. Feature-splitting SAEs have d_sae=768–6144, while standard SAEs have d_sae=24k–131k. The observed differences in explained variance and CE recovery may be width effects, not architecture effects.
- **Missing:** A stratified analysis by **hook point**. The current aggregation across `resid_pre/post/mid`, `mlp_out`, and `attn_out` obscures whether any architecture family performs better at a specific location.
- **Missing:** A sensitivity analysis excluding the **extreme outlier** `res-sce-ajt-l6` (MSE=47k, dead-neuron rate=60%). This checkpoint alone could drive family-level averages for the AJT series.

---

## 5. Reproducibility Score

**Score: 2 / 5**

| Criterion | Status |
|-----------|--------|
| Random seeds fixed | Yes (seed=42) |
| All hyperparameters specified | Partially (L0 targets are implicit from checkpoint selection) |
| Code available | The evaluation scripts (`e1_full_gpt2_pilot.py`, `e1_full_gemma_pilot.py`) exist in `exp/code/`, but the exact absorption and hedging implementations are ad-hoc proxies |
| Data / checkpoint availability | Mixed — GPT-2 SAEs are public, but Gemma-2-2B is gated and blocked the planned experiment |
| Hardware documented | Yes (RTX 4090, 24 GB) |
| Could a competent ML engineer reproduce within 10%? | **No for E1/E3** because the simplified proxies are not standardized. **Possibly for E2** if real SAEBench data are used. |

---

## 6. Top-3 Recommendations

### R1. Fix the absorption metric before any further claims (High Impact / Medium Effort)
The simplified first-letter proxy is degenerate (returns 0.0 for 26 of 27 checkpoints). Replace it with the proper `sae-spelling` pipeline or the SAEBench absorption benchmark. Without this, E1 and E3 cannot support any claim about absorption.

### R2. Stratify E1 by hook point and width, or restrict to matched configurations (High Impact / Low Effort)
Do not compare `resid_pre`, `mlp_out`, and `attn_out` SAEs in the same Pareto front. Either analyze each hook point separately, or restrict the checkpoint corpus to a single hook point with matched widths. This removes a major confounder.

### R3. Clarify the provenance of E2 downstream metrics (High Impact / Low Effort)
The `e2_meta_regression_results.json` contains a pilot note about "synthetic proxies" due to HF rate limits, but the `e2_meta_summary.md` presents the results without this warning. **Determine whether the reported E2 numbers are real SAEBench metrics or synthetic placeholders.** If synthetic, E2 must be re-run with real data before inclusion in any paper.

---

## Summary Judgment

- **E1:** The pipeline runs end-to-end, but the checkpoint selection does not match the research question (no OrtSAE/Matryoshka/JumpReLU), the absorption metric is broken, and hook-point confounding undermines the Pareto analysis.
- **E2:** The regression design is sound, but the data provenance is uncertain due to the synthetic-proxy caveat. This is the most methodologically promising experiment if real data are confirmed.
- **E3:** The task-agnostic metric shows variance, but validation against a degenerate first-letter proxy is methodologically unfair. A proper validation requires either the canonical Chanin et al. metric or a larger, more diverse checkpoint set.
