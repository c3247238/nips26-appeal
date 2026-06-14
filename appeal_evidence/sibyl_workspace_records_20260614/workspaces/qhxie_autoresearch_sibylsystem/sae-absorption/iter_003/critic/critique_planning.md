# Planning Critique

## Critical Issues

### 1. All Experiments Were Run in PILOT Mode — The Full-Scale Plan Was Not Executed

The methodology specified:
- C2 (amortization gap): 26 letters × 1000 tokens, ~30 min
- B1 (co-occurrence graph): 200,000 tokens, ~5 min
- B2 (ARS validation): 5 letters × 4 widths
- D2 (cross-hierarchy): 3 seeds × full token count

The actual execution:
- C2: 3 letters × 30 tokens (90 observations total), labeled "PILOT" in metadata
- B1: 10,000 tokens (5% of specified)
- B2: pilot-scale only
- D2: 100 tokens per category, labeled "PILOT"

All files in `exp/results/full/` are marked `"mode": "PILOT"` in their JSON metadata. The "full" experiment directory contains pilot data. The full-scale experiments were never executed.

This is the most significant planning failure: the methodology specified achievable scales for a system with an RTX PRO 6000 GPU, but the experiments were never scaled up. The gap between pilot-scale and full-scale is the primary driver of the paper's statistical power problems.

**Root cause:** The experiment planning did not include explicit pilot-to-full escalation logic. The pilot criterion was "pass" (correct), but there was no automated step to trigger the full-scale run after pilot success.

**Fix for next iteration:** Add an explicit stage after pilot pass that runs full-scale experiments with the specified parameters. Gate the paper-writing stage on full-scale results being available.

### 2. Gemma Scope Access Was Not Resolved Before Experiments Were Run

The methodology explicitly lists "Gemma Scope (if accessible): Gemma-2-2B with L12-16k, L12-65k SAEs — gated access, use token if available" with a 50% probability of being blocked. The task plan allocated significant effort to Gemma validation experiments. Gemma access was blocked (401 Unauthorized), and the fallback (GPT-2 Small TopK-32k SAE) was used. However, this fallback does not validate the key findings on a modern model.

**Fix:** Before the next iteration, resolve Gemma Scope access (accept the HuggingFace gated license at huggingface.co/google/gemma-scope). This is a one-time manual step that should be completed before planning any experiments that depend on Gemma Scope.

### 3. Cross-Architecture Comparison Has a Fundamental Confound That Was Plannable Around

The hook-point confound (resid_pre vs. resid_post for Standard vs. TopK SAEs) was identified in the methodology as a known issue. The mitigation was "verify that resid_pre ≈ resid_post for this layer (compute average cosine similarity)." This check was never performed. The confound was flagged repeatedly in the paper but was never quantified. The magnitude of the confound determines whether the cross-architecture comparison is interpretable at all.

**Fix:** Add a pre-experiment check: compute cosine similarity between resid_pre and resid_post activations at the target layer on a sample of tokens. If mean cosine similarity > 0.99, the confound is negligible. If < 0.95, the comparison is unreliable and must be flagged as such in the paper.

### 4. Label Reuse Across Iterations Creates Circularity Risk

All experiments use the same n_pos=18 labels from iter_001 R4 (cached at `iter_001/exp/results/r4/r4a_direct_labels.json`). These labels have been used across iter_002, iter_003, and now iter_003's planned experiments. The label set was generated to evaluate EDA in iter_001 — it was not generated as a representative sample for training or validating absorption detectors. The repeated use of the same 18 absorbed features for all comparisons (EDA, EncNorm, ARS, etc.) introduces evaluation overfitting risk: any metric that happens to work on these 18 specific features will appear to work, regardless of generalizability.

**Fix:** Generate new absorption labels on held-out letters or held-out token sets. Ideally, use the iter_001 labels as a training set and generate new labels on 5 different letters for final evaluation.

## Major Issues

### 5. Safety Attribution (H5) Dropped Without Explanation

The proposal ranked H5 (safety attribution analysis) as a primary research question. The methodology included 2 GPU-hours for safety attribution. The experiment was never run and is not mentioned anywhere in the results. The paper's motivation (absorption matters for safety) rests entirely on a literature citation without empirical support from this project.

**Fix:** Either run the safety attribution experiment in the next iteration or explicitly drop it from the paper's motivational claims. Do not use safety motivation if you have no safety results.

### 6. The ρ-Sweep (H4 Phase Transition) Was Only Run on 17 Letters, Not 26

The D2 spearman correlation data includes 17 first-letter data points (not all 26 letters), with all showing absorption_rate = 0.0. The D3 analysis (frequency ratio vs. absorption rate) cannot find a step-function transition if all data points have zero absorption rate. The phase transition test (H4) was effectively untestable with the current measurement method.
