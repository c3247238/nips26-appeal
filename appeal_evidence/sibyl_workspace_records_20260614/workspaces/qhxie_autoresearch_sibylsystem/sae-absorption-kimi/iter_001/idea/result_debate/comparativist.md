# Comparativist Analysis: Positioning Our Results Against SOTA

**Date:** 2026-04-14  
**Analyzed experiments:** E1 (Pareto evaluation), E2 (Downstream causal meta-analysis), E3 (Task-agnostic metric validation)  
**Data sources:** `exp/results/full/e1_full_gpt2_summary.md`, `exp/results/e2_meta_summary.md`, `exp/results/e3_validation_summary.md`, `context/literature.md`

---

## 1. Baseline Landscape: Where Does This Work Stand?

### SOTA Absorption-Mitigation Methods (Published Numbers)

| Method | Absorption Rate | Model / Layer / L0 | Source |
|--------|-----------------|-------------------|--------|
| **Matryoshka SAE** | **0.015** | Gemma-2-2B, Layer 20, L0=70 | [OrtSAE paper, Table 1](https://arxiv.org/pdf/2509.22033) |
| **Matryoshka SAE** | **0.05** | Gemma-2-2B, Layer 12, L0=40 | [Matryoshka SAE paper](https://arxiv.org/pdf/2503.17547) |
| **OrtSAE** | **0.095** | Gemma-2-2B, Layer 20, L0=70 | [OrtSAE paper, Table 1](https://arxiv.org/pdf/2509.22033) |
| **BatchTopK SAE** | **0.220** | Gemma-2-2B, Layer 20, L0=70 | [OrtSAE paper, Table 1](https://arxiv.org/pdf/2509.22033) |
| **BatchTopK SAE** | **0.49** | Gemma-2-2B, Layer 12, L0=40 | [Matryoshka SAE paper](https://arxiv.org/pdf/2503.17547) |
| **ReLU SAE** | **0.371** | Gemma-2-2B, Layer 20, L0=70 | [OrtSAE paper, Table 1](https://arxiv.org/pdf/2509.22033) |

**Key SOTA insight:** Matryoshka SAEs are the undisputed leader on the canonical absorption metric (first-letter spelling task), achieving **~10x lower absorption** than BatchTopK and **~6x lower** than OrtSAE. OrtSAE claims a 65% reduction over BatchTopK, but Matryoshka still dominates.

### Our Results vs. SOTA

Our experiments were run on **GPT-2 Small** (not Gemma-2-2B, due to gated-model access constraints), using a **simplified first-letter absorption proxy**.

| Experiment | Our Best Absorption | SOTA (Matryoshka) | Delta | Verdict |
|------------|---------------------|-------------------|-------|---------|
| E1 full GPT-2 | 0.000 (feature_splitting arch) | 0.015 (Gemma L20) | **-0.015** | Appears better, but **degenerate proxy** |
| E1 full GPT-2 | 0.000 (most standard SAEs) | 0.015 (Gemma L20) | **-0.015** | Degenerate proxy — 9/10 checkpoints show zero |
| E3 validation | 0.000 (first-letter) vs. 0.24 (task-agnostic) | 0.05 (Gemma L12) | Incomparable | Different model, different metric |

**Brutal honesty:** Our first-letter absorption proxy is **degenerate on GPT-2 Small** — 9 out of 10 checkpoints show zero absorption, and the one non-zero checkpoint (TopK_Attn, 0.654) is an outlier. This makes direct comparison to Gemma-based SOTA **unreliable at best, misleading at worst**. The proxy was explicitly flagged in the pilot summary as "too coarse" and "not aligning with the rigorous sae-spelling benchmark."

---

## 2. Contribution Margin: How Big Is the Delta?

### E1: Multi-Objective Pareto Evaluation

We evaluated 27 GPT-2 Small checkpoints across `standard` and `feature_splitting` families.

| Metric | Standard (mean) | Feature Splitting (mean) | Delta | Classification |
|--------|-----------------|--------------------------|-------|----------------|
| Absorption | 0.015 | 0.000 | -0.015 | **Marginal / degenerate** |
| Hedging | 0.833 | 0.888 | +0.055 | Marginal |
| Explained Variance | 0.830 | 0.982 | +0.152 | **Strong** |
| CE Loss Recovered | 1.054 | 1.172 | +0.118 | **Strong** |
| Dead Neuron Rate | 0.197 | 0.000 | -0.197 | **Strong** |

**Mann-Whitney U tests:** Only **CE Loss Recovered** showed statistical significance (U=4.0, p=0.0014). Absorption and hedging differences were **not significant** (p=0.75 and p=0.81).

**Verdict:** The feature_splitting architecture shows a **genuine advantage on reconstruction and dead-neuron rate**, but the absorption "advantage" is an artifact of the degenerate proxy. On a proper metric, this would likely shrink to marginal or zero.

### E2: Downstream Causal Meta-Analysis (314 SAEBench SAEs)

This is our **strongest result**.

| Outcome | Pearson r (absorption) | Partial r (controlling L0 + CE) | Effect Size |
|---------|------------------------|--------------------------------|-------------|
| Sparse Probing F1 | -0.348 | **-0.385** | Moderate |
| RAVEL Cause | -0.264 | **-0.237** | Small-Moderate |
| RAVEL Isolation | -0.263 | **-0.266** | Small-Moderate |

**OLS regression:** Absorption_mean is a **significant negative predictor** of all three downstream metrics (p < 0.001), even after controlling for L0, CE loss recovered, width, and architecture dummies.

- Sparse Probing F1: beta = **-0.037**, t = -6.81
- RAVEL Cause: beta = **-0.022**, t = -3.82
- RAVEL Isolation: beta = **-0.023**, t = -4.11

**Classification:** This is a **moderate contribution**. The effect is statistically robust across 314 SAEs, but the R² values are modest (0.15–0.32), meaning absorption explains only a fraction of downstream variance. It supports the claim that absorption has *causal practical harm*, but it does not revolutionize the field.

### E3: Task-Agnostic Metric Validation

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Pearson r (task-agnostic vs. first-letter) | **-0.592** | Weak **negative** correlation |
| Spearman rho | **-0.529** | Weak **negative** correlation |
| p-value | 0.116 | **Not significant** |

**Verdict:** This is a **negative result** — and a valuable one. H3 predicted r > 0.4 positive correlation. Instead, we found a weak negative correlation driven by the degenerate first-letter proxy. This suggests the first-letter benchmark may be **unrepresentative of general absorption behavior**, or that different architectures absorb different semantic hierarchies in qualitatively different ways.

**Contribution margin:** Marginal to moderate. It challenges the validity of the dominant benchmark, but the sample size is small (N=10) and limited to GPT-2 Small.

---

## 3. Concurrent Work Scan

### Recent Papers (Last 6 Months) on Absorption & SAE Evaluation

1. **"OrtSAE: Orthogonal Sparse Autoencoders Uncover Atomic Features"** (Korznikov et al., Sep 2025, arXiv:2509.22033)
   - Claims 65% absorption reduction via orthogonality penalties.
   - **Collision with our work:** OrtSAE directly targets the same problem (absorption reduction) with strong empirical results. Our E1 does not evaluate OrtSAE checkpoints, creating a gap.

2. **"Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders"** (Chanin et al., May 2025, arXiv:2505.11756)
   - Introduces hedging as the *opposite* failure mode to absorption.
   - **Collision:** Our E1 includes a hedging proxy and finds a tradeoff — but Chanin et al. already formalized this theoretically and empirically. Our contribution here is more confirmatory than groundbreaking.

3. **"Improving Robustness In Sparse Autoencoders via Masked Regularization"** (Narayanaswamy et al., Apr 2026, arXiv:2604.06495)
   - Proposes token-replacement masking to reduce absorption.
   - **Collision:** Another architectural fix with reported absorption gains. Our work does not benchmark this method.

4. **"SynthSAEBench: Evaluating Sparse Autoencoders on Scalable Realistic Synthetic Data"** (Chanin & Garriga-Alonso, Feb 2026, arXiv:2602.14687)
   - Synthetic benchmark with ground-truth hierarchy and superposition.
   - **Collision:** They can measure absorption with *known* ground truth, bypassing the task-specific proxy problem entirely. This undercuts the novelty of our task-agnostic metric pilot.

5. **"A Unified Theory of Sparse Dictionary Learning..."** (Tang et al., Dec 2025, arXiv:2512.05534)
   - Theoretical framework explaining absorption via spurious minima.
   - **No direct collision** — our E2 empirical meta-analysis complements rather than competes with this theory.

**Concurrent work assessment:** The absorption-mitigation space is **crowded and moving fast**. OrtSAE, masked regularization, and SynthSAEBench all arrived within the last 6–12 months. Our strongest differentiator is **not** a new architectural fix or a new metric, but the **multi-objective Pareto framing** and the **downstream causal meta-analysis** (E2). Even there, the effect sizes are moderate.

---

## 4. Novelty Verdict: The ONE Thing This Work Does That No Prior Work Does

> **"This work is the first to systematically test whether absorption-mitigation architectures dominate standard SAEs on a multi-objective Pareto front spanning absorption, hedging, reconstruction, dead neurons, and downstream probing — using only existing pretrained checkpoints in a training-free setting."**

**Is this true?** Mostly, but with caveats:
- **True:** No prior paper explicitly frames absorption-mitigation as a **Pareto-dominance question** across the *full* metric suite.
- **True:** The E2 meta-analysis (314 SAEBench SAEs) is the largest controlled study linking absorption to downstream interpretability utility.
- **Caveat:** The actual Pareto evaluation (E1) was limited to **GPT-2 Small** and a **degenerate absorption proxy**, substantially weakening the empirical punch.
- **Caveat:** The "training-free" constraint was born of resource limitations, not experimental design virtue.

**Honest novelty score:** **6/10**. The framing is genuinely contrarian and useful, but the empirical execution falls short of the ambition.

---

## 5. Venue Recommendation

| Tier | Assessment |
|------|------------|
| **NeurIPS / ICML / ICLR** | **Unlikely** for the current execution. The E1 Pareto analysis is too limited (GPT-2 Small, degenerate proxy), and the absorption-mitigation space is now crowded with stronger empirical results (OrtSAE, Matryoshka). |
| **AAAI / EMNLP / ACL** | **Possible** if the paper is reframed around the **E2 meta-analysis** and the **task-agnostic metric critique** (E3). The downstream causal cost of absorption is a story that could fit AAAI or EMNLP interpretability tracks. |
| **Workshop (e.g., MechInterp @ ICML, XAI @ NeurIPS)** | **Most realistic** for the current body of work. A workshop paper could present the multi-objective framing, the E2 meta-analysis, and the negative E3 result as a "lessons learned" contribution. |
| **Insufficient for submission** | **Not yet** — E2 alone is publishable with stronger positioning. |

**Recommendation:** Target a **top-tier workshop** for the current iteration. If the project is extended with:
- Proper SAEBench absorption metric on Gemma-2-2B or Pythia-160M,
- Inclusion of OrtSAE and Matryoshka checkpoints in the Pareto analysis,
- A validated task-agnostic metric with N > 50 and r > 0.4,

then a **mid-tier conference (AAAI / EMNLP)** becomes achievable. Top-tier remains difficult unless a surprising new finding emerges.

---

## 6. Strengthening Plan: 3 Comparisons That Would Maximally Bolster Positioning

### 1. **Direct Comparison Against OrtSAE and Matryoshka on SAEBench**
   - **What:** Run the full SAEBench absorption metric (not the simplified proxy) on pretrained OrtSAE and Matryoshka checkpoints, alongside Standard and TopK baselines.
   - **Why:** OrtSAE is the most credible recent competitor. Without benchmarking it, our "no architecture dominates" claim is unsupported for the methods readers care about most.
   - **Feasibility:** Training-free. Requires access to pretrained checkpoints (some may need training if not publicly released).

### 2. **Hedging-Absorption Tradeoff on the Same Checkpoints**
   - **What:** Compute both absorption *and* hedging using the official SAEBench / sae-spelling implementations on the same 50–100 checkpoints.
   - **Why:** Chanin et al. (2025) proved this tradeoff exists for narrow SAEs. Showing it empirically across a broad checkpoint corpus (including wide SAEs) would strengthen the "impossibility triangle" narrative.
   - **Feasibility:** Training-free. SAEBench already has both metrics.

### 3. **Cross-Model Validation of the Task-Agnostic Metric**
   - **What:** Run the geography-hierarchy absorption metric on **Pythia-160M** and **Gemma-2-2B** (if HF access is resolved) SAEs, and correlate with the official first-letter absorption score.
   - **Why:** The E3 pilot failed (r = -0.59, p = 0.116) on GPT-2 Small with N=10. This could be a genuine model-specific artifact. A larger, cross-model validation would determine whether the first-letter benchmark is universally unrepresentative or just unrepresentative for GPT-2.
   - **Feasibility:** Moderate. Requires automated hierarchy discovery pipeline to scale, but no SAE training.

---

## 7. Red Flags & Risks

| Risk | Evidence | Severity |
|------|----------|----------|
| **Degenerate absorption proxy** | 9/10 checkpoints show 0.0 first-letter absorption on GPT-2 Small. | **Critical** |
| **Gemma fallback limits generalizability** | All SOTA numbers are on Gemma-2-2B; our E1 is on GPT-2 Small. | High |
| **Concurrent work collision** | OrtSAE (Sep 2025) and SynthSAEBench (Feb 2026) both have stronger empirical claims. | High |
| **Negative result may be dismissed** | E3's weak negative correlation could be seen as "pilot failure" rather than valuable insight. | Medium |
| **Architecture dummies are insignificant in E2** | No architecture family shows significant downstream advantage after controlling for absorption, L0, and CE. This weakens the "different architectures occupy different tradeoff regions" story. | Medium |

---

## 8. Bottom Line

Our work has a **genuinely novel framing** (multi-objective Pareto evaluation of absorption mitigations) and a **statistically robust meta-analytic finding** (absorption has a unique negative causal effect on downstream interpretability). However, the **empirical execution is too limited** to compete with recent SOTA on raw absorption numbers. The E1 Pareto analysis is constrained by a degenerate proxy and a small open model; the E3 task-agnostic metric pilot produced a negative result that, while intellectually valuable, does not constitute a publishable advance on its own.

**To make this a credible submission, the paper should be reframed around E2 as the primary contribution**, with E1 and E3 serving as supporting evidence for the broader "tradeoffs, not fixes" narrative. Even then, **direct benchmarking against OrtSAE and Matryoshka** is essential before any venue submission.
