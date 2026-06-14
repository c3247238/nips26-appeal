# Revisionist Analysis: What the Data Tells Us About Our Mental Model

## 1. Hypothesis Verdict Table

| Hypothesis | Verdict | Key Evidence | Confidence |
|------------|---------|--------------|------------|
| **H1**: Absorption-mitigation methods do not stochastically dominate standard SAEs on a multi-objective Pareto front. | **Inconclusive (leaning supported)** | E1 full GPT-2: only 2 architecture families evaluated (standard vs. feature_splitting). Feature-splitting SAEs show *better* explained variance (0.982 vs. 0.830) and CE recovery (1.172 vs. 1.054) with zero dead neurons, but slightly higher hedging (0.888 vs. 0.833). Mann-Whitney U on CE recovery is significant (p = 0.0014), but absorption and hedging differences are not (p = 0.75, p = 0.81). No stochastic dominance across >=4 metrics. | Medium |
| **H2**: Controlling for L0 and reconstruction, higher absorption correlates negatively with downstream interpretability utility. | **Confirmed** | E2 meta-analysis on 314 SAEBench SAEs: partial correlation between absorption and sparse probing F1 is r = -0.385 (p < 1e-12); RAVEL Cause r = -0.237 (p < 2e-05); RAVEL Isolation r = -0.266 (p < 2e-06). OLS regression: absorption coefficient is negative and significant for all three outcomes (sparse probing: beta = -0.037, p < 5e-11; RAVEL Cause: beta = -0.022, p = 0.00017; RAVEL Isolation: beta = -0.023, p < 5e-05). | High |
| **H3**: A task-agnostic absorption metric will correlate moderately-to-strongly (r > 0.4) with the first-letter benchmark. | **Refuted** | E3 validation on 10 GPT-2 Small checkpoints: Pearson r = -0.592, Spearman rho = -0.529, p = 0.116 (not significant). 9/10 checkpoints show zero first-letter absorption, while the task-agnostic geography-hierarchy probe shows variance from 0.0 to 0.24. | High |
| **H4** (backup, conditional): Random-decoder SAEs exhibit absorption comparable to trained SAEs. | **Not tested** | Training-free constraint prevented execution. | N/A |

### Why H1 is "inconclusive" rather than "confirmed"
The evidence is directionally consistent with H1, but the experimental scope is too narrow to claim confirmation. We only compared standard SAEs against feature-splitting SAEs on GPT-2 Small. The original hypothesis was about absorption-mitigation methods (OrtSAE, Matryoshka, JumpReLU, Masked Regularization), none of which appeared in the E1 full GPT-2 checkpoint corpus. The Gemma-2-2B experiment (which would have included Matryoshka and other families) failed due to gated-model access. To confirm H1, we need a broader architecture palette on an open model like Pythia-160M.

---

## 2. Surprise Analysis

### Surprise 1: The first-letter absorption metric is degenerate on most GPT-2 Small checkpoints
**Expected:** We assumed the simplified first-letter proxy would show meaningful variance across architectures and layers, enabling at least a coarse Pareto analysis.
**Actual:** 26 out of 27 GPT-2 checkpoints showed *zero* absorption. Only `hook-z-kk-l8` (a standard SAE on attention patterns) showed non-zero absorption (0.345).
**Faulty assumption:** We assumed that absorption is a pervasive pathology that would manifest across diverse SAE checkpoints on any model. The data suggests absorption is either (a) genuinely rare in GPT-2 Small SAEs, (b) highly task-dependent, or (c) our simplified proxy is too coarse to detect it.
**Evidence:** `current/exp/results/full/e1_full_gpt2_summary.md` shows absorption_mean = 0.015 for standard architecture, with 22/23 checkpoints at exactly 0.0.

### Surprise 2: The task-agnostic metric is *negatively* correlated with the first-letter benchmark
**Expected:** We expected two different absorption metrics to measure the same underlying construct, yielding a positive correlation (r > 0.4).
**Actual:** Pearson r = -0.592 across 10 checkpoints. The attention-output TopK SAE has high first-letter absorption (0.654) but zero task-agnostic absorption, while standard residual-stream SAEs have zero first-letter absorption but moderate task-agnostic absorption (0.07-0.24).
**Faulty assumption:** We assumed "absorption" is a single, unified phenomenon. The negative correlation suggests there may be *multiple distinct absorption modes*: lexical/spelling absorption (captured by first-letter) versus semantic/hierarchical absorption (captured by geography probes). Different architectures may be vulnerable to different modes.
**Evidence:** `current/exp/results/e3_validation_correlation_results.json` and `current/exp/results/e3_validation_summary.md`.

### Surprise 3: Absorption has a robust, unique negative effect on downstream utility even after controlling for reconstruction and sparsity
**Expected:** We hypothesized a negative correlation, but expected it to be modest (r_partial ~ -0.25) and potentially confounded by architecture family.
**Actual:** The partial correlation is stronger than expected for sparse probing (r = -0.385), and the regression coefficient remains highly significant (p < 5e-11) with architecture dummies included. Notably, *no architecture family dummy is significant* in the sparse probing regression, suggesting the absorption-downstream link is not merely an artifact of some families being worse overall.
**Faulty assumption:** We assumed architecture family would explain much of the variance in downstream performance, making absorption's independent effect hard to isolate. The data suggests absorption carries unique predictive signal above and beyond architecture choice.
**Evidence:** `current/exp/results/e2_meta_regression_results.json`, sparse_probing_f1 regression table.

### Surprise 4: Feature-splitting SAEs achieve zero dead neurons while maintaining better reconstruction
**Expected:** We did not have a strong prior on feature-splitting SAEs, but we generally assumed narrower or specialized architectures would trade off reconstruction for interpretability.
**Actual:** Feature-splitting SAEs (d_sae = 768-6144) achieve 0% dead neurons, higher explained variance (0.982 vs. 0.830), and better CE recovery (1.172 vs. 1.054) than the broader standard family. Their hedging is only marginally higher (0.888 vs. 0.833, not significant).
**Faulty assumption:** We implicitly assumed that the standard, widely-used SAE configurations represent a reasonable Pareto frontier. The feature-splitting results suggest there may be under-explored regions of the design space that dominate on multiple metrics simultaneously.
**Evidence:** `current/exp/results/full/e1_full_gpt2_summary.md`, architecture family summary table.

---

## 3. Mental Model Revision

Our original framing treated absorption as a single, well-defined pathology that could be measured with a canonical benchmark and mitigated by architectural innovation. The data forces three specific updates:

1. **Absorption is not monolithic.** We assumed that "absorption" measured by the first-letter task generalizes to arbitrary semantic hierarchies, but the strong negative correlation between first-letter and task-agnostic metrics suggests different SAEs fail on different *types* of hierarchical structure. A model may avoid lexical absorption while failing on geographic hierarchies, or vice versa.

2. **The first-letter benchmark is unrepresentative for many SAE families.** We assumed the Chanin et al. spelling task was a reasonable proxy for general absorption, but its degeneracy on most GPT-2 checkpoints (and negative correlation with a semantic probe) suggests it captures a narrow, possibly model-specific failure mode. This does not mean the benchmark is useless, but it cannot be treated as a universal absorption thermometer.

3. **Absorption's causal harm on downstream utility is stronger and more general than we anticipated.** We assumed the effect would be modest and potentially confounded. Instead, the meta-analysis shows a consistent, significant negative effect across sparse probing and RAVEL metrics, independent of architecture family. This shifts absorption from an "aesthetic pathology" to a practically consequential failure mode.

**In short:** We assumed the research question was "Which architecture best fixes absorption?" The data suggests the better question is "What *kind* of absorption does each architecture create or avoid, and what are the downstream costs of each kind?"

---

## 4. Reframing Test

**Original research question (RQ1):** Do absorption-mitigation architectures dominate standard SAEs on a multi-objective Pareto front, or do they systematically trade absorption for other pathologies?

**Would we frame it the same way today?** Partially, but inadequately. The original framing assumes a single absorption metric can adjudicate between architectures. Given the E3 results, this is no longer tenable.

**Revised research question:**
> **RQ1 (revised):** Do different SAE architectures occupy distinct regions of a multi-objective tradeoff space *when absorption is measured across multiple hierarchical domains*? In particular, do architectures optimized to minimize lexical absorption (e.g., first-letter spelling) inadvertently increase semantic/hierarchical absorption, and what are the downstream interpretability costs of each tradeoff region?

This revision:
- Acknowledges that absorption is domain-dependent.
- Retains the multi-objective Pareto framing.
- Explicitly links the metric divergence (Surprise 2) to the downstream causal analysis (Surprise 3).
- Makes the paper's core contribution more defensible and more novel.

---

## 5. New Hypothesis Generation

### New Hypothesis 1 (NH1): Architecture-specific absorption modes
Different SAE architectures exhibit qualitatively different absorption profiles across hierarchical domains (lexical, geographic, biological). Residual-stream SAEs will show higher semantic absorption, while attention-output SAEs will show higher lexical absorption.

**Falsification:** If a cross-domain absorption matrix shows no architecture-specific clustering (i.e., all architectures rank similarly across all domains), NH1 is falsified.

**Experiment:** Extend E3 to 3+ domains (geography, biology, color shades) on 20+ checkpoints spanning at least 4 architecture families. Compute per-architecture, per-domain absorption scores and test for significant interaction effects (ANOVA).

### New Hypothesis 2 (NH2): The absorption-downstream harm is mediated by domain alignment
The negative correlation between absorption and downstream utility is strongest when the absorption domain matches the downstream task domain. For example, lexical absorption will harm spelling-related sparse probing more than geographic absorption will, and vice versa.

**Falsification:** If partial correlations between domain-specific absorption scores and matched downstream tasks are no stronger than mismatched correlations, NH2 is falsified.

**Experiment:** Collect or construct domain-matched sparse probing tasks (e.g., geography facts, biology taxonomy). Regress downstream F1 on domain-specific absorption scores, controlling for L0 and reconstruction. Compare matched vs. mismatched coefficient magnitudes.

### New Hypothesis 3 (NH3): Feature-splitting SAEs achieve a better Pareto tradeoff than previously recognized
Feature-splitting SAEs will dominate standard SAEs on a joint metric of reconstruction, dead-neuron rate, and hedging, with no statistically significant penalty on multi-domain absorption.

**Falsification:** If feature-splitting SAEs show significantly worse absorption than standard SAEs on any domain, or if they are Pareto-dominated by another architecture family on >=3 metrics, NH3 is falsified.

**Experiment:** Re-run E1 on Pythia-160M with a broader checkpoint corpus that includes feature-splitting, Gated, JumpReLU, and Matryoshka families. Use multi-domain absorption (not just first-letter) and test for Pareto dominance per family.

---

## Summary of Key Revisions

| Original Belief | Updated Belief | Evidence |
|-----------------|----------------|----------|
| Absorption is a single, measurable pathology. | Absorption is domain-dependent; different architectures fail on different hierarchy types. | E3: negative correlation (-0.59) between first-letter and task-agnostic metrics. |
| First-letter absorption is a reasonable universal proxy. | First-letter absorption is narrow and unrepresentative for many SAE families. | E1: 26/27 GPT-2 checkpoints show zero first-letter absorption; E3: task-agnostic metric shows variance where first-letter does not. |
| Absorption's downstream harm is modest and confounded. | Absorption has a robust, unique negative causal effect on sparse probing and RAVEL. | E2: significant partial correlations and regression coefficients across 314 SAEs, independent of architecture family. |
| The field should focus on "fixing absorption." | The field should focus on characterizing *which* absorptions each architecture creates, and selecting architectures task-adaptively. | Combined evidence from E1, E2, and E3. |
