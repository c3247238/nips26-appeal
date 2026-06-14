# Methodology: Absorption Has a Downstream Cost — Re-evaluating Feature Absorption in Sparse Autoencoders

## Overview

This study conducts a training-free, systematic evaluation of feature absorption in sparse autoencoders (SAEs). The work is strategically reframed from prior rounds: the lead contribution is a **downstream causal cost meta-analysis** (E1), supported by a **critique of the canonical first-letter absorption benchmark** (E2) and a **pilot task-agnostic absorption metric** (E3). A multi-objective Pareto analysis (E4) provides supporting context.

**Key revision from pilot evidence:** The simplified first-letter absorption proxy used in the prior E1 pilot proved degenerate on GPT-2 Small (zero for 26 of 27 checkpoints). All experiments in this round therefore use the **official `sae-spelling` benchmark or SAEBench precomputed absorption metrics**, not the simplified proxy. Gemma-2-2B remains gated and inaccessible; we commit fully to **GPT-2 Small and Pythia-160M** as open-model anchors.

---

## Hypotheses

- **H1 (Downstream Causality):** After controlling for L0 sparsity and reconstruction loss, higher absorption rates correlate negatively with downstream interpretability utility (sparse probing F1, RAVEL Cause/Isolation).
- **H2 (Metric Representativeness):** The first-letter absorption benchmark is unrepresentative for some SAE families, showing near-zero variance on families such as GPT-2 Small Standard/TopK.
- **H3 (Domain Dependence):** A task-agnostic absorption metric shows domain-dependent patterns that do not correlate strongly (r < 0.4) with the first-letter benchmark.
- **H4 (Pareto Tradeoffs, Supporting):** Among evaluable architecture families on open models, absorption-mitigation methods do not unambiguously dominate standard SAEs when absorption, reconstruction, dead-neuron rate, and downstream probing are considered jointly within matched width and hook-point strata.

---

## Experiment 1: Downstream Causal Cost Meta-Analysis (H1)

### Objective
Test whether absorption has a unique negative causal effect on downstream interpretability utility, independent of reconstruction quality and L0 sparsity.

### Data
**Primary anchor:** Pythia-160M SAEBench release (200+ pretrained SAEs across 7 architecture families). SAEBench provides precomputed metrics for absorption, sparse probing F1, RAVEL Cause/Isolation, TPP, SCR, L0, and CE loss recovered.

### Analysis Pipeline
1. Extract absorption, sparse probing F1, RAVEL Cause/Isolation, TPP, SCR, L0, CE loss recovered, dictionary width, architecture family, and training step for all available checkpoints.
2. Compute raw Pearson and Spearman correlations.
3. Compute partial correlations between absorption and each outcome, controlling for L0 and CE loss recovered.
4. Run OLS regression: `outcome ~ absorption + L0 + CE_loss_recovered + width + C(architecture_family)`
5. Use cluster-robust standard errors clustered by architecture family.
6. Report standardized beta coefficients, t-statistics, and p-values.

### Falsification Criterion
H1 is falsified if |r_partial| < 0.15 for both sparse probing and RAVEL, or if the absorption coefficient is not statistically significant (p > 0.05).

### Controls and Baselines
- Architecture family dummy variables control for family-specific mean differences.
- Dictionary width and training step are included as covariates.
- Cluster-robust SEs account for within-family correlation.

### Expected Visualizations
- **Figure 1:** Scatter plot of absorption vs. sparse probing F1 with partial-regression fit line.
- **Figure 2:** Scatter plot of absorption vs. RAVEL Cause/Isolation with partial-regression fit line.
- **Table 1:** Regression table with coefficients, robust SEs, t-statistics, and significance stars.

---

## Experiment 2: First-Letter Benchmark Representativeness (H2)

### Objective
Determine whether the canonical first-letter absorption benchmark is degenerate or unrepresentative for some SAE families.

### Models and Checkpoints
- **GPT-2 Small (117M):** 10-15 checkpoints across Standard, TopK, TopK_MLP, and TopK_Attn families.
- **Pythia-160M:** 15-20 checkpoints across Standard, TopK, BatchTopK, JumpReLU, Gated, P-anneal, and Matryoshka families from SAEBench.

### Metrics
1. **Absorption:** Official `sae-spelling` benchmark or SAEBench precomputed absorption metric (replacing the degenerate simplified proxy).
2. **Dead-neuron rate:** Fraction of latents with near-zero activation frequency, computed on >=50k tokens to ensure stability.
3. **Reconstruction fidelity:** L0 sparsity, explained variance, CE loss recovered (%).

### Analysis Pipeline
1. Load each checkpoint via `SAELens.SAE.from_pretrained()`.
2. Run a forward pass over 50,000 tokens of held-out text (C4 validation subset) to collect activations and compute L0, explained variance, and dead-neuron rate.
3. Run the official `sae-spelling` absorption metric pipeline. For SAEBench Pythia SAEs, extract precomputed absorption scores where available.
4. Report absorption rate distributions per architecture family (mean, median, SD).
5. Test whether any family shows near-zero variance (SD < 0.1).
6. Include dead-neuron rate as a covariate (Neuronpedia finding: dead features artificially lower absorption).

### Falsification Criterion
H2 is falsified if first-letter absorption shows meaningful variance (SD > 0.1) across all evaluated families including GPT-2 Small Standard/TopK.

### Expected Visualizations
- **Figure 3:** Box plots of absorption rate per architecture family for GPT-2 Small and Pythia-160M.
- **Figure 4:** Scatter plot of absorption vs. dead-neuron rate with regression fit.
- **Table 2:** Main results comparison (architecture family × metric) with mean ± std across checkpoints.

---

## Experiment 3: Task-Agnostic Absorption Metric Pilot (H3)

### Objective
Pilot a task-agnostic absorption metric based on automated hierarchy discovery and causal ablation, testing whether absorption is domain-dependent.

**Scope note:** H3 was refuted in prior analysis (task-agnostic metric showed weak negative correlation r = -0.592 with the first-letter benchmark). This experiment is therefore scoped as a **pilot-only negative-result demonstration** rather than a full validation study.

### Model and Domain
- **Model:** Pythia-160M or GPT-2 Small (1 checkpoint, layer 8).
- **Domain:** Geography (continent -> country -> city).
- **Pairs:** 10 parent-child concept pairs.

### Method
1. Define 10 parent-child concept pairs for the geography domain.
2. Collect 100 diverse sentences from C4 containing these concepts.
3. Train logistic regression probes on residual-stream activations for each concept.
4. For probe-true tokens, identify top-k SAE latents.
5. Classify absorption when the parent probe succeeds but no parent-matching latent fires above threshold.
6. Compute domain-specific absorption rate.

### Falsification Criterion
H3 is falsified if Pearson r >= 0.4 between the task-agnostic metric and the first-letter benchmark. Given the pilot scope, we report the pilot rate as a negative-result data point.

### Expected Visualizations
- **Table 3:** Geography-domain absorption rate compared to first-letter benchmark for the same SAE.

---

## Experiment 4: Multi-Objective Pareto Tradeoffs (H4, Supporting)

### Objective
Provide supporting context on whether absorption-mitigation architectures dominate standard SAEs across multiple quality metrics within matched configurations.

### Data
**Pythia-160M SAEBench SAEs:** 7 architecture families (ReLU, TopK, BatchTopK, JumpReLU, Gated, P-anneal, Matryoshka BatchTopK).

### Analysis Pipeline
1. Compute absorption (official metric), reconstruction (L0, explained variance), dead-neuron rate, and sparse probing F1 per checkpoint.
2. Stratify strictly by hook point and dictionary width — do not mix mismatched configurations.
3. Normalize metrics to [0, 1] within each stratum.
4. Compute empirical Pareto fronts per architecture family within each stratum.
5. Test stochastic dominance using pairwise Mann-Whitney U tests (Bonferroni-corrected).

### Falsification Criterion
H4 is falsified if one architecture family shows statistically significant stochastic dominance (Mann-Whitney U test, p < 0.05, Bonferroni-corrected) across >= 3 out of 4 metrics within a matched stratum.

### Expected Visualizations
- **Figure 5:** Pareto front scatter plots (absorption vs. reconstruction, absorption vs. dead-neuron rate) colored by architecture family.
- **Table 4:** Pairwise stochastic dominance test results.

---

## GPU Resource Planning

| Experiment | GPU Count | Strategy | Estimated Time |
|------------|-----------|----------|----------------|
| E1 — Data validation pilot | 0 (CPU) | Single process | 5 min |
| E1 — Full meta-analysis | 0 (CPU) | Single process | 10 min |
| E2 — Metric pipeline pilot (5 checkpoints) | 1 | Single GPU | 15 min |
| E2 — Full GPT-2 (10-15 checkpoints) | 1 | Single GPU | 20 min |
| E2 — Full Pythia (15-20 checkpoints) | 1 | Single GPU | 30 min |
| E3 — Task-agnostic pilot (1 SAE, 1 domain) | 1 | Single GPU | 15 min |
| E4 — Pareto pilot (2 families, 1 stratum) | 1 | Single GPU | 15 min |
| E4 — Pareto full (7 families, stratified) | 1 | Single GPU | 45 min |

All GPU tasks are designed to fit within the 1-hour budget. Multi-GPU is not required because checkpoint evaluations are independent and can be parallelized across tasks if multiple GPUs are available.

---

## Shared Resources

- **Datasets:** `allenai/c4` (via `datasets` library) for activation collection and dead-neuron estimation.
- **Checkpoints:**
  - `gpt2-small-res-jb` and related SAELens releases (Standard, TopK, TopK_MLP, TopK_Attn)
  - Pythia-160M SAEBench release (7 architecture families)
- **Metrics libraries:** `sae-lens`, `sae-spelling` (official absorption benchmark), SAEBench precomputed metrics

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| H1 effect attenuates on real SAEBench data | Downgrade H1 to exploratory; pivot to metric critique (H2) as lead contribution. |
| `sae-spelling` integration fails or is too slow | Fallback to SAEBench's built-in absorption evaluation for Pythia-160M. |
| GPT-2 Small families continue to show near-zero absorption with official metric | This directly supports H2; report as a negative result about benchmark representativeness. |
| Task-agnostic pilot is too noisy | Scope remains pilot-only; report domain rate as a qualitative data point. |

---

## Reproducibility Checklist

- [ ] Fix random seed to 42 for all pilot experiments.
- [ ] Record exact `sae-lens`, `transformer-lens`, `torch`, and `sae-spelling` versions.
- [ ] Use identical hook points (`resid_pre` or `resid_post`) within each comparison.
- [ ] Compute dead-neuron rates on >=50,000 tokens (not the 2k tokens used in the prior pilot).
- [ ] Save raw metric outputs as JSON for each checkpoint.
- [ ] Document any checkpoint exclusions (failed loads, mismatched shapes, gated access).
