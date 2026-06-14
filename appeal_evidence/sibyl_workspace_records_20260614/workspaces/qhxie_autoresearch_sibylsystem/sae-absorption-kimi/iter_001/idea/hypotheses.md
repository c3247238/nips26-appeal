# Testable Hypotheses with Expected Outcomes

## H1: Downstream Causal Cost of Absorption (Front-Runner)

### Hypothesis
After controlling for L0 sparsity and reconstruction loss, higher absorption rates correlate negatively with downstream interpretability utility (sparse probing F1, RAVEL Cause/Isolation). This supports the claim that absorption is not merely an aesthetic pathology but has measurable practical harm.

### Expected Outcome
- **Partial correlation:** Negative partial correlation between absorption and sparse probing F1 (r_partial ≈ −0.25 to −0.45).
- **Partial correlation:** Negative partial correlation between absorption and RAVEL Cause/Isolation (r_partial ≈ −0.20 to −0.35).
- **Regression coefficient:** Absorption coefficient is negative and statistically significant (p < 0.05) in regressions controlling for L0, reconstruction, width, architecture family, and training step.
- **Architecture family dummies:** Not significant predictors of downstream performance after controlling for absorption and reconstruction, suggesting the realized tradeoff matters more than the architecture brand.

### Falsification Criterion
If |r_partial| < 0.15 for both sparse probing and RAVEL, or if the regression coefficient is not statistically significant (p > 0.05), H1 is falsified.

### Measurement Plan
1. Extract absorption, sparse probing F1, RAVEL Cause/Isolation, TPP, SCR, L0, and CE loss recovered for 200+ SAEBench SAEs.
2. Compute Pearson and Spearman correlations (raw).
3. Compute partial correlations controlling for L0 and reconstruction.
4. Run OLS regression with cluster-robust SEs (clustered by architecture family), including width and training step as covariates.

---

## H2: First-Letter Benchmark Representativeness

### Hypothesis
The first-letter absorption benchmark is unrepresentative for many SAE families: it will show near-zero variance on some families (e.g., GPT-2 Small Standard/TopK) despite those families exhibiting absorption-like failures on other hierarchical tasks.

### Expected Outcome
- **GPT-2 Small Standard/TopK:** First-letter absorption rate ≈ 0.0 for the majority of checkpoints (consistent with pilot evidence).
- **Pythia-160M:** First-letter absorption shows meaningful variance across architecture families (range 0.0–0.73, consistent with SAEBench reports).
- **Dead-neuron confound:** After regressing out dead-neuron rate, absorption rankings may shift, confirming Neuronpedia's finding that dead features artificially lower absorption.

### Falsification Criterion
If first-letter absorption shows meaningful variance (SD > 0.1) across all evaluated families including GPT-2 Small Standard/TopK, H2 is falsified.

### Measurement Plan
1. Run official `sae-spelling` or SAEBench absorption metric on 20–50 checkpoints across GPT-2 Small and Pythia-160M.
2. Compute mean, median, and standard deviation of absorption per architecture family.
3. Compute dead-neuron rate on >=50k tokens per checkpoint.
4. Report absorption rankings with and without dead-neuron adjustment.

---

## H3: Domain-Dependent Absorption (Task-Agnostic Metric)

### Hypothesis
A task-agnostic absorption metric, constructed by combining automated hierarchical concept discovery with causal ablation, will show domain-dependent absorption patterns that do not correlate strongly (r < 0.4) with the first-letter benchmark, suggesting absorption is better understood as a domain-local phenomenon than as a single scalar property.

### Expected Outcome
- **Domain-specific rates:** Geography, biology, and color hierarchies yield different absorption rates for the same SAE.
- **First-letter correlation:** Pearson r between task-agnostic metric (averaged across domains) and first-letter benchmark < 0.4.
- **Domain stability:** Coefficient of variation across domains > 0.2 for most SAEs.

### Falsification Criterion
If Pearson r >= 0.4 between the task-agnostic metric and the first-letter benchmark, H3 is falsified.

### Measurement Plan
1. Select 20–30 GPT-2 Small / Pythia-160M checkpoints.
2. For each of 3 domains (geography, biology, colors), define 10 parent-child pairs.
3. Train logistic regression probes on residual-stream activations.
4. Classify absorption when parent probe succeeds but no parent-matching latent fires.
5. Compute per-domain absorption rates and correlate with first-letter benchmark.

---

## H4: Multi-Objective Pareto Tradeoffs (Supporting)

### Hypothesis
Among the architecture families evaluable on open models, absorption-mitigation methods (Matryoshka, JumpReLU, BatchTopK) do not unambiguously dominate standard SAEs when absorption, reconstruction fidelity, dead-neuron rate, and downstream probing are considered jointly within matched width and hook-point strata.

### Expected Outcome
- **Matryoshka / JumpReLU:** Lower absorption than Standard/ReLU.
- **Standard / BatchTopK:** Better reconstruction fidelity (explained variance, CE loss recovered).
- **Pareto front:** Each family occupies a distinct region within matched strata. No single family dominates across all metrics.

### Falsification Criterion
If one architecture family shows statistically significant stochastic dominance (Mann-Whitney U test, p < 0.05, Bonferroni-corrected) across >= 3 out of 4 metrics within a matched stratum, H4 is falsified.

### Measurement Plan
1. Assemble Pythia-160M SAEBench SAEs (7 families).
2. Stratify by hook point and dictionary width.
3. Compute absorption, reconstruction, dead-neuron rate, and sparse probing F1 per checkpoint.
4. Normalize metrics to [0, 1] within stratum.
5. Compute Pareto fronts and test stochastic dominance.

---

## Hypothesis Priority

| Priority | Hypothesis | Role in Paper |
|----------|------------|---------------|
| 1 | H1 | Lead contribution: absorption's causal harm on downstream utility. |
| 2 | H2 | Secondary contribution: critique of first-letter benchmark representativeness. |
| 3 | H3 | Tertiary contribution: domain-dependent absorption pilot. |
| 4 | H4 | Supporting context: multi-objective tradeoffs among evaluable families. |
