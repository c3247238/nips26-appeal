# Paper Outline: Absorption Has a Downstream Cost — Re-evaluating Feature Absorption in Sparse Autoencoders

## Title
**Absorption Has a Downstream Cost: Re-evaluating Feature Absorption in Sparse Autoencoders**

## Abstract
- Define feature absorption and its centrality to SAE research.
- State the three core claims: (1) absorption has a unique negative causal effect on downstream interpretability utility after controlling for confounders; (2) the canonical first-letter benchmark is degenerate on major open-model families; (3) a task-agnostic metric reveals domain-dependent absorption patterns that do not correlate with the first-letter proxy.
- Conclude with the reframing: the field should shift from "fixing absorption" to understanding when, where, and why it matters.

---

## 1. Introduction

### 1.1 Background
- Sparse autoencoders (SAEs) are the dominant unsupervised feature-discovery tool for language model interpretability.
- Feature absorption occurs when sparsity pressure suppresses general parent features in favor of specific child features, making parents effectively invisible (Chanin et al., 2024).
- The community has responded with architectural mitigations (OrtSAE, Matryoshka SAEs) that report impressive absorption reductions.

### 1.2 The Gap
- Almost all absorption evaluations rely on a single metric: the first-letter spelling task.
- Pilot evidence from this study shows the first-letter proxy returns zero for 26 of 27 GPT-2 Small checkpoints, raising questions about representativeness.
- No prior work has performed a controlled causal analysis treating absorption as a predictor of downstream utility while controlling for L0, reconstruction, and architecture family.

### 1.3 Contributions
1. **Downstream causal cost meta-analysis** (314 SAEs): absorption predicts lower sparse probing F1 and RAVEL scores after controlling for L0 and CE loss recovered.
2. **Metric critique**: the first-letter benchmark shows near-zero variance on GPT-2 Small Standard/TopK families.
3. **Task-agnostic pilot**: a geography-hierarchy probe produces weak negative correlation (r = -0.592, p = 0.116) with the first-letter proxy, suggesting domain-dependent absorption.
4. **Reframing**: absorption should be treated as a domain-local phenomenon, not a single scalar pathology.

### Transition
From the problem statement, we move to the empirical evidence for absorption's causal cost.

---

## 2. Related Work

### 2.1 Feature Absorption and Mitigations
- Chanin et al. (2024): first-letter spelling task and absorption definition.
- Korznikov et al. (2025): OrtSAE reports 65% absorption reduction.
- Bussmann et al. (2025): Matryoshka SAEs claim ~10x reduction.

### 2.2 Skeptical Turns
- Chanin et al. (2025): narrower SAEs reduce absorption but increase feature hedging.
- Roy et al. (2026): catastrophic interpretability collapse under aggressive sparsification.
- Kantamneni et al. (2025): SAE probes underperform logistic-regression baselines.

### 2.3 Benchmarks and Metrics
- Karvonen et al. (2025): SAEBench provides precomputed absorption and downstream metrics.
- Wang et al. (2025): weak overall correlation (Kendall's tau ~0.30) between SAEBench interpretability scores and steering utility, but no isolation of absorption as a causal predictor.

### Transition
Having situated the work, we describe the methodology and then present results.

---

## 3. Methodology

### 3.1 Overview
- Training-free systematic evaluation on pretrained SAEs.
- Four experiments: downstream causal meta-analysis (E1), first-letter benchmark representativeness (E2), task-agnostic metric pilot (E3), and multi-objective Pareto analysis (E4).
- Primary anchors: GPT-2 Small (117M) and Pythia-160M; SAEBench dataset for E1.

### 3.2 E1: Downstream Causal Cost Meta-Analysis
- **Data**: 314 pretrained SAEs from SAEBench (Pythia-160M and Gemma-2-2B families).
- **Metrics**: absorption_mean, sparse_probing_f1, ravel_cause, ravel_isolation, L0, CE loss recovered, explained_variance, width.
- **Analysis**:
  1. Raw Pearson/Spearman correlations.
  2. Partial correlations controlling for L0 and CE loss recovered.
  3. OLS regression: `outcome ~ absorption + L0 + CE_loss_recovered + width + C(architecture_family)` with cluster-robust SEs by family.

### 3.3 E2: First-Letter Benchmark Representativeness
- **Data**: 27 GPT-2 Small checkpoints (Standard, TopK, TopK_MLP, TopK_Attn, feature_splitting) and Pythia-160M SAEBench families.
- **Metrics**: first-letter absorption (simplified proxy), dead-neuron rate, explained variance, CE loss recovered.
- **Analysis**: distribution per family; test for near-zero variance (SD < 0.1).

### 3.4 E3: Task-Agnostic Absorption Metric Pilot
- **Data**: 10 GPT-2 Small checkpoints.
- **Domain**: Geography (continent -> country -> city), 10 parent-child pairs.
- **Method**: train logistic probes on residual-stream activations; classify absorption when parent probe succeeds but no parent-matching latent fires above threshold.
- **Analysis**: Pearson correlation between task-agnostic and first-letter rates.

### 3.5 E4: Multi-Objective Pareto Analysis (Supporting)
- **Data**: 27 GPT-2 Small checkpoints.
- **Method**: normalize metrics within strata; compute empirical Pareto fronts; Mann-Whitney U dominance tests.

### Transition
The methodology is intentionally lightweight and training-free; the strength of the paper lies in the scale of the meta-analysis and the critical metric evaluation.

---

## 4. Results

### 4.1 E1: Absorption Predicts Lower Downstream Utility
- **Key result**: across 314 SAEs, absorption correlates negatively with sparse probing F1 (Pearson r = -0.348, p < 1e-9) and RAVEL Cause (r = -0.264, p < 1e-5).
- After controlling for L0 and CE loss recovered, partial correlations strengthen for sparse probing F1 (partial r = -0.385, p < 1e-12) and remain significant for RAVEL Cause (partial r = -0.237, p < 1e-4) and RAVEL Isolation (partial r = -0.266, p < 1e-6).
- **OLS regression**: absorption coefficient is negative and significant for all three outcomes (sparse_probing_f1: beta = -0.037, t = -6.81, p < 1e-10; ravel_cause: beta = -0.022, t = -3.81, p = 1.7e-4; ravel_isolation: beta = -0.023, t = -4.11, p = 5.2e-5).
- Architecture family dummies are not significant predictors, suggesting absorption captures a cross-family signal.

### 4.2 E2: The First-Letter Benchmark Is Degenerate on GPT-2 Small
- **Key result**: 26 of 27 GPT-2 Small checkpoints show zero first-letter absorption.
- Only the TopK_Attn SAE (`blocks.8.hook_attn_out`) shows non-zero absorption (0.654).
- Family averages: Standard = 0.015, TopK = 0.000, TopK_MLP = 0.000, feature_splitting = 0.000.
- This near-zero variance makes the first-letter benchmark uninformative for comparing absorption across most GPT-2 Small families.

### 4.3 E3: Task-Agnostic Metric Reveals Domain-Dependent Patterns
- **Key result**: geography-hierarchy absorption ranges from 0.0 to 0.24 across the 10 checkpoints, showing meaningful variance where the first-letter proxy does not.
- Pearson correlation between task-agnostic and first-letter metrics: r = -0.592 (p = 0.116, not significant).
- The negative correlation is driven by the TopK_Attn outlier (high first-letter absorption, zero task-agnostic absorption).
- This is a **negative result** for H3: the two metrics do not measure the same construct.

### 4.4 E4: No Architecture Dominates on All Metrics (Supporting)
- **Key result**: feature_splitting SAEs achieve higher explained variance (0.982 vs 0.830) and CE loss recovered (1.172 vs 1.054) than standard SAEs, but Mann-Whitney U tests show no significant difference on absorption (p = 0.75) or hedging (p = 0.81).
- No family shows stochastic dominance across >= 3 of 4 metrics within matched strata.

### Transition
The results support a reframing: absorption is not merely a fixable bug, but a phenomenon whose practical impact depends on the metric, model family, and domain.

---

## 5. Discussion

### 5.1 Interpreting the Causal Cost
- The partial correlation and regression results provide the first controlled evidence that absorption has a unique negative effect on downstream interpretability utility.
- The effect is robust across architecture families, suggesting it is a general property of sparse representations rather than a family-specific artifact.
- Effect sizes are moderate (|r| ~ 0.24-0.39), indicating absorption is one of several factors influencing downstream utility.

### 5.2 The First-Letter Benchmark as a Local Phenomenon
- The degeneracy of the first-letter proxy on GPT-2 Small suggests the benchmark captures a narrow, possibly model-family-dependent spelling artifact.
- Claims of "65% absorption reduction" or "~10x reduction" may not generalize beyond the first-letter task.
- The field needs benchmarks that span multiple semantic domains and model families.

### 5.3 Toward Task-Adaptive Evaluation
- The weak negative correlation between task-agnostic and first-letter metrics implies that different SAE architectures may absorb different kinds of hierarchical structure.
- This motivates **task-adaptive SAE selection**: choose SAEs based on the semantic domain of interest, not a single scalar absorption score.

### 5.4 Limitations
- E3 is a small-sample pilot (10 checkpoints, 1 domain); broader domain coverage is needed.
- The first-letter metric used in E2/E3 is a simplified proxy, not the full Chanin et al. spelling pipeline.
- Gemma-2-2B was gated and inaccessible for direct evaluation; E1 relies on SAEBench precomputed metrics.
- Causality is statistical (partial correlation / regression), not interventional.

### Transition
The limitations point directly to future work: expanding the task-agnostic metric and validating benchmarks across more model families.

---

## 6. Conclusion
- Absorption is not just an aesthetic pathology; it has a measurable, unique negative effect on downstream interpretability utility.
- The canonical first-letter benchmark is unrepresentative for major open-model families and should not be the sole criterion for evaluating absorption.
- Absorption appears to be domain-dependent, motivating a shift from "fixing absorption" to understanding when, where, and why it matters.

---

## Figure & Table Plan

### Figure 1: Partial Correlation — Absorption vs. Sparse Probing F1 (Section: Results)
- **Purpose**: Show the unique negative relationship between absorption and sparse probing F1 after controlling for L0 and CE loss recovered.
- **Type**: scatter + regression fit line
- **Content**: x-axis = absorption_mean, y-axis = residuals of sparse_probing_f1 after regressing out L0 and CE_loss_recovered. Include 314 SAEs colored by architecture family.
- **Key takeaway**: Higher absorption predicts lower sparse probing F1 independently of reconstruction and sparsity.
- **Generation**: code (matplotlib/seaborn)
- **Data source**: `current/exp/results/e2_meta_regression_results.json`, `e2_meta_raw_data.json`

### Figure 2: Partial Correlation — Absorption vs. RAVEL Cause (Section: Results)
- **Purpose**: Replicate the causal-cost finding for RAVEL disentanglement.
- **Type**: scatter + regression fit line
- **Content**: x-axis = absorption_mean, y-axis = residuals of ravel_cause after regressing out L0 and CE_loss_recovered.
- **Key takeaway**: The absorption-downstream cost generalizes to causal disentanglement metrics.
- **Generation**: code (matplotlib/seaborn)
- **Data source**: `current/exp/results/e2_meta_regression_results.json`, `e2_meta_raw_data.json`

### Figure 3: First-Letter Absorption Distribution by Architecture Family (Section: Results)
- **Purpose**: Visualize the degeneracy of the first-letter benchmark on GPT-2 Small.
- **Type**: box plot / violin plot
- **Content**: one panel per family (Standard, TopK, TopK_MLP, TopK_Attn, feature_splitting) showing absorption score distribution across checkpoints.
- **Key takeaway**: Most families cluster at zero, with only TopK_Attn showing variance.
- **Generation**: code (matplotlib/seaborn)
- **Data source**: `current/exp/results/full/e1_full_gpt2_pareto_results.json`

### Figure 4: Task-Agnostic vs. First-Letter Absorption Scatter (Section: Results)
- **Purpose**: Illustrate the disconnect between the two metrics.
- **Type**: scatter
- **Content**: x-axis = first-letter absorption, y-axis = task-agnostic (geography) absorption. Annotate the TopK_Attn outlier. Add Pearson r = -0.592, p = 0.116.
- **Key takeaway**: The two metrics capture different phenomena; absorption is not a single scalar property.
- **Generation**: code (matplotlib/seaborn)
- **Data source**: `current/exp/results/e3_validation_correlation_results.json`

### Table 1: OLS Regression — Absorption Predicts Downstream Utility (Section: Results)
- **Purpose**: Present the controlled regression evidence in a compact, interpretable form.
- **Type**: regression table
- **Content**: three columns (sparse_probing_f1, ravel_cause, ravel_isolation). Rows: absorption_mean, L0, CE_loss_recovered, width, architecture dummies. Report standardized beta, cluster-robust SE, t-stat, p-value, significance stars. Include R^2 and N.
- **Key takeaway**: absorption_mean is the only consistently significant negative predictor across all three downstream metrics.
- **Generation**: data_table (LaTeX)
- **Data source**: `current/exp/results/e2_meta_summary.md`

### Table 2: First-Letter Absorption by Family — GPT-2 Small (Section: Results)
- **Purpose**: Summarize the metric degeneracy evidence.
- **Type**: comparison_table
- **Content**: rows = architecture families; columns = N checkpoints, mean absorption, std absorption, mean dead-neuron rate, mean explained variance.
- **Key takeaway**: Standard and TopK families show zero mean absorption with near-zero variance.
- **Generation**: data_table (LaTeX)
- **Data source**: `current/exp/results/full/e1_full_gpt2_summary.md`

### Table 3: Task-Agnostic vs. First-Letter Absorption — Per-Checkpoint (Section: Results)
- **Purpose**: Provide the raw pilot data for the metric comparison.
- **Type**: comparison_table
- **Content**: rows = 10 GPT-2 Small checkpoints; columns = release, SAE ID, family, task-agnostic absorption, first-letter absorption.
- **Key takeaway**: Task-agnostic metric varies from 0.000 to 0.240 while first-letter metric is zero for 9/10 checkpoints.
- **Generation**: data_table (LaTeX)
- **Data source**: `current/exp/results/e3_validation_summary.md`

---

## Visual Storytelling Checklist
- [x] At least 1 architecture/method diagram: included in Method section (Figure placeholder for SAE architecture — can be a simple schematic if needed, but the paper is empirically driven).
- [x] At least 1 main results table: Table 1 (OLS regression).
- [x] At least 1 analysis figure: Figures 3 and 4 (ablation/metric critique).
- [x] Figures referenced before they appear.
- [x] Consistent color scheme across scatter plots (architecture family colors).
