# Paper Outline: Construct Validity of the SAEBench Feature Absorption Metric

## Title
**Does First-Letter Absorption Predict Semantic-Hierarchy Absorption? A Construct-Validity Study of the Dominant SAE Benchmark**

## Abstract (150 words)
- Feature absorption is a central pathology in sparse autoencoders (SAEs), measured by the SAEBench metric using first-letter classification tasks.
- We conduct the first construct-validity study, testing whether first-letter absorption scores generalize to matched-frequency semantic hierarchies from WordNet.
- Across 8 SAE architectures on Pythia-160M, the Pearson correlation between first-letter and semantic-hierarchy absorption is r = 0.463 (95% CI: [-0.389, 0.981]) — inconclusive.
- The metric fails hierarchy specificity: non-hierarchy correlated features show higher absorption than semantic hierarchies (t = -4.748, p = 0.003).
- A Random-SAE control yields semantic-hierarchy absorption of 0.175 (vs. 0.352 for Standard SAE), yet still shows substantial non-zero absorption, suggesting the metric on semantic tasks is not fully specific to learned structure.
- These findings reveal a methodological blind spot in a widely adopted benchmark and suggest the need for domain-specific absorption metrics.

---

## 1. Introduction

### 1.1 The Rise of Feature Absorption as a Central Pathology
- SAEs decompose neural activations into interpretable features but suffer from superposition and polysemanticity.
- Feature absorption (Chanin et al., 2024): a parent feature's information is lost when child features are active, because the SAE allocates representational capacity to children at the parent's expense.
- Analytical proof: sparsity loss incentivizes absorption for hierarchical features (parent-child relationships).
- SAEBench (Karvonen et al., 2025) standardized absorption measurement, making it one of eight canonical evaluations.

### 1.2 The First-Letter Benchmark and Its Limitations
- Current metric uses first-letter classification (e.g., "starts with S" vs. "short") with ground-truth logistic probes.
- Advantages: ground-truth labels, causal ablations tractable, clean hierarchy definition.
- Limitations: artificial task, narrow domain, unknown generalization to real semantic hierarchies.
- Chanin et al. (2024) explicitly noted: "finding examples of feature absorption unrelated to character identification" as future work.

### 1.3 Research Questions
- **RQ1:** Do first-letter absorption scores correlate with semantic-hierarchy absorption scores across diverse SAEs?
- **RQ2:** Is the metric specific to hierarchical features, or does it detect absorption-like behavior in non-hierarchical correlated features?
- **RQ3:** How robust is the correlation across feature-splitting thresholds and base models?

### 1.4 Contributions
- First construct-validity test of the dominant SAE absorption metric on real semantic hierarchies.
- Evidence that the metric lacks hierarchy specificity and is confounded by non-learned structure.
- Open-source replication materials (WordNet hierarchy dataset, evaluation code).
- Direct implications for benchmark design and architecture evaluation.

**Transition:** From the problem statement, we move to the precise measurement protocol.

---

## 2. Method

### 2.1 SAE Selection
- 8 architectures spanning the absorption-rate spectrum (Table 1).
- Source: SAELens releases for Pythia-160M, layer 8, resid_post.
- Random-SAE control: permuted decoder directions from Standard SAE.

### 2.2 First-Letter Absorption (Baseline)
- Official SAEBench evaluator: `sae_bench.evals.absorption.main.run_eval`.
- Config: Pythia-160M-deduped, random_seed=42, batch_size=256.
- Metric: `mean_full_absorption_score` (primary).

### 2.3 Semantic-Hierarchy Construction
- 10 WordNet parent-child hierarchies (Table 2).
- Selection criteria: direct hypernym, single tokens in model vocabulary, concrete concepts.
- Frequency matching: synthetic balanced datasets (N=100 sentences per concept) to control frequency confounds.

### 2.4 Absorption Measurement Protocol
- Ground-truth probe: logistic regression on base-model residual-stream activations (layer 8).
- SAE latent probe: same probe on SAE latents.
- K-sparse probe: probe on top-k SAE latents (k=10).
- Absorption score: `max(0, (resid_acc - sae_acc) / resid_acc, (resid_acc - k_sparse_acc) / resid_acc)`.
- Minimum probe AUROC threshold: 0.7 (all hierarchies achieved AUROC = 1.0).

### 2.5 Non-Hierarchy Control Condition
- 10 semantically correlated but non-hierarchical pairs (e.g., big-large, doctor-hospital).
- Same absorption formula applied to test hierarchy specificity.

### 2.6 Statistical Analysis
- H1: Pearson r with bootstrap 95% CI (B=10,000). Supported if r > 0.6 and CI excludes 0.
- H2: Paired t-test comparing semantic-hierarchy vs. non-hierarchy control.
- H3: Correlation computed at tau_fs in {0.01, 0.03, 0.05}.

**Transition:** With the protocol established, we present the empirical findings.

---

## 3. Results

### 3.1 Main Results: Architecture Comparison
- Table 1: Per-architecture scores across all three conditions.
- First-letter absorption ranges from 0.008 (GatedSAE) to 0.576 (TopK).
- Semantic-hierarchy absorption ranges from 0.064 (PAnneal) to 0.359 (BatchTopK).
- Random-SAE: first-letter 0.030, semantic 0.175, non-hierarchy 0.233.

### 3.2 H1: Construct Validity
- Pearson r = 0.463 (excluding Random SAE), 95% CI: [-0.389, 0.981].
- CI spans from negative to strongly positive — inconclusive.
- With Random SAE included: r = 0.497, CI: [-0.206, 0.958].
- **Assessment:** H1 is neither supported nor rejected; the evidence is too weak.

### 3.3 H2: Hierarchy Specificity
- Mean semantic-hierarchy absorption: 0.235.
- Mean non-hierarchy control absorption: 0.331.
- Paired t-test: t = -4.748, p = 0.0032.
- Non-hierarchy scores are *significantly higher* than hierarchy scores.
- **Assessment:** H2 is rejected. The metric is not hierarchy-specific.

### 3.4 H3: Robustness Across tau_fs
- Correlation stable across thresholds: r = 0.468 (tau_fs=0.01), 0.463 (0.03), 0.471 (0.05).
- All CIs remain wide and inconclusive.
- Hierarchy specificity rejection holds across all thresholds.
- **Assessment:** H3 inconclusive for construct validity; robust for hierarchy specificity failure.

### 3.5 Random-SAE Control
- Random SAE: first-letter 0.030 (near-zero, as expected), semantic-hierarchy 0.175, non-hierarchy 0.233.
- Standard SAE: semantic-hierarchy 0.352, non-hierarchy 0.416 — both higher than Random.
- **Implication:** The metric on semantic tasks partially reflects learned structure (Random < Standard), but the substantial non-zero Random-SAE scores indicate confounding by non-learned factors.

### 3.6 GPT-2 Replication
- Standard SAE: hierarchy 0.000, non-hierarchy 0.025.
- TopK SAE: hierarchy 0.003, non-hierarchy 0.098.
- Pattern differs from Pythia-160M; low absolute scores suggest model-specific behavior.

**Transition:** These results demand careful interpretation of what they mean for the field.

---

## 4. Discussion

### 4.1 Interpreting the Inconclusive Construct Validity
- The point estimate r = 0.463 suggests a moderate positive relationship.
- The wide CI reflects small sample size (n=7 SAEs) and high variance.
- We cannot conclude the metric generalizes, nor can we rule it out.
- **Key takeaway:** The current evidence base is insufficient for confident claims about construct validity.

### 4.2 The Hierarchy Specificity Failure
- Non-hierarchy correlated features show *higher* absorption than semantic hierarchies.
- This contradicts the theoretical motivation: absorption should be specific to hierarchical structure.
- Possible explanations:
  1. The synthetic sentence template introduces spurious correlations.
  2. Semantic relatedness (even non-hierarchical) activates overlapping feature sets.
  3. The k-sparse probing threshold is too coarse for fine-grained semantic distinctions.

### 4.3 The Random-SAE Finding
- Random SAE shows semantic-hierarchy absorption of 0.175 vs. Standard SAE's 0.352.
- The metric on semantic tasks partially reflects learned structure (Random < Standard), but the substantial non-zero Random scores indicate confounding.
- Contrast with first-letter: Random SAE scores near-zero (0.030), confirming the first-letter task does measure learned structure.
- **Implication:** The semantic-hierarchy adaptation of the metric is partially confounded by non-learned factors.

### 4.4 Implications for Benchmark Design
- The SAEBench absorption metric should not be extended to semantic hierarchies without substantial modification.
- Architecture comparisons using absorption as a criterion may be valid for first-letter tasks but not generalizable.
- Future work: domain-specific absorption metrics with validated hierarchy specificity.

### 4.5 Limitations
- Small SAE sample (n=7 trained + 1 random) limits statistical power.
- Single layer (layer 8) on a single model size (Pythia-160M).
- WordNet hierarchies are shallow; deeper hierarchies may behave differently.
- Synthetic sentence templates may not reflect natural language distribution.
- GPT-2 replication shows model-dependent effects; broader model sweep needed.

**Transition:** From implications, we conclude with concrete recommendations.

---

## 5. Conclusion

### 5.1 Summary
- First construct-validity study of the SAEBench absorption metric on semantic hierarchies.
- Construct validity: inconclusive (r = 0.463, wide CI).
- Hierarchy specificity: failed — non-hierarchy features show higher absorption.
- Random-SAE control: semantic-hierarchy metric is partially confounded by non-learned factors.

### 5.2 Recommendations
1. **For benchmark designers:** Do not extend first-letter absorption to semantic tasks without validation.
2. **For architecture researchers:** Absorption-reduction claims should be validated on multiple task types.
3. **For the community:** Invest in domain-specific absorption metrics with demonstrated hierarchy specificity.

### 5.3 Future Work
- Larger SAE cohorts (15-20 architectures) for adequate statistical power.
- Deeper WordNet hierarchies (3-4 levels) and multiple base models.
- Alternative semantic hierarchy sources beyond WordNet (e.g., concept nets, ontologies).
- Causal ablation studies to distinguish "truly missing" from "merely hidden" absorbed features.

---

## Acknowledgments

## References
- Chanin et al. (2024): Feature absorption theory and first-letter metric.
- Karvonen et al. (2025): SAEBench standardization.
- Bussmann et al. (2025): Matryoshka SAEs.
- Korznikov et al. (2025): OrtSAE.
- Zhan et al. (2026): Hierarchical SAEs.

## Appendix
- A.1: Per-hierarchy probe AUROC and absorption scores (Table 4).
- A.2: tau_fs robustness full results (Table 5).
- A.3: GPT-2 replication details.
- A.4: Code and data availability statement.

---

# Figure & Table Plan

## Figure 1: Architecture Ranking Comparison (Section: Results)
- **Purpose**: Show side-by-side first-letter vs. semantic-hierarchy absorption scores across all architectures.
- **Type**: grouped_bar_chart
- **Content**: 8 architectures (x-axis), two grouped bars per architecture (first-letter blue, semantic-hierarchy orange), y-axis = absorption score [0, 0.6].
- **Key takeaway**: The ranking correlation is moderate but not strong; some architectures invert (e.g., GatedSAE low first-letter but moderate semantic).
- **Generation**: code (matplotlib) — already generated as `fig1_architecture_ranking.png`
- **Data source**: `statistical_analysis_summary.json`, field `per_architecture_scores`
- **Caption**: "Comparison of first-letter (blue) and semantic-hierarchy (orange) absorption scores across 8 SAE architectures on Pythia-160M layer 8. The Random-SAE control (rightmost) shows lower semantic-hierarchy absorption than the Standard SAE, confirming partial sensitivity to learned structure, though non-zero Random scores indicate residual confounding."

## Figure 2: Construct Validity Scatter Plot (Section: Results)
- **Purpose**: Visualize the first-letter vs. semantic-hierarchy correlation with regression line and confidence interval.
- **Type**: scatter
- **Content**: Scatter points for 7 SAEs (excluding Random), x = first-letter absorption, y = semantic-hierarchy absorption. Dashed regression line. Bootstrap 95% CI band. Annotation: Pearson r = 0.463, CI = [-0.389, 0.981].
- **Key takeaway**: The wide confidence interval spans from negative to near-perfect correlation, making the result inconclusive.
- **Generation**: code (matplotlib) — already generated as `fig2_firstletter_vs_semantic_scatter.png`
- **Data source**: `statistical_analysis_summary.json`
- **Caption**: "Scatter plot of first-letter vs. semantic-hierarchy absorption scores across 7 trained SAE architectures. Pearson r = 0.463 with bootstrap 95% CI [-0.389, 0.981]. The wide interval reflects small sample size and high variance, yielding an inconclusive construct-validity test."

## Figure 3: Hierarchy Specificity Test (Section: Results)
- **Purpose**: Demonstrate that non-hierarchy correlated features show higher absorption than semantic hierarchies.
- **Type**: grouped_bar_chart
- **Content**: 8 architectures (x-axis), two grouped bars per architecture (semantic-hierarchy coral, non-hierarchy control green). Horizontal line at mean difference.
- **Key takeaway**: Non-hierarchy control consistently exceeds semantic-hierarchy absorption, rejecting hierarchy specificity.
- **Generation**: code (matplotlib) — already generated as `fig3_semantic_vs_nonhierarchy.png`
- **Data source**: `statistical_analysis_summary.json`
- **Caption**: "Hierarchy specificity test: semantic-hierarchy (coral) vs. non-hierarchy correlated-feature (green) absorption scores. Non-hierarchy scores are significantly higher (paired t-test: t = -4.748, p = 0.003), rejecting the hypothesis that the metric is specific to hierarchical structure."

## Figure 4: Robustness Across tau_fs (Section: Results)
- **Purpose**: Show correlation stability across feature-splitting thresholds.
- **Type**: line_plot with error bars
- **Content**: x-axis = tau_fs {0.01, 0.03, 0.05}, y-axis = Pearson r. Points with vertical error bars (95% CI). Horizontal dashed line at r = 0.6 (H1 threshold) and r = 0.
- **Key takeaway**: Correlation is numerically stable but all CIs are wide and inconclusive.
- **Generation**: code (matplotlib) — already generated as `fig4_tau_fs_robustness.png`
- **Data source**: `statistical_analysis_summary.json`, field `tau_fs_robustness`
- **Caption**: "Robustness analysis: Pearson correlation between first-letter and semantic-hierarchy absorption across three feature-splitting thresholds (tau_fs). The correlation is numerically stable (r = 0.468-0.471), but all confidence intervals are wide and span the H1 threshold of r = 0.6, yielding inconclusive evidence."

## Figure 5: GPT-2 Replication (Section: Results / Appendix)
- **Purpose**: Show model-specific differences in semantic-hierarchy absorption.
- **Type**: grouped_bar_chart
- **Content**: 2 architectures (Standard, TopK), two bars each (semantic-hierarchy, non-hierarchy control).
- **Key takeaway**: GPT-2 shows near-zero hierarchy absorption, contrasting with Pythia-160M results.
- **Generation**: code (matplotlib) — already generated as `fig5_gpt2_replication.png`
- **Data source**: `statistical_analysis_summary.json`, field `gpt2_replication`
- **Caption**: "GPT-2 small replication: semantic-hierarchy vs. non-hierarchy control absorption for Standard and TopK SAEs. Absolute scores are near-zero compared to Pythia-160M, indicating model-specific behavior in semantic-hierarchy absorption."

## Table 1: Main Results — Per-Architecture Absorption Scores (Section: Results)
- **Purpose**: Comprehensive numerical summary of all three conditions.
- **Type**: comparison_table
- **Content**: 8 rows (architectures) x 4 columns (First-Letter, Semantic-Hierarchy, Non-Hierarchy Control, Random-SAE). Bold best (lowest) score per column.
- **Key takeaway**: All numerical evidence in one table; Random-SAE anomaly is immediately visible.
- **Generation**: data_table (LaTeX)
- **Data source**: `statistical_analysis_summary.json`, field `per_architecture_scores`

## Table 2: WordNet Semantic Hierarchies (Section: Method)
- **Purpose**: Document the exact hierarchies used for reproducibility.
- **Type**: data_table
- **Content**: 10 rows, columns: Parent, Children, Probe AUROC (mean across SAEs).
- **Key takeaway**: All hierarchies achieved perfect probe AUROC (1.0), validating probe quality.
- **Generation**: data_table (LaTeX)
- **Data source**: `semantic_hierarchy_pythia_results.json`, field `hierarchy_probe_aurocs`

## Table 3: Non-Hierarchy Control Pairs (Section: Method)
- **Purpose**: Document the exact control pairs used.
- **Type**: data_table
- **Content**: 10 rows of word pairs with semantic relationship type.
- **Key takeaway**: Transparent reporting of control condition for reproducibility.
- **Generation**: manual (from methodology.md)

## Table 4: Per-Hierarchy Absorption Scores (Appendix)
- **Purpose**: Full transparency on per-hierarchy breakdown.
- **Type**: data_table
- **Content**: 10 hierarchies x 8 SAEs, cell = absorption score.
- **Key takeaway**: Detailed evidence for readers who want to verify aggregate results.
- **Generation**: data_table (LaTeX)
- **Data source**: `semantic_hierarchy_pythia_results.json`

## Table 5: tau_fs Robustness Full Results (Appendix)
- **Purpose**: Complete statistical results at all thresholds.
- **Type**: data_table
- **Content**: 3 rows (tau_fs values) x 5 columns (Pearson r, CI lower, CI upper, t-statistic, p-value).
- **Key takeaway**: All thresholds yield identical hierarchy-specificity rejection.
- **Generation**: data_table (LaTeX)
- **Data source**: `statistical_analysis_summary.json`, field `tau_fs_robustness`

---

# Transition Logic Summary

| Section | Leads To | Bridge |
|---------|----------|--------|
| Introduction | Method | "To test these questions, we need a precise measurement protocol." |
| Method | Results | "With the protocol established, we present the empirical findings." |
| Results | Discussion | "These results demand careful interpretation of what they mean for the field." |
| Discussion | Conclusion | "From these implications, we distill concrete recommendations." |
| Conclusion | References | "We conclude with a summary of contributions and directions for future work." |

# Paper Length Estimate
- Abstract: 150 words
- Introduction: 800-1000 words
- Method: 1000-1200 words
- Results: 800-1000 words
- Discussion: 1000-1200 words
- Conclusion: 300-400 words
- **Total body**: ~4000-5000 words (suitable for NeurIPS/ICML workshop or short paper)
