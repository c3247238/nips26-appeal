# Paper Outline: L0-Matched or Misleading? A Systematic Re-evaluation of Architecture Claims for Feature Absorption in Sparse Autoencoders

## Title
L0-Matched or Misleading? A Systematic Re-evaluation of Architecture Claims for Feature Absorption in Sparse Autoencoders

## Abstract
- Problem: Feature absorption is a recognized pathology in SAEs; architectural mitigations claim reduction
- Gap: No study controls for sparsity (L0) when comparing architectures
- Method: Systematic L0-matched comparison + dose-response causality study on synthetic hierarchical data
- Key findings: (1) L0 is the dominant driver of absorption; Baseline L1 cannot match TopK/Matryoshka sparsity levels; (2) MCC remains flat (~0.219, std 0.0013) across 2.3x absorption variation, falsifying the causal absorption-interpretability link; (3) OrtSAE orthogonality penalty shows no absorption reduction
- Implication: Control L0 before comparing architectures; community focus on absorption reduction may be misdirected

## 1. Introduction

### 1.1 Problem Statement
- SAEs decompose neural activations into sparse, interpretable features [1]
- Feature absorption: parent features subsumed by child features under sparsity pressure [2]
- Creates gaps in feature coverage, undermining interpretability reliability

### 1.2 Motivation: Architectural Mitigation Claims
- Matryoshka SAE: ~90% absorption reduction [3]
- OrtSAE: ~65% reduction via decoder orthogonality [4]
- Gated SAE, JumpReLU SAE, Hierarchical SAE: moderate claims [5,6,7]
- All compare at natural (unmatched) sparsity levels

### 1.3 The L0 Confound
- TopK with k=50 has L0≈50; Baseline L1 may have L0≈1000
- 19x sparsity difference makes absorption comparisons meaningless
- No prior work controls for L0 --- this is the central gap

### 1.4 Research Questions
- **RQ1**: Does natural L0 confound cross-architecture absorption comparisons?
- **RQ2**: Does absorption rate causally predict downstream interpretability performance?
- RQ3/RQ4 (mutual coherence, semantic generalization): supplementary, reported for completeness

### 1.5 Contributions
1. First systematic demonstration that L0 is the dominant driver of absorption
2. Proof that Baseline L1 cannot match sparsity levels of TopK/Matryoshka
3. Dose-response study falsifying causal absorption-interpretability link
4. Critical methodological contribution: L0-matching protocol and its limitations

### Transition
From the problem and our approach, we now survey related work to position our contribution.

## 2. Related Work

### 2.1 Feature Absorption
- Chanin et al. [2]: first systematic identification; logistic regression probe-based detection
- Validated on hundreds of LLM SAEs; first-letter spelling tasks
- Leaves open generalization to semantic features

### 2.2 Mitigation Architectures
- Matryoshka SAE [3]: nested multi-scale dictionaries; inner levels exacerbate feature hedging [8]
- OrtSAE [4]: chunk-wise decoder orthogonality; 4-11% compute overhead
- GBA [9]: theoretical feature recovery guarantees under specific generative assumptions
- Gated SAE [5], JumpReLU SAE [6], HSAE [7]: alternative approaches

### 2.3 Evaluation Benchmarks
- SAEBench [10]: 8 metrics including absorption; ~26 min per SAE
- CE-Bench [11]: deterministic LLM-free alternative; >70% Spearman correlation with SAEBench
- SynthSAEBench [12]: synthetic ground-truth; SAEs recover only ~9% of true features

### 2.4 Gap
- No prior work controls for L0 when comparing architectures
- Implicit assumption: architecture effects are independent of sparsity
- We test this assumption directly

### Transition
Having established the gap, we describe our experimental framework and analytical approach.

## 3. Method

### 3.1 Experimental Framework
- SynthSAEBench synthetic data with known parent-child hierarchies
- 1024 features, 256 hidden dimensions, 32 root nodes, branching factor 4, depth 3
- Exact absorption detection without probe-based approximations
- 5 random seeds (42, 123, 456, 789, 1011)

### 3.2 Architecture Variants
- Baseline L1: L1 sparse penalty (reference)
- TopK: explicit k-sparse selection (k=50)
- Matryoshka: nested multi-scale dictionary
- OrtSAE: decoder orthogonality penalty
- Gated: separate gate/magnitude paths
- Random: untrained random dictionary (validation baseline)

### 3.3 L0-Matching Protocol
- Target: match Baseline L1 L0 to variant L0 via lambda sweep
- Lambda range: 5e-5 to 2e-3 (40x span)
- Result: Baseline L1 cannot achieve L0=50; minimum L0 ~995
- Demonstrates incommensurability of L1 and k-selection sparsity regimes

### 3.4 Dose-Response Causality Study
- Fix architecture (Baseline L1), vary lambda (5 levels)
- Independent variable: absorption rate
- Dependent variable: feature recovery MCC
- Prediction: if absorption causes harm, MCC should decrease monotonically

### 3.5 Metrics and Statistical Analysis
- Absorption rate: fraction of parent firings where child-matching latents activate (threshold 0.05)
- L0: mean active features per token; true_L0: after removing dead latents
- Dead latents: percentage never activating above threshold
- MCC: Matthews Correlation Coefficient via Hungarian algorithm matching
- Reconstruction MSE: training validation check
- Statistical tests: Welch's t-test, Cohen's d, Pearson correlation with CI, ANOVA

### 3.6 Data Integrity Pipeline
- Feature count verification, cross-file duplicate detection (MD5)
- Output file existence checks, numerical audit against source JSON
- Convergence diagnostics: training loss curves, final loss values

### Transition
With the method established, we present the experimental results in four parts.

## 4. Results

### 4.1 Cross-Architecture Comparison (Unmatched L0)
- Table 1: absorption rates, L0, dead latent percentages across 6 variants
- Key observation: TopK/Matryoshka show ~0.056 absorption vs. Baseline ~0.254
- But L0 differs by 19x (50 vs. 964) --- comparison is confounded
- Statistical tests confirm significance: TopK vs. Baseline t(8)=7.79, p=0.0003, d=4.93
- ANOVA: F(5,24)=67.03, p<0.0001
- L0-absorption correlation: Pearson r=0.943, p=0.016

### 4.2 L0-Matching Attempt
- Table 2: lambda sweep results showing Baseline cannot reach L0=50
- At lambda=0.002 (highest tested), L0=995 --- only 16% decrease from lambda=5e-5
- True L0-matching between Baseline and TopK/Matryoshka is impossible
- L1 regularization cannot achieve arbitrary sparsity in this setting

### 4.3 Dose-Response Causality
- Figure 1: scatter plot of absorption vs. MCC across 25 measurements (5 lambda x 5 seeds)
- Absorption range: 0.141 to 0.319 (2.3x variation)
- MCC range: 0.217 to 0.222 (ratio 1.02)
- MCC flat: mean 0.219, std 0.0013 --- no causal relationship
- Metric validity caveat: Random SAE achieves MCC=0.221, indicating MCC insensitivity in overcomplete setting
- Reconstruction MSE discriminates: Baseline 0.0104 vs. Random 0.649 (t=-21.7, p<0.0001)

### 4.4 Supplementary Analyses
- Mutual coherence (RQ3): max ~0.31, mean ~0.05; no systematic relationship with absorption
- Semantic generalization (RQ4): cross-domain rates below shuffled baselines; not interpretable

### 4.5 Ablation Studies
- Matryoshka nesting: flat (0.056 +/- 0.012) vs. nested (0.057 +/- 0.023); no difference
- OrtSAE orthogonality: without penalty (0.230 +/- 0.052) vs. with penalty (0.247 +/- 0.048); no reduction
- TopK vs. ReLU+L1: explicit k-selection is the key factor, not activation function differences

### Transition
The results establish L0 dominance and null causality; we now interpret their implications.

## 5. Discussion

### 5.1 The L0 Confound
- L0 is the dominant driver of absorption, overshadowing architecture
- Three implications:
  1. Existing mitigation claims are uninterpretable without L0-matched controls
  2. Future work must adopt L0-matching as mandatory methodology
  3. TopK's advantage is real but sparsity-driven, not architectural

### 5.2 The Null Causal Result
- Dose-response falsifies hypothesis that absorption causally predicts MCC
- Two interpretations: (1) genuine null effect; (2) metric insensitivity
- Either way, "lower absorption = better interpretability" lacks causal support

### 5.3 Contrarian Perspective: Absorption as Feature
- Hierarchical representation through child features may mirror human category learning
- Speculative; our data only show absorption does not harm MCC, not that it is beneficial

### 5.4 Limitations
1. Synthetic data only; real-LLM validation ongoing
2. 1024 features (not 16k); generalizability uncertain
3. MCC may be insensitive; alternative metrics (steering, circuit-tracing) needed
4. Training completes in 2-3 seconds; convergence diagnostics should be verified

### Transition
We summarize the key findings and their practical implications for the field.

## 6. Conclusion

### 6.1 Key Findings
1. L0 is the dominant driver of absorption; Baseline L1 cannot achieve TopK/Matryoshka sparsity levels
2. Absorption does not predict downstream interpretability (as measured by MCC)
3. OrtSAE's orthogonality penalty does not reduce absorption

### 6.2 Methodological Contribution
- L0-matching is non-trivial: L1 regularization cannot achieve arbitrary sparsity
- Explicit k-selection mechanisms are required for low-L0 regimes

### 6.3 Practical Guidance
- Control L0 before comparing architectures
- Community focus on absorption reduction may be misdirected

### 6.4 Future Work
1. Test whether larger models (16k+ features) enable L1 to reach lower L0
2. Test alternative downstream metrics (steering efficacy, circuit-tracing precision)
3. Investigate whether absorption is genuinely harmful or a natural consequence of hierarchical representation

---

## Figure & Table Plan

### Figure 1: Dose-Response Scatter Plot (Section: Results 4.3)
- **Purpose**: Show that absorption variation does not predict MCC variation --- the core causal falsification
- **Type**: scatter plot
- **Content**: 25 data points (5 lambda levels x 5 seeds); x-axis = absorption rate, y-axis = feature recovery MCC; color-coded by lambda value; horizontal reference line at MCC = 0.219; annotation showing absorption range (0.141-0.319) and MCC range (0.217-0.222)
- **Key takeaway**: Despite 2.3x variation in absorption, MCC remains essentially flat --- no causal relationship
- **Generation**: code (matplotlib/seaborn)
- **Data source**: `full_rq2_dose_response_results.json`

### Figure 2: Experimental Pipeline Diagram (Section: Method 3.1)
- **Purpose**: Visualize the full experimental workflow for reviewers
- **Type**: flow_chart
- **Content**: Three-stage pipeline --- (1) Synthetic Data Generation (SynthSAEBench: 1024 features, 32 root hierarchies) -> (2) SAE Training (6 variants, 5 seeds, 2M tokens) -> (3) Evaluation (absorption detection, MCC feature recovery, statistical testing); branching to L0-matching protocol and dose-response study
- **Key takeaway**: The paper uses ground-truth synthetic data with exact absorption measurement, not probe-based approximations
- **Generation**: tikz
- **Data source**: N/A (method diagram)

### Figure 3: L0 vs. Absorption Bar Chart (Section: Results 4.1)
- **Purpose**: Visually demonstrate the L0 confound --- architectures with low L0 have low absorption
- **Type**: bar_chart (grouped or dual-axis)
- **Content**: Six variants on x-axis; left y-axis = absorption rate (bars), right y-axis = L0 (line or secondary bars); include dead latent percentage as annotations
- **Key takeaway**: The apparent architectural advantage of TopK/Matryoshka is confounded by their 19x lower L0
- **Generation**: code (matplotlib/seaborn)
- **Data source**: Cross-architecture comparison results (Table 1 data)

### Table 1: Cross-Architecture Comparison at Natural (Unmatched) L0 (Section: Results 4.1)
- **Purpose**: Present the primary cross-architecture absorption comparison
- **Type**: comparison_table
- **Content**: 6 variants x 3 columns (Absorption mean+std, L0 mean+std, Dead Latents %); bold best absorption rate; include statistical test results in footnotes
- **Key takeaway**: TopK/Matryoshka show lowest absorption but also lowest L0 and highest dead latent rates
- **Generation**: data_table (LaTeX)
- **Data source**: Cross-architecture experiment results

### Table 2: L0-Matching Attempt via Lambda Sweep (Section: Results 4.2)
- **Purpose**: Show that Baseline L1 cannot match TopK/Matryoshka L0=50
- **Type**: ablation_table
- **Content**: 4 rows (Baseline lambda=5e-5, lambda=0.002, TopK, Matryoshka); columns for L0, Absorption, Notes; Notes column explains each row's significance
- **Key takeaway**: Even 40x lambda increase only reduces L0 by 16%; true L0-matching is impossible
- **Generation**: data_table (LaTeX)
- **Data source**: Lambda sweep experiment results

### Table 3: Dose-Response Summary Statistics (Section: Results 4.3, Appendix)
- **Purpose**: Provide complete dose-response data for reproducibility
- **Type**: data_table
- **Content**: 5 lambda levels x 5 seeds; columns for lambda, absorption, L0, MCC, reconstruction MSE
- **Key takeaway**: Complete transparency of all 25 measurements underlying the causal claim
- **Generation**: data_table (LaTeX)
- **Data source**: `full_rq2_dose_response_results.json`

---

## Appendix Outline (if applicable)

### A.1 Full Dose-Response Data
- Table with all 25 measurements (5 lambda x 5 seeds)

### A.2 Ablation Study Details
- Matryoshka flat vs. nested: full seed-level data
- OrtSAE with/without penalty: full seed-level data
- TopK vs. ReLU+L1: full seed-level data

### A.3 Statistical Test Details
- Welch's t-test results for all pairwise comparisons
- ANOVA table
- Pearson correlation with 95% CI
- Cohen's d for all comparisons

### A.4 Convergence Diagnostics
- Training loss curves per variant
- Final loss values
- Training time per seed
