# Paper Outline: Why Co-occurrence Clustering Cannot Detect Feature Absorption in Sparse Autoencoders

> Iteration: 3  
> Date: 2026-04-28  
> Author: sibyl-outline-writer  
> Status: Complete

---

## Title

**Why Co-occurrence Clustering Cannot Detect Feature Absorption in Sparse Autoencoders**

Alternative: *Token-Level Mutual Exclusivity Dooms Co-occurrence Clustering for Absorption Detection*

---

## Abstract (250 words)

- Feature absorption: parent features suppressed by child features, creating interpretability illusions
- Existing detection requires supervised ground-truth probes
- Unsupervised co-occurrence clustering (UAD) proposed as training-free alternative
- **Our finding**: UAD fails catastrophically (F1 = 0.00048, indistinguishable from random sampling)
- **Root cause**: absorption features are mutually exclusive at the token level
- **Positive result**: collision rate (top-K feature overlap) is a robust proxy (Spearman r = 0.87, n = 56, 95% CI [0.780, 0.938])
- Contributions: empirical falsification, root cause identification, validated proxy, constructive forward look

---

## 1. Introduction

### 1.1 Feature Absorption: A Critical SAE Failure Mode
- SAEs are the dominant tool for mechanistic interpretability
- Feature absorption: parent features suppressed by child features (Chanin et al., 2024)
- Creates dangerous interpretability illusions

### 1.2 The Detection Bottleneck
- All existing methods require supervised ground truth
- Supervised requirement means absorption only detectable for known concepts
- Unsupervised detection would unlock scale

### 1.3 The Co-occurrence Hypothesis and Our Central Question
- UAD proposes co-occurrence clustering as unsupervised solution
- Central question: does co-occurrence clustering actually detect absorption?
- Preview of answer: no, and we know exactly why

### 1.4 Contributions (4 bullets)
1. Empirical falsification: UAD F1 = 0.0005, identical to random baseline
2. Root cause: token-level mutual exclusivity of absorption features
3. Validation: collision rate proxy (r = 0.87, n = 56)
4. Constructive forward look: decoder weight similarity and causal intervention

**Transition**: After establishing the problem, we review background and related work.

---

## 2. Background and Related Work

### 2.1 Feature Absorption: Definition and Mechanism
- Chanin et al. (2024) definition: parent feature suppressed when child features co-occur
- Example: "animal" suppressed by "dog" in training data
- Creates interpretability illusions

### 2.2 Existing Detection Methods (All Supervised)
- Chanin et al. (2024): supervised probe directions + ablation
- Karvonen et al. (2025): similar supervised approach
- Common requirement: known parent feature + trained probe

### 2.3 Unsupervised Absorption Detection (UAD)
- Proposed method: co-occurrence clustering of SAE features
- Pipeline: dead feature filtering -> phi coefficient matrix -> hierarchical clustering -> pair extraction
- Claim: training-free, no labeled data needed

### 2.4 Collision Rate as Proxy Metric
- Prior work mentions collision rate (top-K feature overlap) as potential proxy
- Not systematically validated at scale

**Transition**: With background established, we describe our experimental methods.

---

## 3. Methods

### 3.1 Experimental Setup
- Model: GPT-2 Small (124M parameters)
- SAE: gpt2-small-res-jb (pretrained, SAELens)
- Layer: 8 (residual stream)
- SAE dictionary size: 24,576 features
- Dataset: OpenWebText (1,000 sequences, max 128 tokens)
- Seed: 42

### 3.2 Ground Truth Definition
- Number hierarchy: one, two, ..., eight (28 pairs)
- Punctuation hierarchy: ., ,, !, ?, ;, :, ", ' (28 pairs)
- Case hierarchy: a/A, b/B, ..., z/Z (26 pairs, control)
- True absorption rate: Jaccard overlap of top-K activating features per concept
- 7 true absorption pairs identified in number hierarchy

### 3.3 UAD Pipeline
1. Feature selection: 500 most active features (mean activation)
2. Co-occurrence matrix: phi coefficients between all pairs
3. Hierarchical clustering: Ward linkage, 50 clusters
4. Pair extraction: all pairs within clusters flagged as candidates
5. Dead feature filtering: remove near-zero variance features

### 3.4 Collision Rate Computation
- For each concept pair: find top-K activating SAE features
- Collision rate = Jaccard overlap = |shared| / |union|
- True absorption rate = same Jaccard metric on absorption feature sets

### 3.5 Ablations and Baselines
- Full UAD pipeline
- Without dead feature filtering
- Without phi coefficient filtering
- Without clustering (all pairs)
- Single linkage clustering
- K-means clustering
- Random baselines: global random, same-cluster random

**Transition**: We now present our results, starting with the failure of UAD.

---

## 4. Results

### 4.1 UAD Fails Catastrophically
- Detected pairs: 4,155
- True positives: 1 out of 7 ground truth pairs
- Precision: 0.024%
- Recall: 14.3%
- F1: 0.00048
- Same-cluster random F1: 0.00048 (identical)

### 4.2 Random Baseline Comparison
- UAD F1 = same-cluster random F1 (0.00048)
- All UAD complexity provides zero value over random sampling within clusters
- Global random F1: 0.00011 (even UAD's tiny advantage is meaningless)

### 4.3 Ablation Results
- All variants fail: full UAD, no dead filter, no phi (all F1 = 0.00048)
- No clustering: F1 = 0.000056 (even worse)
- Single linkage: F1 = 0.0
- K-means (best): F1 = 0.0037, Recall = 85.7% --- but still near-zero precision

### 4.4 Collision Rate Validation
- Spearman r = 0.869 (n = 56 pairs, numbers + punctuation)
- 95% CI: [0.780, 0.938]
- Pearson r = 0.815
- Holds across hierarchy types

### 4.5 Root Cause: Token-Level Mutual Exclusivity
- Absorption features fire on different tokens representing different child concepts
- Feature 11513: only on "three"
- Feature 24189: only on "four" through "eight"
- Complete mutual exclusivity at token level
- Phi coefficients near zero or negative

**Transition**: We now analyze why these results matter and what they imply.

---

## 5. Discussion

### 5.1 Why Co-occurrence Clustering Is the Wrong Tool
- Co-occurrence clustering finds features that fire TOGETHER
- Absorption features fire on mutually exclusive instances
- "Three" and "four" never appear at the same position
- UAD may work for synonyms or contextually related features, but not hierarchical absorption

### 5.2 Why Collision Rate Works
- Measures structural similarity of feature responses, not co-occurrence
- Two child concepts may share absorbing features in top-K without co-occurring
- Captures the right relationship: shared structure, not shared context

### 5.3 Theoretical Implications
- Absorption is not a co-occurrence phenomenon
- Decoder weight similarity may be the right signal (geometric relationship)
- Causal intervention is the gold standard for establishing absorption

### 5.4 Proposed Alternative Approaches
- Decoder weight similarity: cosine similarity of decoder vectors
- Causal intervention: activation patching / ablation
- Semantic similarity clustering: replace co-occurrence with decoder similarity

### 5.5 Limitations
- Single SAE (gpt2-small-res-jb layer 8)
- Small ground truth (7 true absorption pairs)
- Single model (GPT-2 Small)
- No causal validation of proposed alternatives
- Limited hierarchy types (numbers, punctuation)

**Transition**: We conclude with a summary and call to action.

---

## 6. Conclusion

### 6.1 Summary
- Co-occurrence clustering cannot detect feature absorption (F1 = 0.0005)
- Root cause: token-level mutual exclusivity
- Collision rate is a validated proxy (r = 0.87)
- Future work should test decoder weight similarity

### 6.2 Call to Action
- SAE community should abandon co-occurrence approaches for absorption detection
- Focus on structural similarity and causal methods
- Test decoder weight similarity as next step

---

## Figure & Table Plan

### Figure 1: UAD Pipeline Schematic (Section: Methods)
- **Purpose**: Illustrate the full UAD pipeline and where it fails
- **Type**: flow_chart
- **Content**: 5-step pipeline (feature selection -> co-occurrence matrix -> clustering -> pair extraction -> filtering) with annotation showing the mismatch: "Detects features that fire TOGETHER" vs "Absorption features fire on DIFFERENT tokens"
- **Key takeaway**: UAD's detection mechanism is fundamentally mismatched with the nature of absorption
- **Generation**: tikz or manual_diagram
- **Data source**: methodology description

### Figure 2: Token-Level Activation Heatmap (Section: Results / Root Cause)
- **Purpose**: Visualize token-level mutual exclusivity as the root cause
- **Type**: heatmap
- **Content**: 8 tokens (one through eight) x 4 absorption features (F11513, F12413, F22971, F24189), showing activation values. Features fire on disjoint token sets.
- **Key takeaway**: Absorption features are completely mutually exclusive at the token level
- **Generation**: code (matplotlib/seaborn)
- **Data source**: `exp/results/full/f5_false_positive_results.json` (token activation data)

### Figure 3: Collision Rate vs. True Absorption Rate Scatter Plot (Section: Results)
- **Purpose**: Show the strong correlation between collision rate and true absorption rate
- **Type**: scatter
- **Content**: 56 data points (numbers + punctuation pairs), x-axis = collision rate, y-axis = true absorption rate. Color by hierarchy type (numbers = blue, punctuation = orange). Show regression line. Annotate r = 0.87, 95% CI [0.780, 0.938].
- **Key takeaway**: Collision rate is a robust proxy for true absorption rate
- **Generation**: code (matplotlib/seaborn)
- **Data source**: `exp/results/full/f4_collision_correlation_results.json`

### Table 1: UAD Performance Across Variants and Baselines (Section: Results)
- **Purpose**: Summarize the complete failure of UAD and all ablations
- **Type**: ablation_table
- **Content**: Rows = variants (Full UAD, No dead filter, No phi, No clustering, Single linkage, K-means, Global random, Same-cluster random). Columns = Detected Pairs, Precision, Recall, F1, TP/FP/FN.
- **Key takeaway**: All variants fail; K-means achieves best recall but near-zero precision; UAD equals random
- **Generation**: data_table (LaTeX)
- **Data source**: `exp/results/full/f2_uad_ablations_results.json`

### Table 2: Collision Rate Validation Results (Section: Results)
- **Purpose**: Validate collision rate as proxy across hierarchy types
- **Type**: comparison_table
- **Content**: Rows = experiment (Pilot first letters, Full numbers+punctuation, Numbers only, Punctuation only). Columns = N pairs, Spearman r, Pearson r, 95% CI.
- **Key takeaway**: Proxy holds across hierarchy types and sample sizes
- **Generation**: data_table (LaTeX)
- **Data source**: `exp/results/full/f4_collision_correlation_results.json`

### Figure 4: Ablation Comparison Bar Chart (Section: Results)
- **Purpose**: Visual comparison of F1 and Recall across all UAD variants
- **Type**: bar_chart
- **Content**: Grouped bar chart showing F1 (left y-axis, log scale) and Recall (right y-axis) for each variant. K-means highlighted as best variant.
- **Key takeaway**: Even the best variant (K-means) has F1 < 0.004
- **Generation**: code (matplotlib/seaborn)
- **Data source**: `exp/results/full/f2_uad_ablations_results.json`

---

## Transition Logic Summary

| From Section | To Section | Transition Logic |
|-------------|-----------|-----------------|
| Abstract | Introduction | "We begin by establishing the problem..." |
| Introduction | Background | "With the problem defined, we review..." |
| Background | Methods | "Having established the background, we describe..." |
| Methods | Results | "We now present our results, starting with..." |
| Results | Discussion | "These results lead us to ask why..." |
| Discussion | Conclusion | "We conclude with a summary..." |

---

## Paper Metadata

- **Target venue**: NeurIPS / ICML (negative results track or main conference)
- **Paper type**: Negative result with constructive insight
- **Estimated length**: 8 pages (NeurIPS format)
- **Key figures**: 4
- **Key tables**: 2
- **Total visual elements**: 6 (exceeds minimum of 3)
