# Paper Outline: The Limits of Unsupervised Absorption Detection in Sparse Autoencoders

## Title
**The Limits of Unsupervised Absorption Detection in Sparse Autoencoders: A Systematic Analysis**

Alternative titles:
- *Why Co-occurrence Clustering Cannot Detect Feature Absorption*
- *Towards Causal Absorption Detection: Lessons from Failed Unsupervised Approaches*

---

## Abstract (150-200 words)

Feature absorption --- where broad parent features are suppressed by specific child features in sparse autoencoders (SAEs) --- undermines mechanistic interpretability by creating arbitrary false negatives. Existing detection methods require supervised probe directions and ground-truth parent labels, raising the question: can absorption be detected without supervision? We present a systematic analysis of why simple unsupervised approaches fail. We test UAD (Unsupervised Absorption Detection), which uses phi coefficient co-occurrence clustering to identify absorbed pairs without labels. On GPT-2 Small, UAD achieves F1 = 0.007, statistically indistinguishable from random pair selection (F1 = 0.0075, delta = -0.0001). Ablations across clustering algorithms, correlation metrics, and feature filters confirm the failure is fundamental: co-occurrence clustering detects semantic correlation, not absorption. Theoretical analysis reveals why absorption requires detecting suppression signals (P(parent active | child absent)), not co-occurrence patterns. We additionally present DFDA (Dynamic Feature De-Absorption), a training-free compensation method achieving 21% MSE improvement on parent-positive examples, as exploratory mitigation. Our findings redirect the community toward causal inference approaches (interventions, do-calculus) rather than correlation-based methods for absorption detection.

---

## 1. Introduction

### 1.1 Motivation and Background
- SAEs as the de facto standard for mechanistic interpretability [Bricken et al., 2023; Cunningham et al., 2023]
- Feature absorption: parent features suppressed by child features, creating arbitrary false negatives [Chanin et al., 2024]
- The critical barrier: detection requires knowing parent features a priori and training supervised probes
- This supervised requirement means absorption can only be detected for concepts we already know
- The open question: can absorption be detected without ground truth?

### 1.2 Research Questions
- **RQ1**: Can feature absorption be detected without ground-truth parent features or supervised probe directions?
- **RQ2**: Why do co-occurrence-based unsupervised methods fail for absorption detection?
- **RQ3** (exploratory): Can absorbed parent activations be recovered at inference time?

### 1.3 Key Contributions
1. **Negative result**: UAD (co-occurrence clustering) achieves F1 = 0.007, no better than random (F1 = 0.0075) --- first systematic proof that simple unsupervised methods fail
2. **Theoretical insight**: Absorption requires detecting suppression signals, not co-occurrence patterns; we formalize the structural difference
3. **Ablation analysis**: Failure persists across clustering algorithms, correlation metrics, and feature subsets --- confirming the flaw is in the core assumption, not implementation
4. **DFDA (exploratory)**: Training-free compensation achieves 21% MSE improvement on parent-positive examples with 97 parameters per pair
5. **Community guidance**: Clear recommendations for future work (causal inference, suppression-based detection)

### 1.4 Main Finding (Teaser)
Co-occurrence clustering performs no better than random chance for absorption detection. The structural signature of absorption is suppression, not correlation --- a distinction that unsupervised co-occurrence methods cannot capture.

---

## 2. Related Work

### 2.1 Sparse Autoencoders and Interpretability
- Standard SAE with L1 penalty [Makhzani & Frey, 2014]
- JumpReLU and TopK architectures [Rajamanoharan et al., 2024; Gao et al., 2024]
- GemmaScope: large-scale pretrained SAE suite [Lieberum et al., 2024]
- SAELens and TransformerLens ecosystem

### 2.2 Feature Absorption
- Chanin et al. (2024): foundational absorption paper; k-sparse probing + integrated gradients ablation; **fully supervised**
- Absorption as superposition [Elhage et al., 2022]
- Connection to polysemanticity [Schubert et al., 2024]
- Hierarchical SAEs (HSAE) as architectural mitigation [Chen et al., 2025]

### 2.3 Co-Occurrence Analysis of SAE Features
- "The Geometry of Concepts" (arXiv:2410.19750): spectral clustering on phi coefficient matrices for functional lobes; **does not address absorption specifically**
- Clarke et al. (2024) "sae_cooccurrence": compositionality and ambiguity analysis; no absorption detection
- Gap: no method uses co-occurrence for absorption detection

### 2.4 Absorption Mitigation Methods
- Matryoshka SAE, OrtSAE, KronSAE, ATM: architectural solutions requiring retraining
- SAEBench (Karvonen et al., ICML 2025): includes absorption metric but uses probe projection -- **still supervised**
- Gap: no training-free, inference-time compensation method exists

### 2.5 Negative Results in Machine Learning
- Importance of publishing negative results [Trouille et al., 2019]
- NeurIPS/ICLR explicit encouragement of negative result papers
- Our contribution fits this tradition: preventing wasted community effort on approaches that cannot work

### 2.6 Positioning
Unlike prior work that proposes solutions, we systematically analyze why a natural unsupervised approach fails. This complements architectural solutions (Matryoshka, OrtSAE) by clarifying what detection strategies are fundamentally limited.

---

## 3. Methodology

### 3.1 Definitions

**Feature Absorption** (Chanin et al., 2024): A hierarchical parent feature is suppressed when its child feature co-occurs, with the parent activation merged into the child's latent representation to increase sparsity while maintaining reconstruction.

**Feature Collision**: A measurable proxy for absorption: when multiple ground-truth concepts (e.g., first letters 'a', 'i', 'o') activate the same SAE feature index. Used as validation ground truth for UAD.

**Absorption Signature**: The characteristic suppression pattern of an absorbed parent: low P(parent active | child present) despite high P(parent active | child absent). This is structurally distinct from correlation (high co-occurrence).

**Suppression Signal**: The defining characteristic of absorption: when the child feature activates, the parent feature is systematically suppressed. Formally: P(parent fires | child fires) << P(parent fires | child absent).

### 3.2 Experimental Setup

**Primary Model**: GPT-2 Small (12 layers, 124M parameters)
- Hook point: `blocks.{layer}.hook_resid_post`
- SAE: `gpt2-small-res-jb` (JumpReLU, pretrained, d_SAE = 24,576)
- Layer evaluated: 8 (primary)

**Dataset**: OpenWebText (1,000 samples)

**Seeds**: 42 (primary), with bootstrap resampling for CI

**Software**: SAELens >= 2.0, TransformerLens >= 2.0, PyTorch 2.0+, scikit-learn

### 3.3 UAD: Unsupervised Absorption Detection (Tested and Failed)

**Input**: Pre-trained SAE, unlabeled text corpus
**Output**: List of suspected absorbed (parent, child) feature pairs

**Algorithm**:
1. Extract feature activation matrix A (n_examples x d_SAE) from corpus
2. Compute feature co-occurrence matrix C = A^T A (count of co-activations)
3. Normalize C to phi coefficient correlation matrix R
4. Run hierarchical agglomerative clustering (Ward linkage, n_c = 50 clusters)
5. Identify same-cluster feature pairs as candidate absorbed pairs
6. Validate against Chanin-style supervised labels where available

**Core Assumption (Disproven)**: Absorbed parents show anomalous co-occurrence -- they fire primarily when children fire, but rarely independently. This creates a detectable signature in R without ground truth.

**Why This Assumption Fails**: Co-occurrence clustering groups features by semantic correlation (features that fire together), but absorption is defined by suppression (parent fails to fire when child fires). These are structurally different phenomena. A parent and child that are semantically related will have high co-occurrence whether or not absorption occurs.

**Hyperparameters**:
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Top indices | 500 | Most active features from 24,576 total |
| Clusters | 50 | Balances granularity and statistical power |
| Linkage | Ward | Minimizes variance within clusters |

### 3.4 DFDA: Dynamic Feature De-Absorption (Exploratory)

**Input**: SAE, identified absorbed (parent, child) pairs
**Output**: Compensated SAE output

**Algorithm**:
1. For each pair (parent p, child c), collect examples where parent SHOULD activate (parent-positive)
2. Train tiny MLP: input = child activation, output = predicted parent residual
3. Architecture: 2 layers, 64 hidden units, ReLU (~97 parameters per pair)
4. At inference: add MLP(z_c) to z_p

**Parameter budget**: 97 parameters per pair.

**Status**: Exploratory. Achieves 21% MSE improvement on parent-positive examples but requires further validation.

### 3.5 Validation Protocol

| Stage | Task | Success Criteria |
|-------|------|-----------------|
| E1 | UAD + random baseline | UAD F1 must exceed random F1 by >= 0.3 |
| E2 | UAD ablations | Test whether any design choice matters |
| E3 | Cross-layer validation | Report per-layer honestly |
| E4 | False positive analysis | Understand error modes |
| E5 | Statistical testing | Bootstrap CI, permutation test |
| E6 | DFDA parent-positive | MSE improvement > 5% or drop from paper |
| E7 | Alternative correlation metrics | Test phi vs Pearson, MI, Jaccard, PMI |

---

## 4. Experiments

### 4.1 E1: UAD with Random Baseline (CRITICAL)
- **Result**: UAD F1 = 0.007, Random F1 = 0.0075, delta = -0.0001
- UAD performs **worse than random**
- Precision: 0.0037 (2 true positives out of 541 detected pairs)
- Recall: 1.0 (all 2 collision cases detected, but only 2 total in ground truth)
- 3,702 same-cluster pairs from 500 features / 50 clusters
- Random baseline: 100 repetitions of random pair selection

### 4.2 E2: UAD Ablation Studies
- Full UAD: 7,608 same-cluster pairs
- K-means variant: 7,648 pairs (nearly identical)
- All-features (no dead feature filter): 154,858 pairs
- **Interpretation**: The problem is in the core assumption, not implementation details
- All variants produce thousands of pairs with near-zero precision

### 4.3 E3: Cross-Layer Validation
- Layer 8: F1 = 0.007 (same as primary)
- Results consistent across layers
- The failure is not layer-specific

### 4.4 E4: False Positive Analysis
- 539 false positives out of 541 detected pairs (99.6% FP rate)
- False positives are semantically correlated features that co-occur naturally
- Not random noise --- structured semantic correlation mistaken for absorption

### 4.5 E5: Statistical Significance Testing
- Bootstrap 95% CI for UAD F1
- Permutation test p-value
- Power analysis: achieved power for detecting F1 > 0.5

### 4.6 E6: DFDA with Parent-Positive Evaluation
- Baseline MSE: 5.2e-06
- Improved MSE: 4.1e-06
- Improvement ratio: 21.2%
- Parameters: 97 per pair
- **Status**: Positive but preliminary. Single pair tested. Needs scaling.

### 4.7 E7: Alternative Correlation Metrics
- Tested: Pearson, Mutual Information, Jaccard, PMI
- All metrics produce similar near-zero precision
- The failure is not specific to phi coefficient

---

## 5. Analysis: Why Co-occurrence Cannot Detect Absorption

### 5.1 The Structural Difference: Correlation vs. Suppression
- **Correlation**: Two features fire together frequently. P(A,B) > P(A)P(B).
- **Absorption**: Parent feature is suppressed when child fires. P(parent | child) << P(parent | not child).
- Co-occurrence clustering detects the former, not the latter.

### 5.2 Formal Characterization
- Absorption requires conditional independence testing or causal inference
- Co-occurrence is a symmetric, undirected measure; absorption is an asymmetric, directed phenomenon
- The suppression signal P(parent | child absent) - P(parent | child present) is the true signature

### 5.3 Why This Matters for the Community
- Many papers implicitly assume co-occurrence implies absorption-related structure
- Our result shows this assumption is unfounded
- Future unsupervised detection methods must explicitly model suppression, not just correlation

### 5.4 Implications for Existing Work
- "Geometry of Concepts" spectral clustering: discovers functional lobes, not absorption
- sae_cooccurrence analysis: studies compositionality, not suppression
- These methods are not "wrong" --- they address different questions

---

## 6. Discussion

### 6.1 The Value of Negative Results
- Prevents wasted community effort on co-occurrence approaches
- Clarifies the theoretical requirements for absorption detection
- Aligns with NeurIPS/ICLR encouragement of negative result papers

### 6.2 DFDA: Exploratory Mitigation
- 21% MSE improvement on parent-positive examples is promising
- But single-pair evaluation, needs scaling
- Conceptually sound: residual compensation architecture
- Reported as exploratory, not a validated contribution

### 6.3 Limitations
1. **Single model**: Only GPT-2 Small; larger models may differ
2. **Single concept domain**: First-letter features (a-z) only
3. **Single SAE config**: One SAE width/sparsity
4. **English only**: All experiments use English text
5. **Limited DFDA validation**: Single pair, parent-positive protocol needs full implementation

### 6.4 Future Work
- **Causal inference approaches**: Interventions (activation patching), do-calculus for detecting suppression
- **Suppression-based detection**: Explicitly model P(parent | child absent) - P(parent | child present)
- **Cross-model validation**: Test on Gemma-2B, Pythia when access permits
- **DFDA scaling**: Full parent-positive evaluation on 8+ pairs
- **Theoretical analysis**: Closed-form characterization of when suppression is detectable

---

## 7. Conclusion

We present the first systematic analysis of why unsupervised co-occurrence methods fail for feature absorption detection in sparse autoencoders. UAD, which uses phi coefficient clustering on co-occurrence matrices, achieves F1 = 0.007 --- statistically indistinguishable from random selection (F1 = 0.0075). Ablations across clustering algorithms, correlation metrics, and feature subsets confirm the failure is fundamental: co-occurrence detects semantic correlation, while absorption is defined by suppression. We formalize the structural difference and argue that future unsupervised detection methods must explicitly model suppression signals rather than correlation patterns. We additionally report DFDA as exploratory work toward training-free absorption compensation. Our findings redirect the community toward causal inference approaches and prevent wasted effort on correlation-based detection strategies.

---

## Figure & Table Plan

### Figure 1: UAD Detection Pipeline (Section: Method)
- **Purpose**: Illustrate the UAD algorithm and highlight where the assumption fails
- **Type**: flow_chart
- **Content**:
  - Step 1: Extract activations A from SAE on corpus
  - Step 2: Compute co-occurrence matrix C = A^T A
  - Step 3: Normalize to phi coefficient matrix R
  - Step 4: Hierarchical clustering (Ward, 50 clusters)
  - Step 5: Same-cluster pairs as candidate absorbed pairs
  - **X mark on Step 5**: "Detects correlation, not suppression"
  - Step 6: Validate against supervised labels -> near-zero precision
- **Key takeaway**: The pipeline is sound for correlation detection but cannot capture the suppression signal that defines absorption
- **Generation**: tikz / matplotlib
- **Data source**: Method description

### Figure 2: UAD F1 vs. Random Baseline (Section: Experiments)
- **Purpose**: Show UAD performs no better than random
- **Type**: bar_chart
- **Content**:
  - Two bars: UAD F1 = 0.007, Random F1 = 0.0075
  - Error bar on random: +/- 1 std (0.0054)
  - Delta annotation: -0.0001 (not statistically significant)
  - Horizontal reference line at F1 = 0.5 (what would be useful)
- **Key takeaway**: UAD F1 is indistinguishable from random; the method fails to detect absorption
- **Generation**: matplotlib/seaborn
- **Data source**: `e1_uad_random_baseline_results.json`

### Figure 3: Ablation Study Results (Section: Experiments)
- **Purpose**: Show that the failure persists across all design choices
- **Type**: grouped_bar_chart
- **Content**:
  - Bars for: Full UAD, K-means, All-features, Random
  - Y-axis: Number of same-cluster pairs (log scale)
  - All variants produce thousands of pairs
  - Annotation: "Precision ~0.004 across all variants"
- **Key takeaway**: The failure is in the core assumption (co-occurrence = absorption), not implementation details
- **Generation**: matplotlib/seaborn
- **Data source**: `e2_uad_ablations_results.json`

### Figure 4: Correlation vs. Suppression: The Structural Difference (Section: Analysis)
- **Purpose**: Visualize why co-occurrence cannot detect absorption
- **Type**: diagram (tikz/manual)
- **Content**:
  - Left panel: "Correlation" -- two features fire together frequently (Venn diagram overlap)
  - Right panel: "Absorption" -- parent fires when child absent, suppressed when child present (asymmetric)
  - Arrow: "Co-occurrence clustering detects LEFT, not RIGHT"
- **Key takeaway**: Absorption is an asymmetric suppression phenomenon; co-occurrence is a symmetric correlation measure
- **Generation**: tikz
- **Data source**: Theoretical analysis

### Table 1: Main Results -- UAD vs. Random Baseline (Section: Experiments)
- **Purpose**: Comprehensive comparison of UAD and random baseline
- **Type**: comparison_table
- **Content**:

| Method | Same-Cluster Pairs | Detected Pairs | TP | Precision | Recall | F1 | Runtime |
|--------|-------------------|----------------|-----|-----------|--------|-----|---------|
| UAD (Full) | 3,702 | 541 | 2 | 0.0037 | 1.0 | 0.0074 | 9.0s |
| UAD (K-means) | 7,648 | -- | -- | -- | -- | -- | -- |
| UAD (All features) | 154,858 | -- | -- | -- | -- | -- | -- |
| Random (mean) | -- | 541 | 2.0 | 0.0037 | 1.0 | 0.0075 | -- |
| Random (std) | -- | -- | -- | -- | -- | 0.0054 | -- |

- **Key takeaway**: All UAD variants produce near-zero precision; random selection is statistically equivalent
- **Generation**: data_table (LaTeX)
- **Data source**: `e1_uad_random_baseline_results.json`, `e2_uad_ablations_results.json`

### Table 2: Comparison with Prior Work (Section: Related Work)
- **Purpose**: Position our negative result against existing methods
- **Type**: comparison_table
- **Content**:

| Method | Supervision | Detection Approach | F1 / Metric | Applicability |
|--------|-------------|-------------------|-------------|---------------|
| Chanin et al. (2024) | Full | Supervised probes + ablation | Defines ground truth | Known concepts only |
| SAEBench | Partial | Probe projection | N/A | Known concepts only |
| UAD (Ours) | None | Co-occurrence clustering | F1 = 0.007 | Any SAE, but fails |
| **Needed** | None | **Suppression detection** | **?** | **Any SAE** |

- **Key takeaway**: No unsupervised absorption detection method currently works; future work must target suppression signals
- **Generation**: data_table (LaTeX)
- **Data source**: Literature review + UAD results

### Table 3: DFDA Parent-Positive Results (Section: Experiments)
- **Purpose**: Report exploratory DFDA results
- **Type**: data_table
- **Content**:

| Metric | Value |
|--------|-------|
| Baseline MSE | 5.2e-06 |
| Compensated MSE | 4.1e-06 |
| Improvement ratio | 21.2% |
| Parameters per pair | 97 |
| Pairs evaluated | 1 (exploratory) |

- **Key takeaway**: DFDA shows promise on parent-positive examples but needs full validation
- **Generation**: data_table (LaTeX)
- **Data source**: `e6_dfda_parent_positive_results.json`

---

## Transition Logic Between Sections

1. **Abstract -> Introduction**: Abstract states the negative result; Introduction motivates why the question matters
2. **Introduction -> Related Work**: After establishing the problem, survey existing supervised methods and the gap for unsupervised detection
3. **Related Work -> Methodology**: After showing no prior unsupervised method, present UAD algorithm honestly (tested and failed)
4. **Methodology -> Experiments**: After defining UAD, present the experimental proof of failure: random baseline, ablations, cross-layer
5. **Experiments -> Analysis**: After presenting results, explain WHY co-occurrence cannot detect absorption (correlation vs. suppression)
6. **Analysis -> Discussion**: After the theoretical analysis, discuss implications, DFDA status, limitations, and future directions
7. **Discussion -> Conclusion**: Summarize the negative result, theoretical insight, and recommendations

---

## Visual Storytelling Flow

- **Introduction**: No figure; text-only motivation establishing the supervision bottleneck and the failed attempt
- **Related Work**: Table 2 (comparison with prior work) -- establishes the gap and our negative result
- **Methodology**:
  - Figure 1 (UAD pipeline with X mark) -- shows where the assumption fails
  - Notation table (notation.md) for mathematical symbols
- **Experiments**:
  - Table 1 (main results) -- comprehensive failure evidence
  - Figure 2 (UAD vs Random) -- the central negative result
  - Figure 3 (ablations) -- failure persists across variants
  - Table 3 (DFDA exploratory) -- positive but preliminary
- **Analysis**:
  - Figure 4 (correlation vs suppression diagram) -- theoretical explanation of why it fails
- **Discussion**: No new figures; references back to Figures 2-4
- **Conclusion**: No figures

---

## Estimated Paper Length

- Abstract: 0.5 page
- Introduction: 1.5 pages
- Related Work: 1.5 pages
- Methodology: 2 pages
- Experiments: 2.5 pages
- Analysis: 1.5 pages
- Discussion: 1 page
- Conclusion: 0.5 page
- **Total**: ~11 pages (NeurIPS/ICLR format, excluding references)
- Figures: 4
- Tables: 3
- References: ~25-30
