# Paper Outline: Why Co-occurrence Clustering Cannot Detect Feature Absorption in Sparse Autoencoders

## Title

**Why Co-occurrence Clustering Cannot Detect Feature Absorption in Sparse Autoencoders**

> Target: ICBINB ("I Can't Believe It's Not Better") Workshop or NeurIPS/ICML Workshop on Mechanistic Interpretability. Negative result paper with constructive forward look.

---

## Abstract (150 words)

Feature absorption in sparse autoencoders (SAEs) occurs when parent features are suppressed by child features, creating interpretability illusions. While existing detection methods require supervised ground-truth probes, recent work has proposed unsupervised co-occurrence clustering as an alternative. We conduct the first systematic empirical evaluation of this approach and find that it fails catastrophically: F1 = 0.00048 (1 true positive out of 4,155 detected pairs, 6 false negatives out of 7 ground truth pairs). Through careful root-cause analysis, we identify the fundamental reason: absorption features are mutually exclusive at the token level---they fire on different tokens representing different child concepts---making co-occurrence-based detection inherently unsuitable. In parallel, we demonstrate that collision rate (top-K feature overlap) exhibits strong internal consistency as an operationalization of absorption (Spearman r = 0.869, n = 56 pairs, 95% CI [0.780, 0.938]). We discuss why decoder weight similarity and causal intervention methods are theoretically better suited, and propose concrete next steps.

---

## 1. Introduction (1.5 pages)

### 1.1 Feature Absorption: A Critical SAE Failure Mode
- Sparse autoencoders have become the dominant tool for mechanistic interpretability
- Chanin et al. (2024) identified feature absorption as a fundamental failure mode
- When hierarchical features co-occur in training data, the SAE may suppress the parent to increase sparsity while maintaining reconstruction
- Creates dangerous interpretability illusions: a latent appears to track a concept but fails to activate on arbitrary positive examples

### 1.2 The Detection Bottleneck
- All existing absorption detection methods (Chanin et al., 2024; Karvonen et al., 2025) require:
  1. Knowing the parent feature a priori
  2. Training supervised probe directions as ground truth
  3. Running computationally expensive ablation studies
- This supervised requirement means absorption can only be detected for concepts we already know to look for---precisely where SAEs are least needed
- Unsupervised detection would unlock absorption auditing at scale

### 1.3 The Co-occurrence Hypothesis and Our Finding
- The Unsupervised Absorption Detection (UAD) method proposes that absorbed pairs can be discovered via co-occurrence clustering
- Training-free, no labeled data required
- **Our central finding**: UAD achieves F1 = 0.00048---identical to randomly sampling from the same clusters
- The root cause: absorption features are mutually exclusive at the token level
- **Paper roadmap**: Section 2 covers background, Section 3 methods, Section 4 results, Section 5 discussion

### Transition to Section 2
> "To understand why UAD fails, we first review the definitions of feature absorption and the UAD pipeline itself."

---

## 2. Background and Related Work (1 page)

### 2.1 Feature Absorption Definition
- Formal definition from Chanin et al. (2024)
- Parent feature: activates on a broad concept (e.g., "number")
- Child features: activate on sub-concepts (e.g., "three", "four")
- Absorption: parent is suppressed when children are present; SAE allocates activation budget to children
- Consequence: parent appears dead or weak even though it encodes the broad concept

### 2.2 Existing Detection Methods (All Supervised)
- Chanin et al. (2024): supervised probe directions + ablation studies
- Karvonen et al. (2025): learned probes with causal validation
- Common limitation: all require knowing the parent concept in advance

### 2.3 Unsupervised Absorption Detection (UAD)
- Proposed by Chanin et al. (2024) as a training-free alternative
- Pipeline: dead feature filtering -> phi coefficient co-occurrence matrix -> hierarchical clustering -> specificity filtering -> pair extraction
- Claim: features that co-occur on the same tokens are likely to be in an absorption relationship
- **Our contribution**: first systematic evaluation of this claim on a pretrained SAE

### Transition to Section 3
> "We now describe our experimental setup, ground truth operationalization, and the full UAD pipeline as tested."

---

## 3. Methods (1.5 pages)

### 3.1 Experimental Setup
- Model: GPT-2 Small (12 layers, 124M parameters)
- SAE: gpt2-small-res-jb (JumpReLU, pretrained, d_SAE = 24,576, layer 8 residual stream)
- Dataset: OpenWebText (1,000 sequences, seed 42)
- Software: SAELens >= 2.0.0, TransformerLens >= 2.0.0
- Single seed (42); noted as limitation in Section 6

### 3.2 Ground Truth: Operationalization via Top-K Overlap
- For each concept token (e.g., "three"), identify top-K activating SAE features
- Two concepts are in an absorption relationship if they share absorbing features in their top-K sets
- Absorption rate: Jaccard similarity of top-K feature sets
- **Critical distinction**: This operationalization defines what we measure. It is not an independent "proxy" validated against external ground truth. Both collision rate and absorption rate are computed from the same top-K feature sets.
- Collision rate = absorption rate = Jaccard(topK(c1), topK(c2))

### 3.3 Hierarchy Types Tested
- Numbers: one through eight (28 pairs, absorption rate 0.25--1.0)
- Punctuation: ., ,, !, ?, ;, :, ", ' (28 pairs, absorption rate 0.0--1.0)
- Case: a/A through z/Z (26 pairs, absorption rate 0.0, control group)

### 3.4 UAD Pipeline Tested
1. Dead feature filtering: remove features with near-zero activation variance
2. Co-occurrence matrix: compute phi coefficients between all feature pairs on corpus
3. Hierarchical clustering: Ward's linkage on co-occurrence similarity
4. Specificity filtering: filter clusters by activation sparsity patterns
5. Pair extraction: all pairs within clusters flagged as "absorbed"

### 3.5 Ablations and Baselines
- Six ablation variants: full UAD, no dead filter, no phi filter, no clustering, single linkage, K-means
- Two random baselines: same-cluster random (sample N pairs from within UAD clusters), global random (sample N pairs from all possible pairs)
- Bootstrap 95% CI for F1: 1,000 resamples with replacement

### Transition to Section 4
> "With the methodology established, we present our empirical findings."

---

## 4. Results (2.5 pages)

### 4.1 UAD Fails Catastrophically (Negative Result)
- UAD detects 1 true positive out of 4,155 candidate pairs
- Precision: 0.024%, Recall: 14.3%, F1: 0.00048
- Bootstrap 95% CI for F1: [0.00012, 0.00102]
- Table 1 (UAD Performance and Baselines) summarizes
- **Key point**: With only 7 ground truth pairs, statistical power is limited. The conclusion is that UAD fails on this test set, and the failure mode is structurally grounded.

### 4.2 Random Baseline Comparison
- Same-cluster random F1: 0.00048 (identical to UAD)
- Global random F1: 0.00011
- UAD's sophisticated pipeline (phi filtering, dead feature filtering, specificity checks, hierarchical clustering) provides exactly zero improvement over trivial random sampling from the same clusters
- Both methods detect exactly 1 true positive out of 4,155 candidates
- Figure 2 (UAD F1 vs Baselines) visualizes

### 4.3 Ablation Results
- Table 2 (Ablation Results) shows all six variants
- Full UAD / no dead filter / no phi filter: identical (F1 = 0.00048)
- No clustering: F1 = 0.000056 (3 TP, 106,861 FP)
- Single linkage: F1 = 0.0 (0 TP)
- K-means: F1 = 0.0037 (6 TP, 3,237 FP, 85.7% recall, 0.185% precision)
- **K-means analysis**: Achieves high recall because hard assignment forces all features into clusters. Ward linkage's variance-minimizing criterion correctly separates absorption features (phi ~ 0) into different clusters. But precision remains 0.185%, making F1 = 0.0037 still practically unusable.
- Figure 3 (Precision-Recall Trade-off) visualizes

### 4.4 Collision Rate Internal Consistency
- Spearman r = 0.869 across 56 concept pairs (numbers + punctuation)
- 95% CI: [0.780, 0.938], p = 4.2e-18
- Numbers only: r = 0.598; Punctuation only: r = 0.693
- Table 3 (Collision Rate Consistency) summarizes
- **Interpretation**: The operational definition (top-K feature overlap) produces stable, expected patterns across diverse concept pairs. This is internal consistency of the operationalization, not proxy validation against an independent ground truth.
- Figure 4 (Collision Rate Scatter Plot) visualizes

### 4.5 Root Cause: Token-Level Mutual Exclusivity
- Feature 11513 fires only on "three" (activation 29.4)
- Feature 24189 fires on "four" through "eight" (activations 14.9--18.9)
- They never activate on the same token
- Table 4 (Token-Level Activations) shows the full pattern
- **Finding**: Absorption features are mutually exclusive at the token level for tested token-disjoint hierarchies
- Figure 1 (Token Activation Heatmap) visualizes
- **Caveat**: This property holds for token-disjoint hierarchies (numbers, punctuation). Semantic hierarchies (animal/dog) where children co-occur in natural text may show different patterns.

### Transition to Section 5
> "These results reveal a structural mismatch between what UAD measures and what absorption is. We now discuss the implications."

---

## 5. Discussion (1.5 pages)

### 5.1 Why Co-occurrence Clustering Is the Wrong Tool
- Co-occurrence clustering assumes related features activate on the same inputs
- This is true for: synonym features ("happy" and "joyful"), contextually related features ("king" and "queen")
- But absorption in token-disjoint hierarchies is different: absorbed features represent mutually exclusive sub-concepts
- "Three" and "four" never appear at the same position in a number sequence
- UAD is designed to find features that fire TOGETHER, not features that fire on ALTERNATIVE instances of the same abstract concept
- **Caveat**: We only test token-disjoint hierarchies. Semantic hierarchies where children co-occur may show different patterns.

### 5.2 Why Collision Rate Shows Internal Consistency
- Collision rate measures structural similarity of feature responses, not co-occurrence
- Two child concepts may share the same absorbing feature in their top-K even though they never appear together
- Example: both "three" and "four" have feature 13586 in their top-5
- The high correlation (r = 0.869) indicates the operational definition is structurally coherent

### 5.3 Theoretical Implications
1. Absorption is not co-occurrence: the mechanism is more subtle than "features that appear together get merged"
2. Decoder weight similarity may be the right signal: if two child features share a parent in reconstruction, their decoder directions should be geometrically related
3. Causal intervention is the gold standard: only way to definitively establish absorption is to show suppressing a child causes parent recovery

### 5.4 Proposed Alternative Approaches
- **Decoder weight similarity** (highest priority): cosine similarity of decoder weight vectors; training-free; directly measures structural relationship
- **Causal intervention**: activation patching to test whether suppressing child causes parent recovery; causally establishes absorption
- **Semantic similarity clustering**: cluster by decoder weight similarity instead of activation co-occurrence

### Transition to Section 6
> "Before concluding, we acknowledge the limitations of our study."

---

## 6. Limitations (0.5 page)

1. **Single SAE**: gpt2-small-res-jb layer 8 only. Results may not generalize to other layers, models, or SAE architectures.
2. **Small ground truth**: Only 7 true absorption pairs (6 distinct + 1 self-pair). Statistical power is limited; bootstrap CI provided.
3. **Single model**: GPT-2 Small is small by modern standards. Absorption dynamics may differ in larger models.
4. **Token-disjoint hierarchies only**: Numbers and punctuation are token-disjoint. Semantic hierarchies may show different patterns.
5. **No causal validation of alternatives**: Decoder weight similarity and causal intervention are proposed but not empirically tested.
6. **Single seed**: All experiments use seed 42. Sensitivity to corpus sampling is unknown.
7. **Operationalization, not proxy**: Collision rate measures internal consistency of the operational definition, not predictive validity against independent ground truth.

---

## 7. Conclusion (0.5 page)

- Summary: UAD (co-occurrence clustering) fails to detect feature absorption in token-disjoint hierarchies (F1 = 0.00048)
- Root cause: absorption features are mutually exclusive at the token level
- Constructive contribution: collision rate operationalization shows strong internal consistency (r = 0.869)
- Call to action: test decoder weight similarity as an alternative unsupervised detection method
- Negative results prevent wasted effort; honest reporting advances the field

---

## Figure & Table Plan

### Figure 1: Token-Level Activation Heatmap (Section: Results 4.5)
- **Purpose**: Show that absorption features are mutually exclusive at the token level
- **Type**: heatmap
- **Content**: Rows = 4 absorption features (F11513, F12413, F22971, F24189); Columns = 8 tokens ("one" through "eight"); Cell color = activation magnitude
- **Key takeaway**: Each feature fires on a disjoint set of tokens---they never co-occur
- **Generation**: code (matplotlib/seaborn `imshow` or `heatmap`)
- **Data source**: `iter_003/exp/results/pilots/p2_uad_reproduce_results.json` -> `ground_truth.number_features`

### Figure 2: UAD F1 vs Random Baselines (Section: Results 4.2)
- **Purpose**: Show UAD provides zero improvement over trivial random sampling
- **Type**: grouped_bar_chart
- **Content**: Three bars: UAD F1 (0.00048), Same-cluster random F1 (0.00048), Global random F1 (0.00011)
- **Key takeaway**: UAD F1 equals same-cluster random F1; all complexity provides no value
- **Generation**: code (matplotlib/seaborn)
- **Data source**: `iter_003/exp/results/pilots/p3_random_baseline_results.json` -> `uad_results.f1`, `baselines.same_cluster_random.mean_f1`, `baselines.analytical_global_random.mean_f1`

### Figure 3: Precision-Recall Trade-off Across Clustering Variants (Section: Results 4.3)
- **Purpose**: Show that even the best clustering variant (K-means) has near-zero precision
- **Type**: scatter_plot
- **Content**: Six points, one per variant (Full UAD, No dead filter, No phi, No clustering, Single linkage, K-means); x = Precision, y = Recall; point size proportional to F1
- **Key takeaway**: K-means achieves 85.7% recall but 0.185% precision; no variant is practically usable
- **Generation**: code (matplotlib/seaborn)
- **Data source**: `iter_003/exp/results/full/f2_uad_ablations_results.json` -> `ablation_results.*.precision`, `*.recall`, `*.f1`

### Figure 4: Collision Rate vs True Absorption Rate (Section: Results 4.4)
- **Purpose**: Show strong internal consistency of the collision rate operationalization
- **Type**: scatter_plot
- **Content**: 56 points (x = collision rate, y = true absorption rate); color by hierarchy (numbers = blue, punctuation = orange); regression line; annotate r = 0.869, CI [0.780, 0.938]
- **Key takeaway**: Collision rate and absorption rate are strongly correlated, confirming internal consistency of the operational definition
- **Generation**: code (matplotlib/seaborn `regplot`)
- **Data source**: `iter_003/exp/results/full/f4_collision_correlation_results.json` -> `pair_results.numbers` and `pair_results.punctuation` -> `collision_rate`, `true_absorption`

### Table 1: UAD Performance and Random Baselines (Section: Results 4.1--4.2)
- **Purpose**: Summarize the core negative result
- **Type**: comparison_table
- **Content**: Rows = Method (UAD, Same-cluster random, Global random, Frequency-weighted random); Columns = Detected Pairs, TP, Precision, Recall, F1
- **Key takeaway**: UAD F1 = same-cluster random F1 = 0.00048
- **Generation**: data_table (LaTeX `booktabs`)
- **Data source**: `iter_003/exp/results/pilots/p3_random_baseline_results.json`

### Table 2: UAD Ablation Results (Section: Results 4.3)
- **Purpose**: Show all ablation variants fail
- **Type**: ablation_table
- **Content**: Rows = Variant (Full UAD, No dead filter, No phi filter, No clustering, Single linkage, K-means); Columns = Detected Pairs, TP, Precision, Recall, F1
- **Key takeaway**: All variants have F1 <= 0.0037; K-means best but still unusable
- **Generation**: data_table (LaTeX `booktabs`)
- **Data source**: `iter_003/exp/results/full/f2_uad_ablations_results.json`

### Table 3: Collision Rate Internal Consistency (Section: Results 4.4)
- **Purpose**: Summarize collision rate correlation across hierarchy types
- **Type**: comparison_table
- **Content**: Rows = Experiment (Pilot first letters, Full numbers+punctuation, Numbers only, Punctuation only); Columns = N Pairs, Spearman r, 95% CI
- **Key takeaway**: Strong correlation (r = 0.869) across 56 pairs; consistent across hierarchy types
- **Generation**: data_table (LaTeX `booktabs`)
- **Data source**: `iter_003/exp/results/full/f4_collision_correlation_results.json` + `iter_003/exp/results/pilots/p1_collision_proxy_results.json`

### Table 4: Token-Level Activations for Number Sequence (Section: Results 4.5)
- **Purpose**: Provide concrete evidence of token-level mutual exclusivity
- **Type**: data_table
- **Content**: Rows = tokens ("one" through "eight"); Columns = 4 absorption features (F11513, F12413, F22971, F24189); Cells = activation magnitude (0.0 if silent)
- **Key takeaway**: Each token activates exactly one feature; features never co-occur
- **Generation**: data_table (LaTeX `booktabs`)
- **Data source**: `iter_003/exp/results/pilots/p2_uad_reproduce_results.json` -> `ground_truth.number_features`

---

## Section Flow and Dependencies

```
Abstract -> Introduction -> Background -> Methods -> Results -> Discussion -> Limitations -> Conclusion

Results subsections:
  4.1 (UAD Failure) -> 4.2 (Baseline) -> 4.3 (Ablations) -> 4.4 (Collision Rate) -> 4.5 (Root Cause)

Visual element placement:
  Table 1  -> Results 4.1
  Figure 2 -> Results 4.2
  Table 2  -> Results 4.3
  Figure 3 -> Results 4.3
  Table 3  -> Results 4.4
  Figure 4 -> Results 4.4
  Table 4  -> Results 4.5
  Figure 1 -> Results 4.5
```

## Estimated Page Count

| Section | Pages |
|---------|-------|
| Abstract | 0.25 |
| 1. Introduction | 1.5 |
| 2. Background | 1.0 |
| 3. Methods | 1.5 |
| 4. Results | 2.5 |
| 5. Discussion | 1.5 |
| 6. Limitations | 0.5 |
| 7. Conclusion | 0.5 |
| References | 0.5 |
| **Total** | **~9.5 pages** |

> Suitable for workshop submission (8-10 pages). Can expand Discussion for conference submission.

## Writing Notes for Section Authors

1. **Tone**: Honest, not defensive. The negative result is the contribution.
2. **Numbers**: Use exact values from JSON sources. Never round away precision.
3. **Caveats**: Always scope claims to "tested token-disjoint hierarchies" unless evidence supports broader claims.
4. **Terminology**: Use "operationalization" not "proxy" for collision rate. Use "internal consistency" not "validation."
5. **Figures before text**: Reference each figure/table in the text before it appears.
6. **No banned patterns**: Avoid "groundbreaking," "novel" (unquantified), "significantly outperforms," generic openings.
7. **Lead with evidence**: Start paragraphs with the concrete finding, then explain.
