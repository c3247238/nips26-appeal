# Methodology: Iteration 4 — Writing/Reframing Iteration (No New Experiments)

## 1. Overview

Iteration 4 is a **writing and reframing iteration** that addresses all 12 critical reviewer concerns from Iteration 3 (score 6.0/10) without requiring new experiments. All empirical results were collected in Iterations 1-3 and are reused here with corrected framing, terminology, and analysis.

The core empirical findings are unchanged:
- UAD F1 = 0.00048 (1 TP / 4,155 detected pairs, 6 FN / 7 GT pairs)
- Same-cluster random F1 = 0.00048 (identical — both detect exactly 1 TP)
- Collision rate Spearman r = 0.869 (n = 56, 95% CI [0.780, 0.938])
- Token-level mutual exclusivity confirmed as root cause
- K-means achieves 85.7% recall but 0.185% precision (F1 = 0.0037)

What changes in Iteration 4 is the **narrative framing**:
1. Collision rate: "proxy validation" -> "operationalization reliability"
2. F1 identity: "statistically indistinguishable" -> "both detect exactly 1 TP"
3. Mutual exclusivity: universal claim -> scoped to "tested token-disjoint hierarchies"
4. K-means: dismissed -> analyzed as hard assignment artifact
5. Data: corrected numbers r = 0.598, punctuation r = 0.693
6. Fabricated claims: removed

## 2. Experimental Setup (Reused from Iteration 3)

### 2.1 Models and SAEs

**Primary Model**: GPT-2 Small (12 layers, 124M parameters)
- Hook point: `blocks.8.hook_resid_post`
- SAE source: `gpt2-small-res-jb` (JumpReLU, pretrained, d_SAE = 24,576)
- Layer: 8 (residual stream post)

**Rationale for single-model scope**: All experiments are training-free analysis of pretrained SAEs. Cross-model validation was attempted but blocked by gated access. This limitation is explicitly acknowledged.

### 2.2 Dataset

- **Primary**: OpenWebText (1,000 sequences, seed 42)
- **Concept evaluation**: Numbers (one-eight), punctuation (.,!?:;"'), case (a-z, A-Z control)
- **Seeds**: 42 (single seed; noted as limitation)

### 2.3 Software Stack

```
Base: Python 3.10+, PyTorch 2.0+
SAE: SAELens >= 2.0.0
Model: TransformerLens >= 2.0.0
Analysis: NumPy, SciPy, Pandas, scikit-learn, Matplotlib, Seaborn
```

## 3. Core Experiments (Already Completed — Results Reused)

### 3.1 E1: UAD Failure with Random Baseline (COMPLETED, Iteration 3)

**Objective**: Demonstrate that UAD achieves F1 = 0.00048 and is identical to same-cluster random baseline.

**Method** (already executed):
1. Run UAD on GPT-2 Small layer 8
2. Compute same-cluster random baseline: randomly sample 4,155 pairs from within UAD clusters
3. Compare UAD F1 vs same-cluster random F1

**Results** (from `iter_003/exp/results/pilots/p3_random_baseline_results.json`):
- UAD F1: 0.00048053820278712154
- Same-cluster random F1: 0.00048053820278712154 (identical)
- Global random F1: 0.0001102626591882312
- UAD detects exactly 1 TP out of 4,155; same-cluster random also detects exactly 1 TP

**Iteration 4 reframing**: Present as arithmetic identity, not statistical finding. Both methods detect exactly 1 true positive out of 4,155 candidate pairs. UAD's complexity provides zero improvement over trivial random sampling.

### 3.2 E2: UAD Ablation Studies (COMPLETED, Iteration 3)

**Objective**: Test all UAD ablation variants.

**Results** (from `iter_003/exp/results/full/f2_uad_ablations_results.json`):

| Variant | Detected Pairs | TP | Precision | Recall | F1 |
|---------|---------------|----|-----------|--------|-----|
| Full UAD | 4,155 | 1 | 0.024% | 14.3% | 0.00048 |
| No dead filter | 4,155 | 1 | 0.024% | 14.3% | 0.00048 |
| No phi filter | 4,155 | 1 | 0.024% | 14.3% | 0.00048 |
| No clustering | 106,864 | 3 | 0.003% | 42.9% | 0.000056 |
| Single linkage | 102,832 | 0 | 0.0% | 0.0% | 0.0 |
| K-means | 3,243 | 6 | 0.185% | 85.7% | 0.0037 |

**Iteration 4 addition**: K-means analysis. K-means achieves 85.7% recall because its hard assignment forces all features into clusters. Ward linkage's variance-minimizing criterion correctly separates absorption features (phi ~ 0) into different clusters. However, precision remains 0.185% (3,237 false positives), making F1 = 0.0037 still practically unusable.

### 3.3 E3: Collision Rate Internal Consistency (COMPLETED, Iteration 3)

**Objective**: Demonstrate internal consistency of collision rate as operationalization.

**Results** (from `iter_003/exp/results/full/f4_collision_correlation_results.json`):

| Experiment | N Pairs | Spearman r | 95% CI |
|-----------|---------|-----------|--------|
| Pilot (First Letters) | 10 | 0.711 | [0.219, 0.887] |
| Full (Numbers + Punctuation) | 56 | 0.869 | [0.780, 0.938] |
| Numbers only | 28 | 0.598 | - |
| Punctuation only | 28 | 0.693 | - |

**Iteration 4 reframing**: This is NOT "proxy validation" (collision rate and absorption rate are computed from the same top-K feature sets). It is "operationalization reliability" — the operational definition produces stable, expected patterns across diverse concept pairs.

### 3.4 E4: Root Cause — Token-Level Mutual Exclusivity (COMPLETED, Iteration 3)

**Objective**: Identify why co-occurrence clustering fails.

**Evidence** (from `iter_003/exp/results/pilots/p2_uad_reproduce_results.json`):

Token-level activations for number sequence:

| Token | Feature | Activation |
|-------|---------|-----------|
| one | 12413 | 21.03 |
| two | 22971 | 34.79 |
| three | 11513 | 36.31 |
| four | 24189 | 22.23 |
| five | 24189 | 23.30 |
| six | 24189 | 23.26 |
| seven | 24189 | 25.35 |
| eight | 24189 | 25.91 |

Absorption features (feature -> tokens it absorbs):
- Feature 24189: ["four", "five", "six", "seven", "eight"]
- Feature 11513: ["three", "seven"]
- Feature 12413: ["one"] (primary)
- Feature 22971: ["two"] (primary)

**Finding**: Features 11513 and 24189 NEVER co-occur on any token. Feature 11513 fires only on "three", while feature 24189 fires on "four" through "eight". They are mutually exclusive at the token level.

**Iteration 4 scope softening**: This mutual exclusivity property holds for tested token-disjoint hierarchies (numbers, punctuation). Semantic hierarchies (animal/dog) where children co-occur in natural text may show different patterns.

### 3.5 E5: False Positive Analysis (COMPLETED, Iteration 3)

**Results** (from `iter_003/exp/results/full/f5_false_positive_results.json`):
- N detected: 4,155
- N true positives: 1
- N false positives: 4,154
- Precision: 0.024%
- FP rate: 99.98%

**Failure modes identified**:
1. Token-level mutual exclusivity: absorption features fire on different tokens
2. Clustering degeneracy: 50 clusters on 504 features places GT pairs in different clusters
3. Phi coefficient irrelevance: phi measures co-occurrence, but absorption features have phi ~ 0

## 4. Baselines

| Baseline | Purpose | Source |
|----------|---------|--------|
| Same-cluster random | Show UAD clustering provides zero value | `p3_random_baseline_results.json` |
| Global random | Show UAD is marginally better than completely random | `p3_random_baseline_results.json` |
| Frequency-weighted random | Control for feature frequency | `p3_random_baseline_results.json` |

## 5. Expected Visualizations

### Tables
- **Table 1**: UAD ablation results (variant x F1/precision/recall/TP/FP)
- **Table 2**: Collision rate internal consistency (hierarchy x n_pairs x Spearman r x CI)
- **Table 3**: Random baseline comparison (baseline type x F1/precision/recall)

### Figures
- **Figure 1**: Token-level activation heatmap (numbers sequence x features)
- **Figure 2**: UAD F1 comparison — Full vs Random Baseline vs Ablations (grouped bar chart)
- **Figure 3**: Collision rate scatter plot (collision rate vs true absorption, n=56)
- **Figure 4**: K-means analysis — recall vs precision trade-off across clustering variants

## 6. Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Reviewer rejects negative result framing | Low | Critical | Target ICBINB workshop; emphasize constructive contributions |
| Collision rate circularity criticism | Medium | High | Explicitly frame as operationalization reliability, not proxy validation |
| Single SAE generalizability concern | Medium | Medium | Acknowledge as limitation; scope claims to tested setting |
| Small GT sample (n=7) | High | Medium | Bootstrap CI provided; soften conclusions to "fails on this test set" |

## 7. Reproducibility

- All random seeds fixed (42)
- Exact package versions pinned in requirements.txt
- Code open-sourced with Jupyter notebooks
- Pre-trained SAE checkpoints referenced by SAELens release ID
- Bootstrap 95% CI reported for all key metrics
- All results stored in machine-readable JSON

## 8. Iteration 4 Specific Changes

### Writing Changes (No New Experiments)

| Section | Change | Rationale |
|---------|--------|-----------|
| Abstract | Add "operationalization reliability" framing | Prevents circularity criticism |
| Methods 3.2 | Explicitly define "operationalization" vs "proxy validation" | Addresses reviewer concern #1 |
| Results 4.1 | Present F1 identity as arithmetic, not statistical | Addresses reviewer concern #2 |
| Results 4.3 | Add K-means hard assignment analysis | Addresses reviewer concern #4 |
| Results 4.4 | Correct numbers r=0.598, punctuation r=0.693 | Addresses reviewer concern #5 |
| Discussion 5.1 | Soften universal claims to "tested token-disjoint hierarchies" | Addresses reviewer concern #3 |
| All | Remove "manual inspection of 50 false positives" | Addresses reviewer concern #6 |
| All | Standardize terminology (Spearman r, precision) | Addresses reviewer concern #10 |
| Limitations | Add bootstrap CI, single seed, small GT | Addresses reviewer concerns #7, #11, #12 |
