# 3. Methodology

## 3.1 Experimental Design Overview

Our approach is training-free analysis of existing pretrained SAEs. All experiments are completed—no new compute needed. Table 1 summarizes the experimental design.

| Component | Details |
|-----------|---------|
| **Model** | GPT-2 Small (124M parameters) |
| **SAE** | gpt2-small-res-jb, 24,576 latents |
| **Layers** | 0, 4, 8, 10 (hook_resid_pre) |
| **Features** | 26 first-letter features (A-Z) |
| **Samples per feature** | 100 prompts × 100 samples each |
| **Ground truth** | Chanin et al. absorption pairs |
| **Multiple comparison correction** | Bonferroni (alpha=0.00417), BH-FDR (q<0.05) |

**Table 1**: Experimental design summary.

## 3.2 Phase 1: Absorption Detection

### 3.2.1 Differential Correlation Metric

We apply the Chanin et al. (2024) differential correlation metric to detect absorption:

1. Collect activation data for 26 first-letter features (A-Z) across 100 prompts each, 100 samples per feature
2. For each feature $f$, compute the correlation between parent feature activation and child feature activation
3. Ablate the child feature and recompute the parent correlation
4. Absorption rate $A(f)$ = drop in correlation / baseline correlation

### 3.2.2 Results

Absorption rates across 26 features at L4 and L8:
- **L4**: mean=3.91%, max=16.0% (Feature G)
- **L8**: mean=3.38%, max=24.2% (Feature U)

High-absorption features (A > 10%): G (14.6%), J (13.3%), P (14.8%), Q (16.0%), R (14.0%), S (9.9%), W (11.3%) at L4; H (19.0%), S (16.0%), U (24.2%), V (14.7%) at L8.

## 3.3 Phase 2: Downstream Task Evaluation

### 3.3.1 Feature Steering

**Protocol**:
1. For each first-letter feature, steer the SAE latent in the decoder direction $d_f = W_{\text{dec}}[:, f]$ with strengths $s \in \{1.0, 2.0, 5.0, 10.0, 20.0, 50.0\}$
2. Compute steering success: fraction of prompts where the top-5 tokens contain the target letter
3. Apply random baseline subtraction: $\Delta S(f) = S_{\text{feature}}(f) - S_{\text{random}}(f)$

**Success criterion**: Top-5 token contains target letter.

**Random baseline**: Mean steering success with random latent directions = 0.344 (L4), 0.379 (L8).

**Hypotheses tested**:
- **H1**: No correlation between absorption rate and raw steering success
- **H1b**: No correlation between absorption rate and delta-corrected steering success

### 3.3.2 Sparse Probing

**Protocol**:
1. Train k-sparse linear probes at sparsity levels $k \in \{1, 5, 10, 20\}$
2. Evaluate on held-out test data
3. Compute precision, recall, and F1 at each k

**Hypothesis tested**:
- **H2**: No correlation between absorption rate and probing F1

### 3.3.3 EC50 Dose-Response Analysis

**Protocol**:
1. Fit dose-response curves: $S(s) = S_{\text{max}} \cdot s^n / (EC50^n + s^n)$
2. Estimate EC50 (median effective steering strength) for each feature
3. Test correlation between EC50 and absorption rate

**Hypothesis tested**:
- **H4**: No correlation between absorption rate and EC50

### 3.3.4 Cross-Layer Consistency

**Protocol**:
1. Repeat H1 and H2 at both L4 and L8
2. Test whether effect signs are consistent across layers

**Hypothesis tested**:
- **H3**: Cross-layer consistency (same sign at both layers)

## 3.4 Phase 3: Precision-Recall Decomposition

### 3.4.1 H5: Precision-Recall Asymmetry

**Protocol**:
1. For each feature at each k, compute precision (fraction of predicted positives that are true positives) and recall (fraction of true positives that are predicted)
2. Test whether precision varies less than recall across features
3. Correlation: absorption rate vs. recall

**Hypothesis tested**:
- **H5**: Absorption affects recall but not precision (precision invariance, recall varies)

## 3.5 Phase 4: Inhibition Graph Validation

### 3.5.1 Decoder Correlation Graph Construction

1. Compute decoder correlation matrix: $G = W_{\text{dec}}^T W_{\text{dec}}$, $G_{ij} = \langle W_{\text{dec}}[i], W_{\text{dec}}[j] \rangle$
2. For each first-letter feature, extract top-k neighbors by decoder correlation
3. Validate predictions against ground truth Chanin absorption pairs

### 3.5.2 Metrics

- **Precision@k**: Fraction of top-k neighbors that are true absorption pairs
- **Recall@k**: Fraction of true absorption pairs found in top-k neighbors
- **Enrichment**: Precision@k / baseline precision (random chance)

**Hypothesis tested**:
- **H6**: Decoder-correlation graph predicts absorption pairs (precision@20 > 0.10)

**Result**: precision@20 = 0.0 (0/520 predictions correct) — H6 FALSIFIED.

## 3.6 Phase 5: Random SAE Baseline Comparison

### 3.6.1 H10: Trained vs. Random SAE

**Protocol**:
1. Compare Chanin absorption metric on trained SAE vs. random SAE (frozen orthonormal decoder, random encoder)
2. Compute absorption rates for both
3. Statistical test: paired t-test

**Result**: Trained SAE (mean=0.034) vs. Random SAE (mean=0.278), t=-6.745, p<0.001.

**Hypothesis tested**:
- **H7**: Trained SAEs have lower absorption than random baselines

## 3.7 Statistical Methods

### 3.7.1 Multiple Comparison Correction

We conduct 12 statistical tests across H1-H3 hypotheses (2 layers × 2 metrics × 3 hypotheses = 12):

- **Bonferroni correction**: $\alpha_B = 0.05 / 12 = 0.00417$
- **Benjamini-Hochberg FDR**: $q < 0.05$

### 3.7.2 Effect Size Measures

- Pearson $r$ and $R^2$ for correlation strength
- Cohen's $d$ for group differences
- Coefficient of variation (CV) for cross-layer consistency

## 3.8 Summary of Hypothesis Tests

| Hypothesis | Test | Layers | Metric | Correction |
|-----------|------|--------|--------|------------|
| H1 | Pearson correlation | L4, L8 | Steering success | Bonferroni + BH-FDR |
| H1b | Pearson correlation | L4, L8 | Delta-corrected steering | Bonferroni + BH-FDR |
| H2 | Pearson correlation | L4, L8 | Probing F1 | Bonferroni + BH-FDR |
| H3 | Sign consistency + CV | L4, L8 | Slopes | CV < 0.5 |
| H4 | Pearson correlation | L4, L8 | EC50 | Bonferroni + BH-FDR |
| H5 | Precision-recall comparison | L4, L8 | Precision std vs. recall std | Descriptive |
| H6 | Precision@k | L8 | Inhibition graph | Descriptive |
| H7 | Paired t-test | L4, L8 | Trained vs. random absorption | p < 0.05 |

**Table 2**: Summary of all hypothesis tests.

## 3.9 Experimental Assets

- **Pretrained models**: GPT-2 Small (OpenAI), gpt2-small-res-jb SAE (SAELens)
- **Libraries**: TransformerLens for hook-based model access, SAELens for SAE loading
- **Data**: 100 prompts per first-letter feature (A-Z), 100 samples each
- **Compute**: Completed experiments on existing infrastructure; no new training needed

<!-- FIGURES
- None
-->