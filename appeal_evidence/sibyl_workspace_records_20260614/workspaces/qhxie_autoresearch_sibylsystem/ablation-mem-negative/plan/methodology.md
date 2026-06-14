# Methodology: Unsupervised Feature Absorption Detection (UAD) and Dynamic Compensation (DFDA)

## 1. Overview

This iteration focuses on validating the two exploratory directions from Iteration 1 that showed genuine novelty and empirical support: **Unsupervised Absorption Detection (UAD)** and **Dynamic Feature De-Absorption (DFDA)**. The original primary contributions (cross-architecture absorption benchmark, causal downstream assessment) are demoted to supplementary material due to proxy metric issues and insufficient statistical power.

All experiments are **training-free** (using pre-trained SAEs via SAELens) unless explicitly noted.

## 2. Experimental Setup

### 2.1 Models and SAEs

**Primary Model**: GPT-2 Small (12 layers, 124M parameters)
- Hook point: `blocks.{layer}.hook_resid_post`
- SAE source: `gpt2-small-res-jb` (JumpReLU, pretrained, d_SAE=24,576)
- Layer evaluated: 8 (primary), with multi-layer validation on layers 4, 8, 10

**Cross-Validation Models**:
| Model | Size | SAE Source | d_SAE | Status |
|-------|------|-----------|-------|--------|
| Gemma-2B | 2B | GemmaScope (if available) | 16k-24k | Primary cross-validation |
| Pythia-2.8B | 2.8B | SAELens pretrained | 16k-24k | Secondary cross-validation |

**Rationale for GPT-2 Small primary**: Iteration 1 established that pre-trained SAEs are reliably loadable for GPT-2 Small via SAELens. Gemma-2B access was blocked by API issues in Iteration 1; we attempt it again with updated SAELens configuration.

### 2.2 Dataset

- **Primary**: OpenWebText (10k samples for full experiments, 1k for pilots)
- **Concept evaluation**: First-letter features (a-z) as primary validation set; WordNet semantic hierarchies as secondary
- **Seeds**: 42, 123, 456 (3 seeds for all key experiments)

### 2.3 Software Stack

```
Base: Python 3.10+, PyTorch 2.0+
SAE: SAELens >= 2.0.0
Model: TransformerLens >= 2.0.0
Analysis: NumPy, SciPy, Pandas, scikit-learn, Matplotlib, Seaborn
```

## 3. Core Experiments

### 3.1 E1: UAD Cross-Model Validation (CRITICAL - Go/No-Go Gate)

**Objective**: Validate that UAD achieves F1 >= 0.55 on models larger than GPT-2 Small.

**Method**:
1. Load pre-trained SAE for each model (Gemma-2B, Pythia-2.8B)
2. Extract feature activation matrix A (n_examples x n_features) on 10k tokens
3. Compute phi coefficient correlation matrix R from A
4. Run hierarchical agglomerative clustering (Ward linkage, 50 clusters)
5. Identify same-cluster feature pairs
6. Validate against Chanin-style supervised labels (first-letter features)
7. Compute precision, recall, F1

**Metrics**:
- Primary: F1 score (threshold: >= 0.55)
- Secondary: Precision, Recall, AUC-ROC vs random baseline

**Baselines**:
- Random feature pair selection (AUC-ROC baseline)
- Co-occurrence thresholding without clustering

**Go/No-Go**: If F1 < 0.5 on ANY model, trigger PIVOT to backup candidates.

### 3.2 E2: UAD Multi-Seed Robustness

**Objective**: Verify UAD results are stable across random seeds.

**Method**:
1. Run UAD on GPT-2 Small with 3 seeds (42, 123, 456)
2. Each seed uses different token sampling from OpenWebText
3. Compute F1, precision, recall per seed
4. Report mean, std, and 95% CI

**Metrics**:
- Mean F1 >= 0.6, std <= 0.1
- Spearman correlation of feature pair rankings across seeds

### 3.3 E3: DFDA Scaling (8+ Pairs, 2 Models)

**Objective**: Scale DFDA from 4 pairs (Iter 1) to >=8 pairs across 2 models.

**Method**:
1. Identify absorbed pairs using UAD (not just Chanin labels)
2. For each pair (parent p, child c):
   - Collect training examples where c fires but p does not (or fires weakly)
   - Train tiny MLP: input=child activation, output=predicted parent residual
   - MLP architecture: 2 layers, 64 hidden units, ReLU, ~97 params per pair
3. At inference: add MLP(z_c) to z_p
4. Evaluate: per-pair residual MSE, reconstruction MSE change

**Metrics**:
- Mean MSE improvement >= 10%
- >= 60% of pairs show positive improvement
- Reconstruction MSE change < 2%
- Parameter count ratio < 0.01% of SAE params

**Baselines**:
- No-compensation baseline
- Random residual injection (control for spurious improvement)

### 3.4 E4: End-to-End Pipeline Validation

**Objective**: Verify that UAD+DFDA improves downstream sparse probing accuracy.

**Method**:
1. Run UAD to identify absorbed concepts
2. Run DFDA on identified pairs
3. Train linear probes on SAE features for first-letter concepts (before/after DFDA)
4. Compare probe AUROC on absorbed vs non-absorbed concepts

**Metrics**:
- Probe accuracy improvement on absorbed concepts >= 5 percentage points
- Non-absorbed concept accuracy should not change significantly

**Control**: Non-absorbed concept accuracy (should be stable).

### 3.5 E5: True Absorption Validation (Supplementary)

**Objective**: Validate that collision rate correlates with true Chanin absorption.

**Method**:
1. Implement Chanin et al. absorption detection for vowel hierarchy (vowel -> {a,e,i,o,u})
2. Compute true absorption rate and collision rate on same SAE
3. Measure Spearman correlation

**Metrics**:
- Correlation r > 0.3 (minimum for proxy validity)

## 4. Ablation Studies

### 4.1 UAD Ablations

| Ablation | Modification | Purpose |
|----------|-------------|---------|
| No clustering | Pure co-occurrence thresholding | Test clustering contribution |
| No phi coefficient | Use raw co-occurrence counts | Test normalization contribution |
| No dead feature filtering | Include all features | Test dead feature impact |
| Single-link clustering | Replace Ward with single-link | Test linkage sensitivity |

### 4.2 DFDA Ablations

| Ablation | Modification | Purpose |
|----------|-------------|---------|
| Linear only | Single linear layer (no hidden) | Test nonlinearity need |
| Larger MLP | 256 hidden units | Test capacity sensitivity |
| No residual | Predict parent from child without residual connection | Test residual formulation |

## 5. Evaluation Benchmarks

### 5.1 Concept Sets

| Domain | Concepts | Source | Validation Role |
|--------|----------|--------|-----------------|
| First letters | a-z | Chanin et al. baseline | Primary validation |
| Vowels | {a,e,i,o,u} under "vowel" | Chanin hierarchy | True absorption validation |
| Numbers | one-ten | Basic number terms | Secondary generalization |

### 5.2 Baselines

| Baseline | Purpose |
|----------|---------|
| Random pair selection | Null model for UAD detection |
| Co-occurrence thresholding (no clustering) | Ablation baseline |
| No-compensation | DFDA baseline |
| Random residual injection | Control for spurious DFDA improvement |

## 6. Expected Visualizations

- **Figure 1**: UAD precision-recall curve across threshold values
- **Figure 2**: Cross-model F1 comparison bar chart (GPT-2 Small, Gemma-2B, Pythia-2.8B)
- **Figure 3**: Multi-seed F1 stability (mean + error bars across 3 seeds)
- **Figure 4**: DFDA recovery rate per pair (bar chart with baseline)
- **Figure 5**: End-to-end pipeline: probe accuracy before/after DFDA
- **Figure 6**: Ablation study results (grouped bar chart)
- **Table 1**: Main results comparison (model x metric matrix)
- **Table 2**: DFDA per-pair results (pair, MSE before, MSE after, improvement %)

## 7. Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| UAD F1 drops on larger models | Medium | **Critical** | Go/No-Go gate at F1=0.5; pivot to backup candidates if failed |
| Gemma-2B SAE loading fails | Medium | High | Fallback to Pythia-2.8B; if both fail, scope to GPT-2 Small with multi-seed |
| DFDA improvement not significant | Low | High | Pre-register analysis; report negative result if occurs |
| Chanin protocol implementation fails | Low | Medium | Use collision rate as proxy with explicit caveat |
| Co-occurrence clustering finds correlation, not absorption | Medium | High | Validate against Chanin labels; if precision is random, hypothesis falsified |
| Dead feature ratio obscures patterns | Medium | Medium | Filter dead features before clustering; report effective feature count |

## 8. Reproducibility

- All random seeds fixed (42, 123, 456)
- Exact package versions pinned in requirements.txt
- Code open-sourced with Jupyter notebooks for each experiment
- Pre-trained SAE checkpoints referenced by SAELens release ID
- Evaluation data and concept lists included in repository
- Bootstrap 95% CI reported for all key metrics
