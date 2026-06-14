# Methodology: Iteration 2 — UAD Validation with Rigorous Baselines and Ablations

## 1. Overview

This iteration addresses the critical methodological gaps identified in the Iteration 1 critique. The focus is on **strengthening the UAD (Unsupervised Absorption Detection) claim** through:
1. **Random baselines** to anchor F1 claims
2. **Ablation studies** to justify design choices
3. **Statistical significance testing** (bootstrap CI, permutation tests)
4. **False positive analysis** to understand and potentially filter errors
5. **Fixed DFDA evaluation** using parent-positive protocol
6. **Cross-layer validation** (honestly reported, not masked by averages)

All experiments are **training-free** using pre-trained SAEs via SAELens.

Cross-model validation (Gemma-2B, Pythia-2.8B) remains blocked by gated access / missing SAEs. The paper will scope to GPT-2 Small with explicit discussion of this limitation.

## 2. Experimental Setup

### 2.1 Models and SAEs

**Primary Model**: GPT-2 Small (12 layers, 124M parameters)
- Hook point: `blocks.{layer}.hook_resid_post`
- SAE source: `gpt2-small-res-jb` (JumpReLU, pretrained, d_SAE=24,576)
- Layers evaluated: 4, 8, 10 (each reported individually)

**Rationale for single-model scope**: Cross-model validation was attempted but blocked. Multi-layer analysis provides internal validation. This limitation is explicitly acknowledged.

### 2.2 Dataset

- **Primary**: OpenWebText (10k tokens for full experiments)
- **Concept evaluation**: First-letter features (a-z) as primary validation set
- **Seeds**: 42 (primary), with bootstrap resampling for CI estimation

### 2.3 Software Stack

```
Base: Python 3.10+, PyTorch 2.0+
SAE: SAELens >= 2.0.0
Model: TransformerLens >= 2.0.0
Analysis: NumPy, SciPy, Pandas, scikit-learn, Matplotlib, Seaborn
```

## 3. Core Experiments

### 3.1 E1: UAD with Random Baseline (CRITICAL)

**Objective**: Compute UAD F1 with a proper random baseline for interpretation.

**Method**:
1. Run UAD on GPT-2 Small layer 8 (same as Iteration 1)
2. Randomly select N pairs from top 500 features (where N = number of same-cluster pairs found by UAD)
3. Repeat 100 times, compute mean precision/recall/F1
4. Report UAD F1 vs random baseline F1

**Metrics**:
- UAD F1 (expected: ~0.72 based on Iteration 1)
- Random baseline F1 (expected: near 0)
- Delta F1 = UAD_F1 - Random_F1
- Permutation test p-value

**Go/No-Go**: If UAD F1 - Random F1 < 0.3, the contribution is marginal.

### 3.2 E2: UAD Ablation Studies

**Objective**: Test whether each UAD design choice contributes to performance.

**Ablations**:
| Variant | Modification | Purpose |
|---------|-------------|---------|
| Full (UAD) | Phi + HAC Ward + top-500 + dead feature filter | Main method |
| A1: No clustering | Pure co-occurrence thresholding (top phi pairs) | Test clustering contribution |
| A2: No phi coefficient | Raw co-occurrence counts instead of phi | Test normalization contribution |
| A3: No dead feature filter | Include all features (not just active ones) | Test dead feature impact |
| A4: Single-link clustering | Replace Ward with single-link | Test linkage sensitivity |
| A5: K-means clustering | Replace HAC with K-means (k=50) | Test clustering algorithm |

**Metrics**: F1, precision, recall per variant. Report degradation from Full.

### 3.3 E3: UAD Cross-Layer Validation (Honest Reporting)

**Objective**: Test UAD across layers 4, 8, 10 with individual reporting.

**Method**:
1. Run UAD on layers 4, 8, 10 separately
2. Report F1 per layer individually (no averaging)
3. Analyze why layer 4 underperforms (if it does)

**Metrics**:
- F1 per layer
- Number of supervised labels found per layer
- Feature activity statistics per layer

**Expected**: Layer 8 F1 >= 0.6, Layer 10 F1 >= 0.5, Layer 4 F1 may be lower (honestly reported).

### 3.4 E4: False Positive Analysis

**Objective**: Analyze the 22 false positive pairs from Iteration 1 to understand error modes.

**Method**:
1. Run UAD on layer 8, collect all same-cluster pairs
2. Classify each pair: true positive, false positive, or false negative
3. For false positives, analyze:
   - Phi coefficient magnitude
   - Marginal activation frequencies
   - Semantic relationship (topical co-occurrence, shared super-absorber, random)

**Metrics**:
- False positive count and rate
- FP categorization breakdown
- Suggested post-hoc filters (e.g., phi threshold)

### 3.5 E5: Statistical Significance Testing

**Objective**: Add statistical rigor to UAD claims.

**Method**:
1. Bootstrap CI for F1: resample examples with replacement 1000 times
2. Permutation test: shuffle cluster assignments, compute null F1 distribution
3. Report 95% CI and p-value

**Metrics**:
- Bootstrap 95% CI for F1
- Permutation test p-value
- Power analysis (achieved power for detecting F1 > 0.5)

### 3.6 E6: DFDA with Parent-Positive Evaluation (METRIC REBUILD)

**Objective**: Fix the broken DFDA metric by evaluating on parent-positive examples.

**Method**:
1. Identify absorbed pairs using UAD (not just Chanin labels)
2. For each pair (parent p, child c):
   - Collect examples where parent SHOULD activate (based on ground truth concept)
   - Split into: (a) child also fires, (b) child does not fire
   - Train MLP on (a): predict parent activation from child activation
   - Evaluate on (a): does MLP improve parent recovery?
   - Evaluate on (b): measure generalization
3. Report parent-positive MSE improvement (not child-dominant)

**Metrics**:
- Parent-positive MSE improvement (target: > 20%)
- Reconstruction MSE change (target: < 2%)
- Generalization to child-absent examples

**Baselines**:
- No-compensation baseline
- Random residual injection
- Mean-prediction baseline (predict parent mean)

**CRITICAL**: If parent-positive improvement < 5%, DFDA is dropped from the paper.

### 3.7 E7: UAD with Alternative Correlation Metrics

**Objective**: Test whether phi coefficient is the best correlation metric for UAD.

**Method**:
1. Run UAD with alternative correlation metrics:
   - Pearson correlation
   - Mutual information (discretized)
   - Jaccard similarity
   - Pointwise mutual information (PMI)
2. Compare F1 across metrics

**Metrics**: F1, precision, recall per correlation metric.

## 4. Baselines

| Baseline | Purpose | Experiment |
|----------|---------|------------|
| Random pair selection | Null model for UAD | E1 |
| Co-occurrence thresholding (no clustering) | Ablation baseline | E2 |
| No-compensation | DFDA baseline | E6 |
| Random residual injection | Control for spurious DFDA improvement | E6 |
| Mean-prediction baseline | Control for trivial DFDA prediction | E6 |

## 5. Expected Visualizations

- **Figure 1**: UAD precision-recall curve across threshold values
- **Figure 2**: UAD F1 comparison: Full vs Random Baseline vs Ablations (grouped bar chart)
- **Figure 3**: Cross-layer F1 per layer (individual bars, no averaging)
- **Figure 4**: False positive categorization (pie chart or stacked bar)
- **Figure 5**: Bootstrap F1 distribution with 95% CI
- **Figure 6**: DFDA parent-positive recovery per pair (bar chart with baselines)
- **Table 1**: Main results comparison (method × metric matrix)
- **Table 2**: Ablation study results (variant × F1/precision/recall)
- **Table 3**: False positive analysis (pair × phi × category)

## 6. Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Random baseline F1 is high | Low | Critical | If random F1 > 0.3, UAD contribution is marginal; consider pivot |
| Ablations show no design choice matters | Low | High | If all variants achieve similar F1, method is robust but design choices are unimportant |
| DFDA parent-positive improvement < 5% | Medium | High | Pre-registered: drop DFDA from paper, focus on UAD only |
| Layer 4 F1 very low | Medium | Medium | Honestly report; discuss why absorption signatures are layer-dependent |
| Bootstrap CI includes F1 < 0.5 | Low | Medium | Report honestly; discuss sample size limitations |

## 7. Reproducibility

- All random seeds fixed (42)
- Exact package versions pinned in requirements.txt
- Code open-sourced with Jupyter notebooks for each experiment
- Pre-trained SAE checkpoints referenced by SAELens release ID
- Bootstrap 95% CI reported for all key metrics
- Permutation test p-values reported
