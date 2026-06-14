# Methodology: When Benchmarks Become Targets

## Overview

This study tests whether the SAEBench absorption metric measures learned SAE structure or is confounded by base-model geometry, probe artifacts, and co-occurrence correlation. We decompose the metric using random and PCA baselines, correlate absorption with downstream utility, and demonstrate co-occurrence sensitivity. All experiments are training-free (inference-only) on existing SAE checkpoints.

## Models and Checkpoints

**Base model**: Pythia-160M-deduped (EleutherAI), layer 8, resid_post.
**SAE architectures** (8 checkpoints from SAEBench, all trainer_0):
- Standard (L1-sparse autoencoder)
- TopK
- BatchTopK
- MatryoshkaBatchTopK
- GatedSAE
- JumpReLU
- PAnneal
- Random control (permuted decoder, trained encoder)

All checkpoints are available at `iter_002/downloaded_saes/`. Model: 160M params, activation_dim=768, dict_size=16384.

## Experiment 1: Metric Decomposition (H1)

**Goal**: Determine what fraction of the absorption score is attributable to learned SAE structure vs. confounds.

**Conditions**:
1. **Trained SAE** (Standard, TopK, Matryoshka) --- baseline
2. **Random-decoder SAE** (frozen permuted decoder, train only encoder) --- tests geometric confound
3. **PCA-basis SAE** (decoder = top-k principal components of activations) --- tests whether any structured basis works
4. **Fully random SAE** (both encoder and decoder random, matched for L0) --- tests base-model geometry

**Protocol**: For each condition, run the SAEBench absorption evaluator on first-letter, semantic-hierarchy, and non-hierarchy tasks. The random-decoder and fully-random conditions reuse the existing Random SAE from iter_002. The PCA-basis condition requires constructing a new SAE by replacing the decoder with the top-k principal components of the activation covariance matrix (computed from 2k tokens).

**Metrics**:
- Mean absorption score per condition per task type
- One-way ANOVA across conditions

**Analysis**: If random/PCA decoders achieve comparable absorption to trained decoders (p > 0.05), the metric measures geometry, not learned structure.

## Experiment 2: Utility Disconnect (H2)

**Goal**: Test whether absorption scores predict downstream utility.

**Downstream metrics** (from SAEBench public evaluation data where available):
1. **Sparse probing F1** --- concept detection accuracy
2. **RAVEL Cause/Isolation** --- causal disentanglement
3. **Feature consistency (PW-MCC)** --- stability across inputs
4. **Steering efficacy proxy** --- targeted perturbation via SAE latents

**Protocol**: For each of 8 SAEs, collect:
- First-letter absorption score (from existing iter_002 data)
- L0 sparsity (from SAE config)
- Reconstruction loss (CE loss score from SAEBench)
- Each downstream metric (from SAEBench public results or computed via SAELens)

Compute partial correlation: r(absorption, utility | L0, reconstruction). Test against null of r = 0.

**Falsification**: If |partial r| > 0.5 and p < 0.05 for any downstream metric, H2 is falsified.

## Experiment 3: Co-occurrence Confound (H3)

**Goal**: Test whether the absorption metric is sensitive to hierarchical containment or merely to co-occurrence correlation.

**Conditions** (already computed in iter_002):
1. **True hierarchies** (10 WordNet hypernym pairs)
2. **High co-occurrence non-hierarchies** (10 synonym/co-occurring pairs)
3. **Low co-occurrence non-hierarchies** (not yet tested --- new)

**New data needed**: 5 random unrelated word pairs with low co-occurrence (e.g., "zebra-quantum", "volcano-piano").

**Analysis**: Paired t-test comparing hierarchy vs. high co-occurrence non-hierarchy (already done: t = -4.75, p = 0.003). New: compare hierarchy vs. low co-occurrence non-hierarchy. If the metric is hierarchy-specific, low co-occurrence pairs should show lower absorption than hierarchies.

## Experiment 4: Random-Baseline-Corrected Architecture Comparison (H4)

**Goal**: Test whether reported absorption reductions reflect genuine improvement or geometric artifacts.

**Computation**: For each SAE, compute:
```
margin = (trained_score - random_score) / trained_score
```
for semantic-hierarchy absorption.

**Architecture groups**:
- Low-absorption: Matryoshka, Gated, PAnneal
- High-absorption: BatchTopK, TopK, JumpReLU

**Analysis**: Welch's t-test comparing margins across groups. If margins are similar (p > 0.05), reported absorption reductions are geometric artifacts.

## Baselines

1. **Raw absorption scores** (uncorrected) --- Expected to show low-absorption architectures as "better"
2. **Random-baseline margins** --- Expected to shrink or eliminate architecture gaps
3. **PCA-baseline margins** --- Expected to show similar pattern to random baselines
4. **Downstream utility metrics** --- Expected to show null or weak correlation with absorption

## Evaluation Benchmarks

All experiments use existing SAEBench infrastructure:
- First-letter absorption: Official SAEBench evaluator (26 letters)
- Semantic-hierarchy absorption: Custom pipeline (10 WordNet hierarchies)
- Non-hierarchy control: Custom pipeline (10 correlated pairs)
- Downstream metrics: SAEBench public evaluation results or SAELens inference

## Expected Visualizations

- **Figure 1**: Architecture diagram showing the decomposition pipeline
- **Table 1**: Main results --- absorption scores by condition (trained, random-decoder, PCA, fully random)
- **Figure 2**: Bar chart --- mean absorption by condition for each task type (first-letter, semantic, non-hierarchy)
- **Figure 3**: Scatter plot --- absorption vs. downstream utility metrics with partial correlation annotations
- **Figure 4**: Bar chart --- random-baseline-corrected margins by architecture group
- **Figure 5**: Heatmap --- per-hierarchy absorption scores across all SAEs
- **Figure 6**: Box plot --- probe AUROC distribution (ceiling effect demonstration)

## Statistical Methods

- One-way ANOVA for condition comparisons (E1)
- Partial Pearson correlation with bootstrap 95% CI (B=10,000) for utility prediction (E2)
- Paired t-tests for hierarchy vs. non-hierarchy (E3)
- Welch's t-test for architecture group comparison (E4)
- Kendall's tau for ranking consistency (H7)
- Paired t-test for model divergence (H6, GPT-2 vs Pythia)

## Risk Mitigation

| Threat | Mitigation |
|--------|------------|
| PCA SAE construction fails | Pre-compute PCs from 2k tokens; fallback to random if SVD unstable |
| H6 data missing | GPT-2 replication data already collected in iter_002; pure re-analysis |
| Downstream metrics unavailable | Use SAELens to compute sparse probing F1 directly; skip RAVEL if unavailable |
| Low co-occurrence pairs trivial | Verify probe AUROC < 0.7 for these pairs; if all probes perfect, the ceiling effect itself is evidence |
| Results are null | Frame as "absorption reduction does not improve validity" --- still publishable |

## Reuse of iter_002 Data

The following iter_002 results are reused directly:
- First-letter absorption scores for all 8 SAEs
- Semantic-hierarchy absorption scores for all 8 SAEs
- Non-hierarchy control absorption scores for all 8 SAEs
- GPT-2 replication results
- tau_fs robustness analysis

New experiments required:
- PCA-basis SAE construction and absorption evaluation
- Low co-occurrence non-hierarchy pairs
- Downstream metric collection/correlation
- Random-baseline margin computation and architecture comparison
