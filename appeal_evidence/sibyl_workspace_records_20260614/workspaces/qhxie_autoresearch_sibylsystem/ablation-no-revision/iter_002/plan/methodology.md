# Methodology: Quantifying Feature Absorption in Sparse Autoencoders

## Overview

This document details the experimental design to test five hypotheses about feature absorption in Sparse Autoencoders (SAEs). We adopt a **training-free analytical approach** leveraging pretrained SAEs from SAELens (GPT-2 small, `gpt2-small-res-jb` release). All experiments run on a single GPU with modest VRAM requirements (GPT-2 small + SAE fits comfortably on one consumer GPU).

## Experimental Setup

### Model and SAE Configuration
- **Base model**: `gpt2-small` (124M parameters, 12 layers, d_model=768)
- **SAE release**: `gpt2-small-res-jb` (SAELens pretrained residual-stream SAEs)
- **Layers audited**: 0, 2, 4, 6, 8, 10 (6 layers to cover shallow-to-deep spectrum)
- **Dictionary sizes**: 2K, 4K, 8K, 16K per layer (where available in SAELens)
- **Device**: single GPU (CUDA if available, else CPU fallback for small-scale pilots)

### Dataset
- **Analysis corpus**: 1,024 sequences of 128 tokens from `monology/pile-uncopyrighted` (standard SAELens evaluation set)
- **Pilot corpus**: 100 sequences of 128 tokens (seed 42)
- **Token frequency reference**: Token counts computed over the full Pile validation split

### Software Versions
- `transformer-lens>=2.0.0`
- `sae-lens>=0.5.0`
- `torch>=2.0.0`
- `numpy`, `scipy`, `matplotlib`, `seaborn`

---

## Core Metric: Absorption Score

For each latent feature `f`, we define:

1. **Activating tokens**: Tokens where `feature_f > threshold` (threshold = 1% of max activation across corpus).
2. **Co-firing latents**: For each activating token, the top-5 other latents with highest simultaneous activation.
3. **Reconstruction variance explained**: For a token's original residual-stream activation `x`, compute the SAE reconstruction `x_hat = decode(encode(x))`. Then compute the partial reconstruction using only feature `f` and its top-5 co-firers: `x_partial = W_dec[f] * act_f + sum_{c in top5} W_dec[c] * act_c`. The variance explained by the co-firers is `1 - var(x - x_partial) / var(x)`.
4. **Absorption score per latent**: Fraction of activating tokens for which the top-5 co-firers explain >80% of reconstruction variance.

**Rationale**: If a feature's activating tokens are almost perfectly reconstructed by other latents, the feature is not independently representing its concept—it is absorbed.

---

## Hypothesis Experiments

### H1: Prevalence of Feature Absorption
**Hypothesis**: >20% of latents in mid-to-deep layers show absorption score >0.5.
**Falsification**: Aggregate absorption rate <10% across layers 4-10.

**Experiment**:
1. Load SAEs for layers 0, 2, 4, 6, 8, 10 (dictionary size 8K).
2. Run corpus through model, cache residual activations.
3. Compute absorption score for every latent.
4. Report: (a) % latents with absorption score >0.5 per layer, (b) mean absorption score per layer, (c) histogram of scores.

**Baseline**: Random dictionary control—same decoder dimensions but random Gaussian weights, normalized per column. Compute absorption scores on this control to establish a null distribution.

---

### H2: Co-occurrence Frequency Drives Absorption
**Hypothesis**: Low-frequency token latents are absorbed at >=2x the rate of high-frequency token latents.
**Falsification**: Spearman r >= 0 between median token frequency and absorption score, OR lowest-frequency bin does not show >=2x higher absorption than highest-frequency bin.

**Experiment**:
1. For each latent, extract its top-20 activating tokens.
2. Compute median token frequency (log-count over Pile validation) for those tokens.
3. Bin latents into quartiles by median frequency.
4. Compare mean absorption score across quartiles.
5. Compute Spearman correlation between log-frequency and absorption score.

**Baseline**: Shuffle token-frequency mapping (permutation test) to verify the correlation is not spurious.

---

### H3: Sparsity Penalty Trade-off
**Hypothesis**: Higher L1 sparsity penalty (lambda) increases absorption rates monotonically.
**Falsification**: No monotonic increase; or higher-sparsity SAEs show lower absorption.

**Experiment**:
1. Use SAELens SAEs trained with varying L1 coefficients (if available in `gpt2-small-res-jb` release metadata).
2. If no explicit lambda sweep exists, proxy by average L0 (sparsity) per SAE: compute L0 on the pilot corpus and correlate with absorption rate.
3. Alternatively, train 3 small SAEs on layer 8 with L1 = [4e-5, 8e-5, 1.6e-4] for 10M tokens (fast training, ~30 min each) and measure absorption.

**Baseline**: Compare against the standard SAELens checkpoint (default lambda) as reference.

---

### H4: Downstream Impact on Circuit Discovery
**Hypothesis**: Circuits traced through high-absorption SAEs have faithfulness scores >=5 percentage points lower than low-absorption SAEs.
**Falsification**: Faithfulness difference <5 pp regardless of absorption level.

**Experiment**:
1. **Task**: Simple factual recall—"The capital of France is ___" (target: "Paris").
2. **Corruption**: Replace "France" with "Germany" (target shifts to "Berlin").
3. **Activation patching**: Patch SAE latents (or raw residual) from clean run into corrupted run at each layer.
4. **Faithfulness metric**: Fraction of layers/positions where patching restores >=50% of the clean→corrupted logit difference.
5. Compare: (a) raw residual patching, (b) SAE latent patching for a low-absorption layer, (c) SAE latent patching for a high-absorption layer.

**Baseline**: Raw residual stream patching (no SAE bottleneck).

---

### H5: Dictionary Size Effect
**Hypothesis**: Larger dictionary sizes reduce per-feature absorption rates.
**Falsification**: Absorption rate does not decrease with dictionary size; or correlation is positive.

**Experiment**:
1. Load SAEs for layer 8 with dictionary sizes 2K, 4K, 8K, 16K.
2. Compute absorption scores for all latents in each SAE.
3. Report mean absorption rate vs. dictionary size.
4. Fit a log-linear trend: absorption_rate ~ a * log(d_sae) + b.

**Baseline**: Same random dictionary control scaled to each dictionary size.

---

## Task Decomposition

| Task ID | Description | Type | GPU | Est. Time | Depends On |
|---------|-------------|------|-----|-----------|------------|
| setup_env | Install dependencies, verify SAELens/TransformerLens | setup | 0 | 5 min | — |
| setup_data | Download/cache Pile subset, compute token frequencies | setup | 0 | 10 min | setup_env |
| baseline_random | Generate random dictionary controls (2K-16K) | baseline | 1 | 5 min | setup_env |
| h1_pilot | H1 prevalence pilot (100 seqs, layer 8, 8K dict) | pilot | 1 | 10 min | setup_data |
| h1_full | H1 prevalence full (1,024 seqs, layers 0-10, 8K dict) | experiment | 1 | 25 min | h1_pilot |
| h2_pilot | H2 frequency correlation pilot (100 seqs, layer 8) | pilot | 1 | 10 min | h1_pilot |
| h2_full | H2 frequency correlation full (1,024 seqs, layers 0-10) | experiment | 1 | 25 min | h2_pilot |
| h3_pilot | H3 sparsity trade-off pilot (L0 proxy, layer 8) | pilot | 1 | 10 min | h1_pilot |
| h3_full | H3 sparsity trade-off full (multi-layer, or train 3 SAEs) | experiment | 1 | 45 min | h3_pilot |
| h4_pilot | H4 circuit faithfulness pilot (1 task, layer 8) | pilot | 1 | 10 min | h1_pilot |
| h4_full | H4 circuit faithfulness full (3 tasks, layers 4-10) | experiment | 1 | 30 min | h4_pilot |
| h5_pilot | H5 dictionary size pilot (layer 8, 2K/8K) | pilot | 1 | 10 min | h1_pilot |
| h5_full | H5 dictionary size full (layer 8, 2K-16K) | experiment | 1 | 20 min | h5_pilot |
| analysis | Aggregate results, generate figures, compute correlations | analysis | 0 | 15 min | h1-h5 full |

---

## Expected Visualizations

- **Figure 1**: Bar chart—absorption rate (% latents with score >0.5) per layer (H1).
- **Figure 2**: Box plot—absorption score distribution by token-frequency quartile (H2).
- **Figure 3**: Scatter plot—mean L0 vs. mean absorption rate per SAE checkpoint (H3).
- **Figure 4**: Heatmap—faithfulness score (layer x position) for raw vs. low-absorption vs. high-absorption SAE patching (H4).
- **Figure 5**: Line plot—mean absorption rate vs. dictionary size (log scale) with random control baseline (H5).
- **Table 1**: Summary table—hypothesis, operational metric, result, falsification status.

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| SAELens release lacks 2K/4K/16K dicts for GPT-2 | Fallback to available sizes; train missing sizes if time permits |
| High VRAM usage with large dict + full cache | Use selective caching (`names_filter`); process in batches |
| Activation patching on SAE latents is non-trivial | Use `HookedSAETransformer` from SAELens; test on pilot first |
| Token frequency counts unavailable | Compute on-the-fly from Pile validation (one-time, ~5 min) |
| H3 lambda sweep SAEs unavailable | Proxy via L0 correlation; train 3 mini-SAEs if needed |
