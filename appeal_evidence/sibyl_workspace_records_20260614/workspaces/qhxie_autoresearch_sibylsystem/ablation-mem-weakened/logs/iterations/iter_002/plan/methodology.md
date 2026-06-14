# Methodology: The Local Inhibition Graph -- A Neuroscience-Inspired Training-Free Diagnostic for Feature Absorption in Sparse Autoencoders

## Overview

This study proposes and validates the first connection between Rozell et al.'s Locally Competitive Algorithm (LCA) from neuroscience and feature absorption in Sparse Autoencoders (SAEs). We show that the decoder correlation matrix W_dec^T W_dec is exactly the inhibition matrix from the LCA framework, providing a mechanistic explanation for absorption as competitive suppression. We construct a local inhibition graph from decoder correlations and test whether graph edges predict known absorption pairs.

**All experiments are training-free** -- we analyze existing pre-trained SAEs using SAELens and custom evaluation code. No SAE training is required.

**Evidence-Driven Context:** This methodology follows a pivot from a prior correlation-study direction (iterations 1-8) that produced predominantly null results on downstream task degradation. The Local Inhibition Graph framework was selected because it (1) provides a genuinely novel theoretical contribution, (2) explains existing findings mechanistically, and (3) offers a practical diagnostic tool that is training-free and scalable.

## Research Questions

1. **RQ1 (Primary):** Does the local inhibition graph constructed from decoder correlations predict known absorption pairs with precision significantly above chance?
2. **RQ2 (Secondary):** Does the inhibition graph explain the precision-recall asymmetry observed in prior data (precision invariant, recall variable)?
3. **RQ3 (Secondary):** Can the inhibition graph predict which features are at risk of absorption before running the Chanin metric?
4. **RQ4 (Exploratory):** Does the graph structure vary systematically across layers, explaining layer-dependent effects?
5. **RQ5 (Exploratory):** Can homeostatic rebalancing along graph edges restore parent feature firing without degrading reconstruction?

## Hypotheses

| Hypothesis | Type | Statement | Expected Outcome | Falsification |
|------------|------|-----------|------------------|---------------|
| H6 | Primary | Graph edges predict absorption pairs | Precision@20 >= 0.10 (vs ~0.004 chance) | Precision@20 <= 0.05 |
| H7 | Secondary | Inhibition explains precision-recall asymmetry | r(recall, inhibition) < -0.3; r(precision, inhibition) ~ 0 | r(precision, inhibition) significant |
| H8 | Secondary | Graph predicts at-risk features | r(total_inhibition, absorption_rate) > 0.3, p < 0.05 | r < 0.2 or p > 0.05 |
| H9 | Exploratory | Layer-dependent graph structure | Mean edge weight increases with depth: r > 0.3 | No trend with layer |
| H10 | Exploratory | Homeostatic rebalancing restores parent firing | Parent firing +20%, recon error < 5% | Error > 5% or no improvement |

## Theoretical Foundation

### LCA-SAE Structural Correspondence

Rozell et al. (2008) proposed the Locally Competitive Algorithm (LCA) for sparse coding, where neurons compete via lateral inhibition. The dynamics are:

```
tau * du/dt = -u + (b - G * a)
a = T(u)
```

where:
- `u` is the membrane potential
- `b = W_enc^T * x` is the feedforward input
- `G = W_dec^T * W_dec` is the inhibition matrix
- `a` is the activation after thresholding `T`

**Key insight:** In an SAE with tied weights (W_dec = W_enc^T), the decoder correlation matrix W_dec^T W_dec is exactly the LCA inhibition matrix G. Even with untied weights, the structural correspondence holds: decoder correlations encode competitive relationships between latents.

### Competitive Suppression Explains Absorption

When a child feature (specific) and parent feature (general) co-occur:
- The child fires strongly (high activation)
- Via decoder correlation G_ij, the child inhibits the parent
- The parent fails to fire (recall loss)
- The parent's decoder direction is unchanged (precision preserved)

This explains the precision-recall asymmetry: inhibition suppresses true positives (recall loss) but does not cause false positives (precision preserved).

## Experimental Design

### Phase 1: Construct Local Inhibition Graph (H6)

For each latent i in the SAE decoder matrix W_dec (shape d_dict x d_model):
1. Compute decoder correlations: G_ij = <W_dec[i], W_dec[j]> for all j != i
2. Keep top-k neighbors per latent (k=20-50) with highest |G_ij|
3. Edge weight = G_ij (signed correlation)
4. Complexity: O(k * d_dict * d_model) --- feasible for 24K-1M latents

**Ground truth:** Use Chanin et al.'s absorption detection on first-letter features (A-Z) from prior experiments. For each absorption pair (parent latent i, absorbing latent j), check if j is in N(i).

**Metrics:**
- Precision@k: fraction of top-k neighbors that are absorption pairs
- Recall@k: fraction of absorption pairs found in top-k neighbors
- Fisher exact test for enrichment vs. random baseline
- AUPR (area under precision-recall curve) across k values

**Baselines:**
1. **Random graph:** Shuffle latent indices; expected precision@20 ~ 0.004 (20/24000)
2. **Non-absorbed control:** Test graph edges for correlated but non-absorbed pairs
3. **Identity graph:** Only self-loops; tests whether correlations beyond self-similarity matter

### Phase 2: Validate Precision-Recall Asymmetry Explanation (H7)

For each first-letter feature:
1. Compute total incoming inhibition: sum of |G_ij| for all neighbors j
2. Compute total outgoing inhibition: sum of |G_ij| for all neighbors i inhibits
3. Test correlation: total_inhibition vs. recall (predicted: negative)
4. Test correlation: total_inhibition vs. precision (predicted: none)

**Data source:** Existing probing results from prior experiments (precision/recall per feature at layers 4, 8).
**GPU required:** No (analysis only).

### Phase 3: Test At-Risk Feature Prediction (H8)

For each first-letter feature latent:
1. Compute graph statistics: degree, total_inhibition, centrality, clustering coefficient
2. Test correlation: each statistic vs. absorption_rate
3. Compare top quartile vs. bottom quartile absorption rates

**Prediction task:** Can we predict absorption rate from graph statistics alone (without running Chanin metric)?

### Phase 4: Layer-Dependent Graph Structure (H9)

Construct graphs for layers 0, 4, 8, 10 of GPT-2 Small:
1. Compute graph statistics per layer: mean edge weight, density, clustering coefficient
2. Test correlation with layer depth
3. Compare layer 8 (where H1b was significant) vs. other layers

**GPU required:** Yes (1 GPU, ~20 min for all 4 layers).

### Phase 5: Homeostatic Rebalancing (H10, Exploratory)

For input activation a:
1. Compute original latents: z = f(W_enc * a + b_pre)
2. Compute inhibition per latent: inh_i = sum_{j in N(i)} G_ij * z_j
3. Apply boost: z'_i = z_i + alpha * inh_i
4. Clip negative values
5. Constrain: ||a - W_dec * z'||_2 <= (1 + epsilon) * ||a - W_dec * z||_2

**Test:** Does rebalancing restore parent feature firing?
- Measure parent firing rate before/after on test prompts
- Measure reconstruction error increase
- Sweep alpha in [0.1, 0.5, 1.0, 2.0, 5.0]

**GPU required:** Yes (1 GPU, ~30 min).

### Phase 6: Cross-Model Validation (Optional)

Replicate primary experiments on Gemma-2-2B (GemmaScope SAE, 16K latents):
- Construct inhibition graph
- Validate against absorption pairs
- Test H6-H8

**Rationale:** Single model limits generalizability.
**GPU required:** Yes (1 GPU, ~30 min).
**Decision criteria:** Only run if GPU is readily available.

## SAE Configurations and GPU Requirements

| Model | Layer | Dict Size | Experiment | GPU | Est. Time |
|-------|-------|-----------|------------|-----|-----------|
| GPT-2 Small | 0 | 24K | Graph construction | 1 | ~5 min |
| GPT-2 Small | 4 | 24K | Graph construction + validation | 1 | ~10 min |
| GPT-2 Small | 8 | 24K | Graph construction + validation | 1 | ~10 min |
| GPT-2 Small | 10 | 24K | Graph construction | 1 | ~5 min |
| GPT-2 Small | all | 24K | H7, H8 analysis | 0 | ~10 min |
| GPT-2 Small | all | 24K | H10 rebalancing | 1 | ~30 min |
| Gemma-2-2B | 8 | 16K | Cross-validation (optional) | 1 | ~30 min |

**Total GPU time (primary):** ~1.5 hours
**Total GPU time (optional):** ~0.5 hours

## Metrics

### Graph Construction Metrics
- **Precision@k:** Fraction of top-k neighbors that are true absorption pairs
- **Recall@k:** Fraction of absorption pairs found in top-k neighbors
- **Fisher exact test:** Enrichment p-value vs. random baseline
- **AUPR:** Area under precision-recall curve

### Inhibition Statistics
- **Total incoming inhibition:** sum_j |G_ij| for neighbors j
- **Total outgoing inhibition:** sum_j |G_ij| for latents i inhibits
- **Graph density:** fraction of possible edges present
- **Clustering coefficient:** local clustering per node

### Precision-Recall Metrics
- **Precision:** Fraction of predicted positives that are true positives
- **Recall:** Fraction of true positives that are predicted
- **Correlation:** Pearson/Spearman between inhibition and recall/precision

### Rebalancing Metrics
- **Parent firing rate:** Fraction of prompts where parent feature fires
- **Reconstruction error:** ||a - W_dec * z'||_2 relative to original
- **Alpha sweep:** Optimal alpha value

## Baselines

1. **Random graph baseline:** Shuffle latent indices; expected precision@20 ~ 0.004.
2. **Non-absorbed pair control:** Test graph edges for correlated but non-absorbed pairs.
3. **Identity graph:** Only self-loops; tests whether correlations beyond self-similarity matter.
4. **Prior correlation data:** Use existing absorption rates, precision/recall, steering results from iterations 1-8 as validation targets.

## Expected Visualizations

- **Figure 1:** Architecture diagram showing LCA-SAE structural correspondence and inhibition graph construction
- **Table 1:** Main results -- precision@k, recall@k, Fisher test for H6 across layers
- **Figure 2:** Precision-recall curves for graph prediction at different k values
- **Figure 3:** Correlation plots -- total inhibition vs. recall (H7)
- **Figure 4:** Correlation plots -- total inhibition vs. precision (H7, predicted null)
- **Figure 5:** Bar chart -- graph statistics by layer (H9)
- **Figure 6:** Homeostatic rebalancing -- parent firing rate vs. alpha (H10)
- **Figure 7:** Homeostatic rebalancing -- reconstruction error vs. alpha (H10)
- **Table 2:** Hypothesis testing summary (H6-H10)

## Software Stack

- **SAELens:** Loading pre-trained SAEs
- **TransformerLens:** Model loading, activation caching
- **PyTorch:** Core tensor operations, graph construction
- **NumPy/SciPy:** Statistical analysis, correlation tests
- **Matplotlib/Seaborn:** Visualization
- **NetworkX (optional):** Graph statistics (clustering, centrality)

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Graph edges don't correspond to absorption pairs | Medium | High | Structural correspondence is mathematically exact. If edges don't match, report as finding about decoder correlation limitations. Fallback: diagnostic-only claims. |
| Homeostatic rebalancing degrades reconstruction | Medium | Medium | Alpha is tunable; sweep to find values that improve absorption without degrading reconstruction. Fallback: report Pareto frontier. |
| Local graph misses long-range absorption | Medium | Medium | Test multiple k values (10, 20, 50, 100). Fallback: hierarchical clustering. |
| Gemma-2-2B access issues | High | Medium | Primary experiments on GPT-2 Small; Gemma as validation only. |
| Neuroscience analogy criticized as superficial | Low | Medium | Ground claims in exact mathematical correspondence (W_dec^T W_dec = G_LCA), not metaphor. |

## Reproducibility

- All experiments use fixed random seed (42)
- All SAEs are from publicly available releases (SAELens)
- Code and evaluation protocol will be released
- Results reported with confidence intervals and exact p-values
- Graph construction is deterministic (no training randomness)
