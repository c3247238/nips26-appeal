# Methodology: Feature Absorption as Optimal Compression

## Overview

This document describes the experimental methodology for the paper "Feature Absorption as Optimal Compression: Evidence that SAEs Correctly Handle Hierarchical Features." All primary experiments (H1-H6) have been completed across iterations 1-8. This plan covers the two remaining exploratory hypotheses (H9, H10) and consolidates the full experimental design for paper writing.

## Model and SAE Configuration

- **Primary Model**: GPT-2 Small (85M parameters)
- **SAE Release**: `gpt2-small-res-jb` (24,576 latents, d_model=768)
- **Layers Tested**: 0, 4, 8, 10 (hook_resid_pre)
- **Features**: 26 first-letter features (A-Z)
- **Samples per feature**: 100
- **Seed**: 42
- **Device**: Single GPU (NVIDIA, ~12-16GB VRAM sufficient)

## Completed Experiments (H1-H6)

### Phase 1: Absorption Detection (COMPLETED)

**Method**: Chanin et al. differential correlation metric on 26 first-letter features.

For each first-letter feature (e.g., "A"), we:
1. Generate 100 prompts containing words starting with that letter
2. Run model forward pass and extract SAE features at target layer
3. Identify the top-activating latent for the feature
4. Compute absorption rate = fraction of child prompts where parent latent does NOT fire but child DOES

**Results**: Mean absorption 2.1-3.9% across layers; max 24.2% (Feature U at layer 8).

### Phase 2: Downstream Task Evaluation (COMPLETED)

#### Feature Steering
- Strengths tested: [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
- Metric: relative probability lift of target letter tokens
- Random baseline subtraction for delta-corrected analysis
- Success criterion: top-5 token contains target letter

#### Sparse Probing
- k-sparse linear probes at k=1, 5, 10, 20
- Precision-recall decomposition
- F1 score as summary metric

#### EC50 Analysis
- Dose-response curve fitting via linear interpolation
- Hill equation: S(s) = S_max * s^n / (EC50^n + s^n)
- Correlation with absorption rate

**Results**: No significant correlation (H1-H4) after multiple comparison correction (12 tests, Bonferroni alpha=0.00417).

### Phase 3: Inhibition Graph Validation (COMPLETED, FALSIFIED)

**Method**:
1. Construct decoder correlation graph: G[i,j] = W_dec[i] dot W_dec[j]
2. For each first-letter feature, extract top-20 neighbors by correlation
3. Validate against Chanin absorption pairs

**Result**: 0/520 predictions correct. Precision@20 = 0.0. Fisher p = 1.0.

### Phase 4: Precision-Recall Decomposition (COMPLETED)

**Result (H5)**: Precision = 1.0 universally at k >= 5; recall varies (0.05-1.0). Absorption affects coverage, not selectivity.

---

## Remaining Experiments (H9, H10)

### H9: Co-occurrence Strength vs. Absorption Rate

**Hypothesis**: Features with stronger parent-child co-occurrence (higher p_11) exhibit higher absorption rates.

**Method**:
1. For each first-letter feature, generate 100 child prompts (words starting with that letter)
2. For each prompt, check if the parent feature (first-letter latent) fires
3. Compute p_11 = fraction of child prompts where parent also fires
4. Test correlation: absorption_rate vs. p_11
5. Prediction: r > 0.3, p < 0.05

**Falsification criterion**: r < 0.2 or p > 0.05.

**GPU count**: 1
**Estimated time**: 25 minutes
**Multi-GPU strategy**: single

### H10: Random SAE Baseline

**Hypothesis**: Random SAE baselines (frozen decoder weights) exhibit absorption-like patterns, confirming absorption is partially a structural artifact.

**Method**:
1. Load pre-trained SAE from SAELens (`gpt2-small-res-jb`)
2. Create random baseline by freezing decoder weights at initialization (or use orthogonal random matrix)
3. Run Chanin absorption metric on same 26 first-letter features
4. Compare: trained vs. random absorption rates
5. Prediction: Random SAE shows non-zero absorption (structural artifact)

**Implementation**:
```python
# Random SAE: freeze decoder, use random encoder
random_sae = copy.deepcopy(sae)
random_sae.W_dec = torch.nn.Parameter(torch.randn_like(sae.W_dec))
random_sae.W_dec.data = F.normalize(random_sae.W_dec.data, dim=1)
random_sae.W_enc = torch.nn.Parameter(torch.randn_like(sae.W_enc))
# Run same absorption detection pipeline
```

**GPU count**: 1
**Estimated time**: 20 minutes
**Multi-GPU strategy**: single

---

## Baselines

1. **Random steering baseline**: Mean success = 0.344 (L4), 0.379 (L8). Used for delta-corrected analysis.
2. **Multiple comparison correction**: Bonferroni (alpha=0.00417) and BH-FDR (q<0.05) applied to all 12 tests.
3. **Cross-layer validation**: Tests repeated at L4 and L8; opposite-sign slopes falsify H3.
4. **Cross-model validation**: Pythia-70M pilot; inconclusive due to limited feature overlap.
5. **Random SAE baseline** (H10): Tests whether absorption is structural or learned.

## Evaluation Metrics

| Metric | Description | Used For |
|--------|-------------|----------|
| Absorption rate | Fraction of child prompts where parent latent is suppressed | Primary detection |
| Steering success | Fraction of prompts where target letter appears in top-5 | Downstream evaluation |
| Probing F1 | k-sparse linear probe F1 score | Downstream evaluation |
| Precision@k | Fraction of top-k predictions that are true absorption pairs | Graph validation |
| EC50 | Median effective steering strength | Dose-response analysis |
| Pearson r / Spearman rho | Correlation coefficients | Hypothesis testing |
| Bonferroni p / BH-FDR q | Multiple comparison correction | Statistical rigor |

## Expected Visualizations

- **Figure 1**: Absorption rate distribution across layers (bar chart)
- **Figure 2**: Steering success vs. absorption rate scatter plot (L4 and L8)
- **Figure 3**: Precision-recall decomposition (precision invariant, recall variable)
- **Figure 4**: EC50 vs. absorption rate scatter plot
- **Figure 5**: Inhibition graph precision@k curve (flat at 0)
- **Figure 6**: Rate-distortion schematic (conceptual diagram)
- **Table 1**: Hypothesis testing summary (all 12 tests with corrected p-values)
- **Table 2**: Feature-level absorption and steering data

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|----------|
| H9 produces null result | Medium | Low | Report honestly; null result is still informative |
| H10 random SAE has no detectable features | Low | Medium | Use feature-agnostic metric or synthetic data fallback |
| Paper dismissed as "we found nothing" | High | High | Strong framing: honest null results + methodological contributions + falsified hypothesis |
| Single-model limitation questioned | Medium | Medium | Acknowledge; Pythia cross-validation attempted; frame as pilot |

## Shared Resources

```json
{
  "shared_resources": [
    {"type": "dataset", "name": "first-letter prompts", "path": "exp/code/prompts"},
    {"type": "checkpoint", "name": "gpt2-small-res-jb SAE", "path": "SAELens pretrained"},
    {"type": "code", "name": "absorption detection", "path": "exp/code/full_absorption_gpt2.py"},
    {"type": "code", "name": "steering + probing", "path": "exp/code/full_steering_probing_gpt2.py"},
    {"type": "code", "name": "correlation analysis", "path": "exp/code/correlation_analysis.py"},
    {"type": "code", "name": "precision-recall", "path": "exp/code/precision_recall_analysis.py"},
    {"type": "code", "name": "EC50 analysis", "path": "exp/code/ec50_analysis.py"}
  ]
}
```

## Dependencies

- `torch>=2.0.0`
- `transformer-lens>=2.0.0`
- `sae-lens>=0.5.0`
- `numpy`, `scipy`, `matplotlib`, `seaborn`
- `tqdm`

All experiments are training-free (analysis of pretrained SAEs only).
