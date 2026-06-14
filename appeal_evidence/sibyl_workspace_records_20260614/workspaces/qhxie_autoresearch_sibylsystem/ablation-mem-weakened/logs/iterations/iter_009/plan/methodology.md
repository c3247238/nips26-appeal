# Methodology: Feature Absorption as Optimal Compression

## Overview

This project systematically investigates whether feature absorption in Sparse Autoencoders (SAEs) is a failure mode requiring mitigation, or a benign structural artifact that training reduces. We reframe absorption as rate-distortion optimal compression behavior.

## Model and SAE Configuration

| Component | Details |
|-----------|---------|
| Model | GPT-2 Small (124M parameters) |
| SAE | gpt2-small-res-jb (24,576 latents) |
| Layers | 0, 4, 8, 10 (hook_resid_pre) |
| Features | 26 first-letter features (A-Z) |
| Samples | 100 per feature |

## Phase 1: Absorption Detection

### Method
Chanin et al. differential correlation metric:
1. Generate 100 child prompts (words starting with target letter)
2. Measure parent latent firing probability with and without child suppression
3. Absorption rate = fraction of child prompts where parent does NOT fire but child DOES

### Results
- Mean absorption: 2.1-3.9% across layers
- Max absorption: 24.2% (Feature U at layer 8)
- Layer 4 mean: 3.91%
- Layer 8 mean: 3.38%

## Phase 2: Downstream Task Evaluation

### Feature Steering
- **Strengths tested**: [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
- **Metric**: Relative probability lift of target letter tokens
- **Random baseline**: Mean success = 0.344 (L4), 0.379 (L8)
- **Delta correction**: Subtract random baseline to isolate absorption-specific effects
- **Success criterion**: Top-5 token contains target letter

### Sparse Probing
- **Method**: k-sparse linear probes at k=1, 5, 10, 20
- **Decomposition**: Precision-recall analysis
- **Metric**: F1 score

### EC50 Analysis
- **Method**: Dose-response curve fitting via linear interpolation
- **Equation**: S(s) = S_max * s^n / (EC50^n + s^n)
- **Metric**: Correlation between EC50 and absorption rate

## Phase 3: Inhibition Graph Validation

### Method
1. Construct decoder correlation graph: G[i,j] = W_dec[i] dot W_dec[j]
2. Top-20 neighbors per first-letter feature
3. Validate against Chanin absorption pairs

### Results
- **FALSIFIED**: Precision@20 = 0.0 (0/520 predictions correct)
- Enrichment = 0.0x (predicted >= 25x)
- Fisher p = 1.0

## Phase 4: Random Baseline Comparison

### Method
1. Create random SAE (frozen orthonormal decoder, random encoder)
2. Run Chanin absorption metric on same 26 first-letter features
3. Compare trained vs. random SAE absorption

### Results
| Metric | Trained SAE | Random SAE |
|--------|-------------|------------|
| Mean | 0.034 | 0.278 |
| Std | 0.069 | 0.169 |
| Max | 0.242 | 0.676 |

- Difference: -0.244 (random > trained), p < 0.001
- Correlation between trained and random: r = 0.023, p = 0.913

## Statistical Rigor

| Aspect | Details |
|--------|---------|
| Total tests | 12 |
| Correction methods | Bonferroni (alpha = 0.00417), BH-FDR (q < 0.05) |
| Cross-layer validation | L4 and L8 |
| Cross-model validation | Pythia-70M pilot |

## Hypothesis Results Summary

| Hypothesis | Result | Key Evidence |
|------------|--------|--------------|
| H1: Absorption does not degrade steering | SUPPORTED (null) | r=+0.008 (L4), r=-0.301 (L8), p>0.05 |
| H1b: Delta-corrected steering | NOT SUPPORTED (after correction) | p=0.028 uncorrected, p=0.334 Bonferroni |
| H2: Absorption does not degrade probing | SUPPORTED (null) | r=-0.003 (L4), r=-0.107 (L8), p>0.05 |
| H3: Cross-layer consistency | SUPPORTED (null) | CV=1.079, opposite signs |
| H4: EC50 unaffected | SUPPORTED (null) | r=-0.166 (L4), r=+0.180 (L8), p>0.05 |
| H5: Precision invariant, recall variable | SUPPORTED | Precision=1.0, recall varies |
| H6: Decoder graph predicts absorption | FALSIFIED | precision@20=0.0 |
| H7: Trained < random absorption | SUPPORTED | trained=0.034, random=0.278, p<0.001 |
| H9: Co-occurrence predicts absorption | TAUTOLOGICAL | Excluded from paper |

## Rate-Distortion Interpretation

Under hierarchical co-occurrence and sparsity constraints, absorption minimizes rate (sparsity loss) while preserving decoder alignment (reconstruction quality):

| Finding | Interpretation |
|---------|---------------|
| Precision = 1.0 | Decoder alignment preserved; no false positives |
| Recall varies | Encoder coverage reduced; parent suppressed |
| Feature U (24.2% abs, 100% steering) | Information redistributed, not destroyed |
| H1-H4 null results | No downstream degradation in this regime |
| H6 falsified | Decoder correlations do not capture mechanism |
| H7 (trained < random) | Training reduces structural artifacts |

## Expected Visualizations

1. **Figure 1**: Absorption rate distribution across layers (bar chart, A-Z features)
2. **Figure 2**: Steering success vs. absorption rate (scatter plot, H1/H1b results)
3. **Figure 3**: Precision-recall decomposition (grouped bar/violin, H5)
4. **Figure 4**: Inhibition graph precision@k (line plot, H6 falsification)
5. **Figure 5**: Rate-distortion schematic (conceptual diagram)
6. **Figure 6**: Random vs. trained SAE absorption (box plot, H10)
7. **Table 1**: Hypothesis testing summary with corrected p-values
8. **Table 2**: Feature-level absorption and downstream data
9. **Table 3**: Rate-distortion interpretation of findings