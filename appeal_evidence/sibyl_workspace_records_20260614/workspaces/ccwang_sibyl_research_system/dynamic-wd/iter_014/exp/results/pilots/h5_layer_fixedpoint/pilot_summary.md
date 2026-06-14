# H5: Per-Layer Fixed-Point Differentiation — Pilot Summary

## Hypothesis

Under alignment-modulated WD (UDWDC), per-layer gradient-to-weight ratios r*_l converge to layer-specific values that anti-correlate with per-layer steady-state alignment delta*_l. Under FixedWD, r* should be uniform across layers (CV < 0.15).

## Predictions and Thresholds

- CV(r*) under FixedWD < 0.15 (uniform r* across layers)
- Spearman(r*, delta*) under UDWDC/UDWDC-v2 < -0.3 (anti-correlation)
- Falsification: Spearman rho > -0.3 falsifies H5

## Results by Architecture

| Model | Norm | CV(r*) FixedWD | Spearman UDWDC | Spearman UDWDC-v2 | UDWDC Pass | v2 Pass |
|-------|------|----------------|----------------|-------------------|------------|---------|
| ResNet-50 | BN | 2.356 | -0.613 (p<0.001) | -0.161 (p=0.246) | YES | NO |
| ResNet-101 | BN | 3.485 | N/A | -0.785 (p<0.001) | N/A | YES |
| ViT-S/16 | LN | 2.615 | N/A | -0.318 (p=0.025) | N/A | YES (marginal) |

## Key Findings

1. **CV(r*) prediction REFUTED**: Under FixedWD, r* is NOT uniform across layers (CV ranges 2.36-3.49, far above 0.15). Even with constant WD, architecture effects create large per-layer ratio differences.

2. **UDWDC vs UDWDC-v2 divergence on ResNet-50**: Original UDWDC shows strong anti-correlation (rho=-0.61) but UDWDC-v2 does NOT (rho=-0.16). The EMA smoothing and floor clipping in v2 may suppress the differentiation effect. This is an important design tension.

3. **ResNet-101 shows STRONGEST differentiation**: UDWDC-v2 achieves rho=-0.79 on ResNet-101, contradicting earlier pilot findings. Deeper networks may have more pronounced per-layer differentiation.

4. **ViT marginal**: rho=-0.32 just barely passes the threshold, consistent with the hypothesis that BN architectures show stronger differentiation than LayerNorm.

5. **fc.weight outlier**: The final classification layer consistently has the highest r* and most negative delta*, driving Pearson correlations. Spearman (rank-based) is more appropriate.

## Caveats

- **Only 2 pilot epochs**: H5 is about steady-state fixed-point behavior (last 10 epochs of 90-200 epoch training). At 2 epochs, the network is far from equilibrium.
- **High CV under FixedWD may be transient**: The prediction of uniform r* under FixedWD may hold at convergence but not during early training.
- **UDWDC-v2 failure on ResNet-50 may be pilot artifact**: With only 2 epochs, the floor clipping dominates the effective WD, suppressing differentiation.

## Verdict

**GO** — All pass criteria met (data is computable, sufficient layers). The analysis pipeline works correctly and produces meaningful results. Full-training results (90-200 epochs) are needed to test H5 properly, as 2-epoch pilot data is far from steady-state equilibrium.

## Figures Generated

- `h5_scatter_resnet50_UDWDC.png` — r* vs delta* scatter, ResNet-50, UDWDC
- `h5_scatter_resnet50_UDWDC-v2.png` — r* vs delta* scatter, ResNet-50, UDWDC-v2
- `h5_scatter_resnet101_UDWDC-v2.png` — r* vs delta* scatter, ResNet-101, UDWDC-v2
- `h5_scatter_vit_s_16_UDWDC-v2.png` — r* vs delta* scatter, ViT-S/16, UDWDC-v2
- `h5_bar_resnet50_comparison.png` — Per-layer r* FixedWD vs UDWDC-v2, ResNet-50
- `h5_bar_resnet101_comparison.png` — Per-layer r* FixedWD vs UDWDC-v2, ResNet-101
- `h5_bar_vit_s_16_comparison.png` — Per-layer r* FixedWD vs UDWDC-v2, ViT-S/16
- `h5_cross_architecture_comparison.png` — Cross-architecture CV and Spearman comparison
