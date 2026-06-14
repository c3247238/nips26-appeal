# H5: Per-Layer Fixed-Point Differentiation — Pilot Summary

## Hypothesis

Under alignment-modulated WD on networks with normalized layers, per-layer gradient-to-weight ratios r*_l converge to layer-specific values that anti-correlate with per-layer steady-state alignment delta*_l.

## Predictions

- CV(r*) under FixedWD < 0.15 (uniform r* across layers)
- Spearman(r*, delta*) under UDWDC-v2 < -0.3 (anti-correlation)

## Results by Architecture

| Model | Norm | CV(r*) FixedWD | Spearman rho UDWDC-v2 | H5 Pass? |
|-------|------|----------------|----------------------|----------|
| resnet50 | BatchNorm | 2.3558 | -0.1605 | NO |
| resnet101 | BatchNorm | 3.4850 | -0.7852 | YES |
| vit_s_16 | LayerNorm | 2.6146 | -0.3177 | YES |

## Verdict

H5 holds for ResNet-50 (BN architecture) but also holds for ResNet-101 and also holds for ViT (LayerNorm).

**Recommendation**: H5 does not hold even for ResNet-50. Consider revising hypothesis.

**Pilot GO/NO-GO**: GO
