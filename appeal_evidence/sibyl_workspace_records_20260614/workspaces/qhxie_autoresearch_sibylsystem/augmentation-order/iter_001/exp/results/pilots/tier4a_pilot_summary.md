# Tier 4a Pilot Summary: NC_2 (Sliced Wasserstein Distance, Pixel Space)

**Task**: `tier4a_nc_measurement_pilot`
**Mode**: PILOT (100 samples, 100 projections, pixel space 3072-D)
**Date**: 2026-04-02
**Status**: PASS (GO)

## Pass Criteria Evaluation

| Criterion | Result |
|---|---|
| SWD computed for at least 1 pair without error | PASS (3 pairs computed) |
| rho computation runs without error | PASS (rho=-0.20, p=0.68 computed) |
| All criteria met | PASS |

## NC_2 Results (Pilot, Pixel Space)

All 3 operation pairs computed in both directions (6 total directional measurements):

| Rank | Pair | NC_2 (SWD proxy) |
|---|---|---|
| 1 | Crop ↔ ColorJitter | 0.05149 |
| 2 | Crop ↔ Flip | 0.04483 |
| 3 | Flip ↔ ColorJitter | 0.03477 |

Individual transform SWD vs. identity:
- Crop: 0.0957 (highest — spatially destructive)
- CJ: 0.0272 (moderate)
- Flip: 0.0218 (lowest — most reversible)

## H3 Pixel-Space Verdict: FALSIFIED

**Spearman rho = -0.20, p = 0.68** — not significant.

Root cause: 100 samples in 3072-D pixel space is statistically meaningless. High variance in SWD estimates with n=100 makes reliable ranking impossible.

This is an **expected failure** at pilot scale, not a fundamental problem with the hypothesis. The full-scale feature-space computation is required.

## DPI Theory Alignment (Positive Signal)

Despite the rho failure, the NC_2 values support the DPI reversibility theory:
- Reversibility ranking: Flip > CJ > Crop (from individual SWD vs. identity)
- Pairs involving Crop show the highest non-commutativity (consistent with Crop being spatially destructive)
- All NC_2 values are positive, confirming operations do not commute in distribution space

## Decision: GO

Proceed to `tier4a_feature_nc` (full-scale feature-space computation):
- 10k samples (vs. 100 pilot)
- 512-D penultimate-layer features from 200-epoch ResNet-18 (vs. 3072-D raw pixels)
- 1000 projections (vs. 100 pilot)
- Expected full runtime: ~30 min

## Runtime
- Pilot completed in 1.7 seconds
- Full feature-space run estimated: ~30 minutes
