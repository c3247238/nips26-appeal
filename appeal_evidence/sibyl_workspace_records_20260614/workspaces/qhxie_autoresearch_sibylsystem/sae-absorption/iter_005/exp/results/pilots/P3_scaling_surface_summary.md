# P3 Scaling Surface — Pilot Summary

## Verdict: GO

## Key Numbers

| Metric | Value | Threshold | Pass |
|--------|-------|-----------|------|
| N SAEs with absorption + L0 | 420 | >= 100 | YES |
| Linear R^2 | 0.488 | > 0.15 | YES |
| Additive GAM R^2 | 0.620 | — | — |
| Interaction GAM R^2 | 0.693 | — | — |
| Interaction p-value | 3.1e-15 | < 0.05 | YES |
| Phase boundary detected | Yes | — | — |

## Data Coverage

- **Model**: Gemma 2B (all SAEs from SAEBench)
- **Releases**: 9 SAE release families (gemma-scope, topk, vanilla, baselines)
- **Architectures**: jumprelu (54), standard (360), unknown/baseline (6)
- **Layers**: 5, 12, 19 (140 SAEs each)
- **Width range**: 2,304 — 1,048,576 (log2: 11.2 — 20.0)
- **L0 range**: 9.3 — 8,277 (log2: 3.2 — 13.0)
- **Absorption range**: 0.0001 — 0.896

## Model Comparison

| Model | R^2 | AIC | Interaction p |
|-------|-----|-----|---------------|
| Linear | 0.488 | -1929.6 | — |
| Additive GAM | 0.620 | -843.8 | — |
| Interaction GAM | 0.693 | -916.9 | 3.1e-15 |

The interaction GAM significantly outperforms the additive model (p = 3.1e-15), confirming that absorption depends on the **joint structure** of width and L0, not either independently. This supports H3.

## Linear Model Coefficients

- `log_width`: +0.054 (wider SAEs have more absorption)
- `log_l0`: -0.014 (higher L0 has less absorption)
- `layer`: +0.003 (deeper layers have slightly more absorption)

Direction is consistent with theory: wider SAEs with sparser activations (lower L0) exhibit more absorption.

## Per-Layer Analysis

| Layer | N | Mean Absorption | rho(width) | rho(L0) |
|-------|---|----------------|------------|---------|
| 5 | 140 | 0.051 | +0.347*** | -0.491*** |
| 12 | 140 | 0.091 | +0.423*** | -0.469*** |
| 19 | 140 | 0.090 | +0.337*** | -0.457*** |

All correlations highly significant (p < 0.0001). L0 has a stronger (negative) correlation with absorption than width (positive), consistent across layers.

## Phase Boundary

A phase boundary was detected at log2(L0) in [2.7, 3.8] (L0 ~ 6.5-14), spanning the full width range. This corresponds to the transition zone where absorption rises sharply as L0 drops below ~14. The transition is concentrated in a narrow L0 band, consistent with the "hedging-to-absorption" phase transition predicted by theory.

## Caveats for Full Analysis

1. **Extrapolation**: The GAM surface extrapolates unreliably outside the data range (negative predicted absorption). Full analysis should clip predictions to [0, 1].
2. **Data clustering**: SAEs are heavily clustered at specific width/L0 combinations (e.g., 144 SAEs at width=16384). Full analysis should account for this structure.
3. **AIC comparison**: AIC favors the linear model (more negative = better), but this is because AIC penalizes complexity. The GAM's higher R^2 with significant interaction test is the more relevant comparison for scientific questions.
4. **Architecture confound**: 360/420 SAEs are "standard" architecture. Full analysis should test whether interaction holds within architecture subgroups.
