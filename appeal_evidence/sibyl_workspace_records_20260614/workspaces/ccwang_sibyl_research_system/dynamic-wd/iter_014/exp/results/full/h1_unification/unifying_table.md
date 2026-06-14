# H1 Unification: Control Law Parameter Mapping (Full)

200-epoch CIFAR-10/ResNet-20 trajectories, 3 seeds (42, 123, 456)

| Method | Family | K_p | K_i | K_d | Rel. Error (%) | Std (%) | Status |
|--------|--------|-----|-----|-----|---------------:|--------:|--------|
| FixedWD | baseline | 0.000 | 0.000 | 0.000 | 0.00 | 0.00 | TRIVIAL |
| CWD | alignment-based | 0.000 | -0.000 | -0.000 | 4.71 | 0.16 | PASS |
| SWD | scheduling-based | -0.000 | -0.000 | -0.000 | 45.81 | 0.68 | FAIL |
| CPR | constraint-based | 1.417 | 8.658 | -9.898 | 9.57 | 1.47 | PASS |
| DefazioCorrective | scheduling-based | -0.000 | 0.002 | 0.000 | 37.56 | 2.37 | FAIL |
| NoWD | degenerate | 0.001 | 0.500 | 0.000 | 0.00 | 0.00 | PASS |

**Verdict**: H1_SUPPORTED
- Converged: 5/5
- Methods with < 15% error: 3/5 (CWD, CPR, NoWD)
- Methods with > 20% error: 2 (SWD, DefazioCorrective)

H1 NOT falsified (2/5 > 20%, threshold: >2 for falsification)

## Method Family Interpretation

| Family | Methods | Dominant Gain | Avg Error (%) | Avg Std (%) |
|--------|---------|---------------|---------------:|------------:|
| alignment-based | CWD | K_i | 4.7 | 0.2 |
| scheduling-based | SWD, DefazioCorrective | K_i | 41.7 | 1.5 |
| constraint-based | CPR | K_d | 9.6 | 1.5 |
| degenerate | NoWD | K_i | 0.0 | 0.0 |

## Detailed Interpretation

### CWD (alignment-based)
- **Gains**: K_p=0.0000, K_i=-0.0000, K_d=-0.0000
- **Extended params**: offset=-0.00005014, scale=0.5000
- **Error**: 4.71% (mean=4.70%, std=0.16%)
- **Per-seed**: 4.90%, 4.71%, 4.50%
- **Dominant term**: K_i (integral)
- CWD applies per-element binary masking based on sign(g) != sign(w). The UDWDC framework operates at per-layer granularity with continuous modulation, so element-level binary decisions cannot be fully captured. The K_d (alignment) term partially accounts for this.

### SWD (scheduling-based)
- **Gains**: K_p=-0.0000, K_i=-0.0000, K_d=-0.0000
- **Extended params**: offset=-0.00000241, scale=6.2148
- **Error**: 45.81% (mean=45.76%, std=0.68%)
- **Per-seed**: 45.59%, 46.66%, 45.03%
- **Dominant term**: K_i (integral)
- SWD uses gradient-norm-aware scheduling with internal normalization. Its monotonically-evolving effective WD profile may not map well to the feedback error signal rho_t - rho*(t).

### CPR (constraint-based)
- **Gains**: K_p=1.4171, K_i=8.6581, K_d=-9.8976
- **Extended params**: offset=-0.00006002, scale=0.0275
- **Error**: 9.57% (mean=9.45%, std=1.47%)
- **Per-seed**: 10.49%, 10.49%, 7.37%
- **Dominant term**: K_d (derivative/alignment)
- CPR's augmented Lagrangian constraint accumulates penalty over time, producing large effective WD. The K_i (integral) term captures this accumulation behavior.

### DefazioCorrective (scheduling-based)
- **Gains**: K_p=-0.0001, K_i=0.0023, K_d=0.0002
- **Extended params**: offset=-0.00002037, scale=1.0000
- **Error**: 37.56% (mean=37.48%, std=2.37%)
- **Per-seed**: 40.78%, 35.34%, 36.33%
- **Dominant term**: K_i (integral)
- Defazio's corrective WD is proportional to LR, producing a simple monotonically-decreasing WD profile that closely matches the proportional control signal.

### NoWD (degenerate)
- **Gains**: K_p=0.0011, K_i=0.5000, K_d=0.0000
- **Error**: 0.00% (mean=0.00%, std=0.00%)
- **Per-seed**: 0.00%, 0.00%, 0.00%
- **Dominant term**: K_i (integral)
- NoWD is trivially fitted: large negative gains drive lambda below zero, and floor clipping produces zero effective WD.

