# IGSD Ablation Study -- Pilot Summary

**Task**: igsd_ablation_refined | **Status**: GO | **Elapsed**: 76.1 min

## Part 1: T_draft Sweep (tau=0.9 fixed)

| T_draft | GSM8K Acc | Speedup | Acc Retention | Combined QAS |
|---------|-----------|---------|---------------|--------------|
| 16      | 0.440     | 2.50x   | 60.3%         | 1.26         |
| 32      | 0.495     | 1.71x   | 67.8%         | 1.03         |
| 48      | 0.535     | 1.22x   | 73.3%         | 0.82         |

**Finding**: T_draft dominates the speed-quality trade-off. Each doubling of T_draft loses ~0.6x speedup but gains ~7% accuracy retention.

## Part 2: tau Sweep (T_draft=32 fixed)

| tau  | GSM8K Acc | Speedup | Accept Rate | Combined QAS |
|------|-----------|---------|-------------|--------------|
| 0.7  | 0.485     | 1.78x   | 96.7%       | 1.07         |
| 0.85 | 0.495     | 1.73x   | 95.9%       | 1.05         |
| 0.9  | 0.495     | 1.71x   | 95.3%       | 1.03         |

**Finding**: tau has modest effect. All accept rates are very high (>95%), indicating that at T_draft=32, most tokens are already confident enough to pass any threshold.

## Part 3: Confidence Partitioning (T_draft=32)

| Config  | GSM8K Acc | Speedup | Accept Rate | Combined QAS |
|---------|-----------|---------|-------------|--------------|
| tau=0.0 | 0.480     | 2.00x   | 100%        | 1.20         |
| tau=0.9 | 0.495     | 1.71x   | 95.3%       | 1.03         |

**Finding**: The confidence gate **does not** improve QAS. Pure 32-step drafting (tau=0.0) achieves 16% higher QAS than draft+refine (tau=0.9). The refine phase adds +1.5% accuracy but loses 17% speed.

## Part 4: KL Divergence Profile

- **Samples**: 100/100 GSM8K (seed=42)
- **Pattern**: Monotonically increasing (NOT inverted-U)
- **Early KL** (steps 1-15): 0.061
- **Mid KL** (steps 16-47): 0.089
- **Late KL** (steps 48-63): 0.206
- **Peak**: step 62, KL=0.344
- **Entropy**: Monotonically decreasing from 3.57 to 0.73

**Finding**: H6 (inverted-U hypothesis) is **refuted**. KL increases throughout denoising as the model makes increasingly significant distribution changes in late steps.

## Key Implications

1. T_draft (not tau) is the dominant IGSD hyperparameter
2. Confidence partitioning adds overhead without benefit at T_draft>=32
3. Monotonic KL profile explains diminishing returns of late-stage refinement
4. IGSD should be reframed as "reduced-step denoising" rather than "speculative denoising with confidence partition"
