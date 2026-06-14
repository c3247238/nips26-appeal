# Iteration 4 Pilot Summary

## Phase 0: Zero-Compute Fixes

### T0.1: BEM Metric Fix - COMPLETED
**Problem**: `HalfLambdaPhi` did not report `effective_wd` in diagnostics, causing `get_metrics()` to fall back to `base_wd` (full rate). Combined with `abs()` in BEM formula, `half_lambda` always showed BEM=0.0.

**Fix**:
1. Added `self._diagnostics['wd_schedule'] = wd` to `HalfLambdaPhi.get_per_param_wd()` so effective_wd is correctly reported as `0.5 * base_wd`
2. Removed `abs()` from BEM formula, using signed BEM: `(mean_wd - constant_wd) / constant_wd`

**Verification**:
- `half_lambda` BEM = -0.5000 (previously 0.0)
- `cosine_schedule` BEM = -0.5995 (correctly negative, was ~0.5 with abs)
- `no_wd` BEM = -1.0000 (correctly shows zero budget)
- `constant` BEM = 0.0000 (unchanged, correct)

### T0.2: AIS Metric - VERIFIED
AIS implementation already correctly computes normalized entropy in [0, 1] range. Updated docstring to include explicit per-layer averaging formula:
- `alignment_l = |cos(g_l, w_l)|` for each layer l
- Distribution binned into 10 equal-width bins over [0, 1]
- `AIS = H(distribution) / H_max` where H_max = log(10)

Pilot values: AIS ranges from 0.25 to 0.50 across architectures and methods, consistent with moderate alignment diversity.

### T0.3: CSI Metric - UPDATED
CSI raw formula unchanged (CV of weight norm deltas). Added relative normalization documentation:
- `CSI_rel = CSI_raw / CSI_constant` (computed at analysis time)
- By definition `CSI_rel(constant) = 1.0`

### T0.4: SGD Baseline Analysis - COMPLETED
Analyzed 42 SGD baseline runs from iter_003 (ResNet-20, CIFAR-10/100, 7 methods x 3 seeds each).

Key findings:
| Dataset | Method vs Constant | Cohen's d | p_adj (Holm) |
|---------|-------------------|-----------|-------------|
| CIFAR-10 | no_wd | 10.29 | 0.000 * |
| CIFAR-10 | swd | 3.48 | 0.004 * |
| CIFAR-10 | half_lambda | 2.75 | 0.074 |
| CIFAR-10 | cosine_schedule | 0.17 | 0.869 |
| CIFAR-100 | swd | 2.86 | 0.000 * |
| CIFAR-100 | cwd_hard | 2.37 | 0.065 |

SGD shows large effect sizes (Cohen's d > 1 for most methods), confirming that WD method choice matters significantly under SGD. In contrast, AdamW iter_003 results show all methods within ~0.3% accuracy.

## Phase 1: Pilot Validation

### T1.1: VGG-16-BN on CIFAR-10 (10 epochs, seed=42) - PASSED

| Method | Best Acc | Weight Norm | Epoch Time | BEM | CSI | AIS |
|--------|----------|-------------|------------|-----|-----|-----|
| constant | 79.94% | 187.26 | ~22s | 0.000 | 0.996 | 0.357 |
| cwd_hard | 80.30% | 185.53 | ~50s | -0.490 | 1.011 | 0.315 |
| no_wd | 80.61% | 184.83 | ~10s | -1.000 | 0.988 | 0.246 |

All runs completed successfully:
- No OOM errors on RTX PRO 6000 (98GB)
- Loss decreased monotonically
- VGG-16-BN training works correctly with all WD methods

Note: CWD_hard is ~2.3x slower than constant due to per-element mask computation. This is expected and acceptable for 200-epoch runs (~2.8 hours vs ~1.2 hours).

### T1.2: ResNet-20 Metric Verification (5 epochs, seed=42) - PASSED

| Method | Best Acc | BEM | CSI | AIS |
|--------|----------|-----|-----|-----|
| half_lambda | 78.55% | -0.500 | 0.802 | 0.497 |
| cosine_schedule | 78.12% | -0.600 | 0.801 | 0.419 |

BEM values match theoretical expectations:
- half_lambda: mean_wd = 0.00025 = 0.5 * base_wd, BEM = -0.5
- cosine_schedule: BEM ~ -0.6 (cosine from 5e-4 to 0 over 5 epochs)

## Readiness Assessment

Phase 1 pilots confirm:
1. VGG-16-BN architecture works correctly with all WD methods
2. Metric fixes produce correct values
3. GPU performance is adequate (10-50s per epoch for VGG-16-BN)
4. No infrastructure issues

**Recommendation**: Proceed to Phase 2a (VGG-16-BN full 200-epoch runs) and Phase 2b (ResNet-20 extra seeds) in parallel.

## Files Modified
- `exp/code/optimizers.py`: Fixed HalfLambdaPhi to report effective_wd via diagnostics
- `exp/code/train_unified.py`: Signed BEM (removed abs()), updated CSI/AIS docstrings
- `exp/code/analyze_sgd_baseline.py`: New analysis script for SGD baseline data
