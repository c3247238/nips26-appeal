# RACFG Pilot Summary — Countdown-16

## Task: pilot_racfg
- **Date**: 2026-03-10
- **GPU**: 2 (NVIDIA RTX PRO 6000 Blackwell Server Edition)
- **Elapsed**: 2.8 minutes
- **Verdict**: **NO-GO** (stability variance not meaningful)

## Configuration
- remask_pct: 10%
- w_base: 1.0, w_max: 2.0
- schedule_type: threshold_70_30
- stability_ema_lambda: 0.7
- 16 samples, seed 42, 128 steps, gen_len=256, temperature=0.4

## Results

| Method  | Accuracy | rep-2  | rep-3  | distinct-3 | Avg Time |
|---------|----------|--------|--------|------------|----------|
| Vanilla | 0.0%     | 0.1401 | 0.0786 | 0.8755     | 3.7s     |
| RACFG   | 0.0%     | 0.1377 | 0.0786 | 0.8781     | 6.6s     |

## Stability Analysis
- Overall stability mean: 0.9970 (very high — predictions barely change step-to-step)
- Overall stability std: 0.0061 (very low variance)
- Per-sample stability std mean: 0.0111
- **Variance meaningful**: False (std < 0.01 threshold)

## Key Findings

1. **Both methods achieve 0% accuracy on Countdown-16** — this is consistent with prior iterations showing Dream-7B struggles with Countdown at small sample sizes (vanilla baseline was 4.7% on Countdown-500 in iteration 3).

2. **Stability scores are uniformly high (~0.997)** — this means the JSD between consecutive denoising steps is near zero. The model's token-level probability distributions are already very stable across steps, leaving no meaningful "instability signal" for RACFG to exploit.

3. **No degeneration** — RACFG does not harm generation quality (rep-2/rep-3/distinct-3 all comparable to vanilla).

4. **~1.8x compute overhead** — RACFG takes ~6.6s vs vanilla 3.7s per sample due to the extra forward pass for unconditional input.

## Root Cause Analysis

The stability-based re-masking strategy assumes that cross-step JSD reveals "uncertain" or "reasoning-critical" positions. However, on Dream-7B:
- The EMA-smoothed probability distributions converge very quickly (lambda=0.7 is too aggressive)
- By the time guidance kicks in (mask_rate < 70% per threshold_70_30 schedule), most positions have already stabilized
- The JSD signal is dominated by noise, not meaningful uncertainty

## Recommendations

1. **Reduce EMA lambda** to 0.3-0.5 to amplify step-to-step differences
2. **Test with raw JSD** (no EMA smoothing) to see if the underlying signal exists
3. **Consider alternative instability signals**: entropy of current predictions, or rank changes in top-k tokens
4. **Test on Countdown-100** where baseline accuracy is nonzero for better signal
5. If stability signal remains uninformative, fall back to confidence-based re-masking (A-CFG style) which doesn't require cross-step history
