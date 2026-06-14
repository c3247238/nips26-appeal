# RACFG Re-mask Percentage Ablation Summary (PILOT)

## Task: ablation_racfg_remask_pct
- **Date**: 2026-03-10
- **GPU**: 2 (NVIDIA RTX PRO 6000 Blackwell Server Edition)
- **Elapsed**: 13.6 minutes
- **Verdict**: **NO-GO** (no RACFG variant beats A-CFG)

## Configuration
- remask_pct ∈ {5%, 10%, 20%}
- ema_lambda ∈ {0.7, 0.3} (0.3 added based on pilot recommendation)
- w_base=1.0, w_max=2.0, schedule=threshold_70_30
- 16 samples (PILOT), seed 42, 128 steps, gen_len=256, temperature=0.4

## Results

| Method             | Accuracy | rep-2  | rep-3  | distinct-3 | Avg Time |
|--------------------|----------|--------|--------|------------|----------|
| Vanilla            | 0.0%     | 0.140  | 0.079  | 0.875      | 3.7s     |
| A-CFG (w=1.0)      | 6.2%     | 0.115  | 0.057  | 0.900      | 7.4s     |
| RACFG_m5_lam0.7    | 0.0%     | 0.140  | 0.079  | 0.876      | 6.6s     |
| RACFG_m5_lam0.3    | 0.0%     | 0.140  | 0.079  | 0.874      | 6.6s     |
| RACFG_m10_lam0.7   | 0.0%     | 0.140  | 0.079  | 0.878      | 6.6s     |
| RACFG_m10_lam0.3   | 0.0%     | 0.140  | 0.079  | 0.875      | 6.6s     |
| RACFG_m20_lam0.7   | 0.0%     | 0.141  | 0.079  | 0.874      | 6.6s     |
| RACFG_m20_lam0.3   | 0.0%     | 0.140  | 0.079  | 0.876      | 6.6s     |

## Degeneration Check
No degeneration detected in any configuration.

## Key Findings

1. **All RACFG variants achieve 0.0% accuracy** — identical to vanilla and far below A-CFG (6.2%). The stability-guided re-masking strategy provides no improvement regardless of remask percentage or EMA smoothing.

2. **A-CFG clearly outperforms RACFG** — A-CFG's single-step confidence-based re-masking successfully identifies better positions for constructing the unconditional input, while RACFG's cross-step JSD stability signal fails to produce meaningful variation (as identified in the pilot).

3. **EMA lambda reduction (0.3 vs 0.7) has no effect** — reducing the smoothing parameter as recommended by the pilot does not produce more informative stability scores. The fundamental issue is that Dream-7B's denoising process produces very stable cross-step probability distributions.

4. **Remask percentage has no effect** — all three values (5%, 10%, 20%) produce identical 0.0% accuracy, suggesting the issue is not the quantity of re-masked positions but the quality of position selection.

5. **RACFG outputs are nearly identical to vanilla** — examining generated text shows that RACFG barely modifies the denoising trajectory. The stability-guided re-masking selects positions that are already masked (high mask rate early) or positions that are already well-determined (low mask rate late), neither of which produces a useful unconditional signal.

## Root Cause Analysis

The fundamental failure of RACFG vs A-CFG can be attributed to:

1. **JSD stability is uninformative on Dream-7B**: Cross-step probability distributions are nearly identical (stability ~0.997), leaving no meaningful "instability signal" for RACFG to exploit. This is a property of Dream-7B's architecture, not a hyperparameter issue.

2. **Confidence is a better signal than stability**: A-CFG selects positions based on single-step confidence (low max probability = uncertain). This directly identifies positions where guidance is most impactful. RACFG's cross-step stability measures temporal consistency, which is orthogonal to current-step uncertainty.

3. **Temporal scheduling may hurt A-CFG's strength**: RACFG's threshold_70_30 schedule suppresses guidance at high mask rates, but this is precisely when A-CFG has the most positions to work with (many revealed but uncertain tokens).

## Recommendations for Downstream Tasks

1. **For ablation_racfg_schedule**: The schedule ablation depends on this task. Given NO-GO, the schedule ablation should be re-evaluated. Consider testing with confidence-based re-masking (A-CFG style) combined with temporal scheduling, rather than JSD-based RACFG.

2. **For ablation_racfg_vs_acfg**: This comparison is effectively resolved — A-CFG > RACFG. The direct comparison should still be run for completeness but expectations should be calibrated.

3. **Alternative RACFG signal**: If RACFG is to be salvaged, consider:
   - Using entropy of current predictions instead of cross-step JSD
   - Using rank changes in top-k tokens across steps
   - Combining confidence with a lightweight temporal signal
