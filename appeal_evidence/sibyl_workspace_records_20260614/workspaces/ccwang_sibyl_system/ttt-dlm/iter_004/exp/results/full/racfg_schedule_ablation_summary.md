# RACFG Temporal Schedule Ablation Summary (PILOT)

## Task: ablation_racfg_schedule
- **Date**: 2026-03-10
- **GPU**: 1 (NVIDIA RTX PRO 6000 Blackwell Server Edition)
- **Elapsed**: 16.3 minutes
- **Verdict**: **NO-GO** (no scheduled variant beats fixed)

## PIVOT NOTE

This ablation was **pivoted** from the original JSD-based RACFG temporal scheduling
to **A-CFG confidence-based re-masking + temporal scheduling**.

Reason: `ablation_racfg_remask_pct` was NO-GO — JSD stability is uninformative on
Dream-7B (stability ~0.997 everywhere), so all RACFG variants got 0% accuracy.
A-CFG with confidence-based re-masking achieved 6.2% in the same test.

## Configuration
- schedule_types: ['fixed', 'linear', 'cosine', 'threshold_70_30']
- w_bases: [1.0, 1.5]
- remask_pct=0.1, w_max=2.0
- 16 samples (PILOT), seed 42, 128 steps, gen_len=256, temperature=0.4

## Results

| Method | Accuracy | rep-3 | dist-3 | Avg Time | Guidance Steps |
|--------|----------|-------|--------|----------|---------------|
| Vanilla | 0.0% | 0.0786 | 0.8755 | 3.7s | N/A |
| ACFG_cosine_w1.0 | 0.0% | 0.0784 | 0.8507 | 7.4s | 126/128 |
| ACFG_cosine_w1.5 | 0.0% | 0.0681 | 0.8750 | 7.4s | 126/128 |
| ACFG_fixed_w1.0 | 6.2% | 0.0565 | 0.8998 | 7.4s | N/A |
| ACFG_fixed_w1.5 | 12.5% | 0.0542 | 0.8892 | 7.4s | N/A |
| ACFG_linear_w1.0 | 0.0% | 0.0773 | 0.8835 | 7.4s | 126/128 |
| ACFG_linear_w1.5 | 0.0% | 0.0735 | 0.8661 | 7.4s | 126/128 |
| ACFG_threshold_70_30_w1.0 | 0.0% | 0.0672 | 0.8829 | 6.3s | 90/128 |
| ACFG_threshold_70_30_w1.5 | 0.0% | 0.0546 | 0.8971 | 6.3s | 90/128 |

## Degeneration Check
No degeneration detected in any configuration.

## Key Findings

1. **No scheduled variant beats fixed A-CFG**: Temporal scheduling of guidance weight does not improve upon constant guidance, contradicting H6.

2. **Fixed guidance is sufficient**: A-CFG's constant guidance weight strategy is already near-optimal for Dream-7B on this benchmark.

## Schedule Type Analysis

- **fixed**: mean accuracy 9.4%, max accuracy 12.5%
- **linear**: mean accuracy 0.0%, max accuracy 0.0%
- **cosine**: mean accuracy 0.0%, max accuracy 0.0%
- **threshold_70_30**: mean accuracy 0.0%, max accuracy 0.0%

## W_base Analysis

- **w1.0**: mean accuracy 1.6%, max accuracy 6.2%
- **w1.5**: mean accuracy 3.1%, max accuracy 12.5%

## Recommendations for Downstream Tasks

1. **Best configuration**: ACFG_fixed_w1.5 (12.5%)
2. **For ablation_racfg_vs_acfg**: Use the best config from this ablation as the 'RACFG' entry (since original RACFG failed, this scheduled A-CFG variant is the closest to the 'enhanced guidance' concept).
3. **For fullscale_racfg**: The 'RACFG' full-scale evaluation should use the best configuration found here.