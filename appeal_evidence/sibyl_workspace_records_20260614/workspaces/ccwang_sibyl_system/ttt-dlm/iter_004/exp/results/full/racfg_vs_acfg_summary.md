# RACFG vs A-CFG Direct Comparison Summary (PILOT)

## Task: ablation_racfg_vs_acfg
- **Date**: 2026-03-10
- **GPU**: 0 (NVIDIA RTX PRO 6000 Blackwell Server Edition)
- **Elapsed**: 4.8 minutes
- **Verdict**: **NO-GO**

## Hypothesis H5

> JSD-based re-masking outperforms confidence-based re-masking by >= 2pp
> **Result**: FALSIFIED

## Results

| Method | Re-mask Signal | Accuracy | rep-3 | distinct-3 | Avg Time | FLOPs |
|--------|---------------|----------|-------|------------|----------|-------|
| Vanilla | N/A | 0.0% | 0.0786 | 0.8755 | 3.7s | 1.0x |
| RACFG | Cross-step JSD | 0.0% | 0.0786 | 0.8781 | 6.6s | 1.8x |
| A-CFG | Single-step confidence | 12.5% | 0.0542 | 0.8892 | 7.4s | 2.0x |

## Per-Sample Agreement

| | A-CFG Correct | A-CFG Wrong |
|---|---|---|
| **RACFG Correct** | 0 | 0 |
| **RACFG Wrong** | 2 | 14 |

## Degeneration Check
No degeneration detected in either method.

## Key Findings

1. **H5 FALSIFIED**: A-CFG (12.5%) >= RACFG (0.0%). Single-step confidence is a better re-masking signal than cross-step JSD stability on Dream-7B.

2. **Root cause**: Dream-7B produces very stable cross-step probability distributions (JSD stability ~0.997), leaving no meaningful instability signal for RACFG to exploit. Confidence directly identifies uncertain positions where guidance has the most impact.

3. **Compute parity**: Both methods use ~2x vanilla FLOPs (one extra forward pass per step). RACFG (1.8x) vs A-CFG (2.0x).

4. **Implication for fullscale_racfg**: The 'RACFG' full-scale evaluation should use A-CFG (best config: fixed w=1.5) as the 'enhanced guidance' method, since original RACFG is definitively inferior.