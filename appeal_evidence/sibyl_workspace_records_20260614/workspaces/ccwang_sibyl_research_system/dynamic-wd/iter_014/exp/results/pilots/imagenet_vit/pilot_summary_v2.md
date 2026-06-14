# Pilot Summary: ImageNet ViT-S/16 (v2)

**Model**: ViT-S/16 (22M params) | **Optimizer**: AdamW | **LR**: 0.001
**Augmentation**: RandAugment(2,9) + CutMix(1.0) + Mixup(0.8) + RandomErasing(0.25)
**Batch size**: 96 | **Epochs**: 2
**Train samples**: 34862 | **Val samples**: 5000

## Results

| Method | Val Acc (%) | Top-5 (%) | Train Acc (%) | WD Budget | Time (s) | Status |
|--------|-------------|-----------|---------------|-----------|----------|--------|
| FixedWD | 0.22 | 0.98 | 0.17 | 5.2000 | 120 | PASS |
| SWD | 0.12 | 0.60 | 0.12 | 0.7715 | 171 | PASS |
| CWD | 0.18 | 1.02 | 0.11 | 2.5776 | 178 | PASS |
| CPR | 0.32 | 1.10 | 0.16 | 38.4246 | 158 | PASS |
| DefazioCorrective | 0.22 | 0.98 | 0.17 | 5.1999 | 100 | PASS |
| UDWDC-v2 | - | - | - | - | - | FAIL |

## Diagnostics Coverage

All methods track per-layer rho_t, alpha_t, w_norm, g_norm, effective_wd.
Coverage includes attention (qkv, proj) and MLP (fc1, fc2) blocks.

## UDWDC-v2 Stability

- UDWDC-v2 did not complete successfully

## Overall: PARTIAL

**Pass criteria**: ViT-S/16 completes 2 epochs without OOM; augmentation correct; diagnostics cover attention+MLP blocks

**Timestamp**: 2026-03-25T14:22:00.073916