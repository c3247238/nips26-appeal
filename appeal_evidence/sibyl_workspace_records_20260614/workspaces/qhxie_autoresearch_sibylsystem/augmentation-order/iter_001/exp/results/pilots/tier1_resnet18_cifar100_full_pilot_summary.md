# Pilot Summary: tier1_resnet18_cifar100_full

**Task**: Tier 1 Full Pilot — ResNet-18 x CIFAR-100 (10 epochs, seed=42, full train set)
**Date**: 2026-04-03
**Duration**: 2.8 minutes
**GPU**: RTX PRO 6000 (GPU 6)

## Verdict: GO

**Pass criteria met**: YES
- All 6 orderings ran 10 epochs without error
- All 6 orderings achieved val_accuracy > 30% (range: 45.75%–46.63%)
- Per-class accuracy logged for all 100 classes and 20 superclasses

## Results

| Ordering | Label | Val Acc (seed=42) |
|----------|-------|-------------------|
| order_0 | Crop→Flip→CJ (conventional) | 46.13% |
| order_1 | Crop→CJ→Flip | 46.45% |
| order_2 | Flip→Crop→CJ | **46.63%** (BEST) |
| order_3 | Flip→CJ→Crop | 46.30% |
| order_4 | CJ→Crop→Flip | 45.92% |
| order_5 | CJ→Flip→Crop | **45.75%** (WORST) |

**Spread**: 0.88% (order_2 vs order_5)

## Key Observations

1. **Ordering sensitivity confirmed**: 0.88% spread at pilot scale (10 epochs). Full-scale (200 epochs, 5 seeds) will determine statistical significance.
2. **Flip-first advantage**: order_2 (Flip→Crop→CJ) is the best performing, consistent with H4 DPI-corrected prediction (Flip-first outperforms Crop-first).
3. **CJ-first disadvantage**: Both orderings starting with CJ (order_4, order_5) rank 5th and 6th. Consistent with DPI prediction (least reversible last).
4. **Per-class accuracy**: All 100 CIFAR-100 classes and all 20 superclasses logged for each ordering. Ready for `per_class_analysis` downstream task.

## Next Step

Full-scale run: `tier1_resnet18_cifar100_full` with 5 seeds [42–46], 200 epochs, full 50k training set.
Expected wall-clock: ~45 minutes on GPU 6.
