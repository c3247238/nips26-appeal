# Pilot Summary: tier1_resnet18_cifar100

**Task**: Tier 1 — Full Factorial ResNet-18 x CIFAR-100  
**Mode**: PILOT (10 epochs, seed 42, full 50k train set)  
**Timestamp**: 2026-04-02T14:08:44

## Verdict: GO (HIGH_CONFIDENCE)

Pass criteria: val_accuracy > 30% at epoch 10 for >= 3 orderings  
Result: **6/6 orderings exceeded 30%** (all reached ~46%)

## Results Summary

| Ordering | Label | Val Acc @ep10 |
|---|---|---|
| order_2 | Flip→Crop→CJ | 0.4663 (BEST) |
| order_1 | Crop→CJ→Flip | 0.4645 |
| order_3 | Flip→CJ→Crop | 0.4630 |
| order_0 | Crop→Flip→CJ (conventional) | 0.4613 |
| order_4 | CJ→Crop→Flip | 0.4592 |
| order_5 | CJ→Flip→Crop | 0.4575 (WORST) |

- **Spread**: 0.0088 (0.88%) — HIGH_CONFIDENCE
- **Best**: order_2 (Flip→Crop→CJ) = 0.4663
- **Worst**: order_5 (CJ→Flip→Crop) = 0.4575
- **Runtime**: 2.8 min

## Key Observations

1. All 6 orderings converge to ~46% after 10 epochs — training is stable, no NaN loss.
2. The 0.88% spread at epoch 10 (single seed) suggests a meaningful ordering effect on CIFAR-100.
3. Flip-first orderings (order_2, order_3) outperform CJ-first orderings (order_4, order_5).
4. Conventional ordering (Crop→Flip→CJ, order_0) falls in the middle, not the best.
5. CJ-first orderings (reversibility-first principle) actually perform worst in this pilot.

## Recommendation

**PROCEED to full experiment** (6 orderings × 5 seeds × 200 epochs on CIFAR-100).
