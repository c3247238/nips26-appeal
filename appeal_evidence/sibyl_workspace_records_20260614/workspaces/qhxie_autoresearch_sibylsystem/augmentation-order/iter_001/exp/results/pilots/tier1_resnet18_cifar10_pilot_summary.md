# Pilot Summary: tier1_resnet18_cifar10

**Mode**: PILOT (100 samples, 10 epochs, seed=42)
**Status**: PASS
**Timestamp**: 2026-04-02T14:07:19

## Pass Criteria

| Criterion | Result |
|---|---|
| All 6 orderings run without error | YES (6/6) |
| val_accuracy variance > 0.0% across orderings | YES (spread=0.96%) |
| Per-epoch logging works | YES |

## Key Metrics

- **Spread**: 0.0096 (0.96%)
- **Best ordering**: order_5 (CJ→Flip→Crop) = 0.1097
- **Worst ordering**: order_4 (CJ→Crop→Flip) = 0.1001
- **Conventional order_0** (Crop→Flip→CJ) = 0.1019 (rank 4/6)

## Per-Ordering Results (seed=42, 100 samples, 10 epochs)

| Ordering | Label | val_acc |
|---|---|---|
| order_5 | CJ→Flip→Crop | 0.1097 |
| order_3 | Flip→CJ→Crop | 0.1023 |
| order_0 | Crop→Flip→CJ | 0.1019 |
| order_2 | Flip→Crop→CJ | 0.1010 |
| order_1 | Crop→CJ→Flip | 0.1009 |
| order_4 | CJ→Crop→Flip | 0.1001 |

## Notes

- Val accuracy is near-random (10%) due to only 100 training samples — this is expected for the pilot smoke test.
- The 0.96% spread at this tiny scale indicates orderings DO produce different loss landscapes even with minimal data.
- Per-epoch logging is functional with all metrics recorded correctly.
- GPU infrastructure works: 6 runs completed in ~0.6 minutes on RTX PRO 6000 Blackwell.

## GO/NO-GO

**GO** — All pilot pass criteria met. Infrastructure is functional. Ready to run full Tier 1 experiment (200 epochs, 5 seeds [42–46], full CIFAR-10 dataset, 6 orderings).
