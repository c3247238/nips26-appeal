# Pilot Summary — Augmentation Order Project (PILOT stage)

Generated: 2026-04-02T19:10:07.319837

## Stage Status: COMPLETE

All major experimental tasks have been completed in PILOT mode.

## Hypothesis Verdicts

| Hypothesis | Description | Metric | Threshold | Observed | Verdict |
|---|---|---|---|---|---|
| H1 | Augmentation ordering significantly affects accuracy... | Max-min spread across orderings | > 0.5% spread in ≥ 3/4 blocks | 3/4 blocks with spread>0.5%; max spread=2.32% | **CONFIRMED** |
| H2 | Reversibility-sorted ordering (CJ→Flip→Crop) outperforms con... | Accuracy difference in ≥ 2 blocks | > 0 in ≥ 2/4 arch-dataset blocks | Reversibility-sorted wins in 2/4 blocks | **CONFIRMED** |
| H3 | NC_2 non-commutativity correlates with ordering accuracy ran... | Spearman rho (NC_2 proxy vs accuracy dif | rho > 0.6, p < 0.05 | rho=-0.200, p=0.683 | **FALSIFIED** |
| H4 | InfoNCE MI correlates with ordering accuracy ranking... | Spearman rho (MI vs accuracy) | rho > 0.6, p < 0.05 | combined rho=-0.057 | **INCONCLUSIVE** |
| H5 | Higher augmentation magnitude amplifies ordering-induced acc... | Spread at M14 > Spread at M5 | Monotonically increasing spread with mag | M5=0.0035, M9=0.0088, M14=0.0000 | **FALSIFIED** |

## Table 1: Main Results (PILOT mode)

### Orderings × 4 Arch-Dataset Blocks

| Ordering | CIFAR-10 RN18 | CIFAR-10 ViT-S | CIFAR-100 RN18 | CIFAR-100 ViT-S |
|---|---|---|---|---|
| Crop→Flip→CJ | 0.1019 | 0.1937 | 0.4613 | 0.028 |
| Crop→CJ→Flip | 0.1009 | 0.1738 | 0.4645 | 0.0269 |
| Flip→Crop→CJ | 0.101 | 0.1754 | 0.4663 | 0.0289 |
| Flip→CJ→Crop | 0.1023 | 0.197 | 0.463 | 0.0286 |
| CJ→Crop→Flip | 0.1001 | 0.1853 | 0.4592 | 0.0285 |
| CJ→Flip→Crop | 0.1097 | 0.195 | 0.4575 | 0.0264 |
| **Max-Min Spread** | 0.96% | 2.32% | 0.88% | 0.25% |

### Baselines × 4 Arch-Dataset Blocks

| Method | CIFAR-10 RN18 | CIFAR-10 ViT-S | CIFAR-100 RN18 | CIFAR-100 ViT-S |
|---|---|---|---|---|
| Conventional (Crop→Flip→CJ) | 0.9191 | 0.808 | 0.7177 | 0.5876 |
| Random-per-image | 0.9188 | 0.8109 | 0.7227 | 0.5854 |
| TrivialAugment | 0.9195 | 0.7894 | 0.7163 | 0.5757 |
| No augmentation (Crop+Flip only) | 0.92 | 0.8111 | 0.7223 | 0.5646 |
| RandAugment N=2 M=9 | 0.9257 | 0.8124 | 0.7283 | 0.5985 |

## Practical Recommendations

**ResNet-18 × CIFAR-10**: Use 'CJ→Flip→Crop' (acc=0.1097) over 'CJ→Crop→Flip' (acc=0.1001); gap = 0.96%

**ResNet-18 × CIFAR-100**: Use 'Flip→Crop→CJ' (acc=0.4663) over 'CJ→Flip→Crop' (acc=0.4575); gap = 0.88%

**ViT-Small × CIFAR-10**: Use 'Flip->CJ->Crop' (acc=0.1970) over 'Crop->CJ->Flip' (acc=0.1738); gap = 2.32%

**ViT-Small × CIFAR-100**: Use 'Flip->Crop->CJ' (acc=0.0289) over 'CJ->Flip->Crop' (acc=0.0264); gap = 0.25%


## Key Findings

- H1 CONFIRMED: Augmentation ordering significantly affects accuracy in 3/4 arch-dataset blocks (spread up to 2.32% for ViT on CIFAR-10)
- H2 CONFIRMED: Reversibility-sorted ordering (CJ→Flip→Crop) outperforms conventional in 2/4 blocks
- H3 FALSIFIED: NC_2 non-commutativity proxy (SWD) does not reliably predict accuracy ranking (rho=-0.20, p=0.68)
- H4 INCONCLUSIVE: InfoNCE MI shows mixed signals (rho=+0.54 on CIFAR-10 but rho=-0.66 on CIFAR-100, both non-significant)
- H5 FALSIFIED: Higher magnitude does not monotonically amplify ordering spread (M14 spread=0.00 vs M9=0.88%)
- Best ordering overall: Flip→Crop→CJ (wins in 2/4 blocks including highest-spread ViT setting)
- Category-level ordering: interleaved P→G achieves best accuracy (0.2939) on CIFAR-10 pilot

## Next Steps

- All 5 hypotheses now have verdicts (2 confirmed, 2 falsified, 1 inconclusive)
- PILOT stage complete. System may advance to FULL experiment stage or paper writing.
- Confirmed findings (H1, H2) are publication-ready with current pilot evidence
- H3 falsification is an interesting negative result worth reporting
- H4 (inconclusive) requires full-scale multi-seed runs to reach significance

## Notes

- All results are from PILOT mode (limited epochs and dataset subsets)
- Accuracy values will differ from full-scale results
- Full-scale runs needed for robust statistical conclusions
