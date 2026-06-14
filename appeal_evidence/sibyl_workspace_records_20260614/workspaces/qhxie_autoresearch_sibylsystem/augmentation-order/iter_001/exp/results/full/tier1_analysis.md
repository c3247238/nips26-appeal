# Tier 1 Statistical Analysis — PILOT MODE

**Generated**: 2026-04-02 18:00:23  
**Mode**: Pilot (10 epochs, 100-sample subsets, 1 seed)
**Note**: All statistics are indicative only. Full analysis requires 200 epochs, 5 seeds.

---

## Per-Block Spread Summary

| Block | Best Ordering | Best Acc | Worst Ordering | Worst Acc | Spread | Confidence |
|-------|--------------|----------|----------------|-----------|--------|------------|
| resnet18 × cifar10 | CJ→Flip→Crop (order_5) | 0.1097 | CJ→Crop→Flip (order_4) | 0.1001 | 0.960% | high |
| resnet18 × cifar100 | Flip→Crop→CJ (order_2) | 0.4663 | CJ→Flip→Crop (order_5) | 0.4575 | 0.880% | high |
| vit × cifar10 | Flip->CJ->Crop (order_3) | 0.1970 | Crop->CJ->Flip (order_1) | 0.1738 | 2.320% | high |
| vit × cifar100 | Flip->Crop->CJ (order_2) | 0.0289 | CJ->Flip->Crop (order_5) | 0.0264 | 0.250% | cautious |

---

## Detailed Per-Block Rankings

### resnet18 × cifar10

| Rank | Ordering ID | Label | Val Accuracy |
|------|------------|-------|-------------|
| 1 | order_5 | CJ→Flip→Crop | 0.1097 |
| 2 | order_3 | Flip→CJ→Crop | 0.1023 |
| 3 | order_0 | Crop→Flip→CJ | 0.1019 |
| 4 | order_2 | Flip→Crop→CJ | 0.1010 |
| 5 | order_1 | Crop→CJ→Flip | 0.1009 |
| 6 | order_4 | CJ→Crop→Flip | 0.1001 |

**Paired t-test**: n/a (1 seed — need ≥2 for t-test)
**Cohen's d**: 2.7115

### resnet18 × cifar100

| Rank | Ordering ID | Label | Val Accuracy |
|------|------------|-------|-------------|
| 1 | order_2 | Flip→Crop→CJ | 0.4663 |
| 2 | order_1 | Crop→CJ→Flip | 0.4645 |
| 3 | order_3 | Flip→CJ→Crop | 0.4630 |
| 4 | order_0 | Crop→Flip→CJ | 0.4613 |
| 5 | order_4 | CJ→Crop→Flip | 0.4592 |
| 6 | order_5 | CJ→Flip→Crop | 0.4575 |

**Paired t-test**: n/a (1 seed — need ≥2 for t-test)
**Cohen's d**: 2.6700

### vit × cifar10

| Rank | Ordering ID | Label | Val Accuracy |
|------|------------|-------|-------------|
| 1 | order_3 | Flip->CJ->Crop | 0.1970 |
| 2 | order_5 | CJ->Flip->Crop | 0.1950 |
| 3 | order_0 | Crop->Flip->CJ | 0.1937 |
| 4 | order_4 | CJ->Crop->Flip | 0.1853 |
| 5 | order_2 | Flip->Crop->CJ | 0.1754 |
| 6 | order_1 | Crop->CJ->Flip | 0.1738 |

**Paired t-test**: n/a (1 seed — need ≥2 for t-test)
**Cohen's d**: 2.2749

### vit × cifar100

| Rank | Ordering ID | Label | Val Accuracy |
|------|------------|-------|-------------|
| 1 | order_2 | Flip->Crop->CJ | 0.0289 |
| 2 | order_3 | Flip->CJ->Crop | 0.0286 |
| 3 | order_4 | CJ->Crop->Flip | 0.0285 |
| 4 | order_0 | Crop->Flip->CJ | 0.0280 |
| 5 | order_1 | Crop->CJ->Flip | 0.0269 |
| 6 | order_5 | CJ->Flip->Crop | 0.0264 |

**Paired t-test**: n/a (1 seed — need ≥2 for t-test)
**Cohen's d**: 2.4733

---

## Consensus Best/Worst Orderings for Tier 3

- **Best overall**: `order_2` — Flip→Crop→CJ (wins: 2/4 blocks)
- **Worst overall**: `order_5` — CJ→Flip→Crop (losses: 2/4 blocks)

---

## Pilot Assessment

- Blocks with high-confidence spread: **3/4**
- Blocks with cautious spread: **1/4**
- Blocks with low spread (noisy): **0/4**

**STATUS**: PASS — spread values written for all available blocks; best/worst ordering IDs confirmed
