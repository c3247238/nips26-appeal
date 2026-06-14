# Per-Class Ordering Sensitivity Analysis (PILOT)

**Mode**: Pilot | **Data source**: real_pilot | **Timestamp**: 2026-04-03T13:37:25.532845

> WARNING: Pilot results use 100-sample pilot data.
> Full-scale analysis blocked on `tier1_resnet18_cifar100_full` completion.

## Pass Criteria
- Classes with sensitivity > aggregate spread: **26/100** → PASS
- Taxonomy grouping: **20 superclasses** → PASS
- **Overall: GO**

## Aggregate Statistics
- Aggregate spread (best - worst ordering): `0.0088`
- Classes analyzed: `100/100`

## Category Type Sensitivity Ranking
| Category Type | Mean Spread | Class Count |
|---|---|---|
| vehicles | 0.0770 | 10 |
| natural | 0.0695 | 20 |
| animals | 0.0556 | 45 |
| objects | 0.0300 | 15 |
| people | 0.0180 | 5 |
| structures | 0.0100 | 5 |

## Superclass Sensitivity Ranking (Top 10)
| Superclass | Mean Spread | Std | Category Type |
|---|---|---|---|
| large_omnivores_and_herbivores | 0.2560 | 0.3381 | animals |
| fish | 0.1760 | 0.3470 | animals |
| vehicles_1 | 0.1540 | 0.3030 | vehicles |
| large_natural_outdoor_scenes | 0.1260 | 0.1764 | natural |
| flowers | 0.0940 | 0.1203 | natural |
| food_containers | 0.0820 | 0.1360 | objects |
| fruit_and_vegetables | 0.0580 | 0.1160 | natural |
| non_insect_invertebrates | 0.0320 | 0.0412 | animals |
| people | 0.0180 | 0.0312 | people |
| small_mammals | 0.0160 | 0.0206 | animals |

## Top 10 Most Ordering-Sensitive Classes
| Class | Spread | Best Ordering | Category | Superclass |
|---|---|---|---|---|
| aquarium_fish | 0.8700 | order_3 | animals | fish |
| elephant | 0.8400 | order_1 | animals | large_omnivores_and_herbivores |
| bus | 0.7600 | order_5 | vehicles | vehicles_1 |
| plain | 0.4500 | order_0 | natural | large_natural_outdoor_scenes |
| chimpanzee | 0.4400 | order_0 | animals | large_omnivores_and_herbivores |
| can | 0.3500 | order_4 | objects | food_containers |
| pear | 0.2900 | order_0 | natural | fruit_and_vegetables |
| sunflower | 0.2900 | order_0 | natural | flowers |
| cloud | 0.1800 | order_5 | natural | large_natural_outdoor_scenes |
| poppy | 0.1800 | order_2 | natural | flowers |

## Key Findings
- Most ordering-sensitive category: vehicles (mean_spread=0.0770)
- Least ordering-sensitive category: structures (mean_spread=0.0100)
- Most sensitive superclass: large_omnivores_and_herbivores (mean_spread=0.2560)
- 26/100 classes show ordering sensitivity > aggregate spread (0.0088)
- Animals less ordering-sensitive than vehicles (0.0556 vs 0.0770)

## Notes
- PILOT: Uses pilot-scale data or synthetic data for pipeline validation. Per-class ordering effects at full scale (5 seeds, 200 epochs) may differ substantially. Full analysis blocked on tier1_resnet18_cifar100_full completion.