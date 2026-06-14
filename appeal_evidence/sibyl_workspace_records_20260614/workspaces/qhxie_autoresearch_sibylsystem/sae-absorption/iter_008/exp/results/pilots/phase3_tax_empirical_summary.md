# Phase 3: Absorption Tax Empirical Validation (PILOT)

## Theoretical Absorption Tax T(G)

**T(G) = 0.413962**

Interpretation: An absorption-free SAE needs approximately 0.4140 additional L0 budget
per token compared to one that uses absorption. This is the "tax" for maintaining full
feature resolution under hierarchy.

Top contributors to T(G):
| Letter | p_c (frequency) | R_pc (redundancy) | Contribution |
|--------|-----------------|-------------------|-------------|
| j | 0.0597 | 0.4585 | 0.027352 |
| f | 0.0511 | 0.4971 | 0.025422 |
| m | 0.0568 | 0.4359 | 0.024765 |
| o | 0.0455 | 0.5055 | 0.022978 |
| v | 0.0455 | 0.4733 | 0.021516 |
| h | 0.0398 | 0.4932 | 0.019614 |
| r | 0.0426 | 0.4446 | 0.018944 |
| p | 0.0540 | 0.3491 | 0.018844 |
| n | 0.0540 | 0.3490 | 0.018837 |
| t | 0.0511 | 0.3610 | 0.018461 |

## Prediction Results Summary

| Prediction | Verdict | Key Metric |
|-----------|---------|-----------|
| P1: Absorption-MSE trade-off | NOT_SUPPORTED | Spearman=0.07881104062391006 |
| P2: Width power law | NOT_SUPPORTED | gamma=-0.38661449235135914 |
| P3: R_pc predicts absorption | NOT_SUPPORTED | Spearman=0.1581 |
| P4: Matryoshka advantage | PRELIMINARY | N=1 configs |

## Prediction 3 Detail: R_pc vs Per-Letter Absorption

Spearman rho = 0.15805137504669234
Pearson r = 0.16162913183622793
N letters = 24

| Letter | R_pc | Absorption Rate | N_tokens | N_FN |
|--------|------|----------------|----------|------|
| a | 0.3998 | 0.0909 | 12 | 1 |
| b | 0.5477 | 0.1000 | 11 | 1 |
| c | 0.3830 | 0.0833 | 15 | 1 |
| d | 0.4815 | 0.0000 | 13 | 0 |
| e | 0.4035 | 0.0833 | 14 | 1 |
| f | 0.4971 | 0.1333 | 18 | 2 |
| g | 0.4213 | 0.0000 | 15 | 0 |
| h | 0.4932 | 0.0000 | 14 | 0 |
| i | 0.3328 | 0.0000 | 16 | 0 |
| j | 0.4585 | 0.0000 | 21 | 0 |
| k | 0.4718 | 0.0000 | 13 | 0 |
| l | 0.5059 | 0.0000 | 9 | 0 |
| m | 0.4359 | 0.0000 | 20 | 0 |
| n | 0.3490 | 0.0000 | 19 | 0 |
| o | 0.5055 | 0.0909 | 16 | 1 |
| p | 0.3491 | 0.0000 | 19 | 0 |
| q | 0.4757 | 0.0000 | 5 | 0 |
| r | 0.4446 | 0.0714 | 15 | 1 |
| s | 0.4418 | 0.1818 | 12 | 2 |
| t | 0.3610 | 0.0000 | 18 | 0 |
| u | 0.2672 | 0.0769 | 19 | 1 |
| v | 0.4733 | 0.0000 | 16 | 0 |
| w | 0.4760 | 0.0000 | 11 | 0 |
| y | 0.0217 | 0.0000 | 8 | 0 |

## SAE Configurations Tested

| Config | Width | MSE | NMSE | L0 | Absorption |
|--------|-------|-----|------|-----|-----------|
| L12_16k_canonical | 16384 | 945.136841 | 6.745128 | 444.6 | 0.0390 |
| saebench_topk_width-2pow14 | 16384 | 2.407331 | 0.017180 | 20.0 | 0.2667 |
| saebench_vanilla_width-2pow14 | 16384 | 1.258918 | 0.008984 | 532.2 | 0.2667 |
| saebench_topk_width-2pow12 | 4096 | 2.590265 | 0.018486 | 20.0 | 0.3333 |
| saebench_topk_width-2pow16 | 65536 | 2.142205 | 0.015288 | 20.0 | 0.2667 |
| saebench_vanilla_width-2pow12 | 4096 | 1.671787 | 0.011931 | 358.7 | 0.2667 |
| saebench_vanilla_width-2pow16 | 65536 | 1.122717 | 0.008012 | 562.9 | 0.1333 |
| L12_65k_canonical | 65536 | 25161.308594 | 179.567923 | 1439.7 | 0.0667 |

## Pilot Assessment

- **Predictions testable:** 4/4
- **T(G) computed:** Yes (0.413962)
- **Pilot pass:** YES
- **Elapsed time:** 0.3 minutes
