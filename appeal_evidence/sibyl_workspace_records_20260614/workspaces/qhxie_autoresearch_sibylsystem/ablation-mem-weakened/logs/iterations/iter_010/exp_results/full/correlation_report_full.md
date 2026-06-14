# Full Experiment Results: Feature Absorption and Downstream SAE Reliability

## Configuration
- Model: GPT-2 Small (85M params)
- SAE: gpt2-small-res-jb (24K latents)
- Features: 26 first-letter features (A-Z)
- Layers tested: 4, 8
- Statistical tests performed: 12 (4 hypotheses x 2 layers x 2 metrics)

## Layer 4 Results

### H1: Absorption vs Raw Steering Effectiveness
- Pearson r = 0.0077, p = 0.9701
- Spearman rho = 0.0288, p = 0.8888
- R^2 = 0.0001
- **Uncorrected passes (r<0, p<0.05)**: False

### H1b: Absorption vs Delta Steering (feature-specific minus random baseline)
- Pearson r = 0.2452, p = 0.2272
- Spearman rho = 0.2313, p = 0.2557
- R^2 = 0.0601
- Mean random baseline success = 0.344
- **Uncorrected passes (r<0, p<0.05)**: False

### H2: Absorption vs Probing F1
- Pearson r = -0.0033, p = 0.9873
- Spearman rho = 0.0242, p = 0.9068
- R^2 = 0.0000
- **Uncorrected passes (r<0, p<0.05)**: False

### Top Absorption Features
| Feature | Absorption Rate | Steering Success | Probing F1 |
|---------|----------------|------------------|------------|
| Q | 0.160 | 0.800 | 0.581 |
| P | 0.148 | 0.700 | 0.444 |
| G | 0.146 | 0.800 | 0.688 |
| R | 0.140 | 0.400 | 0.444 |
| J | 0.133 | 1.000 | 0.519 |
| W | 0.113 | 0.900 | 0.400 |
| S | 0.099 | 1.000 | 0.261 |
| A | 0.077 | 1.000 | 0.333 |
| B | 0.000 | 0.800 | 0.462 |
| C | 0.000 | 0.850 | 0.182 |

## Layer 8 Results

### H1: Absorption vs Raw Steering Effectiveness
- Pearson r = -0.3005, p = 0.1358
- Spearman rho = -0.2224, p = 0.2748
- R^2 = 0.0903
- **Uncorrected passes (r<0, p<0.05)**: False

### H1b: Absorption vs Delta Steering (feature-specific minus random baseline)
- Pearson r = -0.4313, p = 0.0278
- Spearman rho = -0.5024, p = 0.0089
- R^2 = 0.1860
- Mean random baseline success = 0.379
- **Uncorrected passes (r<0, p<0.05)**: True

### H2: Absorption vs Probing F1
- Pearson r = -0.1067, p = 0.6040
- Spearman rho = -0.0061, p = 0.9765
- R^2 = 0.0114
- **Uncorrected passes (r<0, p<0.05)**: False

### Top Absorption Features
| Feature | Absorption Rate | Steering Success | Probing F1 |
|---------|----------------|------------------|------------|
| U | 0.242 | 1.000 | 0.462 |
| H | 0.190 | 0.550 | 0.400 |
| S | 0.160 | 0.650 | 0.182 |
| V | 0.147 | 0.700 | 0.667 |
| A | 0.080 | 1.000 | 0.444 |
| B | 0.060 | 0.700 | 0.667 |
| C | 0.000 | 1.000 | 0.333 |
| D | 0.000 | 0.850 | 0.400 |
| E | 0.000 | 1.000 | 0.261 |
| F | 0.000 | 0.750 | 0.261 |

## H3: Cross-Layer Consistency
- CV of H1 slopes: 1.0789
- CV of H2 slopes: 0.9315
- H1 slopes by layer: [0.023902829950215133, -0.6298740079534727]
- H2 slopes by layer: [-0.010121035106683636, -0.28550852663262205]
- **Passes (CV < 0.5)**: False

## Multiple Comparisons Correction

**Total statistical tests performed**: 12
- H1 (raw steering): Pearson + Spearman x 2 layers = 4 tests
- H1b (delta steering): Pearson + Spearman x 2 layers = 4 tests
- H2 (probing): Pearson + Spearman x 2 layers = 4 tests
- **Total = 12 tests**

### Bonferroni Correction
- Corrected alpha = 0.05 / 12 = 0.00417
- Significant results after correction: **0**

### Benjamini-Hochberg FDR Correction
- Significant results at q < 0.05: **0**

### Corrected P-values by Test

| Test | Raw p | Bonferroni p | BH q-value | Bonferroni sig? | BH sig? |
|------|-------|-------------|-----------|-----------------|---------|
| H1_L4_Pearson | 0.9701 | 1.0000 | 0.9873 | No | No |
| H1_L4_Spearman | 0.8888 | 1.0000 | 0.9873 | No | No |
| H1b_L4_Pearson | 0.2272 | 1.0000 | 0.5495 | No | No |
| H1b_L4_Spearman | 0.2557 | 1.0000 | 0.5495 | No | No |
| H2_L4_Pearson | 0.9873 | 1.0000 | 0.9873 | No | No |
| H2_L4_Spearman | 0.9068 | 1.0000 | 0.9873 | No | No |
| H1_L8_Pearson | 0.1358 | 1.0000 | 0.5431 | No | No |
| H1_L8_Spearman | 0.2748 | 1.0000 | 0.5495 | No | No |
| H1b_L8_Pearson | 0.0278 | 0.3339 | 0.1670 | No | No |
| H1b_L8_Spearman | 0.0089 | 0.1068 | 0.1068 | No | No |
| H2_L8_Pearson | 0.6040 | 1.0000 | 0.9873 | No | No |
| H2_L8_Spearman | 0.9765 | 1.0000 | 0.9873 | No | No |

## Summary
- H1 passes (uncorrected): False
- H1b passes (uncorrected): True
- H2 passes (uncorrected): False
- H3 passes: False
- Any significant after Bonferroni: False
- Any significant after BH FDR: False

**Conclusion**: With 12 statistical tests, no hypothesis survives multiple comparison correction. The uncorrected H1b result at layer 8 (p=0.028) does not reach significance after Bonferroni (p=0.334) or BH-FDR (q=0.107).
