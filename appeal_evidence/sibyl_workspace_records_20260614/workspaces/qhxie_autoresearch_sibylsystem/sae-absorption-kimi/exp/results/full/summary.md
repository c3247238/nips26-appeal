# Full Experiment Results: Component-Isolated SAE Absorption Study

## Date: 2026-04-25

## Results Summary (5 replicates each, seeds 42, 123, 456, 789, 1011)

| Variant | Absorption Rate | MCC | MSE | L0 | Hedging |
|---------|----------------|-----|-----|----|---------|
| Baseline ReLU | 0.252 ± 0.046 | 0.216 | 0.0104 | 964.0 | 0.240 |
| TopK (k=50) | 0.056 ± 0.021 | 0.214 | 0.0077 | 50.0 | 0.237 |
| MultiScale | 0.055 ± 0.024 | 0.220 | 0.0071 | 50.0 | 0.235 |
| Orthogonality | 0.245 ± 0.050 | 0.222 | 0.0000 | 550.2 | 0.240 |
| Gated | 0.261 ± 0.050 | 0.217 | 0.0082 | 965.9 | 0.233 |
| Full Matryoshka | 0.066 ± 0.029 | 0.220 | 0.0071 | 50.0 | 0.233 |
| Random Control | 0.534 ± 0.050 | 0.221 | 0.6491 | 1029.2 | 0.236 |

## Key Findings

### 1. TopK Sparsity is the Dominant Driver of Absorption Reduction
- **TopK reduces absorption by 78.0%** (0.252 -> 0.056)
- Cohen's d = 4.93 (extremely large effect size)

### 2. Component Ranking by Effect Size
1. **TopK (k=50)**: Cohen's d = 4.93 (78.0% reduction)
2. **MultiScale**: Cohen's d = 4.81 (78.3% reduction)
3. **Full Matryoshka**: Cohen's d = 4.31 (73.7% reduction)
4. **Orthogonality**: Cohen's d = 0.13 (2.7% reduction)
5. **Gated**: Cohen's d = -0.17 (-3.6% reduction)

### 3. Hypothesis Tests

| Hypothesis | Status | Evidence |
|------------|--------|----------|
| H1_TopK_dominance | SUPPORTED |  |
| H2_sparsity_mediation | SUPPORTED |  |
| H3_orthogonality_null | SUPPORTED |  |

### 4. L0-Absorption Correlation
- Pearson r = 0.865, p = 0.0119
- Strong positive correlation confirms absorption is driven by sparsity level

## Statistical Tests

- ANOVA: F = 73.36, p = 0.0000***

## Notes

- All experiments use synthetic hierarchical data (1024 features, 256 hidden dim)
- Training: 2M samples, batch size 1024, d_sae = 2048
- Each replicate takes ~15-20 seconds on RTX PRO 6000 Blackwell