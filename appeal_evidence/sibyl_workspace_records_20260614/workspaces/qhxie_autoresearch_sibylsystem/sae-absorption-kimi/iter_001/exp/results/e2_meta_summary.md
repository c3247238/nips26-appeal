# E2 Meta-Analysis: Downstream Causal Cost of Absorption

**Task:** e2_meta  
**SAEs analyzed:** 314 (from SAEBench cached data)  
**Data source:** SAEBench HF dataset (adamkarvonen/sae_bench_results_0125)  

## Descriptive Statistics

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| absorption_mean | 0.1108 | 0.1534 | 0.0000 | 0.7316 |
| l0 | 460.9326 | 1177.0541 | 18.7975 | 8284.9795 |
| ce_loss_recovered | 0.9541 | 0.1662 | -0.1579 | 1.0000 |
| explained_variance | 0.7467 | 0.3005 | -1.3906 | 1.0000 |
| sparse_probing_f1 | 0.6329 | 0.0968 | 0.2602 | 0.9046 |
| ravel_cause | 0.5518 | 0.0904 | 0.2473 | 0.7838 |
| ravel_isolation | 0.5647 | 0.0905 | 0.3352 | 0.8565 |
| width | 26308.9231 | 24382.2389 | 4096.0000 | 65536.0000 |

## Correlations (absorption vs downstream)

| Outcome | Pearson r | p-value | N |
|---------|-----------|---------|---|
| sparse_probing_f1 | -0.348 | 0.000 | 314 |
| ravel_cause | -0.264 | 0.000 | 314 |
| ravel_isolation | -0.263 | 0.000 | 314 |

## Partial Correlations (controlling for L0 and CE loss recovered)

| Outcome | Partial r | p-value | N |
|---------|-----------|---------|---|
| sparse_probing_f1 | -0.385 | 0.000 | 314 |
| ravel_cause | -0.237 | 0.000 | 314 |
| ravel_isolation | -0.266 | 0.000 | 314 |

## OLS Regression Summaries

### sparse_probing_f1 (R² = 0.321, N = 312)

| Variable | Coefficient | SE | t-stat | p-value |
|----------|-------------|----|--------|---------|
| absorption_mean | -0.0373 | 0.0055 | -6.813 | 0.000 |
| l0 | 0.0196 | 0.0097 | 2.020 | 0.044 |
| ce_loss_recovered | 0.0571 | 0.0092 | 6.206 | 0.000 |
| width | 0.0058 | 0.0052 | 1.104 | 0.271 |
| architecture_GatedSAE | -0.0149 | 0.0134 | -1.111 | 0.267 |
| architecture_JumpRelu | -0.0188 | 0.0134 | -1.411 | 0.159 |
| architecture_MatryoshkaBatchTopK | -0.0089 | 0.0136 | -0.653 | 0.514 |
| architecture_PAnneal | -0.0107 | 0.0134 | -0.803 | 0.423 |
| architecture_Standard | -0.0187 | 0.0105 | -1.784 | 0.075 |
| architecture_TopK | -0.0062 | 0.0100 | -0.618 | 0.537 |

### ravel_cause (R² = 0.166, N = 312)

| Variable | Coefficient | SE | t-stat | p-value |
|----------|-------------|----|--------|---------|
| absorption_mean | -0.0216 | 0.0057 | -3.815 | 0.000 |
| l0 | 0.0324 | 0.0100 | 3.229 | 0.001 |
| ce_loss_recovered | 0.0434 | 0.0095 | 4.572 | 0.000 |
| width | 0.0025 | 0.0054 | 0.460 | 0.646 |
| architecture_GatedSAE | 0.0213 | 0.0139 | 1.538 | 0.125 |
| architecture_JumpRelu | 0.0039 | 0.0138 | 0.283 | 0.778 |
| architecture_MatryoshkaBatchTopK | 0.0167 | 0.0140 | 1.196 | 0.233 |
| architecture_PAnneal | 0.0269 | 0.0138 | 1.946 | 0.053 |
| architecture_Standard | -0.0157 | 0.0108 | -1.447 | 0.149 |
| architecture_TopK | -0.0018 | 0.0103 | -0.171 | 0.864 |

### ravel_isolation (R² = 0.150, N = 312)

| Variable | Coefficient | SE | t-stat | p-value |
|----------|-------------|----|--------|---------|
| absorption_mean | -0.0235 | 0.0057 | -4.107 | 0.000 |
| l0 | 0.0097 | 0.0101 | 0.962 | 0.337 |
| ce_loss_recovered | 0.0336 | 0.0096 | 3.508 | 0.001 |
| width | -0.0000 | 0.0054 | -0.003 | 0.998 |
| architecture_GatedSAE | -0.0074 | 0.0140 | -0.532 | 0.595 |
| architecture_JumpRelu | 0.0068 | 0.0139 | 0.490 | 0.624 |
| architecture_MatryoshkaBatchTopK | 0.0106 | 0.0141 | 0.750 | 0.454 |
| architecture_PAnneal | 0.0074 | 0.0139 | 0.530 | 0.597 |
| architecture_Standard | 0.0215 | 0.0109 | 1.971 | 0.050 |
| architecture_TopK | 0.0183 | 0.0104 | 1.755 | 0.080 |
