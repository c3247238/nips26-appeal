# E1 Full: Downstream Causal Cost Meta-Analysis (Pythia-160M)

**Task:** e1_full  
**SAEs analyzed:** 126 (Pythia-160M from SAEBench)  
**Data source:** SAEBench HF dataset (adamkarvonen/sae_bench_results_0125)  

## Descriptive Statistics

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| absorption_mean | 0.1410 | 0.1780 | 0.0074 | 0.7316 |
| l0 | 212.7917 | 209.7400 | 18.7975 | 999.9365 |
| ce_loss_recovered | 0.9776 | 0.0195 | 0.9080 | 0.9993 |
| explained_variance | 0.8720 | 0.0758 | 0.6947 | 0.9911 |
| sparse_probing_acc | 0.9339 | 0.0043 | 0.9222 | 0.9418 |
| sparse_probing_top1_acc | 0.7511 | 0.0227 | 0.7033 | 0.8049 |
| scr_metric_10 | 0.1720 | 0.0827 | -0.1452 | 0.2872 |
| scr_metric_20 | 0.2138 | 0.1283 | -0.2884 | 0.3815 |
| scr_metric_50 | 0.2186 | 0.1954 | -0.4224 | 0.4995 |
| tpp_metric_10 | 0.1370 | 0.0959 | 0.0075 | 0.3122 |
| tpp_metric_20 | 0.1847 | 0.0919 | 0.0079 | 0.3316 |
| tpp_metric_50 | 0.2259 | 0.0957 | 0.0228 | 0.3604 |
| width | 28672.0000 | 26651.0828 | 4096.0000 | 65536.0000 |
| layer | 8.0000 | 0.0000 | 8.0000 | 8.0000 |

## Correlations (absorption vs downstream)

| Outcome | Pearson r | p-value | N |
|---------|-----------|---------|---|
| sparse_probing_acc | 0.274 | 0.002 | 126 |
| sparse_probing_top1_acc | -0.592 | 0.000 | 126 |
| scr_metric_10 | 0.009 | 0.919 | 126 |
| tpp_metric_10 | -0.579 | 0.000 | 126 |

## Partial Correlations (controlling for L0 and CE loss recovered)

| Outcome | Partial r | p-value | N |
|---------|-----------|---------|---|
| sparse_probing_acc | 0.263 | 0.003 | 126 |
| sparse_probing_top1_acc | -0.573 | 0.000 | 126 |
| scr_metric_10 | -0.293 | 0.001 | 126 |
| tpp_metric_10 | -0.481 | 0.000 | 126 |

## OLS Regression Summaries

### sparse_probing_acc (R² = 0.670, N = 126)

| Variable | Coefficient | SE | t-stat | p-value |
|----------|-------------|----|--------|---------|
| absorption_mean | -0.0001 | 0.0003 | -0.280 | 0.780 |
| l0 | -0.0041 | 0.0004 | -11.692 | 0.000 |
| ce_loss_recovered | 0.0037 | 0.0004 | 9.848 | 0.000 |
| width | 0.0011 | 0.0003 | 4.032 | 0.000 |
| layer | -0.0000 | 0.0000 | -0.000 | 1.000 |
| architecture_GatedSAE | 0.0019 | 0.0006 | 3.138 | 0.002 |
| architecture_JumpRelu | -0.0005 | 0.0006 | -0.833 | 0.407 |
| architecture_MatryoshkaBatchTopK | -0.0013 | 0.0006 | -2.096 | 0.038 |
| architecture_PAnneal | 0.0021 | 0.0006 | 3.531 | 0.001 |
| architecture_Standard | 0.0072 | 0.0007 | 10.778 | 0.000 |
| architecture_TopK | 0.0000 | 0.0006 | 0.022 | 0.982 |

### sparse_probing_top1_acc (R² = 0.559, N = 126)

| Variable | Coefficient | SE | t-stat | p-value |
|----------|-------------|----|--------|---------|
| absorption_mean | -0.0097 | 0.0019 | -5.130 | 0.000 |
| l0 | -0.0101 | 0.0021 | -4.702 | 0.000 |
| ce_loss_recovered | 0.0147 | 0.0022 | 6.547 | 0.000 |
| width | -0.0056 | 0.0017 | -3.259 | 0.001 |
| layer | -0.0000 | 0.0000 | -0.000 | 1.000 |
| architecture_GatedSAE | -0.0139 | 0.0037 | -3.817 | 0.000 |
| architecture_JumpRelu | -0.0036 | 0.0036 | -1.017 | 0.311 |
| architecture_MatryoshkaBatchTopK | 0.0036 | 0.0036 | 0.999 | 0.320 |
| architecture_PAnneal | -0.0090 | 0.0036 | -2.509 | 0.013 |
| architecture_Standard | 0.0077 | 0.0040 | 1.917 | 0.058 |
| architecture_TopK | -0.0003 | 0.0036 | -0.072 | 0.943 |

### scr_metric_10 (R² = 0.536, N = 126)

| Variable | Coefficient | SE | t-stat | p-value |
|----------|-------------|----|--------|---------|
| absorption_mean | -0.0268 | 0.0071 | -3.776 | 0.000 |
| l0 | -0.0712 | 0.0080 | -8.899 | 0.000 |
| ce_loss_recovered | 0.0207 | 0.0084 | 2.465 | 0.015 |
| width | 0.0030 | 0.0064 | 0.463 | 0.644 |
| layer | 0.0000 | 0.0000 | 0.000 | 1.000 |
| architecture_GatedSAE | 0.0445 | 0.0136 | 3.261 | 0.001 |
| architecture_JumpRelu | 0.0308 | 0.0133 | 2.308 | 0.023 |
| architecture_MatryoshkaBatchTopK | -0.0430 | 0.0135 | -3.178 | 0.002 |
| architecture_PAnneal | 0.0495 | 0.0134 | 3.697 | 0.000 |
| architecture_Standard | 0.0219 | 0.0150 | 1.461 | 0.147 |
| architecture_TopK | 0.0109 | 0.0134 | 0.813 | 0.418 |

### tpp_metric_10 (R² = 0.786, N = 126)

| Variable | Coefficient | SE | t-stat | p-value |
|----------|-------------|----|--------|---------|
| absorption_mean | -0.0275 | 0.0056 | -4.913 | 0.000 |
| l0 | 0.0225 | 0.0063 | 3.558 | 0.001 |
| ce_loss_recovered | 0.0269 | 0.0066 | 4.057 | 0.000 |
| width | -0.0059 | 0.0050 | -1.169 | 0.245 |
| layer | 0.0000 | 0.0000 | 0.000 | 1.000 |
| architecture_GatedSAE | -0.0469 | 0.0108 | -4.362 | 0.000 |
| architecture_JumpRelu | -0.0062 | 0.0105 | -0.586 | 0.559 |
| architecture_MatryoshkaBatchTopK | 0.0314 | 0.0107 | 2.941 | 0.004 |
| architecture_PAnneal | -0.0632 | 0.0106 | -5.979 | 0.000 |
| architecture_Standard | -0.1335 | 0.0118 | -11.304 | 0.000 |
| architecture_TopK | 0.0043 | 0.0106 | 0.406 | 0.685 |
