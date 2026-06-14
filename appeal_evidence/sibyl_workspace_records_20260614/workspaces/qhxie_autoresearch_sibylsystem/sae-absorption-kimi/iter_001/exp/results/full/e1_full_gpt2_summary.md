# E1 Full: GPT-2 Small Pareto Evaluation

**Checkpoints Evaluated:** 27/27
**Pareto Points:** 17
**Total Time:** 675s

## Architecture Family Summary

| Architecture | Count | Absorption | Hedging | Explained Var | CE Recovered | L0 | Dead Neurons |
|---|---|---|---|---|---|---|---|
| standard | 23 | 0.015 | 0.833 | 0.830 | 1.054 | 33.9 | 0.1971 |
| feature_splitting | 4 | 0.000 | 0.888 | 0.982 | 1.172 | 39.5 | 0.0000 |

## Per-Checkpoint Results

| Checkpoint | Arch | L0 | ExpVar | Dead | CE Rec | Absorption | Hedging | Pareto |
|---|---|---|---|---|---|---|---|---|
| res-jb-l8-resid_pre-24k | standard | 67.0 | 0.990 | 0.0005 | 1.135 | 0.000 | 0.985 | Yes |
| res-jb-l0-resid_pre | standard | 16.8 | 0.928 | 0.1951 | 1.017 | 0.000 | 0.806 | Yes |
| res-jb-l4-resid_pre | standard | 34.7 | 0.996 | 0.0255 | 1.044 | 0.000 | 0.985 | Yes |
| resid-post-v5-32k-l8 | standard | 32.0 | 0.987 | 0.0663 | 1.003 | 0.000 | 1.000 | Yes |
| resid-post-v5-128k-l8 | standard | 32.0 | 0.989 | 0.2710 | 1.003 | 0.000 | 0.894 | No |
| resid-post-v5-32k-l0 | standard | 32.0 | 0.984 | 0.0819 | 0.991 | 0.000 | 0.745 | No |
| resid-post-v5-32k-l4 | standard | 32.0 | 0.996 | 0.0374 | 1.031 | 0.000 | 0.985 | Yes |
| resid-mid-v5-32k-l8 | standard | 32.0 | 0.988 | 0.0732 | 1.014 | 0.000 | 1.000 | Yes |
| resid-mid-v5-128k-l8 | standard | 32.0 | 0.991 | 0.2846 | 1.016 | 0.000 | 0.909 | No |
| mlp-out-v5-32k-l8 | standard | 32.0 | 0.732 | 0.0027 | 0.879 | 0.000 | 0.611 | Yes |
| mlp-out-v5-128k-l8 | standard | 32.0 | 0.773 | 0.1137 | 0.936 | 0.000 | 0.548 | No |
| mlp-out-v5-32k-l0 | standard | 32.0 | 0.972 | 0.0379 | 0.992 | 0.000 | 0.811 | No |
| attn-out-v5-32k-l8 | standard | 32.0 | 0.804 | 0.0873 | 0.680 | 0.000 | 0.323 | No |
| attn-out-v5-128k-l8 | standard | 32.0 | 0.819 | 0.2030 | 0.595 | 0.000 | 0.284 | No |
| attn-out-v5-32k-l0 | standard | 32.0 | 0.973 | 0.1002 | 0.985 | 0.000 | 0.955 | No |
| hook-z-kk-l8 | standard | 26.3 | 0.632 | 0.4397 | 3.290 | 0.345 | 0.751 | Yes |
| hook-z-kk-l0 | standard | 7.5 | 0.662 | 0.3227 | 0.815 | 0.000 | 0.924 | Yes |
| mlp-tm-l8 | standard | 30.8 | 0.575 | 0.0028 | 0.606 | 0.000 | 0.909 | Yes |
| mlp-tm-l0 | standard | 25.2 | 0.927 | 0.0600 | 1.010 | 0.000 | 0.760 | Yes |
| res-jb-fs-768 | feature_splitting | 33.9 | 0.976 | 0.0000 | 1.181 | 0.000 | 0.820 | Yes |
| res-jb-fs-1536 | feature_splitting | 38.6 | 0.981 | 0.0000 | 1.173 | 0.000 | 0.909 | Yes |
| res-jb-fs-3072 | feature_splitting | 41.8 | 0.984 | 0.0000 | 1.175 | 0.000 | 0.884 | Yes |
| res-jb-fs-6144 | feature_splitting | 43.7 | 0.986 | 0.0000 | 1.158 | 0.000 | 0.939 | Yes |
| res-sce-ajt-l6 | standard | 32.6 | 0.440 | 0.5997 | 1.012 | 0.000 | 1.000 | No |
| res-scl-ajt-l6 | standard | 67.4 | 0.728 | 0.4324 | 1.067 | 0.000 | 1.000 | Yes |
| res-sle-ajt-l6 | standard | 45.9 | 0.531 | 0.5423 | 1.016 | 0.000 | 0.970 | No |
| res-sll-ajt-l6 | standard | 41.2 | 0.675 | 0.5521 | 1.100 | 0.000 | 1.000 | Yes |

## Dominance Tests (Mann-Whitney U)

### standard_vs_feature_splitting
| Metric | U | p | Mean A | Mean B |
|---|---|---|---|---|
| absorption | 48.0 | 0.7545 | 0.015 | 0.000 |
| hedging | 50.0 | 0.8104 | 0.833 | 0.888 |
| explained_variance | 31.0 | 0.3362 | 0.830 | 0.982 |
| ce_loss_recovered | 4.0 | 0.0014 | 1.054 | 1.172 |
