# E4 Pilot: Pareto Tradeoff Validation (BatchTopK vs TopK)

**Task:** e4_pilot  
**Stratum:** Pythia-160M, resid_post_layer_8, width=16384  
**Total Time:** 0.0 min  

## Selected Families

- BatchTopK
- TopK

## Pareto Fronts (absorption vs explained_variance)

### BatchTopK (1 non-dominated)

| Trainer | Absorption | Explained Var | Dead Neurons | Sparse Probing Acc |
|---------|------------|---------------|--------------|--------------------|
| trainer_3 | 0.0346 | 0.9794 | 0.3577 | 0.9386 |

### TopK (1 non-dominated)

| Trainer | Absorption | Explained Var | Dead Neurons | Sparse Probing Acc |
|---------|------------|---------------|--------------|--------------------|
| trainer_3 | 0.0757 | 0.9768 | 0.1649 | 0.9375 |

## Stochastic Dominance Tests (Mann-Whitney U)

| Family A | Family B | Metric | U-stat | p-value | Significant |
|----------|----------|--------|--------|---------|-------------|
| BatchTopK | TopK | official_absorption_full | 1.0 | 0.667 | No |
| BatchTopK | TopK | explained_variance | 3.0 | 0.667 | No |
| BatchTopK | TopK | dead_neuron_fraction | 4.0 | 0.333 | No |
| BatchTopK | TopK | sparse_probing_acc | 3.0 | 0.667 | No |