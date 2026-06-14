# E4 Full: Multi-Objective Pareto Analysis (Pythia-160M)

**Task:** e4_full  
**Total Time:** 0.0 min  
**Checkpoints Evaluated:** 14  
**Families:** 7  
**Strata:** 1  

## Stratum: resid_post_layer_8, width=16384

**Checkpoints:** 14  
**Families:** BatchTopK, GatedSAE, JumpRelu, MatryoshkaBatchTopK, PAnneal, Standard, TopK

### Pareto Fronts (2D: absorption vs explained_variance)

#### BatchTopK (1 non-dominated)

| Trainer | Absorption | Explained Var | Dead Neurons | Sparse Probing Acc |
|---------|------------|---------------|--------------|--------------------|
| trainer_3 | 0.0346 | 0.9794 | 0.3577 | 0.9386 |

#### GatedSAE (1 non-dominated)

| Trainer | Absorption | Explained Var | Dead Neurons | Sparse Probing Acc |
|---------|------------|---------------|--------------|--------------------|
| trainer_0 | 0.0070 | 0.9944 | 0.5087 | 0.9350 |

#### JumpRelu (1 non-dominated)

| Trainer | Absorption | Explained Var | Dead Neurons | Sparse Probing Acc |
|---------|------------|---------------|--------------|--------------------|
| trainer_3 | 0.0225 | 0.9769 | 0.6106 | 0.9348 |

#### MatryoshkaBatchTopK (1 non-dominated)

| Trainer | Absorption | Explained Var | Dead Neurons | Sparse Probing Acc |
|---------|------------|---------------|--------------|--------------------|
| trainer_3 | 0.0198 | 0.9778 | 0.4755 | 0.9372 |

#### PAnneal (1 non-dominated)

| Trainer | Absorption | Explained Var | Dead Neurons | Sparse Probing Acc |
|---------|------------|---------------|--------------|--------------------|
| trainer_0 | 0.0117 | 0.9887 | 0.3703 | 0.9383 |

#### Standard (1 non-dominated)

| Trainer | Absorption | Explained Var | Dead Neurons | Sparse Probing Acc |
|---------|------------|---------------|--------------|--------------------|
| trainer_0 | 0.0273 | 0.9796 | 0.2532 | 0.9389 |

#### TopK (1 non-dominated)

| Trainer | Absorption | Explained Var | Dead Neurons | Sparse Probing Acc |
|---------|------------|---------------|--------------|--------------------|
| trainer_3 | 0.0757 | 0.9768 | 0.1649 | 0.9375 |

## Pairwise Stochastic Dominance Tests (Mann-Whitney U)

| Family A | Family B | Metric | U-stat | p-value | Significant |
|----------|----------|--------|--------|---------|-------------|
| BatchTopK | GatedSAE | official_absorption_full | 3.0 | 0.667 | No |
| BatchTopK | GatedSAE | explained_variance | 1.0 | 0.667 | No |
| BatchTopK | GatedSAE | dead_neuron_fraction | 2.0 | 1.000 | No |
| BatchTopK | GatedSAE | sparse_probing_acc | 3.0 | 0.667 | No |
| BatchTopK | JumpRelu | official_absorption_full | 3.0 | 0.667 | No |
| BatchTopK | JumpRelu | explained_variance | 3.0 | 0.667 | No |
| BatchTopK | JumpRelu | dead_neuron_fraction | 0.0 | 0.333 | No |
| BatchTopK | JumpRelu | sparse_probing_acc | 4.0 | 0.333 | No |
| BatchTopK | MatryoshkaBatchTopK | official_absorption_full | 3.0 | 0.667 | No |
| BatchTopK | MatryoshkaBatchTopK | explained_variance | 3.0 | 0.667 | No |
| BatchTopK | MatryoshkaBatchTopK | dead_neuron_fraction | 1.0 | 0.667 | No |
| BatchTopK | MatryoshkaBatchTopK | sparse_probing_acc | 3.0 | 0.667 | No |
| BatchTopK | PAnneal | official_absorption_full | 3.0 | 0.667 | No |
| BatchTopK | PAnneal | explained_variance | 1.0 | 0.667 | No |
| BatchTopK | PAnneal | dead_neuron_fraction | 1.0 | 0.667 | No |
| BatchTopK | PAnneal | sparse_probing_acc | 2.0 | 1.000 | No |
| BatchTopK | Standard | official_absorption_full | 3.0 | 0.667 | No |
| BatchTopK | Standard | explained_variance | 1.0 | 0.667 | No |
| BatchTopK | Standard | dead_neuron_fraction | 3.0 | 0.667 | No |
| BatchTopK | Standard | sparse_probing_acc | 1.0 | 0.667 | No |
| BatchTopK | TopK | official_absorption_full | 1.0 | 0.667 | No |
| BatchTopK | TopK | explained_variance | 3.0 | 0.667 | No |
| BatchTopK | TopK | dead_neuron_fraction | 4.0 | 0.333 | No |
| BatchTopK | TopK | sparse_probing_acc | 3.0 | 0.667 | No |
| GatedSAE | JumpRelu | official_absorption_full | 1.0 | 0.667 | No |
| GatedSAE | JumpRelu | explained_variance | 3.0 | 0.667 | No |
| GatedSAE | JumpRelu | dead_neuron_fraction | 0.0 | 0.333 | No |
| GatedSAE | JumpRelu | sparse_probing_acc | 4.0 | 0.333 | No |
| GatedSAE | MatryoshkaBatchTopK | official_absorption_full | 1.0 | 0.667 | No |
| GatedSAE | MatryoshkaBatchTopK | explained_variance | 3.0 | 0.667 | No |
| GatedSAE | MatryoshkaBatchTopK | dead_neuron_fraction | 2.0 | 1.000 | No |
| GatedSAE | MatryoshkaBatchTopK | sparse_probing_acc | 3.0 | 0.667 | No |
| GatedSAE | PAnneal | official_absorption_full | 2.0 | 1.000 | No |
| GatedSAE | PAnneal | explained_variance | 3.0 | 0.667 | No |
| GatedSAE | PAnneal | dead_neuron_fraction | 2.0 | 1.000 | No |
| GatedSAE | PAnneal | sparse_probing_acc | 2.0 | 1.000 | No |
| GatedSAE | Standard | official_absorption_full | 2.0 | 1.000 | No |
| GatedSAE | Standard | explained_variance | 3.0 | 0.667 | No |
| GatedSAE | Standard | dead_neuron_fraction | 2.0 | 1.000 | No |
| GatedSAE | Standard | sparse_probing_acc | 1.0 | 0.667 | No |
| GatedSAE | TopK | official_absorption_full | 1.0 | 0.667 | No |
| GatedSAE | TopK | explained_variance | 3.0 | 0.667 | No |
| GatedSAE | TopK | dead_neuron_fraction | 2.0 | 1.000 | No |
| GatedSAE | TopK | sparse_probing_acc | 3.0 | 0.667 | No |
| JumpRelu | MatryoshkaBatchTopK | official_absorption_full | 3.0 | 0.667 | No |
| JumpRelu | MatryoshkaBatchTopK | explained_variance | 1.0 | 0.667 | No |
| JumpRelu | MatryoshkaBatchTopK | dead_neuron_fraction | 4.0 | 0.333 | No |
| JumpRelu | MatryoshkaBatchTopK | sparse_probing_acc | 2.0 | 1.000 | No |
| JumpRelu | PAnneal | official_absorption_full | 3.0 | 0.667 | No |
| JumpRelu | PAnneal | explained_variance | 1.0 | 0.667 | No |
| JumpRelu | PAnneal | dead_neuron_fraction | 4.0 | 0.333 | No |
| JumpRelu | PAnneal | sparse_probing_acc | 0.0 | 0.333 | No |
| JumpRelu | Standard | official_absorption_full | 2.0 | 1.000 | No |
| JumpRelu | Standard | explained_variance | 1.0 | 0.667 | No |
| JumpRelu | Standard | dead_neuron_fraction | 4.0 | 0.333 | No |
| JumpRelu | Standard | sparse_probing_acc | 0.0 | 0.333 | No |
| JumpRelu | TopK | official_absorption_full | 1.0 | 0.667 | No |
| JumpRelu | TopK | explained_variance | 2.0 | 1.000 | No |
| JumpRelu | TopK | dead_neuron_fraction | 4.0 | 0.333 | No |
| JumpRelu | TopK | sparse_probing_acc | 1.0 | 0.667 | No |
| MatryoshkaBatchTopK | PAnneal | official_absorption_full | 3.0 | 0.667 | No |
| MatryoshkaBatchTopK | PAnneal | explained_variance | 1.0 | 0.667 | No |
| MatryoshkaBatchTopK | PAnneal | dead_neuron_fraction | 2.0 | 1.000 | No |
| MatryoshkaBatchTopK | PAnneal | sparse_probing_acc | 1.0 | 0.667 | No |
| MatryoshkaBatchTopK | Standard | official_absorption_full | 2.0 | 1.000 | No |
| MatryoshkaBatchTopK | Standard | explained_variance | 1.0 | 0.667 | No |
| MatryoshkaBatchTopK | Standard | dead_neuron_fraction | 4.0 | 0.333 | No |
| MatryoshkaBatchTopK | Standard | sparse_probing_acc | 1.0 | 0.667 | No |
| MatryoshkaBatchTopK | TopK | official_absorption_full | 1.0 | 0.667 | No |
| MatryoshkaBatchTopK | TopK | explained_variance | 2.0 | 1.000 | No |
| MatryoshkaBatchTopK | TopK | dead_neuron_fraction | 4.0 | 0.333 | No |
| MatryoshkaBatchTopK | TopK | sparse_probing_acc | 1.0 | 0.667 | No |
| PAnneal | Standard | official_absorption_full | 1.0 | 0.667 | No |
| PAnneal | Standard | explained_variance | 3.0 | 0.667 | No |
| PAnneal | Standard | dead_neuron_fraction | 4.0 | 0.333 | No |
| PAnneal | Standard | sparse_probing_acc | 1.0 | 0.667 | No |
| PAnneal | TopK | official_absorption_full | 1.0 | 0.667 | No |
| PAnneal | TopK | explained_variance | 3.0 | 0.667 | No |
| PAnneal | TopK | dead_neuron_fraction | 4.0 | 0.333 | No |
| PAnneal | TopK | sparse_probing_acc | 3.0 | 0.667 | No |
| Standard | TopK | official_absorption_full | 1.0 | 0.667 | No |
| Standard | TopK | explained_variance | 3.0 | 0.667 | No |
| Standard | TopK | dead_neuron_fraction | 4.0 | 0.333 | No |
| Standard | TopK | sparse_probing_acc | 3.0 | 0.667 | No |

## Summary Statistics

- Total dominance tests: 84
- Significant dominance tests (p < 0.05): 0