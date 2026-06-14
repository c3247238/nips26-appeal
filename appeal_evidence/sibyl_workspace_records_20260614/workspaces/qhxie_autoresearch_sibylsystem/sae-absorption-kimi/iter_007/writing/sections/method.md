## 3. Method

### 3.1 Experimental Framework

We use SynthSAEBench, a synthetic data generator with known parent-child feature hierarchies. Each synthetic feature has a known semantic relationship, enabling exact absorption detection without probe-based approximations. The data has 1024 features, 256 hidden dimensions, and a hierarchy of 32 root nodes with branching factor 4 and depth 3.

**Architecture variants.** We evaluate six SAE architectures plus a random control:

| Variant | Core Mechanism | Prior Absorption Claim |
|---------|---------------|----------------------|
| Baseline L1 | L1 sparse penalty | Reference (high) |
| TopK | Explicit k-sparse selection | Lower than ReLU [10] |
| Matryoshka | Nested multi-scale dictionary | ~90% reduction [3] |
| OrtSAE | Decoder orthogonality penalty | ~65% reduction [4] |
| Gated | Separate gate/magnitude paths | Moderate reduction [5] |
| Random | Untrained random dictionary | Validation baseline |

**Training.** All variants are trained for 2M tokens with batch size 1024, learning rate 1e-3, on a single GPU. Each variant is run with 5 random seeds (42, 123, 456, 789, 1011).

**Absorption metric.** We compute ground-truth absorption rate as the fraction of parent feature firings where child-matching SAE latents also activate (threshold = 0.05). This is exact because the synthetic data has known feature hierarchies.

### 3.2 L0-Matching Attempt

We attempted to match Baseline L1's L0 to other architectures via lambda sweep. The protocol was:
1. Train each variant with default hyperparameters and measure its achieved L0
2. Train Baseline L1 with tuned lambda to approach that L0
3. Compare absorption rates

**Result.** Baseline L1 cannot match the low L0 values of TopK/Matryoshka. Even with lambda spanning a 40× range (5e-5 to 0.002), Baseline L0 decreases by only ~16% (1082→995). This demonstrates that L1 regularization cannot achieve arbitrary sparsity levels in this synthetic setting, making true L0-matching with TopK/Matryoshka impossible.

### 3.3 Dose-Response Causality Study

To test whether absorption causally predicts downstream interpretability, we fix the Baseline L1 architecture and vary lambda across five levels (5e-5 to 2e-3). This creates a sparsity gradient that naturally varies absorption. We measure:
- Absorption rate (independent variable)
- Feature recovery MCC (dependent variable)

If absorption causes downstream harm, MCC should decrease monotonically with absorption.

### 3.4 Data Integrity Pipeline

Following lessons from prior iterations, we enforce five validation checks:
1. **Feature count verification**: `num_features` in results matches the plan spec
2. **Cross-file duplicate detection**: MD5 hash of replicate metrics across variants
3. **Output file existence**: Every planned experiment has a corresponding result file
4. **Numerical audit**: All paper numbers traceable to a single source-of-truth JSON
5. **Convergence diagnostics**: Training loss curves and final loss values recorded
