# Testable Hypotheses

## H1: Proximal Unification Completeness

**Statement**: All four WD sub-approaches (scheduling, alignment-aware, decoupled, norm-matched) can be expressed as instances of a time-varying proximal operator prox_{eta*R_t}(w) with specific structural choices for the regularizer R_t. The resulting taxonomy covers all first-order methods with additive WD.

**Expected outcome**: Each method maps to a distinct proximal operator with a closed-form expression. The soft-CWD approximation (sigmoid with beta -> infinity) converges to hard CWD with bounded error O(1/beta).

**Falsification**: If any commonly used WD method cannot be expressed in the proximal framework (e.g., its WD operator is not the gradient of any regularizer), the unification claim is weakened. The most likely failure case is CWD's binary mask, which requires the smooth approximation.

**Test**: Mathematical derivation + experimental verification that soft CWD with beta=100 is indistinguishable from hard CWD (<0.05% accuracy difference).

---

## H2: WD Stability Condition

**Statement**: The time-varying Lyapunov function V_t = f(w_t) + R_t(w_t) decreases in expectation if and only if the regularizer changes at most as fast as the optimization is converging:

```
|R_{t+1}(w_t) - R_t(w_t)| <= (1 - rho) * eta * ||nabla f(w_t)||^2 + sigma^2
```

WD warmup from 0 to lambda_max in K < 1/(eta*lambda_max) steps violates this condition when gradients are small (flat initialization region).

**Expected outcome**: Training runs with WD warmup K < K_critical show elevated loss variance in the first 1000 steps. K_critical approximately equals 1/(eta*lambda_max).

**Falsification**: If loss variance does not change across warmup durations K, the stability condition is vacuous. If all practical schedules trivially satisfy the condition, it has no discriminative power.

**Test**: ResNet-20/CIFAR-10 with eta=0.1, lambda=0.001. Vary K in {1, 10, 50, 200, 1000}. Measure training loss variance in first 1000 steps. Predict K < 100 shows elevated variance.

---

## H3: Budget Equivalence at Scale

**Statement**: Under equal compute budgets, temporal WD scheduling (cosine, linear, SWD, inverse-sqrt) achieves statistically indistinguishable final accuracy from optimally tuned constant WD when the time-averaged WD strength is matched. This holds for CIFAR-10/100 and ImageNet across ResNet and VGG architectures.

**Expected outcome**: |accuracy(schedule) - accuracy(constant at mean lambda)| < 0.3% for all temporal schedules, with p > 0.05 on paired t-test.

**Falsification**: Any temporal schedule achieves >0.3% improvement over mean-matched constant WD with p < 0.05 across 3+ seeds on 2+ benchmarks.

**Test**: Implement 5 temporal WD schedules. For each, compute mean lambda and compare against constant WD at same mean. CIFAR-10/100 (ResNet-20, VGG-16-BN) + ImageNet (ResNet-50), 3 seeds each.

**Prior evidence**: Iterations 0-2 confirmed budget equivalence on CIFAR-10/ResNet-20. Extension to CIFAR-100 and ImageNet is the key validation.

---

## H4: CWD Mechanism Decomposition

**Statement**: CWD's improvement over standard AdamW is substantially (>50%) attributable to reduced effective WD strength (because the mask blocks ~50% of parameters from receiving WD), rather than to alignment-awareness as claimed.

**Expected outcome**:
- CWD mask ratio is approximately 50% and roughly constant during training
- Effective-lambda-matched constant WD achieves >50% of CWD's improvement over standard AdamW
- Random binary mask with same sparsity achieves comparable performance to CWD

**Falsification**: CWD outperforms effective-lambda-matched constant WD by >0.3% (p<0.05) AND random-mask WD with matched sparsity performs >0.3% worse than CWD (both conditions required).

**Test**: CWD falsification battery on CIFAR-10/100 and ImageNet. Track mask ratio, effective lambda, and weight norm trajectories.

**Prior evidence**: Iterations 0-2 found alignment signal "uninformative at nonconvex scale." CWD's own ablation showed random mask underperforms, but did not test effective-lambda matching.

---

## H5: Metric Predictiveness

**Statement**: The proposed standardized metrics (CSI and AIS) have predictive power for WD method rankings. CSI correlates with final test accuracy (Spearman rho > 0.5), and AIS values differentiate alignment-aware from alignment-agnostic methods.

**Expected outcome**:
- CSI rank-orders methods consistently with accuracy on at least 3/4 benchmark settings
- AIS values are low (<0.1) for most settings, confirming alignment is uninformative (a negative but valuable result)
- BEM normalization does not change method rankings (confirming budget equivalence)

**Falsification**: CSI does not correlate with accuracy (rho < 0.3) across method-architecture-dataset combinations. AIS is consistently high (>0.2), contradicting the alignment limitation finding.

**Test**: Compute all three metrics for 7 methods on 4 benchmark settings. Cross-validate: compute metrics on CIFAR-10, predict rankings on CIFAR-100 and ImageNet.

---

## H6: Spatial Modulation Superiority

**Statement**: The spatial dimension of dynamic WD (per-parameter, per-layer modulation) provides genuine benefit through effective learning rate redistribution, while the temporal dimension (scheduling lambda over time) provides near-zero benefit under budget equivalence. The mechanism of spatial modulation is effective LR redistribution, not regularization optimization.

**Expected outcome**:
- Per-layer WD (AlphaDecay-style) shows measurable improvement over constant WD on architectures with heterogeneous layer structures (VGG, ViT)
- CWD's spatial modulation (per-parameter mask) provides benefit even when temporal scheduling does not
- Effective-LR-matched control (direct LR adjustment matching CWD's effective LR trajectory with constant WD) reproduces CWD's performance

**Falsification**: Temporal scheduling shows significant improvement (>0.3%, p<0.05) even after budget normalization, or spatial modulation shows no benefit beyond effective LR redistribution.

**Test**: Factor analysis separating temporal and spatial contributions. Test effective-LR equivalence for CWD.

---

## H7: Composition Orthogonality

**Statement**: Temporal scheduling and alignment-aware modulation are orthogonal (compose beneficially), while alignment-aware and norm-matched modulation are redundant (CWD already controls norms as a side effect).

**Expected outcome**:
- CWD + Cosine WD outperforms both CWD alone and Cosine alone (beneficial composition)
- CWD + AdamWN does not outperform the better of CWD and AdamWN (redundant composition)

**Falsification**: CWD + Cosine performs no better than CWD alone (composition not beneficial), or CWD + AdamWN shows clear improvement (they address different phenomena).

**Test**: Composition experiments on CIFAR-100/ResNet-20 and VGG-16-BN, 3 seeds.
