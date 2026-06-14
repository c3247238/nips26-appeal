# 6. Discussion

## 6.1 Why Constant WD Wins at Standard $\rho$

Theorem 1 predicts this outcome through a quantitative tradeoff. At standard $\rho \approx 0.5$ under AdamW with $\lambda = 5 \times 10^{-4}$, batch-normalized networks satisfy two conditions that make constant WD optimal:

1. **BN's scale-invariance limits alignment benefit.** BN layers are invariant to weight scaling: $\text{BN}(\alpha \theta) = \text{BN}(\theta)$. This drives weights toward an equilibrium norm (Kosson et al., 2023), making the gradient-weight alignment signal $\hat{\delta}_t$ geometrically constrained rather than freely informative. AIS values in our BN experiments (0.18--0.40) fall below the Theorem 1 threshold.

2. **AdamW's per-parameter scaling subsumes $\phi$ effects.** AdamW's second-moment normalization ($\hat{v}_t^{-1/2}$) already provides per-parameter adaptive scaling. The additional modulation from $\phi$ is redundant: it adds noise to an already well-calibrated update without providing new information. This explains why even a 10$\times$ variation in effective WD budget (BEM from 0.0 to 0.90) produces $< 0.25\%$ accuracy change.

## 6.2 When Adaptive WD Should Help

The theory predicts three regimes where adaptive WD becomes beneficial:

**Elevated $\rho$.** When $\rho > \rho^*$, the alignment benefit exceeds the stability cost (Theorem 1). Our $\rho$-low data (constant only, 90.13% $\pm$ 0.07) and SGD data ($\rho \approx 0.005$, spread = 0.91%) provide two points on the spread-vs-$\rho$ curve. Full $\rho$-high ($\rho \approx 5.0$) data is needed to confirm the predicted increase in method sensitivity.

**Without BN.** NoBN experiments show higher AIS ($\sim$0.50 vs. $\sim$0.35), consistent with the prediction that removing BN's scale-invariance increases alignment informativeness. The available NoBN data (2 methods, constant slightly outperforming CWD by 0.10%) does not yet resolve whether the AIS increase is sufficient to cross the Theorem 1 threshold, but the direction is consistent with theory.

**Large-batch training.** Proposition 1's noise constraint relaxes as batch size increases: $\sigma^2/n$ decreases, lowering the Theorem 1 threshold. At LLM-scale batch sizes (4K--64K tokens), the raw alignment signal becomes more informative, potentially enabling direct alignment-based WD modulation without EMA smoothing.

## 6.3 PMP-WD as a Principled Alternative

PMP-WD is the first WD algorithm derived from an optimality condition on the training dynamics. Three properties distinguish it from existing methods:

1. **State-feedback vs. feedforward.** PMP-WD measures $\hat{\rho}_t$ and corrects deviations from $\rho^*$ in real time. Cosine schedules and AdamC (Defazio 2025) follow predetermined trajectories regardless of actual training state. When the actual trajectory deviates from the plan---due to data distribution shift, batch noise, or learning rate schedule interactions---feedforward methods cannot compensate; PMP-WD does.

2. **Dual derivation.** The same functional form emerges from both stochastic PMP (control theory) and RG beta function analysis (statistical physics). This convergence from independent mathematical frameworks provides stronger theoretical grounding than either derivation alone.

3. **Minimal overhead.** PMP-WD requires only per-layer scalar EMA of $\|g_l\| / \|\theta_l\|$---a computation already performed for gradient norm tracking in standard training loops. The additional cost is $O(L)$ per step, negligible relative to the forward/backward pass.

PMP-WD is not empirically validated in this paper; its implementation and evaluation at elevated $\rho$ is deferred to future work.

## 6.4 Implications for Practitioners

Three practical recommendations follow from the theory and experiments:

1. **Under AdamW at standard hyperparameters ($\lambda \sim 10^{-4}$--$10^{-3}$, BN architectures): use constant WD.** Dynamic scheduling adds implementation complexity without accuracy benefit. The 0.25% Phi spread on CIFAR-10 and 0.16% on VGG-16-BN are within inter-seed variance.

2. **New dynamic WD methods should demonstrate gains at elevated $\rho$ or at scale.** Standard CIFAR settings with AdamW are in the Phi-invariance regime; positive results here do not generalize to the claim that the method is universally better. The gradient-to-weight ratio should be reported as a context variable.

3. **If using alignment feedback, EMA smoothing with $k \geq 10$ is mandatory.** Proposition 1 shows that raw single-step alignment signals are unreliable at batch sizes $\leq$ 256. CWD's binary mask is partially robust (sign is low-dimensional), but continuous alignment methods must aggregate temporally.

## 6.5 Limitations

1. **Scale.** CIFAR-10/100 with ResNet-20 (270K parameters) and VGG-16-BN (15M parameters) are small by current standards. ImageNet and LLM experiments are not included. The theory's predictions at larger scale remain unvalidated.

2. **$\rho$-high data gap.** Full 200-epoch $\rho = 5.0$ sweep failed due to training instability. Only 5-epoch pilot data (77.69% at epoch 5) is available. The critical test of Theorem 1's prediction---that method sensitivity increases at elevated $\rho$---is incomplete.

3. **Matched-$\rho$ SGD incomplete.** Seed 42 constant shows anomalous 76.12% (5 epochs only), and CWD has only 1 seed. The confound resolution between optimizer identity and $\rho$ is preliminary.

4. **NoBN incomplete.** Only 2 of 7 methods have 3-seed data. Conclusive Phi spread for NoBN requires additional methods.

5. **PMP-WD not implemented.** The algorithm is derived but not empirically evaluated. Its predicted advantage at elevated $\rho$ is theoretical.

6. **Three seeds.** With 3 seeds, statistical power is limited for effect sizes below 0.3%. TOST equivalence testing at margin $\pm 0.3\%$ requires larger $n$ for definitive equivalence claims.

7. **Fixed hyperparameters.** All methods share the same base $\lambda$ and lr. Per-method hyperparameter tuning (e.g., CWD's $\beta$ parameter) might reveal larger differences, at the cost of confounding method identity with hyperparameter optimization quality.

<!-- FIGURES
- None
-->
