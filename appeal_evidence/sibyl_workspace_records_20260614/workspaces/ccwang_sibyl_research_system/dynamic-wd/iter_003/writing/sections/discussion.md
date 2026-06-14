# 6. Discussion

## 6.1 The Phi Invariance Conjecture

Our experimental results reveal a striking pattern: across seven weight decay strategies spanning the full range of modulation approaches---temporal, directional, spatial, and budget ablation---no method produces statistically distinguishable accuracy from the constant baseline under AdamW. We formalize this observation as the following conjecture.

**Conjecture 1** (Phi Invariance under AdamW). *For a neural network trained with AdamW to convergence on a sufficiently overparameterized problem, the final test accuracy is invariant to the choice of Phi modulator $\varphi(t, \boldsymbol{\theta}, \mathbf{g})$, up to the effective weight decay budget $\mathbb{E}[\lambda_{\mathrm{eff}}]$.*

Formally, let $\mathrm{acc}(\varphi)$ denote the test accuracy achieved by training with modulator $\varphi$. Then for any two budget-equivalent modulators $\varphi_1$ and $\varphi_2$ (i.e., $\sum_t \lambda \cdot \mathbb{E}[\varphi_1(t)] = \sum_t \lambda \cdot \mathbb{E}[\varphi_2(t)]$):
$$
|\mathrm{acc}(\varphi_1) - \mathrm{acc}(\varphi_2)| \leq \epsilon_{\mathrm{noise}}
$$
where $\epsilon_{\mathrm{noise}}$ is bounded by training stochasticity (seed variance), not by the functional form of $\varphi$. Our experiments show that this bound holds even for non-budget-equivalent modulators: the accuracy difference between $\mathrm{BEM} = 0.0$ and $\mathrm{BEM} = 1.0$ is less than 0.5\%.

**Mechanistic hypothesis.** **We conjecture that AdamW's per-parameter adaptive scaling subsumes the effect of any Phi modulator, rendering the functional form of $\varphi$ irrelevant to generalization.** AdamW's adaptive learning rate $\eta_t / (\sqrt{\hat{v}_t} + \epsilon)$ already scales updates inversely with gradient magnitude on a per-parameter basis, providing implicit equalization of the effective regularization strength. When a Phi modulator redistributes weight decay across parameters or time steps, it operates on a system that has already "pre-equalized" the effective regularization per parameter through its adaptive step size. The Phi modulation is therefore a second-order effect, overwhelmed by AdamW's first-order adaptive dynamics.

Three lines of evidence from our experiments support this mechanistic hypothesis:

1. **Weight norm convergence** (Section 5.4): All seven methods converge to nearly identical weight norms (95.89--97.04, a 1.2\% range) despite a ten-fold variation in effective weight decay, demonstrating that AdamW's adaptive scaling implicitly controls weight norms regardless of the explicit decay schedule.

2. **AIS as an intrinsic property** (Section 5.3): The Alignment Informativeness Score is consistent across all methods ($\mathrm{AIS} \in [0.280, 0.410]$), including random\_mask and no\_wd. The gradient-weight alignment signal exists as a geometric property of the loss landscape but is not meaningfully exploitable by weight decay modulation under AdamW.

3. **Budget insensitivity** (Section 5.2): Even at $\mathrm{BEM} = 1.0$ (zero weight decay), accuracy is equivalent to the constant baseline, indicating that the total weight decay budget is irrelevant to generalization at this scale---the strongest possible evidence that weight decay modulation cannot help.

**Boundary conditions: when the conjecture may fail.** We identify several settings where the Phi Invariance Conjecture is unlikely to hold:

- **SGD (non-adaptive optimizers).** SGD lacks per-parameter adaptive scaling, so the distributional effect of $\varphi$ across parameters is not pre-equalized. Weight decay modulation may be first-order important under SGD.
- **Very large weight decay ($\lambda \gg$ standard range).** At extreme decay strengths, the weight decay term dominates the gradient update, and modulation may matter because the implicit norm control mechanism saturates.
- **Long training at scale (LLMs).** In near-single-epoch training of large language models, weight decay timing effects may compound over many tokens. Wang & Aitchison (2024) showed that optimal weight decay scales with model and dataset size, suggesting scale-dependent behavior.
- **Severe overfitting regimes.** All our experiments are in the well-generalized regime. When the model is heavily overfitting, weight decay's role as a regularizer may reassert itself, potentially making modulation strategy relevant.
- **Different architectures.** Vision Transformers with layer normalization, which interacts differently with weight decay than batch normalization, may respond differently to Phi modulation.

## 6.2 Implications for Weight Decay Research

Our findings have several practical and methodological implications.

**For AdamW practitioners.** The choice of weight decay schedule does not matter under the conditions tested. Practitioners should use constant weight decay with a grid-searched $\lambda$; the simplest approach is optimal. The engineering effort of implementing and tuning dynamic weight decay strategies offers no measurable benefit on CIFAR-scale AdamW experiments.

**For weight decay method developers.** New dynamic weight decay methods should be primarily evaluated under conditions where the Phi Invariance Conjecture's boundary conditions are more likely violated: with SGD, at ImageNet or LLM scale, or in severely overfitting regimes. Evaluating novel weight decay strategies solely on CIFAR with AdamW is misleading---differences will be indistinguishable from noise, not reflective of the method's potential.

**For benchmark designers.** Comparing weight decay methods on CIFAR-scale AdamW experiments is insufficient. This may explain why many dynamic weight decay papers report improvements only in specific settings (e.g., CWD with Lion/Muon, SWD with SGD): the benefits may be optimizer-specific or scale-dependent, and evaluations under AdamW at small scale mask genuine differences.

**The Phi framework as infrastructure.** Even under the Phi Invariance Conjecture, the framework retains substantial value. It provides a common mathematical language and programmatic interface for weight decay research, enables principled composition of methods (Proposition 1), and the diagnostic metrics---CSI, AIS, BEM---characterize *how* methods differ in their training dynamics even when they produce identical final accuracy. This infrastructure is essential for future work investigating the conjecture's boundary conditions.

## 6.3 Limitations

We acknowledge several limitations of this study.

1. **Scale.** All experiments are restricted to CIFAR-10/100 with ResNet-20 ($\sim$270K parameters). The Phi Invariance Conjecture may not hold at ImageNet scale (ResNet-50, $\sim$25M parameters) or LLM scale (billions of parameters), where weight decay's dynamics-modifying role may be more consequential.

2. **Architecture diversity.** Only ResNet-20 with batch normalization is tested. VGG-16-BN (which lacks skip connections) and Vision Transformers (which use layer normalization) may exhibit different sensitivity to weight decay modulation.

3. **Optimizer scope.** All experiments use AdamW. SGD, which lacks adaptive per-parameter scaling, is not included. The Phi Invariance Conjecture explicitly identifies SGD as a likely boundary condition; SGD experiments are a high-priority follow-up.

4. **Statistical power.** Three seeds per configuration provide limited statistical power. Effect sizes below $\sim$0.3\% may be undetectable. Equivalence testing (e.g., TOST with a $\pm 0.3\%$ margin) would provide stronger evidence for the null result.

5. **Fixed hyperparameters.** Using identical hyperparameters across methods ensures fair comparison of the Phi modulation effect but may disadvantage methods with strong hyperparameter sensitivity. For example, CWD with different $\beta$ values in soft mode, or SWD with different gradient-norm sensitivity scales, might exhibit different behavior under optimal tuning.

6. **Overfitting regime.** All experiments operate in the well-generalized regime (generalization gaps of $\sim$9.7\% on CIFAR-10 and $\sim$25.6\% on CIFAR-100). Methods may differentiate in heavily overfitted settings where weight decay's regularization role is more critical.
