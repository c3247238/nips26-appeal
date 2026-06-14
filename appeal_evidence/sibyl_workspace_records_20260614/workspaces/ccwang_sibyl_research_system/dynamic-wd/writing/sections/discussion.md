# 6. Discussion

## 6.1 The Phi Invariance Conjecture

Our experiments reveal a striking pattern: across seven weight decay strategies spanning the full range of modulation approaches, no method produces statistically distinguishable accuracy from the constant baseline under AdamW---but several methods produce measurably different results under SGD.

**Conjecture 1** (Phi Invariance under AdamW). *For a neural network trained with AdamW to convergence on a sufficiently overparameterized problem with batch normalization, the final test accuracy is invariant to the choice of Phi modulator $\varphi(t, \boldsymbol{\theta}, \mathbf{g})$, up to training stochasticity.*

Formally, let $\mathrm{acc}(\varphi)$ denote the test accuracy achieved by training with modulator $\varphi$. For any two modulators $\varphi_1$ and $\varphi_2$:
$$
|\mathrm{acc}(\varphi_1) - \mathrm{acc}(\varphi_2)| \leq \epsilon_{\mathrm{noise}}
$$
where $\epsilon_{\mathrm{noise}}$ is bounded by training stochasticity (seed variance), not by the functional form of $\varphi$. Our experiments show this bound holds even for non-budget-equivalent modulators: the accuracy difference between $\mathrm{BEM} = 0.0$ and $\mathrm{BEM} = 1.0$ is less than 0.5% under AdamW but reaches 1.71% under SGD on CIFAR-100.

**Mechanistic hypothesis.** AdamW's per-parameter adaptive scaling $\eta_t / (\sqrt{\hat{v}_t} + \epsilon)$ already equalizes the effective regularization strength per parameter. When a Phi modulator redistributes weight decay, it operates on a system that has already pre-equalized effective regularization through adaptive step sizes. The modulation is a second-order effect, overwhelmed by AdamW's first-order adaptive dynamics.

Four lines of evidence support this hypothesis:

1. **Weight norm convergence** (Section 5.6): Under AdamW, all methods converge to nearly identical weight norms (95.89--97.04, a 1.2% range) despite ten-fold WD variation. Under SGD, the same methods produce norms spanning 64.57--127.06 (97% range).

2. **AIS invariance** (Section 5.5): AIS is consistent across all methods ($\mathrm{AIS} \in [0.280, 0.410]$), including random\_mask and no\_wd. The alignment signal is an intrinsic landscape property, not exploitable by WD modulation under AdamW.

3. **Budget insensitivity** (Section 5.4): At $\mathrm{BEM} = 1.0$ (zero WD), accuracy is equivalent to the constant baseline under AdamW---the strongest evidence that WD modulation cannot help.

4. **SGD boundary condition** (Section 5.2): Under SGD, which lacks adaptive scaling, no\_wd drops by 0.92% (CIFAR-10) and 1.71% (CIFAR-100). This confirms the conjecture's prediction: adaptive scaling is the enabling mechanism for Phi Invariance.

## 6.2 The Role of Batch Normalization

The VGG-16-BN results (Section 5.3) add nuance. Under SGD + VGG-16-BN, the accuracy spread is only 0.16pp---even narrower than AdamW + ResNet-20 (0.25pp). This suggests that batch normalization, not just adaptive optimizer scaling, contributes to Phi Invariance.

BN normalizes activations by their mean and variance, making the effective forward pass invariant to weight scaling within each layer. When weight norms change due to different WD strategies, BN compensates by adjusting the normalization statistics. The effective learning rate under BN scales as $\eta / \|\boldsymbol{\theta}_l\|^2$ (Li et al., 2020), providing an implicit norm-dependent LR adjustment analogous to AdamW's per-parameter adaptation.

The implication: **Phi Invariance may hold whenever the training system includes any mechanism that decouples effective updates from weight scale---whether through adaptive optimization (AdamW), activation normalization (BN), or both.** Testing architectures without BN (plain ResNets, Transformers with only LayerNorm) under SGD would further clarify this interaction.

## 6.3 Implications for Weight Decay Research

**For AdamW practitioners.** The choice of weight decay schedule does not matter under the conditions tested. Constant weight decay with a grid-searched $\lambda$ is optimal; the engineering effort of implementing dynamic WD strategies offers no measurable benefit on CIFAR-scale experiments.

**For weight decay method developers.** New methods should be evaluated under conditions where Phi Invariance is less likely to hold: with SGD on architectures without batch normalization, at ImageNet or LLM scale, or in severely overfitting regimes. Evaluating novel WD strategies solely on CIFAR with AdamW is misleading---differences will be indistinguishable from noise.

**For benchmark designers.** CIFAR-scale AdamW experiments are insufficient for discriminating WD methods. CWD's improvements with Lion/Muon and SWD's gains with SGD may reflect genuine optimizer-specific benefits masked by AdamW's adaptive scaling.

**The Phi framework as infrastructure.** Even under Phi Invariance, the framework provides a common mathematical language and programmatic interface for WD research, enables principled composition (Proposition 1), and the diagnostic metrics characterize *how* methods differ even when they produce identical accuracy.

## 6.4 Limitations

1. **Scale.** Experiments are restricted to CIFAR-10/100 with ResNet-20 ($\sim$270K parameters) and VGG-16-BN ($\sim$15M parameters). The conjecture may not hold at ImageNet scale (ResNet-50, $\sim$25M parameters) or LLM scale.

2. **Architecture diversity.** Both tested architectures use batch normalization. Vision Transformers with layer normalization may respond differently to Phi modulation.

3. **Optimizer scope.** SGD results show the boundary condition, but we have not tested other adaptive optimizers (Lion, Muon, Scion) where CWD reports improvements.

4. **Statistical power.** Three seeds provide limited power. Effect sizes below $\sim$0.3% may be undetectable.

5. **Fixed hyperparameters.** Identical base hyperparameters ensure fair comparison but may disadvantage methods with strong hyperparameter sensitivity.

6. **Overfitting regime.** All experiments operate in the well-generalized regime. Methods may differentiate in heavily overfitted settings where weight decay's regularization role is more critical.

<!-- FIGURES
- None
-->
