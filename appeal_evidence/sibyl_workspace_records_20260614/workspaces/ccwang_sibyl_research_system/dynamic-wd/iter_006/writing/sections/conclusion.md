# 8. Conclusion

This paper introduced the phi modulator framework, a unified formulation that expresses every published weight decay method as $\lambda_{\text{eff}}(t) = \phi(t, w, g) \cdot \lambda_{\text{base}}$, exposing the modulation strategy as the sole differentiator across methods. Using a composite Lyapunov function $V_t = f(w_t) + \mu_t\|w_t\|^2$, we derived a certified convergence band $[\lambda_{\min}(t), \lambda_{\max}(t)]$ (Theorem 1) and showed that constant WD, CWD, cosine-scheduled WD, SWD, and PMP-WD all lie within this band for at least 95% of training steps (Theorem 3). Applying Pontryagin's Maximum Principle, we derived PMP-WD (Theorem 4), the optimal bang-bang controller that uses the momentum buffer as a zero-cost costate approximation.

Experiments on CIFAR-10 and CIFAR-100 with ResNet-20 confirm the **weight decay illusion**: on batch-normalized architectures, all six tested methods achieve accuracy within a 0.49 percentage point spread on CIFAR-10 and 0.76 points on CIFAR-100. No pairwise difference reaches statistical significance. PMP-WD achieves the highest mean accuracy (90.29 $\pm$ 0.12%) consistent with the optimality prediction, but its advantage over constant WD (89.80 $\pm$ 0.31%) is within the noise floor. Weight norm trajectories, Lyapunov function curves, and the three diagnostic metrics (BEM, CSI, AIS) provide converging evidence that batch normalization narrows the certified band to the point where method choice is irrelevant.

These results carry a practical message: for BN architectures, constant weight decay suffices. The effort spent designing and tuning dynamic WD methods is better allocated to other hyperparameters. The certified band framework predicts that dynamic WD should matter on non-BN architectures (transformers, plain CNNs), where scale invariance does not constrain the band -- a prediction we plan to test on ImageNet with ResNet-50 and Vision Transformers in future work.

<!-- FIGURES
- None
-->
