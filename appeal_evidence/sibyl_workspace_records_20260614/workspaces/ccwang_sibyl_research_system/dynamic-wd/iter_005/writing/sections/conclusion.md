# 7. Conclusion

Three theorems provide the first stability-optimal control theory for weight decay scheduling. Theorem 1 formalizes the tradeoff between alignment benefit and stability cost, predicting that constant WD is optimal when AIS falls below a noise-dependent threshold---a condition satisfied at standard $\rho$ in batch-normalized networks. This prediction is confirmed across all 7 tested configurations (2 optimizers $\times$ 2 datasets $\times$ 2 architectures), with Phi spread $\leq$ 0.25% under AdamW (CIFAR-10) and $\leq$ 0.16% on VGG-16-BN. Theorem 2 bounds the generalization penalty from per-parameter WD variation, explaining why binary masking methods (CWD, random mask) incur hidden stability costs despite moderate aggregate CSI. Theorem 3, derived independently from stochastic PMP and RG beta function theory, yields PMP-WD: $\lambda^*(t) = \text{clip}(\kappa \cdot (\rho^* - \hat{\rho}_t)^+, 0, \lambda_{\max})$---a state-feedback WD controller that adapts to the measured gradient-to-weight ratio. Proposition 1 establishes EMA smoothing ($k \geq 10$) as a necessary design constraint for any alignment-aware method operating at batch sizes $\leq$ 256.

SGD shows 3.7$\times$ larger method sensitivity than AdamW on CIFAR-10 (0.91% vs. 0.25%), partially attributed to its 100$\times$ lower $\rho$ rather than optimizer identity alone. The practical implication is direct: under AdamW at standard hyperparameters, constant WD is sufficient; dynamic WD methods should target elevated-$\rho$ or large-scale regimes where the alignment benefit exceeds the stability cost.

Future work includes ImageNet-scale validation, PMP-WD empirical evaluation at elevated $\rho$, Vision Transformer architectures, and LLM-scale experiments where large batch sizes relax Proposition 1's noise constraint and long training horizons amplify WD timing effects.

<!-- FIGURES
- None
-->
