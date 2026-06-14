## 8. Conclusion

### 8.1 Summary of Findings

We introduced the Phi Modulator Framework, a unified mathematical abstraction that recovers seven major dynamic weight decay methods as special cases along four modulation axes (temporal, directional, spatial, and target-norm). Through 87+ controlled experiments under identical optimization conditions, we established three central findings:

First, **conditional equivalence under AdamW**: at standard settings ($\rho = \lambda/\eta = 0.5$), all seven weight decay strategies---including the complete absence of weight decay---are statistically indistinguishable on CIFAR-10 (accuracy range $< 0.3\%$, all $p_{\text{adj}} > 0.05$) and CIFAR-100. This equivalence holds across the full BEM spectrum from $0.000$ to $-1.000$, meaning neither the modulation pattern nor the total budget of weight decay affects outcomes in this regime.

Second, **optimizer-dependent sensitivity**: SGD exhibits fundamentally different behavior, with the constant vs.\ no\_wd comparison yielding Cohen's $d = 10.29$ ($p_{\text{adj}} = 0.002$)---an effect-size ratio of approximately 18.3$\times$ relative to AdamW. This asymmetry is attributable to AdamW's implicit $\ell_\infty$ constraint, which absorbs weight decay perturbations in a mechanism absent from SGD.

Third, **the $\rho = \lambda/\eta$ regime boundary**: we formalized these observations through the Phi Invariance Conjecture, which posits $\rho$ as the order parameter governing when weight decay strategy choice matters. The conjecture connects to Xie \& Li's (2024) constraint radius and Defazio's (2025) gradient-weight equilibrium through a novel dual characterization (Theorem 1), and makes falsifiable predictions at specific $\rho$ values.

Additionally, we contributed three diagnostic metrics---BEM, CSI, and AIS---as standardized tools for characterizing weight decay strategies, and demonstrated the practical utility of the cosine schedule's anomalously low variance ($\sigma = 0.07\%$ vs.\ $\sim 0.30\%$) as a stability-enhancing property even when it provides no mean accuracy benefit.

### 8.2 Future Work

Several directions warrant investigation:

1. **Formal proof of the Phi Invariance Conjecture.** A rigorous proof under simplified settings (e.g., quadratic loss with diagonal Hessian under AdamW) would elevate the conjecture to a theorem and sharpen the regime boundary estimates.

2. **$\rho$ regime sweep.** Systematic variation of $\lambda \in \{5 \times 10^{-5}, 5 \times 10^{-4}, 5 \times 10^{-3}, 5 \times 10^{-2}\}$ (corresponding to $\rho \in \{0.05, 0.5, 5, 50\}$) would empirically locate the Regime I/II boundary and test the trichotomy's predictions.

3. **Large-scale validation.** ImageNet experiments with ResNet-50 and LLM pre-training would test whether the $\rho$-based framework generalizes across scales. Wang \& Aitchison's (2024) scaling results suggest the regime boundary may be scale-independent, but this requires direct verification.

4. **BN ablation.** Experiments with architectures lacking batch normalization would disentangle the contributions of BN scale-invariance (D'Angelo et al., 2024) from AdamW's implicit $\ell_\infty$ constraint to the observed invariance.

5. **Optimizer generalization.** Testing the framework with other adaptive optimizers (Lion, Muon, Shampoo) would determine whether the $\ell_\infty$ absorption mechanism is specific to AdamW or common to a broader class of sign-based optimizers.

### 8.3 Broader Impact

Our work delivers a practically useful message: knowing *when not to optimize* is as valuable as knowing *how to optimize*. For the substantial community of practitioners using AdamW at standard settings, our results suggest that effort invested in dynamic weight decay strategy selection is likely wasted---the simplest constant schedule suffices. This redirects optimization effort toward hyperparameters and design choices that genuinely affect outcomes. At the same time, our $\rho$-based framework identifies the precise conditions under which weight decay strategy *does* matter, providing a principled guide rather than a universal dismissal.
