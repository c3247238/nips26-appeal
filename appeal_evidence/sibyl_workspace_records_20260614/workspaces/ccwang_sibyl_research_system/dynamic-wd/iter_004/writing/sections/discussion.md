## 7. Discussion

### 7.1 Practical Implications

Our findings deliver a clear, actionable message to practitioners: **under standard AdamW settings ($\lambda = 5 \times 10^{-4}$, $\eta = 10^{-3}$, batch-normalized architectures), there is no benefit to adopting dynamic weight decay strategies.** Constant weight decay, cosine-annealed weight decay, CWD, SWD, half-budget weight decay, and even no weight decay at all produce statistically indistinguishable results. Practitioners can save hyperparameter tuning effort and implementation complexity by using the simplest option.

This recommendation comes with explicit boundary conditions. Weight decay strategy *does* matter when:
- **Using SGD**: Our experiments show a $> 10\times$ effect-size amplification under SGD compared to AdamW for weight decay presence. Practitioners using SGD should carefully select and tune their weight decay strategy.
- **Using high weight decay coefficients**: The Phi Invariance Conjecture predicts that as $\rho = \lambda/\eta$ increases beyond $\sim 1$, modulation strategy becomes progressively important. This is directly testable via $\lambda$ sweep experiments (planned as P1).
- **Architectures without batch normalization**: The interaction between BN scale-invariance and weight decay dynamics remains an open question. Our analysis is validated on BN architectures; NoBN architectures may exhibit different sensitivity.

We propose a simple decision heuristic for practitioners:
1. Compute $\rho = \lambda/\eta$.
2. If $\rho < 1$ and using AdamW with BN: use constant weight decay.
3. If $\rho > 1$ or using SGD: consider dynamic strategies, with CWD as a principled starting point.
4. Use BEM to verify that observed differences are not merely budget effects.

### 7.2 Theoretical Implications

The $\rho = \lambda/\eta$ order parameter provides a unifying lens that connects several independent lines of recent work:

**Dual characterization.** We show (Theorem 1) that Xie \& Li's (2024) constraint radius $\tau^* = \eta/\lambda = 1/\rho$ and Defazio's (2025) steady-state gradient-to-weight ratio $R_* = \lambda/\eta = \rho$ are dual characterizations of the same quantity. This connection was not made explicit in either prior work and suggests that the $\ell_\infty$ constraint picture and the gradient-weight equilibrium picture are two faces of the same phenomenon.

**Consistency with Wang \& Aitchison (2024).** Their finding that optimal weight decay scales as an EMA timescale constant across model sizes is consistent with our framework: if $\rho$ remains in Regime I across scales, then the specific weight decay schedule is immaterial, and the ``optimal'' weight decay reduces to choosing the right magnitude rather than the right modulation pattern. Their EMA timescale $\approx 1/\rho$ in our notation.

**Beyond D'Angelo et al.\ (2024).** D'Angelo et al.\ showed that weight decay is never useful as explicit regularization, attributing this to batch normalization's scale-invariance. Our finding is complementary but distinct: we show that weight decay *modulation* is absorbed by AdamW's adaptive scaling, a dynamic argument that goes beyond static reparameterization. Whether BN scale-invariance or AdamW's $\ell_\infty$ constraint is the primary mechanism remains an important open question addressable through NoBN ablation experiments.

**Chou (2025) as a special case.** Their proposal to scale weight decay proportionally to the learning rate schedule ($\lambda \propto \gamma$) is a specific functional form within our temporal-axis modulation. Our results show that in Regime I, this schedule is no better (nor worse) than any other---including no schedule at all.

### 7.3 Scope and Boundary Conditions

We explicitly delineate the verified scope of our findings and the boundaries beyond which our results are predictive but unconfirmed:

**Verified.** CIFAR-10 and CIFAR-100 with ResNet-20 (270K parameters, batch normalization), AdamW at $\rho = 0.5$, SGD at standard settings. Seven weight decay methods spanning all four modulation axes. 87+ controlled experiments with 3 seeds per configuration.

**Predicted but unverified.** The $\rho$ regime boundary suggests that conditional equivalence should hold at $\rho < 1$ across architectures and datasets, and should break at $\rho > 1$. This prediction is directly testable through $\lambda$ sweep experiments. VGG-16-BN pilot results (Section 6.3) are consistent with this prediction but are not conclusive (10 epochs, 1 seed).

**Unknown.** Large-scale settings (ImageNet, LLM pre-training), non-BN architectures (Vision Transformers without BN, pure LayerNorm), and other adaptive optimizers (Lion, Muon, Shampoo) are beyond our current experimental scope. We expect the $\rho$ framework to generalize to any optimizer with implicit norm constraints, but this remains a conjecture.

### 7.4 Limitations

We acknowledge the following limitations with full candor:

1. **Statistical power.** With $n = 3$ seeds (2 degrees of freedom), our paired $t$-tests have limited power. The minimum detectable effect at 80\% power is approximately 0.7\% accuracy. TOST equivalence testing with margin $\delta = 0.5\%$ requires approximately $n = 10$ for adequate power given our observed standard deviations ($\sigma \approx 0.3\%$). Our claim of equivalence is therefore ``consistent with'' rather than ``proven by'' the data. Increasing to $n = 5$ or $n = 7$ seeds is a planned follow-up.

2. **Architecture coverage.** Our primary results use only ResNet-20 (270K parameters)---a small model by modern standards. While the $\rho$ framework predicts architecture-independent behavior in Regime I, this has not been empirically validated on larger architectures, Vision Transformers, or models without batch normalization.

3. **Dataset scale.** CIFAR-10 and CIFAR-100 are standard benchmarks but limited in scale and complexity. ImageNet validation, even at pilot scale, would significantly strengthen our claims. We note that the $\rho$ value ($\lambda/\eta$) is independent of dataset size, suggesting the regime boundary should transfer, but this requires verification.

4. **$\rho$ regime boundary precision.** We have tested only $\rho = 0.5$ (standard AdamW). The regime boundary $\rho_1$ between Regime I and Regime II is predicted but not experimentally located. A systematic $\lambda$ sweep ($\rho \in \{0.05, 0.5, 5, 50\}$) is needed to empirically characterize the transition.

5. **Conjecture, not theorem.** The Phi Invariance Conjecture is supported by empirical evidence and motivated by theoretical arguments (Lemmas 1--3), but a formal proof under realistic conditions is not provided. The critical assumption---Adam saturation ($\epsilon / (\sqrt{\hat{v}_i} \cdot |w_i|) < 0.1$)---holds empirically but is not guaranteed in all settings.

6. **SGD hyperparameter asymmetry.** Our SGD experiments use $\lambda = 5 \times 10^{-3}$ (standard for SGD+ResNet+CIFAR) while AdamW uses $\lambda = 5 \times 10^{-4}$. This means the SGD/AdamW comparison confounds optimizer choice with $\lambda$ magnitude. A matched-$\lambda$ SGD control ($\lambda = 5 \times 10^{-4}$) would isolate the optimizer effect; this is planned as a follow-up experiment.

7. **BEM metric history.** During development, we discovered and fixed a bug in the BEM computation for the half\_lambda method: the `HalfLambdaPhi` module did not correctly report effective weight decay in its diagnostics, causing BEM to read 0.000 instead of the correct $-0.500$. The signed BEM formula (removing absolute value) was also adopted. All results reported here use the corrected implementation. We disclose this to maintain full transparency about our development process.
