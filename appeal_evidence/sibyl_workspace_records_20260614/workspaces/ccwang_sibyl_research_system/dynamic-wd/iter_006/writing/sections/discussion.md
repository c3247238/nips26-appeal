# 7. Discussion

## 7.1 The Weight Decay Illusion on BN Architectures

The central empirical finding of this work is that weight decay method choice has negligible impact on accuracy when batch normalization is present. On CIFAR-10/ResNet-20, the 0.49 percentage point spread across 6 methods (including no WD) is smaller than the inter-seed standard deviation of most individual methods. On CIFAR-100, the spread widens to 0.76 percentage points but remains statistically insignificant.

The Lyapunov certified band provides a theoretical explanation. On BN architectures, scale invariance constrains the range of effective WD values that produce distinct optimization trajectories. Proposition 1 shows the band width shrinks as training loss decreases, and the weight norm trajectories in Figure 3 confirm that all methods converge to the same final norm ($\approx 96$) regardless of modulation strategy. The practical implication is direct: **practitioners using BN architectures should use constant WD and allocate hyperparameter tuning budget elsewhere.**

## 7.2 When Dynamic WD Matters

The weight decay illusion is specific to BN architectures. The certified band analysis predicts that without BN, the band widens because the loss is no longer scale-invariant, and WD directly regularizes the effective weights. Prior work supports this prediction: CWD (Chen et al., ICLR 2026) reports improvements primarily on LLMs (which do not use BN), and D'Angelo et al. (NeurIPS 2024) show WD's dynamics-modifying role is qualitatively different with and without BN.

Three conditions should widen the certified band and make dynamic WD matter:

1. **Architectures without BN.** Transformers (ViTs, GPTs), plain CNNs, and architectures using LayerNorm or no normalization at all break scale invariance for a subset of parameters.

2. **Very long training.** As training progresses to hundreds of epochs or more, the cumulative effect of WD budget differences (captured by BEM) may compound. The weak BEM-accuracy correlation ($r = 0.61$) could strengthen at longer horizons.

3. **Large-scale models.** Weight norm dynamics in large models may exhibit more layer-level heterogeneity, increasing AIS and making alignment-aware modulation more informative.

## 7.3 PMP-WD: Theory Matches Practice

PMP-WD achieves the highest mean accuracy (90.29%) with the lowest standard deviation (0.12%) among all tested methods, consistent with the Pontryagin optimality prediction. The bang-bang switching pattern (Figure 5) validates Theorem 4: the switching function $\sigma(t) = \langle m_t, w_t \rangle$ oscillates around zero, producing approximately balanced decay application (switch rate $\approx 0.55$).

The structural similarity between PMP-WD and CWD is noteworthy. Both implement bang-bang control with binary $\phi \in \{0, 1\}$, but PMP-WD uses the global costate-weight inner product while CWD uses per-parameter sign alignment. PMP-WD's marginal accuracy advantage (90.29% vs. 89.98%) and lower variance (0.12 vs. 0.41) suggest the global switching criterion is more stable, though neither difference is statistically significant. The random mask control from iter_003 data (90.12% on CIFAR-10) further suggests that even random binary modulation achieves comparable results, reinforcing the narrow-band conclusion.

## 7.4 Diagnostic Metrics as Predictive Tools

The three proposed metrics (BEM, CSI, AIS) serve complementary roles:

- **BEM** predicts the total regularization budget. Methods with BEM $\approx 0.5$ (CWD, PMP-WD, cosine) apply roughly half the constant WD budget. Interestingly, reducing the budget does not hurt accuracy on BN architectures, consistent with the view that WD's regularization effect is dominated by the dynamics effect.

- **CSI** measures optimization stability. CWD achieves the lowest CSI (0.867), indicating the most stable coupling, likely because its binary mask eliminates the continuous-valued WD fluctuations that cosine schedule introduces (CSI = 0.956).

- **AIS** captures alignment informativeness. PMP-WD achieves the highest AIS (0.381), suggesting its costate-based switching responds to more layer-level alignment variation than other methods. The difference is small on BN architectures but may widen on architectures where alignment varies more across layers.

## 7.5 Limitations

**Scale.** All experiments use CIFAR-10/100 with ResNet-20 (0.27M parameters) and AdamW. The weight decay illusion may not hold at ImageNet/ViT scale, where BN is absent and models are orders of magnitude larger. The project constraints specify ImageNet experiments (ResNet-50), which remain as future work for this iteration.

**Optimizer.** We evaluate only AdamW. SGD with momentum may exhibit different WD sensitivity because the preconditioner structure differs. The matched-rho SGD experiments (Table 2 data not shown) suggest similar narrow-band behavior with SGD, but systematic comparison awaits.

**Theoretical gaps.** The Lyapunov certificate (Theorem 1) provides sufficient but not necessary conditions for convergence. Methods may converge outside the certified band through mechanisms not captured by the composite Lyapunov function. The cumulative alignment bound (Theorem 2) improves on the worst-case bound but has not been validated against the actual generalization gap trajectory due to the computational cost of full-batch alignment estimation at every epoch.

**Number of methods.** We evaluate 6 methods out of $>15$ published since 2023. Notable omissions include AdamO, AlphaDecay, and ADANA, each requiring non-trivial implementation effort. The phi modulator framework accommodates these methods theoretically (Table 1 can be extended), but empirical validation is incomplete.

<!-- FIGURES
- None
-->
