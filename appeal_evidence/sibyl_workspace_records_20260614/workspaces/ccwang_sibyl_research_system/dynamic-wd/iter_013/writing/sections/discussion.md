# Analysis and Discussion

## Why Does EqWD Excel on ImageNet but Not Decisively on CIFAR-100?

The disparity in EqWD's performance across benchmarks---ranking first on ImageNet but third (with default $\beta$) on CIFAR-100---reveals an important property of equilibrium-based modulation: its effectiveness depends on the information content of the ratio deviation signal.

CIFAR-100 with ResNet-20 is a relatively simple optimization problem. The model has 278K parameters, the input resolution is 32$\times$32 pixels, and training converges within 200 epochs with smooth, predictable dynamics. In this regime, weight norms stabilize quickly, ratio deviations from equilibrium are small and transient, and the modulation factor $\varphi_l(t)$ rarely exceeds 1.1. There is insufficient deviation signal for EqWD to exploit meaningfully with the default $\beta = 1.0$---which is why higher $\beta$ values (amplifying the limited signal) improve CIFAR-100 performance in our ablation.

ImageNet with ResNet-50, by contrast, presents a substantially more complex optimization landscape. The model has 25.6M parameters across four residual stages with different channel dimensions, the input resolution is 224$\times$224 with heavy data augmentation, and the 45-epoch training schedule produces prolonged transitional phases. Ratio deviations are larger, more sustained, and structurally informative---reflecting genuine differences in how different layers respond to the evolving loss surface. In this setting, EqWD's per-layer modulation captures meaningful dynamics that a fixed coefficient ignores.

This observation suggests that adaptive weight decay methods based on training dynamics are most beneficial for tasks where the optimization trajectory exhibits rich, heterogeneous behavior. However, we note that this hypothesis is based on two benchmarks that differ on many dimensions simultaneously (dataset size, resolution, architecture scale, training length), and more systematic investigation across a continuum of task complexities would be needed to establish a firm scaling relationship.

## EqWD vs. SWD: The Source of Lower Variance

EqWD and SWD both adapt weight decay dynamically, yet EqWD tends to exhibit lower seed-to-seed variance on ImageNet (0.20\% vs. 0.40\%), though as noted in Section 4.2, this variance estimate has limited precision at $n = 3$. We hypothesize that the difference arises from signal quality.

SWD modulates weight decay based on the raw gradient norm $\|g_t\|$. Gradient norms are inherently noisy: a single batch with outlier samples can spike $\|g_t\|$ by an order of magnitude, causing SWD to dramatically reduce weight decay for that step. This introduces high-frequency noise into the regularization schedule that varies across random seeds, potentially amplifying seed-to-seed divergence.

EqWD, by contrast, uses the *deviation* from an EMA-tracked equilibrium. The EMA smoothing ($\alpha = 0.9$) filters out per-step noise, and the deviation measure is inherently self-normalizing: a large $r_t^l$ that is consistent with the recent trajectory (high $r^{*,l}$) produces small deviation, while the same $r_t^l$ following a period of low ratios (low $r^{*,l}$) produces large deviation. This context-sensitivity makes EqWD responsive to genuine training transitions while being robust to stochastic fluctuations, resulting in more consistent behavior across seeds.

## Why Do CWD and CPR Underperform on ImageNet?

A notable finding in our experiments is that two recent methods---CWD (ICLR 2026) and CPR (NeurIPS 2024)---both underperform the simple FixedWD baseline on ImageNet, despite being designed to improve upon it. These two methods have distinct mechanisms and likely distinct failure modes, which we discuss separately.

**CWD** uses a binary sign-alignment mask: weight decay is applied only to parameters where $\text{sign}(g_t) = \text{sign}(w_t)$. At ImageNet scale, the high-dimensional gradient and weight vectors are often near-orthogonal, making the per-element sign alignment essentially random for many parameters. The binary decision amplifies this noise: parameters flicker between "decay" and "no decay" states based on stochastic sign fluctuations rather than meaningful alignment structure. Our alignment diagnostic (Appendix F) provides supporting evidence: the mutual information between cosine alignment and test accuracy, conditioned on gradient and weight norms, is near zero across all layers. We note that this result is specific to our 45-epoch training regime and CWD's masking mechanism may behave differently under longer training.

**CPR** enforces per-matrix norm constraints via a smooth augmented Lagrangian. Unlike CWD's binary masking, CPR's modulation is continuous, but its formulation enforces a fixed norm target rather than adapting to training dynamics. At ImageNet scale with proper learning rate scheduling and data augmentation, the norm constraint may not be the binding bottleneck for generalization. We hypothesize that the Lagrangian multiplier updates can introduce oscillatory behavior that hurts convergence, though we have not directly measured this and leave detailed analysis of CPR's failure mode for future investigation.

EqWD avoids both issues: its continuous modulation signal is smoother than CWD's binary mask, and its deviation-based formulation adapts to the actual training dynamics rather than enforcing a fixed target.

## The CAWD Negative Result: Implications for Alignment-Based Regularization

CAWD (continuous cosine alignment modulation) underperforms FixedWD on ImageNet (71.44\% vs. 71.89\%). This is a noteworthy negative finding that has implications beyond our specific method comparison. It suggests that cosine alignment between gradient and weight vectors, which has been the theoretical motivation for several recent methods \cite{chen2026cwd, sun2025cvpr}, is not a useful *modulation signal* in isolation---even when implemented with the same EMA-smoothed framework that makes EqWD effective. As our AIS diagnostic shows, the alignment information is *redundant given the gradient and weight norms*. The ratio $r_t^l = \|g_t^l\| / \|w_t^l\|$ already captures the generalization-relevant information that alignment encodes, at lower computational cost and with greater noise robustness. This finding suggests that future work on alignment-based regularization should consider whether the alignment signal provides information beyond what norm-based quantities already capture.

## The 45-Epoch ImageNet Regime

Our ImageNet experiments use a 45-epoch accelerated training regime, which is half the canonical 90-epoch schedule \cite{he2016resnet} and yields absolute accuracies approximately 4\% below standard benchmarks. We adopted this regime to enable multi-seed comparison (3 seeds) within a fixed compute budget, consistent with the resource constraints of this study. Several considerations are relevant:

**Internal validity.** All methods are compared under identical conditions (same epochs, batch size, learning rate schedule, augmentation, seeds), ensuring that relative comparisons are fair within this regime. The key comparisons---EqWD vs. FixedWD, EqWD vs. SWD---are internally valid.

**Precedent.** Short training schedules for ImageNet ResNet-50 are common in the literature when the research question concerns relative method comparison rather than absolute state-of-the-art accuracy. He et al. \cite{he2019bag} use 120-epoch schedules with various enhancements; Goyal et al. \cite{goyal2017accurate} focus on 90 epochs but validate learning rate scaling with shorter runs; and several weight decay studies \cite{xie2023swd, franke2024cpr} evaluate under varied epoch budgets.

**Potential concern.** Method rankings can in principle change with training length: methods that help during transitional phases (EqWD's claimed advantage) might contribute less once the optimizer has more time to recover naturally. Conversely, EqWD's modulation of the cosine decay transition (which occurs proportionally later in shorter schedules) could be more pronounced with shorter training. We cannot rule out either possibility without 90-epoch experiments, which we leave for future work.

**Recommendation.** Practitioners should validate EqWD at their target training length. Our 45-epoch results demonstrate that EqWD is competitive and potentially advantageous under accelerated training, which is itself a practically relevant regime for hyperparameter search and rapid prototyping.

## Limitations

We acknowledge several limitations of the current work:

1. **Statistical power.** With $n = 3$ seeds, our comparisons have limited statistical power. The EqWD vs. SWD improvement on ImageNet (+0.23\%) has a bootstrap 95\% CI that includes zero, meaning we cannot definitively claim EqWD outperforms SWD. The EqWD vs. FixedWD comparison (+0.38\%, Cohen's $d = 1.72$) is more robust but would benefit from validation with 5--10 seeds. We recommend that future work use at least 5 seeds for key comparisons.

2. **Effective weight decay inflation.** Because EqWD only increases weight decay relative to $\lambda_{\text{base}}$ ($\varphi_l(t) \geq 1$ always), the effective average weight decay over training is higher than FixedWD with the same base coefficient. Part of the accuracy gain could reflect this higher average regularization strength rather than better-timed modulation. The critical missing experiment is FixedWD with a tuned higher $\lambda$ (e.g., $6 \times 10^{-4}$ or $7 \times 10^{-4}$), which would isolate the strength-vs-timing confound. The CAWD baseline partially controls for this (CAWD also modulates upward with the same EMA structure) and underperforms FixedWD, providing evidence that the signal choice (ratio vs. alignment) matters beyond effective strength. However, CAWD's alignment signal may be independently detrimental, so this control is not perfectly clean.

3. **Accelerated ImageNet training.** Our 45-epoch ImageNet regime is shorter than the canonical 90 epochs. While all methods are compared under identical conditions, method rankings may change with training length (Section 5.5). A 90-epoch validation, even at single-seed, is an important next step.

4. **Optimizer scope.** Our experiments use SGDW exclusively. While EqWD's formulation is optimizer-agnostic in principle, the ratio dynamics under Adam-family optimizers (where the effective gradient is scaled by the second moment) may differ fundamentally. The common AdamW + Transformer paradigm \cite{loshchilov2019adamw, dosovitskiy2021vit} represents the dominant training configuration in modern practice, and EqWD has not been validated in this setting. Extension to AdamW and optimizers such as Lion \cite{chen2023lion} requires separate investigation.

5. **Architecture scope.** We evaluate on convolutional architectures (ResNet, VGG). Vision Transformers and large language models may exhibit different ratio dynamics due to their distinct layer structures (attention, LayerNorm vs. BatchNorm).

6. **Single-seed $\beta = 5.0$ result.** The highest CIFAR-100 accuracy (66.07\%) was observed at $\beta = 5.0$ with a single seed. This aggressive setting may not be robust across seeds, and multi-seed validation at high $\beta$ values is needed.

7. **NoBN failure.** Our VGG-16 experiments without batch normalization produced 1\% accuracy across all methods (including all baselines). This is a known training stability issue with VGG without BN at learning rate 0.1, not specific to EqWD, but it highlights that the ratio equilibrium analysis presumes the presence of normalization layers.

## When Should Practitioners Use EqWD?

Based on our empirical evidence, we offer practical guidance:

- **Recommended for** large-scale training (ImageNet-scale and above) with SGDW, where ratio deviations are substantial and persistent.
- **Marginal benefit** on smaller-scale tasks (CIFAR-level) with the default $\beta = 1.0$; practitioners may consider higher $\beta$ values (2.0--5.0) if operating in this regime, though multi-seed validation is advisable.
- **Not yet validated** for AdamW-based training, Transformer architectures, or alternative optimizers such as Lion. We recommend fixed weight decay as the safer default in these settings until EqWD is validated.
- **Default hyperparameters** ($\beta = 1.0$, $\alpha = 0.9$) are robust and require no tuning for the settings tested.
