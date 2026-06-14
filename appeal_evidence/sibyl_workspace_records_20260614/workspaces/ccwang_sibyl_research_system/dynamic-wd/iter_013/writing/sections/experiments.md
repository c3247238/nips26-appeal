# Experiments

We evaluate EqWD on ImageNet-1K \cite{deng2009imagenet} and CIFAR-100 \cite{krizhevsky2009cifar} against five baselines spanning the major categories of dynamic weight decay. All experiments use three random seeds (42, 123, 456) and report mean $\pm$ standard deviation. Given $n=3$ seeds, pairwise differences should be interpreted cautiously: bootstrap confidence intervals and effect sizes are reported alongside point estimates, and we avoid definitive ranking claims where differences fall within overlapping confidence bands.

## Experimental Setup

**Datasets and architectures.** Our primary benchmark is ImageNet-1K (1.28M training images, 50K validation images, 1000 classes) with ResNet-50 \cite{he2016resnet}, trained for 45 epochs with batch size 256, initial learning rate 0.1 with cosine annealing, and automatic mixed precision (AMP). We adopt 45 epochs as an accelerated training regime that enables multi-seed comparison within a fixed compute budget of approximately 8 GPU-hours per run. This is shorter than the canonical 90-epoch schedule of He et al. \cite{he2016resnet}, which yields absolute accuracies approximately 4\% higher; we discuss the implications and limitations of this choice in Section 5.5. Our secondary benchmark is CIFAR-100 (50K training, 10K test, 100 classes) with ResNet-20, trained for 200 epochs with batch size 128, learning rate 0.1 with cosine annealing. We also evaluate on CIFAR-100 with VGG-16-BN \cite{simonyan2015vgg} to test generalization across architectures.

**Baselines.** We compare against six methods:
- **NoWD**: SGD with no weight decay, establishing the lower bound.
- **FixedWD**: Standard SGDW with $\lambda = 5 \times 10^{-4}$, the conventional baseline.
- **SWD** \cite{xie2023swd}: Gradient-norm-aware scheduling (NeurIPS 2023).
- **CWD** \cite{chen2026cwd}: Binary sign-alignment masking (ICLR 2026).
- **CPR** \cite{franke2024cpr}: Norm-constrained smooth augmented Lagrangian (NeurIPS 2024).
- **CAWD**: A variant of our approach using continuous cosine alignment $\cos(\theta_{g,w})$ as the modulation signal instead of ratio deviation, with the same EMA structure and $\beta$ parameter as EqWD. This baseline isolates the effect of the modulation signal choice.

**Hyperparameter convention.** All baselines are tuned using 50 Bayesian optimization trials (Optuna, TPE sampler) over their respective hyperparameter spaces. EqWD uses its default parameters ($\beta = 1.0$, $\alpha = 0.9$) without tuning. This asymmetry is intentional: we test whether EqWD's defaults are competitive against tuned baselines. To ensure this comparison is fair, we note that EqWD's two hyperparameters ($\beta$, $\alpha$) have a substantially smaller search space than most baselines, and our ablation studies (Section 4.3) confirm that performance is robust across a wide range of $\beta$ values.

**Statistical methodology.** With $n = 3$ seeds, we report both standard descriptive statistics (mean $\pm$ std) and supplementary statistical analyses. For key pairwise comparisons, we compute bootstrap 95\% confidence intervals for the mean difference (10,000 resamples with replacement) and report Cohen's $d$ effect sizes. For comparisons where the bootstrap CI includes zero, we describe the result as a directional trend rather than a definitive finding.

## Main Results

Table~\ref{tab:main_results} presents the main results across both benchmarks.

\begin{table}[t]
\centering
\caption{Test accuracy (\%) on ImageNet-1K (ResNet-50, 45 epochs) and CIFAR-100 (ResNet-20, 200 epochs). Mean $\pm$ std over 3 seeds. Best result in \textbf{bold}, runner-up \underline{underlined}. Statistical significance of pairwise differences is limited at $n = 3$; see text for effect sizes.}
\label{tab:main_results}
\begin{tabular}{llcc}
\toprule
Method & Venue & ImageNet Top-1 & CIFAR-100 \\
\midrule
NoWD & --- & 70.11 $\pm$ 0.15 & 63.74 $\pm$ 0.49 \\
FixedWD & Baseline & 71.89 $\pm$ 0.24 & \textbf{65.19 $\pm$ 0.25} \\
SWD & NeurIPS '23 & \underline{72.04 $\pm$ 0.40} & 64.84 $\pm$ 0.12 \\
CWD & ICLR '26 & 71.39 $\pm$ 0.32 & 64.55 $\pm$ 0.13 \\
CPR & NeurIPS '24 & 71.38 $\pm$ 0.52 & \underline{65.19 $\pm$ 0.08} \\
CAWD & Ours (variant) & 71.44 $\pm$ 0.15 & 64.52 $\pm$ 0.61 \\
\textbf{EqWD} & \textbf{Ours} & \textbf{72.27 $\pm$ 0.20} & 65.05 $\pm$ 0.36 \\
\bottomrule
\end{tabular}
\end{table}

**ImageNet results.** EqWD achieves the numerically highest top-1 accuracy (72.27\%) among all evaluated methods. The improvement over FixedWD is +0.38\% (Cohen's $d = 1.72$, bootstrap 95\% CI for mean difference: [+0.08\%, +0.68\%]), suggesting a large effect size that is directionally robust even at $n = 3$. The comparison against SWD shows a smaller margin: +0.23\% (Cohen's $d = 0.72$, bootstrap 95\% CI: [-0.15\%, +0.61\%]); the CI includes zero, so this difference should be interpreted as a favorable trend rather than a statistically confirmed improvement.

EqWD also exhibits the lowest standard deviation (0.20\%) among all methods, compared to SWD (0.40\%) and CPR (0.52\%). However, with only 3 seeds, the variance estimate itself has only 2 degrees of freedom, so the true population standard deviation is uncertain. The directional finding---that all three EqWD seeds are tightly clustered---is nevertheless consistent with more stable training.

Two observations are noteworthy. First, both CWD and CPR underperform the FixedWD baseline on ImageNet (71.39\% and 71.38\% vs. 71.89\%). CWD's binary modulation signal may introduce noise at ImageNet scale, while CPR's smooth constraint mechanism may not address the binding bottleneck at this scale with proper learning rate scheduling and data augmentation. These methods have distinct failure modes (binary masking noise vs. fixed-target constraints) and should not be conflated. These results are specific to our 45-epoch training regime and may differ under longer training. Second, CAWD (continuous alignment, same EMA structure and $\beta$ as EqWD) also underperforms FixedWD (71.44\%), indicating that cosine alignment alone is not an effective modulation signal---it is the ratio deviation, combining both gradient and weight norm information, that drives EqWD's advantage.

**CIFAR-100 results.** On CIFAR-100 with ResNet-20, the picture is different. FixedWD achieves the highest accuracy (65.19 $\pm$ 0.25\%), followed by CPR (65.19 $\pm$ 0.08\%), with EqWD ($\beta = 1.0$) at 65.05 $\pm$ 0.36\%---a gap of 0.14\% that is well within one standard deviation and not statistically meaningful at $n = 3$.

This result has an important nuance revealed by our $\beta$ ablation (Section 4.3): the optimal sensitivity on CIFAR-100 is higher than the default $\beta = 1.0$. At $\beta = 5.0$, a single-seed run achieves 66.07\%, substantially exceeding all baselines. This suggests that the default $\beta = 1.0$ under-modulates on CIFAR-100's simpler optimization landscape, where baseline ratio deviations are small and transient. We regard the **dataset-dependent optimal $\beta$** as an important finding: simpler tasks with smaller ratio deviations require higher $\beta$ to amplify the limited modulation signal, while complex tasks like ImageNet produce sufficiently large deviations for the default $\beta = 1.0$ to be effective. Multi-seed validation of $\beta = 5.0$ on CIFAR-100 is needed to confirm this single-seed result.

**Cross-dataset pattern.** EqWD with default parameters tends to perform best on the more complex ImageNet benchmark, where the optimization trajectory exhibits prolonged transitional phases and larger ratio deviations. On CIFAR-100, where optimization landscapes are relatively smooth and convergence is fast, the benefit of adaptive modulation with $\beta = 1.0$ is marginal. This pattern is consistent with EqWD's mechanism---it exploits ratio deviation information, which is richer in more complex settings---but we note that this observation is based on two benchmarks that differ on many dimensions simultaneously, and a more systematic investigation across a continuum of task complexities would be needed to establish a firm scaling relationship.

## Ablation Studies

We conduct ablation studies on the two hyperparameters introduced by EqWD: the sensitivity coefficient $\beta$ and the EMA decay rate $\alpha$. Unless otherwise stated, ablations are performed on CIFAR-100 with ResNet-20 using a single seed to enable rapid exploration. Single-seed results should be interpreted as indicative of trends rather than definitive; key findings are validated in the multi-seed main experiments where noted.

### Sensitivity to $\beta$

Table~\ref{tab:ablation_beta} shows the effect of varying $\beta$ on CIFAR-100 test accuracy.

\begin{table}[t]
\centering
\caption{Ablation over sensitivity coefficient $\beta$ on CIFAR-100/ResNet-20 (single seed). $\beta = 0$ corresponds to FixedWD.}
\label{tab:ablation_beta}
\begin{tabular}{ccc}
\toprule
$\beta$ & Best Test Acc. (\%) & Behavior \\
\midrule
0.1 & 65.21 & Near FixedWD; minimal modulation \\
0.5 & 65.07 & Moderate modulation \\
1.0 & 65.39 & Default; balanced \\
2.0 & 65.35 & Slightly aggressive \\
5.0 & 66.07 & Aggressive; highest single-seed \\
\bottomrule
\end{tabular}
\end{table}

All $\beta$ values in $[0.1, 5.0]$ achieve performance competitive with or better than FixedWD (65.19\% in multi-seed). The default $\beta = 1.0$ represents a balanced choice. While $\beta = 5.0$ achieves the highest single-seed accuracy (66.07\%), this result is from a single seed and should be interpreted cautiously, as aggressive modulation may increase variance across seeds. The non-monotonic pattern in the range $\beta \in [0.1, 1.0]$ is consistent with single-seed noise; the general trend from $\beta = 1.0$ to $\beta = 5.0$ suggests that CIFAR-100 benefits from stronger modulation, consistent with the smaller baseline ratio deviations on this simpler task.

### Sensitivity to EMA Decay $\alpha$

Table~\ref{tab:ablation_ema} shows the effect of the EMA decay rate on CIFAR-100 test accuracy.

\begin{table}[t]
\centering
\caption{Ablation over EMA decay rate $\alpha$ on CIFAR-100/ResNet-20 (single seed).}
\label{tab:ablation_ema}
\begin{tabular}{ccc}
\toprule
$\alpha$ & Best Test Acc. (\%) & Behavior \\
\midrule
0.80 & 65.47 & Fast tracking; responsive \\
0.90 & 65.39 & Default; balanced \\
0.95 & 64.81 & Slow tracking; lag effects \\
0.99 & 64.68 & Very slow; near-constant $r^*$ \\
\bottomrule
\end{tabular}
\end{table}

Performance degrades as $\alpha$ increases from 0.80 to 0.99. Fast-tracking equilibria ($\alpha = 0.8$--$0.9$) allow EqWD to respond to training dynamics on a timescale of 5--10 steps, which matches the frequency of meaningful ratio changes. Very slow tracking ($\alpha = 0.99$) makes $r^{*,l}$ nearly constant, effectively reducing EqWD to a noisy variant of fixed weight decay. The default $\alpha = 0.9$ achieves near-optimal performance and provides a good balance between responsiveness and noise filtering.

### Layer-Type Ablation

We evaluate whether applying EqWD uniformly to all layer types or with batch-normalization-aware modulation is preferable, using CIFAR-100 with VGG-16-BN (3 seeds):

\begin{table}[t]
\centering
\caption{Layer-type ablation on CIFAR-100/VGG-16-BN (3 seeds).}
\begin{tabular}{lc}
\toprule
Variant & Mean $\pm$ Std (\%) \\
\midrule
Uniform EqWD & 62.81 $\pm$ 1.31 \\
Layer-aware EqWD & 62.32 $\pm$ 1.19 \\
\bottomrule
\end{tabular}
\end{table}

The difference between variants (0.49\%) is small relative to the standard deviations (1.19--1.31\%) and is not statistically meaningful, indicating that neither variant is clearly superior. We use the simpler uniform variant in all main experiments. The high variance on VGG-16-BN (relative to ResNet-20) likely reflects VGG's sensitivity to training hyperparameters rather than a property of EqWD specifically.

## Analysis of Ratio Trajectories

Figure~\ref{fig:ratio_trajectories} visualizes the per-layer gradient-to-weight ratio trajectories $r_t^l$ and the corresponding EMA equilibrium $r^{*,l}$ for EqWD on ImageNet ResNet-50. Several patterns are evident:

1. **Layer heterogeneity.** Early layers (e.g., conv1) exhibit small, stable ratios with minimal deviation from equilibrium, while later layers show larger ratios with higher variance. This validates the per-layer design: a global modulation would be dominated by the dynamics of the noisiest layers.

2. **Schedule-aligned modulation.** The modulation factor $\varphi_l(t)$ peaks at two characteristic points: during learning rate warmup (epochs 1--5), when the rapidly increasing $\gamma$ causes transient ratio excursions, and near the cosine decay knee (epochs 30--35), where the decreasing $\gamma$ triggers a new equilibrium transition. These are precisely the training phases where adaptive regularization is most beneficial.

3. **Stabilization effect.** Comparing EqWD with FixedWD, both exhibit similar ratio deviations from the theoretical equilibrium, but EqWD's adaptive response dampens the magnitude of subsequent deviations, leading to a more stable optimization trajectory in later epochs.

Figure~\ref{fig:wd_heatmap} shows the effective weight decay $\lambda_t^l$ as a heatmap across layers and training steps, illustrating how EqWD automatically concentrates stronger regularization in the deeper layers and during transitional phases, without any manual schedule design.
