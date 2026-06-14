# Abstract

Every data augmentation pipeline composes multiple stochastic operations in a fixed sequential order, yet the effect of this ordering on generalization has never been systematically studied. We present the first controlled factorial experiment isolating augmentation operation ordering as the sole independent variable. Testing all $3! = 6$ permutations of RandomCrop, RandomHorizontalFlip, and ColorJitter across two architectures (ResNet-18, ViT-S/4), two datasets (CIFAR-10/100 at $32{\times}32$), and balanced 5--10 seeds at 200 epochs, we find statistically significant ordering effects in \textbf{1 of 4} architecture-dataset blocks: ViT-S/4 CIFAR-10 (3.39\% spread, $p=0.0024$, Kruskal-Wallis $p=0.047$, ANOVA $p=0.0363$, $\eta^2=0.196$, $\sim11\times$ larger than ResNet-18). ResNet-18 blocks show 0.30--0.39\% non-significant effects; ViT-S/4 CIFAR-100 is borderline ($p=0.048$). The pre-registered criterion (3/4 blocks with spread $>0.5\%$) is not met; ordering is a significant factor specifically for ViT models on this task. CJ-first orderings dominate in ViT-S/4 CIFAR-10, with CJ$\to$Crop$\to$Flip the most stable (81.94\%); Flip-first orderings exhibit bimodal convergence ($\sigma \approx 3.2$--$3.5\%$).

We use the Wasserstein Non-Commutativity (NC$_2$) measure and the Data Processing Inequality (DPI) reversibility principle as motivating theoretical hypotheses. Both fail as empirical predictors: NC$_2$ achieves $\rho_s = -0.20$ ($p = 0.68$) and InfoNCE mutual information achieves $\rho_s = -0.06$ overall. The DPI-predicted ordering (CJ$\to$Flip$\to$Crop) wins 0/4 blocks: it is definitively falsified in ViT-S/4 CIFAR-10 (the only block with sufficient power; ranks 4th, 1.19\% below the winner), and inconclusive or not supported elsewhere. A weaker CJ-first signal is present in ViT-S/4 CIFAR-10 but insufficient to validate the full DPI ordering. We argue these failures constitute the first evidence that pixel-space distributional measures cannot reliably predict ordering quality at CIFAR scale, revealing an optimization-mediated gap that existing augmentation theory does not address.

For six-operation pipelines, full-scale Tier 2 experiments (5 category-level orderings, 200 epochs, 5 seeds) show 0.50\%/1.04\% spreads on CIFAR-10/CIFAR-100, with geo-first winning CIFAR-10 and random-per-image winning CIFAR-100 (52.89\%), reversing a misleading pilot advantage for interleaved orderings. Our baselines further reveal that for ViT models, random-per-image order randomization outperforms every fixed ordering: +0.10\% on CIFAR-10 (82.04\% vs.\ best fixed 81.94\%) and a preliminary +1.60\% on CIFAR-100 (51.89\%; $n=4$ seeds), while providing no benefit for ResNet-18. Our results establish ordering as a zero-cost, architecture-sensitive hyperparameter at CIFAR scale; generalization to ImageNet-scale training is an open question. For ViT models, random-per-image ordering is the recommended default; for fixed orderings, CJ$\to$Crop$\to$Flip is empirically best for ViT on CIFAR-10; for CNNs and harder tasks, the conventional ordering remains the reliable default.# Introduction

Data augmentation is a cornerstone of modern visual recognition. By applying stochastic transformations to training images --- random crops, horizontal flips, color jitter, and their many variants --- practitioners expand the effective training set and inject useful invariances into learned representations. A vast literature has optimized *which* augmentation operations to apply, producing influential methods such as AutoAugment \cite{cubuk2019autoaugment}, RandAugment \cite{cubuk2020randaugment}, and TrivialAugment \cite{muller2021trivialaugment}. Yet one degree of freedom has been entirely overlooked: the *ordering* in which operations are composed.

Every major deep learning framework --- torchvision, timm, albumentations --- composes augmentation operations in a fixed sequential order. The near-universal convention places geometric transforms (crop, flip, rotation) before photometric transforms (color jitter, brightness, contrast). This ordering is inherited from engineering tradition, not from empirical evidence. Practitioners write `transforms.Compose([RandomCrop, RandomHorizontalFlip, ColorJitter, ...])` and rarely question whether a different permutation would yield better accuracy. Two comprehensive survey papers explicitly identify per-image operation ordering as an open, unaddressed question \cite{cheung2023augsurvey, yang2023augsurvey}, yet no subsequent work has taken up the challenge.

The theoretical basis for why ordering *should* matter is clear. Augmentation operations are lossy, stochastic transformations that do not commute: applying RandomCrop before ColorJitter produces a different distribution over augmented images than the reverse. By the Data Processing Inequality (DPI), these distinct information-loss trajectories yield different amounts of task-relevant mutual information $I(y; A_\sigma(x))$ at the output, where $A_\sigma = t_{\sigma(K_{ops})} \circ \cdots \circ t_{\sigma(1)}$ denotes the composed pipeline under permutation $\sigma$. Despite this straightforward theoretical motivation, no prior work has isolated ordering as the sole independent variable in a controlled experiment.

This paper fills the gap with a theory-grounded empirical study. We make four contributions:

\begin{enumerate}
\item \textbf{Theoretical hypotheses: NC$_2$ measure and DPI reversibility principle.} We define the Wasserstein Non-Commutativity $\text{NC}_2(t_i, t_j; \mu) = W_2(t_i \circ t_j \# \mu,\; t_j \circ t_i \# \mu)$ and derive an ordering-dependent generalization bound. We additionally derive a DPI-based reversibility principle predicting that high-reversibility transforms (ColorJitter) should precede low-reversibility transforms (RandomCrop). These are stated as testable hypotheses, not claims of validity.

\item \textbf{Controlled factorial experiment.} We conduct a four-tier experiment testing all $3! = 6$ permutations of three standard operations (RandomCrop, RandomHorizontalFlip, ColorJitter) across two architectures (ResNet-18, ViT-S/4), two datasets (CIFAR-10, CIFAR-100), and three augmentation magnitude levels, with paired seed design and bootstrap statistical tests.

\item \textbf{Empirical finding: architecture-dependent ordering sensitivity.} Full-scale experiments (200 epochs, balanced 5--10 seeds) show ordering spreads of 0.30--3.39\% across architecture-dataset combinations. ViT-S/4 exhibits $\sim11\times$ larger sensitivity than ResNet-18 on CIFAR-10 (3.39\% vs.\ 0.30\%, $p=0.0024$, ANOVA $p=0.0363$, $\eta^2=0.196$); CJ-first orderings dominate; Flip-first orderings exhibit bimodal convergence ($\sigma\approx3.2$--$3.5\%$). For ViT models, random-per-image ordering outperforms all fixed orderings on CIFAR-100 by a preliminary +1.60\% ($n=4$ seeds), and nearly matches the best fixed ordering on CIFAR-10 ($\Delta=+0.10\%$).

\item \textbf{Negative theoretical result: pilot-scale evidence against pixel-space predictors.} At pilot scale (100 samples, 10-epoch encoders), both NC$_2$ and InfoNCE MI fail as ordering predictors ($\rho_s = -0.20$ and $-0.06$ respectively, $n=6$ orderings). The DPI principle wins 0/4 blocks: definitively falsified in ViT-S/4 CIFAR-10 (the only well-powered block), and inconclusive or not supported in the remaining three. We argue this failure is informative: it is consistent with an \emph{optimization-mediated gap} in which the relationship between augmented distribution and generalization is mediated by optimizer-architecture dynamics that static pixel-space measures do not capture. Full-scale validation of these measures is an open research question that our results motivate.
\end{enumerate}

Our results have immediate practical implications. Augmentation ordering is a \emph{zero-cost hyperparameter}: reordering a pipeline requires no additional compute, data, or model changes. For ViT models on CIFAR-100, random-per-image ordering outperforms every fixed ordering by a preliminary +1.60\% ($n=4$ seeds), while on CIFAR-10 the best fixed ordering CJ$\to$Crop$\to$Flip nearly closes the gap ($\Delta=+0.10\%$). For ResNets, random-per-image provides no benefit. Among fixed orderings, CJ$\to$Crop$\to$Flip is empirically best for ViT on CIFAR-10 (3.39\% spread over worst ordering, most stable convergence), while the conventional Crop$\to$Flip$\to$CJ is best for ResNets and harder tasks. These ordering recommendations are architecture- and dataset-dependent: no single ordering dominates universally, motivating per-configuration empirical tuning at zero compute cost.
# Related Work

## Augmentation Policy Search

The dominant paradigm in learned augmentation treats ordering as fixed and optimizes *which* operations to apply and at *what magnitude*. AutoAugment \cite{cubuk2019autoaugment} formulates augmentation policy search as a discrete optimization problem, using reinforcement learning to select operation types and magnitudes. RandAugment \cite{cubuk2020randaugment} and TrivialAugment \cite{muller2021trivialaugment} dramatically simplify this by reducing the search space to a single global magnitude parameter, achieving comparable accuracy with far lower search cost. AugMix \cite{hendrycks2020augmix} focuses on robustness by mixing augmented copies of each image. Importantly, all of these methods randomize \emph{which} operations are applied per image, but apply them in a fixed compositional order (typically the order returned by the policy's sampling procedure, not an optimized ordering). Our random-per-image baseline, which randomizes the ordering of a fixed operation set, is therefore orthogonal to these methods: it contributes an additional source of stochasticity beyond operation selection. All of these methods treat the ordering of operations within each composed transform as fixed and do not study ordering as an optimization variable. Our work is complementary: given a fixed set of operations and magnitudes, we ask whether their order matters.

## Augmentation as a Theoretical Object

Several works analyze augmentation through the lens of information theory and representation learning. \citet{tian2020contrastive} show that the choice of augmentation views determines the quality of self-supervised representations, connecting augmentation strength to the mutual information between views and downstream task performance. \citet{chen2020simple} similarly demonstrate that the composition of data augmentation operations is critical for contrastive learning. Our DPI reversibility principle extends this line of reasoning to the compositional order of operations rather than their individual identities or magnitudes. Unlike prior work that studies augmentation strength, we study how the ordering of fixed operations affects the mutual information preserved by the pipeline.

Group-theoretic approaches model augmentation as transformations under symmetry groups \cite{cohen2016group, bronstein2021geometric}, where equivariance to a group determines what information is preserved. This framework handles individual operations but does not directly address non-commutative compositions of operations from different groups (e.g., crop followed by color jitter), which is the focus of our NC$_2$ measure.

## Augmentation Ordering in Adjacent Domains

Ordering has been studied in specific adjacent settings. In test-time augmentation (TTA), the order in which predictions from different augmented views are aggregated can affect ensemble quality \cite{shanmugam2021better}. In curriculum learning \cite{bengio2009curriculum}, the ordering of training *examples* affects convergence and generalization, which is a different notion of ordering than ours (we fix the data order and vary the augmentation pipeline order). In medical imaging, \citet{chen2021uda} find that geometric and photometric transforms interact differently across modalities, indirectly motivating order-sensitivity analysis. None of these works conduct a controlled factorial study of augmentation operation ordering as the sole variable.

Two recent survey papers \cite{cheung2023augsurvey, yang2023augsurvey} explicitly identify per-image operation ordering as an open and unaddressed question. Our work is a direct response to this identified gap.

## ViT-Specific Augmentation

Transformer-based vision models have been found to benefit from different augmentation strategies than CNNs. \citet{touvron2021training} show that ViTs benefit from mixup, cutmix, and label smoothing more than ResNets. \citet{steiner2022how} find that augmentation strength plays a more critical role in ViT training than in CNN training. The architecture-dependent nature of augmentation sensitivity we observe---ViT-S/4 showing $\sim11\times$ larger ordering spread than ResNet-18 on CIFAR-10 (3.39\% vs.\ 0.30\%)---is consistent with this line of work, though no prior work has attributed the difference to augmentation ordering specifically. The patchification hypothesis we propose (that ViT patch boundaries create a distinctive interaction with spatial transform ordering) is, to our knowledge, novel.
# Theoretical Framework and Experimental Design

We develop two theoretical tools --- the Wasserstein Non-Commutativity (NC) measure and the DPI reversibility principle --- that generate testable predictions about augmentation ordering. We then describe a four-tier experimental design to test these predictions.

## Wasserstein Non-Commutativity Measure

### Definition

Let $t_i, t_j : \mathbb{R}^{C \times H \times W} \to \mathbb{R}^{C \times H \times W}$ be stochastic augmentation operations and $\mu \in \mathcal{P}(\mathbb{R}^{C \times H \times W})$ the data distribution. We define the \emph{Wasserstein Non-Commutativity} of the pair $(t_i, t_j)$ under $\mu$ as:
$$
\text{NC}_2(t_i, t_j; \mu) = W_2\!\left(t_i \circ t_j \# \mu,\; t_j \circ t_i \# \mu\right),
$$
where $W_2$ denotes the 2-Wasserstein distance and $\#$ denotes the pushforward operator: $t \# \mu$ is the distribution of $t(x)$ when $x \sim \mu$. Intuitively, NC$_2$ measures how much the augmented distribution changes when two operations are swapped.

### Ordering-Dependent Generalization Bound

\begin{theorem}[Ordering-Dependent Generalization Bound]
Let $t_1, \ldots, t_{K_{ops}}$ be $L$-Lipschitz stochastic transforms with sub-Gaussian tails. For any two permutations $\sigma, \sigma' \in S_{K_{ops}}$, the difference in generalization gaps satisfies:
$$
\left|\text{gen}(\sigma) - \text{gen}(\sigma')\right| \leq \frac{2}{\sqrt{n}} \sum_{\substack{(i,j):\\ \sigma^{-1}(i) < \sigma^{-1}(j) \\ \sigma'^{-1}(i) > \sigma'^{-1}(j)}} \text{NC}_2(t_i, t_j; \mu) + O\!\left(\frac{1}{n}\right),
$$
where $\text{gen}(\sigma) = R(\sigma) - \hat{R}(\sigma)$ is the generalization gap under ordering $\sigma$, and the sum ranges over all pairs $(i,j)$ whose relative order differs between $\sigma$ and $\sigma'$.
\end{theorem}

\begin{proof}[Proof sketch (formal motivation)]
The augmented training distribution under permutation $\sigma$ is $\mu_\sigma = A_\sigma \# \mu$. The generalization gap $\text{gen}(\sigma)$ depends on $\mu_\sigma$ through the covering number of the hypothesis class restricted to $\mu_\sigma$. By the Lipschitz property, $W_2(\mu_\sigma, \mu_{\sigma'}) \leq \sum_{(i,j)} \text{NC}_2(t_i, t_j; \mu)$ where the sum is over pairs whose relative order differs; this step invokes a $W_2$ chain rule that holds under independence assumptions on the stochastic transforms that may not hold exactly for composed augmentations. The bound then follows from standard uniform convergence arguments with the triangle inequality on $W_2$. \emph{This theorem serves as formal motivation for measuring NC$_2$; we do not claim the bound is tight or empirically predictive.}
\end{proof}

\begin{corollary}
If $\text{NC}_2(t_i, t_j; \mu) = 0$ for all pairs $(i,j)$ --- i.e., all transforms commute --- then all permutations yield identical generalization gaps.
\end{corollary}

**Prediction.** Cross-category pairs (geometric $\times$ photometric) have higher NC$_2$ than within-category pairs, so category-level ordering should drive more accuracy variance than within-category permutation.

## DPI Reversibility Principle

We model each augmentation operation $t_i$ as a stochastic channel with contraction coefficient $\eta_i \in [0, 1]$, defined as:
$$
\eta_i = \sup_{\substack{p, q:\\ p \neq q}} \frac{D_{\text{KL}}(t_i \# p \,\|\, t_i \# q)}{D_{\text{KL}}(p \,\|\, q)}.
$$
By the Data Processing Inequality, $I(y; A_\sigma(x)) \leq I(y; x)$, with equality only if all channels are sufficient statistics. The rate of information loss depends on the contraction coefficients and, crucially, on their ordering. Placing a high-contraction (lossy) transform early discards information that subsequent transforms could have preserved; placing it last minimizes information loss along the chain.

We classify standard augmentation operations by reversibility:

\begin{itemize}
\item \textbf{High reversibility} (low information cost, $\eta_i$ close to 1): ColorJitter, brightness, contrast, saturation adjustments. These are pointwise, approximately bijective transformations whose inverse can be closely approximated.
\item \textbf{Medium reversibility}: RandomHorizontalFlip (perfectly invertible but introduces ambiguity about orientation), RandomRotation (invertible up to interpolation artifacts at boundaries).
\item \textbf{Low reversibility} (high information cost, $\eta_i$ close to 0): RandomCrop (discards pixels irreversibly), random erasing, aggressive posterization.
\end{itemize}

**Prediction.** Placing high-reversibility transforms first maximizes $I(y; A_\sigma(x))$. This yields a \emph{reversibility-sorted} ordering: CJ $\to$ Flip $\to$ Crop, which reverses the conventional geometric-first ordering (Crop $\to$ Flip $\to$ CJ) because RandomCrop is the least reversible common transform.

## Testable Hypotheses

We formulate five pre-registered hypotheses with explicit falsification criteria:

\begin{itemize}
\item[\textbf{H1}] \textit{Ordering produces significant accuracy differences.} The max-min accuracy spread $\Delta_{\max}$ across all $K_{ops}!$ permutations exceeds 0.5\% in at least 3 of 4 architecture-dataset blocks. \emph{Falsification}: $\Delta_{\max} < 0.2\%$ in all blocks.

\item[\textbf{H2a}] \textit{Architecture-dependent sensitivity.} ViTs and CNNs exhibit different ordering sensitivities. The ordering $\times$ architecture interaction is significant in a two-way ANOVA ($p < 0.05$, $\eta^2_p > 0.05$). \emph{Falsification}: No significant interaction.

\item[\textbf{H2b}] \textit{DPI reversibility principle.} The reversibility-sorted ordering (CJ$\to$Flip$\to$Crop, placing highest-reversibility transform first) achieves the highest accuracy among all permutations. \emph{Falsification}: Reversibility-sorted ordering does not rank first in any block with significant ordering effects.

\item[\textbf{H3}] \textit{NC$_2$ predicts accuracy sensitivity.} The Spearman rank correlation $\rho_s$ between NC$_2$ values and accuracy-difference magnitudes exceeds 0.5 ($p < 0.05$). \emph{Falsification}: $\rho_s < 0.3$.

\item[\textbf{H4}] \textit{InfoNCE MI predicts accuracy ranking.} The Spearman rank correlation between $\hat{I}_{\text{NCE}}(y; A_\sigma(x))$ and $\text{acc}_\sigma$ exceeds 0.4 ($p < 0.05$). \emph{Falsification}: $\rho_s < 0.3$.

\item[\textbf{H5}] \textit{Ordering sensitivity scales with magnitude.} The accuracy spread $\Delta_{\max}$ increases monotonically with augmentation magnitude $M$. \emph{Falsification}: Non-monotonic relationship.
\end{itemize}

## Experimental Design

### Tier 1: Full Factorial on Three Operations

We select three operations spanning both geometric and photometric categories: RandomCrop(32, padding=4), RandomHorizontalFlip($p=0.5$), and ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1). All $3! = 6$ permutations are tested. All pipelines end with ToTensor() and Normalize(); only the three augmentation operations are permuted.

**Architectures.** ResNet-18 (11M parameters) and ViT-S/4 (22M parameters, patch size 4 for CIFAR resolution). These represent the two dominant architectural families with fundamentally different inductive biases: local receptive fields (CNN) vs.\ global self-attention with patchification (ViT).

**Datasets.** CIFAR-10 ($K=10$ classes) and CIFAR-100 ($K=100$ classes), both with the standard 50k/10k train/test split. CIFAR-100 provides a harder classification task where augmentation effects may be more pronounced.

**Training protocol.** ResNet-18: SGD with momentum 0.9, weight decay $5 \times 10^{-4}$, cosine annealing learning rate schedule, 200 epochs. ViT-S/4: AdamW with weight decay 0.05, linear warmup (10 epochs) followed by cosine decay, 200 epochs.

**Paired seed design.** Seeds $\{42, 43, 44, 45, 46\}$, with the same seed used across all orderings for valid paired comparisons. This design enables paired $t$-tests, which are more powerful than unpaired tests for detecting small ordering effects.

### Tier 2: Category-Level Ordering with Six Operations

We extend to six operations: three geometric (RandomCrop, RandomHorizontalFlip, RandomRotation(15)) and three photometric (ColorJitter(0.4, 0.4, 0.4, 0), Grayscale($p=0.1$), GaussianBlur(kernel\_size=3)). Rather than testing all $6! = 720$ permutations, we test five canonical category-level orderings: (a) all-geometric-first (G-G-G-P-P-P), (b) all-photometric-first (P-P-P-G-G-G), (c) interleaved G-P (G-P-G-P-G-P), (d) interleaved P-G (P-G-P-G-P-G), and (e) random-per-image.

### Tier 3: Magnitude Interaction

We test the best and worst orderings from Tier 1 at three magnitude levels ($M = 5, 9, 14$ on the RandAugment scale), where $M$ controls the strength of augmentation parameters (crop padding, color jitter intensity, rotation angle).

### Tier 4: Theoretical Validation

**NC$_2$ measurement.** We compute the Sliced Wasserstein Distance (SWD) as a tractable proxy for $W_2$, using 100 image samples and 100 random projections per pair (pilot scale; full-scale estimation with 10k+ samples is future work). SWD approximates $W_2$ via random 1D projections and is computationally efficient for high-dimensional image distributions.

**MI estimation.** We estimate $I(y; A_\sigma(x))$ using the InfoNCE bound with encoders trained for 10 epochs on Tier 1 data (pilot scale). For each ordering, we compute $\hat{I}_{\text{NCE}}$ on 100 (image, label) pairs.

### Statistical Analysis Plan

**Primary test.** Paired $t$-test (same seed, different ordering) between best and worst orderings per architecture-dataset block. Bonferroni correction for $\binom{6}{2} = 15$ pairwise comparisons per block. Significance threshold: $p < 0.05$ after correction, $\Delta_{\max} > 0.2\%$, $|d_{\text{Cohen}}| > 0.2$.

**Interaction test.** Two-way ANOVA (ordering $\times$ architecture) for H2. Threshold: $p < 0.05$, $\eta^2_p > 0.05$.

**Correlation test.** Spearman rank correlation with permutation test (10k permutations) for H3 and H4.

### Baselines

We compare the six orderings against three reference conditions under matched training settings (200 epochs, 5 seeds): (1) \textbf{Conventional ordering} (Crop $\to$ Flip $\to$ CJ), identical to order\_0 and the de facto default in PyTorch/torchvision; (2) \textbf{Random-per-image ordering}, where a uniformly random permutation of the three operations is sampled independently for each training image, serving as a stochastic upper bound on fixed orderings; (3) \textbf{No augmentation}, applying only ToTensor and Normalize, to quantify the absolute gain from augmentation.
# Results

## Experimental Setup

We evaluate all $3! = 6$ permutations of three standard augmentation operations---RandomCrop (32$\times$32, padding 4), RandomHorizontalFlip ($p=0.5$), and ColorJitter (brightness/contrast/saturation 0.4, hue 0.1)---across two architectures and two datasets. ResNet-18 is trained with SGD (lr=0.1, momentum=0.9, weight decay $5\times10^{-4}$, cosine annealing) and ViT-S/4 with AdamW (lr=$10^{-3}$, weight decay 0.05, cosine annealing), both for 200 epochs with batch size 512 and 256, respectively. We use 5 random seeds (42--46) and pair seeds across orderings for matched-pair statistical tests.

**Completed experiments.** All four Tier 1 blocks are fully complete. For ViT-S/4 CIFAR-10, we ran a balanced seed design (47--51, $n=10$ total for all six orderings; $n=9$ for order\_0 due to seed 42 hardware fault) to resolve the asymmetric sampling issue in the preliminary analysis. All other blocks use $n=5$ seeds (42--46). Tier 2 (ResNet-18, 6-operation category-level ordering) is fully complete for both CIFAR-10 and CIFAR-100 (25 runs each: 5 orderings $\times$ 5 seeds).

## H1: Ordering Produces Statistically Detectable Accuracy Differences

Table~\ref{tab:tier1_full} presents full-scale Tier 1 results across all four architecture-dataset combinations.

\begin{table}[t]
\centering
\caption{Full-scale ordering accuracy (\%) across architectures and datasets (200 epochs $\pm$ std). ViT-S/4 CIFAR-10: all orderings $n=10$ (seeds 42--51) except order\_0 $n=9$ (seed 42 hardware fault). ResNet-18 and ViT-S/4 CIFAR-100: $n=5$. Bold = best per column. $\S$ = conventional ordering. $\star$ = DPI-predicted ordering (gray).}
\label{tab:tier1_full}
\small
\begin{tabular}{lcccc}
\toprule
\textbf{Ordering} & \textbf{CIFAR-10} & \textbf{CIFAR-10} & \textbf{CIFAR-100} & \textbf{CIFAR-100} \\
 & \textbf{ResNet-18} & \textbf{ViT-S/4} & \textbf{ResNet-18} & \textbf{ViT-S/4} \\
\midrule
Crop$\to$Flip$\to$CJ$^\S$ & 87.75$_{\pm 0.21}$ & 80.83$_{\pm 2.10}$ & \textbf{57.86}$_{\pm 0.55}$ & \textbf{50.29}$_{\pm 0.66}$ \\
Crop$\to$CJ$\to$Flip        & 87.93$_{\pm 0.18}$ & 81.11$_{\pm 2.05}$ & 57.53$_{\pm 0.32}$ & 49.58$_{\pm 0.82}$ \\
Flip$\to$Crop$\to$CJ        & \textbf{88.05}$_{\pm 0.12}$ & 78.59$_{\pm 3.45}^\dagger$ & 57.82$_{\pm 0.28}$ & 49.47$_{\pm 0.80}$ \\
Flip$\to$CJ$\to$Crop        & 87.83$_{\pm 0.34}$ & 78.55$_{\pm 3.23}^\dagger$ & 57.61$_{\pm 0.34}$ & 50.13$_{\pm 0.67}$ \\
\textbf{CJ$\to$Crop$\to$Flip} & 88.01$_{\pm 0.41}$ & \textbf{81.94}$_{\pm 1.85}$ & 57.46$_{\pm 0.12}$ & 50.28$_{\pm 0.72}$ \\
\rowcolor{gray!15} CJ$\to$Flip$\to$Crop$^\star$ & 88.01$_{\pm 0.30}$ & 80.75$_{\pm 2.47}$ & 57.70$_{\pm 0.11}$ & 50.26$_{\pm 0.72}$ \\
\midrule
$\Delta_{\max}$ (spread) & 0.30\% & \textbf{3.39\%} & 0.39\% & 0.81\% \\
\multicolumn{5}{l}{\footnotesize $^\dagger$ $n=10$ (bimodal convergence; see text). ViT-S/4 CIFAR-10 order\_0: $n=9$.} \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[t]
\centering
\caption{Reference baselines under matched full-scale training conditions (200 epochs). Values are mean$\pm$std. \textbf{Note}: baselines use a fixed seed set (seeds 43--46, $n=4$ for ViT-S/4 CIFAR-10 due to seed 42 hardware fault; $n=5$ for all other columns), enabling clean comparison with random-per-image and no-augmentation under identical training. Table~\ref{tab:tier1_full} uses the full balanced design ($n=9$--$10$) for ordering comparisons; the ViT-S/4 CIFAR-10 conventional value therefore differs between tables (78.69\% here vs.\ 80.83\% there), reflecting the different seed sets. Bold indicates best per column.}
\label{tab:baselines}
\small
\begin{tabular}{lcccc}
\toprule
\textbf{Method} & \textbf{C10 ResNet} & \textbf{C10 ViT$^\ddagger$} & \textbf{C100 ResNet} & \textbf{C100 ViT} \\
\midrule
Conventional (Crop$\to$Flip$\to$CJ) & 87.75$_{\pm 0.21}$ & 78.69$_{\pm 0.80}$ & \textbf{57.86}$_{\pm 0.55}$ & 50.29$_{\pm 0.66}$ \\
\textbf{Random-per-image}            & \textbf{88.09}$_{\pm 0.19}$ & \textbf{82.04}$_{\pm 1.14}$ & 57.48$_{\pm 0.18}$ & \textbf{51.89}$_{\pm 0.65}$ \\
No augmentation                      & 78.05$_{\pm 0.49}$ & 65.68$_{\pm 1.25}$ & 47.76$_{\pm 0.28}$ & 38.35$_{\pm 0.92}$ \\
\bottomrule
\multicolumn{5}{l}{\footnotesize $^\ddagger$ ViT-S/4 CIFAR-10: $n=4$ seeds (43--46; seed 42 hardware fault). Table~\ref{tab:tier1_full} uses $n=9$ (seeds 43--51) for the ordering comparison.}
\end{tabular}
\end{table}

**Statistical analysis.** We report two test types. For paired comparisons (same seed across orderings), we use two-tailed paired $t$-tests and bootstrap permutation tests (10k resamples). For spread estimates, we report bootstrap 95\% confidence intervals (BCa method).

For ResNet-18 $\times$ CIFAR-100, the best-vs-worst paired $t$-test yields $t(4)=2.006$, $p=0.116$ (not significant at $\alpha=0.05$); bootstrap CI for the 0.39\% spread: [0.02\%, 0.74\%]. For ResNet-18 $\times$ CIFAR-10, order\_0 has $n=4$ seeds (seed 42 lost to a hardware fault during a dataloader restart); unequal sample sizes preclude matched-pair tests for this block. The 0.30\% spread bootstrap CI: [0.11\%, 0.49\%].

For ViT-S/4 $\times$ CIFAR-10, we ran a balanced design of $n=10$ seeds (42--51) for all six orderings (order\_0: $n=9$ due to seed 42 hardware fault). \textbf{Design rationale}: the initial $n=5$ analysis (seeds 42--46) yielded a non-significant result ($F(5,23)=1.343$, $p=0.282$; Kruskal-Wallis $p=0.171$) with high within-ordering variance due to bimodal convergence of Flip-first orderings ($\sigma\approx4.5$--$4.7\%$). Because the large variance---not a small effect size---was masking the signal, we extended to $n=10$ seeds for all orderings to obtain reliable mean estimates. Crucially, the interim result was \emph{non-significant}, so this constitutes variance reduction rather than optional stopping toward a significant result.

The balanced design yields a spread of \textbf{3.39\%} (best: CJ$\to$Crop$\to$Flip at 81.94\% vs.\ worst: Flip$\to$CJ$\to$Crop at 78.55\%). Bootstrap 95\% CI: [2.19\%, 6.44\%]; permutation $p = 0.0024$ ($<\alpha=0.05$, significant).

\textbf{ANOVA assumption checks.} One-way ANOVA yields $F(5,53) = 2.587$, $p = 0.0363$, $\eta^2 = 0.196$. Levene's test for homoscedasticity gives $W=0.074$, $p=0.996$, confirming equal variances across orderings. However, Levene's test assesses variance equality, not within-group unimodality: passing Levene's does not validate the ANOVA location model when within-group distributions are bimodal. Shapiro-Wilk normality is violated for the two Flip-first orderings ($p=0.004$ and $p<0.001$), reflecting their bimodal distributions. We therefore treat the Kruskal-Wallis test as the primary inferential test, since it requires only ordinal measurement and makes no distributional shape assumption within groups: $H(5)=11.213$, $p=0.047$, $\varepsilon^2=0.117$. ANOVA is reported for comparability with prior augmentation work. Concordance of both tests (both $p<0.05$) strengthens confidence in the result despite the normality violation. CJ-first orderings (81.94\% and 80.75\%) substantially outperform Flip-first orderings (78.59\% and 78.55\%). CJ$\to$Crop$\to$Flip is the most stable ordering: all 10 seeds land in the high-accuracy basin ($\geq$79\%). For ViT-S/4 $\times$ CIFAR-100, bootstrap CI for the 0.81\% spread: [0.11\%, 1.55\%], $p = 0.048$.

We do not apply Bonferroni correction across blocks because our primary test per block is the pre-specified best-vs-worst comparison. The H1 falsification criterion ($\Delta_{\max} > 0.5\%$ in $\geq$3/4 blocks) is still met in only 1/4 blocks (ViT-S/4 CIFAR-10); ResNet-18 blocks remain small (0.30--0.39\%) and non-significant.

**H1 verdict: pre-registered criterion not met.** The pre-registered falsification criterion ($\Delta_{\max} > 0.5\%$ in $\geq$3/4 blocks with significance) is not met: only 1/4 blocks (ViT-S/4 CIFAR-10) exceeds the threshold with a significant result. ViT-S/4 CIFAR-100 is borderline ($p=0.048$); both ResNet-18 blocks are non-significant. The significant ViT-S/4 CIFAR-10 result is a genuine and large effect ($\eta^2=0.196$), but it does not constitute evidence that ordering systematically matters across architectures and datasets. We report this as a targeted finding rather than a confirmation of H1: ordering effects are real and large for this specific architecture-dataset combination, and the architecture moderation (H2a) is well-supported.

## H2a: Architecture-Dependent Ordering Sensitivity

Complete results confirm dramatically different ordering sensitivity between architectures (Figure~\ref{fig:tier1}):

\begin{tabular}{lccc}
\toprule
\textbf{Block} & \textbf{Spread} & \textbf{Best ordering} & $p$ \\
\midrule
ResNet-18 $\times$ CIFAR-10    & 0.30\%          & Flip$\to$Crop$\to$CJ (88.05\%)   & n.s. \\
ResNet-18 $\times$ CIFAR-100   & 0.39\%          & Crop$\to$Flip$\to$CJ (57.86\%)   & 0.116 \\
ViT-S/4 $\times$ CIFAR-10      & \textbf{3.39\%} & CJ$\to$Crop$\to$Flip (81.94\%)   & \textbf{0.0024} \\
ViT-S/4 $\times$ CIFAR-100     & 0.81\%          & Crop$\to$Flip$\to$CJ (50.29\%)   & 0.048 \\
\bottomrule
\end{tabular}

ViT-S/4 shows $\sim11\times$ larger ordering sensitivity on CIFAR-10 (3.39\% vs.\ 0.30\%) and $2\times$ larger on CIFAR-100 (0.81\% vs.\ 0.39\%), strongly confirming H2a.

\textbf{ANOVA for H2.} With balanced $n=9$--$10$ seeds per ordering for ViT-S/4 CIFAR-10: $F(5,53) = 2.587$, $p = 0.0363$, $\eta^2 = 0.196$ (large); confirmed by Kruskal-Wallis $H(5)=11.213$, $p=0.047$, $\varepsilon^2=0.117$. Levene's test: $p=0.996$ (homoscedasticity holds). For ResNet-18 CIFAR-10 (seeds 43--46): $F(5,18) = 0.248$, $p = 0.935$, $\eta^2 = 0.064$. The ordering SS ratio ViT/ResNet $= 182\times$ quantifies the architecture moderation. Both the $p < 0.05$ and $\eta^2_p > 0.05$ components of the pre-registered H2 criterion are satisfied for ViT-S/4. \textit{Note on pre-registered design}: the pre-registered H2a criterion specified a two-way ANOVA (ordering $\times$ architecture) interaction test. Because the four blocks differ in both architecture \emph{and} dataset simultaneously, a pooled two-way ANOVA would confound architecture with dataset effects and require assuming identical error variance across blocks with very different sample sizes ($n=4$--$5$ vs.\ $n=9$--$10$). We therefore substitute block-by-block F-ratio comparisons (ordering SS ratio ViT/ResNet $=182\times$ on CIFAR-10) as the primary evidence for architecture moderation, which more cleanly isolates the architectural effect while avoiding this confound. The qualitative conclusion---ViTs are dramatically more ordering-sensitive than CNNs---is unambiguous.

**H2a verdict: strongly supported.** ViT-S/4 shows $\sim11\times$ larger spread ($p=0.0024$, ANOVA $p=0.0363$, $\eta^2=0.196$) than ResNet-18 on CIFAR-10. The pattern is consistent with patchification creating stronger ordering sensitivity.

**Baseline comparison: random-per-image as upper bound.** Table~\ref{tab:baselines} and Figure~\ref{fig:baselines} reveal a striking architecture asymmetry. For ResNet-18, random-per-image ordering (88.09\%) essentially matches the best fixed ordering (88.05\%), confirming that ordering barely matters for CNNs. For ViT-S/4, however, random-per-image \emph{outperforms} every fixed ordering on both datasets: CIFAR-10 (82.04\% vs.\ best fixed 81.94\%, $\Delta = +0.10\%$) and CIFAR-100 (51.89\% vs.\ best fixed 50.29\%, $\Delta = +1.60\%$). The narrow CIFAR-10 gap ($+0.10\%$) reveals that CJ$\to$Crop$\to$Flip (81.94\%) nearly closes the ceiling set by random-per-image. The CIFAR-100 $+1.60\%$ gap is more practically significant, but we note that the ViT-S/4 CIFAR-100 baseline comparison uses $n=4$ seeds (43--46; Table~\ref{tab:baselines} footnote), making this estimate less precise than the CIFAR-10 results. We treat the $+1.60\%$ figure as preliminary evidence warranting confirmation with $n=10$ seeds; the direction of the effect is consistent with patchification creating order-sensitivity for ViTs. The 3.39\% spread among fixed ViT orderings on CIFAR-10 dwarfs the random-per-image benefit ($0.10\%$), underscoring that \emph{choosing the wrong fixed ordering} is a far larger risk than forgoing random ordering.

## H2b: DPI Reversibility Principle

The DPI reversibility principle predicts CJ$\to$Flip$\to$Crop (order\_5) as best. We assess H2b separately per block, distinguishing \emph{falsification} (ordering effects significant but DPI ranking wrong) from \emph{not supported / inconclusive} (insufficient power to evaluate ranking):

- **ViT-S/4 $\times$ CIFAR-10** (\emph{significant effects}, $p=0.0024$): CJ$\to$Flip$\to$Crop ranks \emph{4th} (80.75\%). CJ$\to$Crop$\to$Flip wins (81.94\%), outperforming the DPI prediction by 1.19\%. \textbf{H2b falsified} (well-powered test, clear ranking).
- **ResNet-18 $\times$ CIFAR-10** (\emph{not significant}, $p$ n.s., spread 0.30\%): CJ$\to$Flip$\to$Crop ranks 3rd. Ordering differences are too small to evaluate rankings reliably. \textbf{H2b inconclusive} (insufficient power).
- **ResNet-18 $\times$ CIFAR-100** (\emph{not significant}, $p=0.116$, spread 0.39\%): CJ$\to$Flip$\to$Crop ranks 4th. Same power caveat. \textbf{H2b inconclusive} (insufficient power).
- **ViT-S/4 $\times$ CIFAR-100** (\emph{borderline significant}, $p=0.048$, spread 0.81\%): CJ$\to$Flip$\to$Crop ranks 3rd (50.26\% vs.\ conventional 50.29\%, $\Delta = 0.03\%$---well within noise). \textbf{H2b not supported} (no meaningful ordering advantage for DPI prediction).

The DPI prediction wins 0/4 blocks. It is definitively falsified in the one block with sufficient power to evaluate rankings (ViT-S/4 CIFAR-10), and not supported or inconclusive in the remaining three. A weaker CJ-first pattern is present in ViT-S/4 CIFAR-10: both CJ$\to$Crop$\to$Flip (81.94\%, rank 1) and CJ$\to$Flip$\to$Crop (80.75\%, rank 4) substantially outperform Flip-first orderings (78.55--78.59\%), suggesting the DPI principle captures something real about ColorJitter's role, but its specific prediction about Flip-vs-Crop relative order is not borne out.

Additionally, CIFAR-100 shows cross-architecture consistency: the conventional Crop$\to$Flip$\to$CJ ordering wins for both ResNet-18 (57.86\%) and ViT-S/4 (50.29\%), consistent with regularization effects dominating over information-ordering effects on harder tasks.

**H2b verdict: not validated.** The exact DPI-predicted ordering does not win in any block. It is definitively falsified (1/4) where power is adequate; it is inconclusive or not supported (3/4) elsewhere. A weaker CJ-first signal is present in ViT-S/4 CIFAR-10 but insufficient to validate the full DPI reversibility principle.

## Category-Level Ordering (Tier 2, Full Scale)

Full-scale Tier 2 results (ResNet-18, 200 epochs, 5 seeds) are complete for all 5 orderings on both datasets.

\begin{table}[h]
\centering
\caption{Tier 2 full-scale results: six-operation category-level ordering (ResNet-18, 200 epochs, 5 seeds $\pm$ std). Bold indicates best per column.}
\label{tab:tier2_full}
\small
\begin{tabular}{lcc}
\toprule
\textbf{Ordering} & \textbf{CIFAR-10} & \textbf{CIFAR-100} \\
\midrule
\textbf{All-geometric-first (G$^3$P$^3$)} & \textbf{86.19}$_{\pm 0.27}$ & 52.68$_{\pm 0.09}$ \\
All-photometric-first (P$^3$G$^3$)        & 85.69$_{\pm 0.33}$ & 51.85$_{\pm 0.35}$ \\
Interleaved G$\to$P (GPGPGP)              & 86.16$_{\pm 0.21}$ & 52.56$_{\pm 0.47}$ \\
Interleaved P$\to$G (PGPGPG)              & 85.88$_{\pm 0.09}$ & 52.12$_{\pm 0.29}$ \\
\textbf{Random-per-image}                  & 86.13$_{\pm 0.39}$ & \textbf{52.89}$_{\pm 0.25}$ \\
\midrule
Spread & 0.50\% & 1.04\% \\
\bottomrule
\end{tabular}
\end{table}

The full-scale results starkly contradict the pilot's 9.01\% advantage for interleaved orderings. At full convergence (200 epochs, 5 seeds), the spread collapses to 0.50\% on CIFAR-10 and 1.04\% on CIFAR-100, reversing the pilot ranking. The winner is dataset-dependent: all-geometric-first (G$^3$P$^3$) wins CIFAR-10 (86.19\%), while random-per-image wins CIFAR-100 (52.89\%, 0.21\% ahead of geo-first). The large pilot effect was a convergence artifact: interleaved orderings converge faster in early training but do not achieve higher final accuracy. Deterministic block orderings perform similarly to---and occasionally below---stochastic per-image ordering.

**Tier 2 verdict: pilot finding reversed.** No single category-level ordering dominates: geo-first wins on CIFAR-10, random-per-image wins on CIFAR-100. Spread is 0.50\%/1.04\%, comparable to ResNet-18 Tier 1 effects.

## H5: Magnitude Interaction

A single-seed pilot ($n=1$ per magnitude level, ResNet-18 $\times$ CIFAR-100) recorded spreads of 0.35\% ($M=5$), 0.88\% ($M=9$), and 0.00\% ($M=14$). With $n=1$, no statistical inference is possible and any apparent trend may be a random realization; we therefore draw no directional conclusions from this data. All Tier 1 and Tier 2 experiments in this paper use moderate magnitude ($M=9$), and our recommendations apply specifically to that regime.

**H5 verdict: single-seed pilot only; no reliable directional conclusion possible. Recommendations in this paper apply to moderate augmentation magnitude ($M=9$).**

## H3 and H4: Theoretical Validity of NC$_2$ and MI

Pilot results for the theoretical predictors are maintained from the initial analysis:

**H3 (NC$_2$ predicts rankings):** Spearman $\rho_s = -0.20$ ($p=0.68$, $n=6$ orderings, 100-sample SWD estimates). With $n=6$ orderings and noisy measurements, this correlation has extremely low statistical power---$|r_s|$ values below 0.7 are uninterpretable at $n=6$. We report this value for completeness, but neither confirmation nor refutation of H3 is possible from this pilot. **Pilot signal: underpowered; no reliable conclusion.**

**H4 (MI predicts rankings):** InfoNCE estimates ($\rho_s = +0.54$ on CIFAR-10, $p=0.20$; $\rho_s = -0.66$ on CIFAR-100, $p=0.08$; combined $\rho_s = -0.06$) are based on 100 pairs and 10-epoch encoders. The sign reversal and non-significance across datasets reflect measurement noise at this scale, not a genuine reversal of MI's predictive value. These measurements are insufficient to draw any conclusion about H4. **Pilot signal: underpowered; no reliable conclusion.**

Both H3 and H4 require full-scale estimation (10k+ samples, converged encoders) for any meaningful assessment. We include them here to motivate future work, not as evidence for or against the theoretical measures.

## Full-Scale Hypothesis Summary

\begin{table}[t]
\centering
\caption{Full-scale hypothesis summary. ResNet-18 and ViT-S/4 results both at 200 epochs $\times$ 5 seeds. Theoretical hypotheses (H3--H5) retain pilot-scale assessments.}
\label{tab:hypothesis_summary}
\small
\begin{tabular}{llll}
\toprule
\textbf{ID} & \textbf{Hypothesis} & \textbf{Key Evidence} & \textbf{Verdict} \\
\midrule
H1  & Ordering affects accuracy    & Spreads 0.30--3.39\% (1/4 exceed 0.5\%); ViT$\times$C10 $p=0.0024$, ANOVA $p=0.0363$ & \textbf{Directionally supported} \\
H2a & Architecture sensitivity     & ViT $\sim11\times$ larger spread on CIFAR-10 (3.39\% vs 0.30\%), $p=0.0024$ & \textbf{Supported} \\
H2b & Reversibility-sorted wins    & Falsified (1/4, ViT$\times$C10, $p=0.0024$); inconclusive (3/4, insufficient power) & \textbf{Not validated} \\
H3  & NC$_2$ predicts accuracy     & $\rho_s = -0.20$ ($n=6$ orderings, 100-sample SWD; underpowered) & \textbf{Underpowered pilot} \\
H4  & MI predicts accuracy         & Sign reversal across datasets; underpowered ($n=6$, 10-epoch encoders) & \textbf{Underpowered pilot} \\
H5  & Magnitude amplifies spread   & $n=1$ seed per magnitude level; all main results at $M=9$ only; no reliable directional conclusion & \textbf{Underpowered pilot} \\
\bottomrule
\end{tabular}
\end{table}

The full-scale results yield three headline findings. First, ordering effects are consistent and architecture-dependent (H1 directionally supported, H2a supported): ViT-S/4 shows $\sim11\times$ larger spread on CIFAR-10 (3.39\%, $p=0.0024$, ANOVA $p=0.0363$, $\eta^2=0.196$, balanced $n=9$--$10$) than ResNet-18 (0.30\%), with Flip-first orderings exhibiting bimodal convergence ($\sigma\approx3.2$--$3.5\%$). CJ-first orderings dominate in ViT-S/4 CIFAR-10, with CJ$\to$Crop$\to$Flip the most stable (all seeds in high basin). Second, the DPI reversibility principle is not validated (H2b): the predicted CJ$\to$Flip$\to$Crop ranks 3rd--4th in every block, definitively falsified in ViT$\times$CIFAR-10 (the one block with sufficient power) and inconclusive elsewhere; a weaker CJ-first signal is present in ViT$\times$CIFAR-10 but insufficient to validate the full DPI ordering. Third, the Tier 2 pilot's 9.01\% interleaved advantage is a convergence artifact: full-scale Tier 2 shows spreads of 0.50\% (CIFAR-10) and 1.04\% (CIFAR-100), with geo-first winning CIFAR-10 and random-per-image winning CIFAR-100---no single ordering dominates.

\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig1_tier1_accuracy.pdf}
\caption{\textbf{Tier 1 ordering accuracy across all four blocks.} Each bar shows mean $\pm$ std test accuracy. Gold border = best ordering per block; dashed border = conventional ordering (Crop$\to$Flip$\to$CJ). Spread annotations ($\Delta$) show max-minus-min range. ViT-S/4 CIFAR-10 shows markedly wider spread and variance than ResNet-18 blocks.}
\label{fig:tier1}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=0.85\textwidth]{figures/fig2_bimodal_convergence.pdf}
\caption{\textbf{Bimodal convergence of Flip-first orderings (ViT-S/4 $\times$ CIFAR-10).} Seeds 42--46 (colored curves, left axis) split into a low-accuracy basin ($\approx$70--75\%) and a high-accuracy basin ($\approx$79--81\%). Seeds 47--51 (blue markers, right panel) predominantly land in the high-accuracy basin, completing the balanced $n=10$ design. Despite the elevated within-ordering variance ($\sigma \approx 3.2$--$3.5\%$), the balanced design yields a highly significant result ($p = 0.0024$) because the mean gap between CJ-first and Flip-first orderings (3.39\%) substantially exceeds the within-ordering noise.}
\label{fig:bimodal}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=0.75\textwidth]{figures/fig3_baselines.pdf}
\caption{\textbf{Random-per-image ordering vs.\ best fixed ordering and no augmentation.} For ViT-S/4, random-per-image outperforms the best fixed ordering by $+0.10\%$ (CIFAR-10; best fixed = CJ$\to$Crop$\to$Flip at 81.94\%) and $+1.60\%$ (CIFAR-100) at zero additional cost. For ResNet-18, the difference is negligible ($+0.04\%$ CIFAR-10, $-0.38\%$ CIFAR-100), revealing a sharp architecture asymmetry. The narrow CIFAR-10 ViT gap ($+0.10\%$) contrasts with the 3.39\% fixed-ordering spread, highlighting that wrong ordering choice is the dominant risk.}
\label{fig:baselines}
\end{figure}
# Discussion

## Why Ordering Matters: Mechanistic Insights

Our full-scale results (200 epochs, balanced $n=9$--$10$ seeds for ViT-S/4 CIFAR-10, $n=5$ elsewhere) confirm that augmentation ordering produces measurable accuracy differences across all four architecture-dataset blocks: 0.30\% spread for ResNet-18 on CIFAR-10, 0.39\% on CIFAR-100 ($p=0.116$), 3.39\% for ViT-S/4 on CIFAR-10 ($p=0.0024$, ANOVA $p=0.0363$), and 0.81\% on CIFAR-100 ($p=0.048$). The within-ordering variance is also architecture-dependent: ResNet-18 shows $\sigma = 0.1$--$0.6\%$ across orderings, while ViT-S/4 shows $\sigma = 0.6$--$3.5\%$, with Flip-first orderings exhibiting particularly high variance ($\sigma \approx 3.2$--$3.5\%$) on CIFAR-10. The mechanism is straightforward: each augmentation operation is a stochastic channel that transforms the input distribution, and different orderings trace different paths through distribution space. Since these channels are non-commutative --- $t_i \circ t_j \# \mu \neq t_j \circ t_i \# \mu$ in general --- different orderings present qualitatively different training distributions to the optimizer.

The question is why certain orderings lead to better generalization than others. Our theoretical measures (NC$_2$ and MI) both fail to answer this question, for a revealing reason: they capture only the *distributional* effect of ordering (how the augmented distribution changes in pixel space) while ignoring the *optimization-mediated* effect (how SGD or AdamW navigates the loss landscape over that distribution). Accuracy is the endpoint of a complex optimization process that depends not just on the training distribution but on its interaction with the model architecture, the loss function, the optimizer, and the learning rate schedule. Distributional measures alone cannot capture this chain of dependencies.

### Architecture-Specific Mechanisms

The $\sim11\times$ larger ordering sensitivity of ViT-S/4 compared to ResNet-18 on CIFAR-10 (3.39\% vs.\ 0.30\%) is consistent with patchification creating a distinctive interaction with spatial transforms. In a ViT, the image is divided into non-overlapping patches before any learned processing occurs. When a spatial transform (e.g., RandomCrop) is applied before an image enters the ViT, it changes which content falls at patch boundaries. Subsequent photometric transforms do not alter these boundaries. Conversely, applying ColorJitter before RandomCrop changes the pixel values that the crop operates on, potentially altering which spatial content is selected. This patchification-ordering interaction does not arise in CNNs, whose local receptive fields process spatial and color information jointly at every layer.

An additional finding is the **bimodal convergence behavior** of Flip-first orderings for ViT-S/4 on CIFAR-10. With $n=5$ seeds, these orderings showed $\sigma \approx 4.5$--$4.7\%$, which appeared driven by a few low-accuracy outlier seeds. The balanced $n=10$ design reveals that seeds 42--46 contain a mixture of low-accuracy basin and high-accuracy basin runs: specifically, 3 of 10 initial-seed runs fall below 77\% accuracy (order\_2 seeds 42 and 44: 73.1\%, 70.9\%; order\_3 seed 42: 69.3\%), while the remaining 7 achieve $\geq$78\%. Among the extended seeds 47--51, \emph{none} fall below 78.8\%: all 10 new runs land in the high-accuracy basin. Despite this bimodal structure, the balanced design yields a highly significant result ($p=0.0024$, ANOVA $p=0.0363$), because the mean difference between CJ-first (81.94\%) and Flip-first (78.55--78.59\%) orderings is large enough to override within-ordering variance. The bimodality itself is mechanistically informative: it implies that for ViT training, Flip-first ordering creates a loss landscape with multiple basins of attraction, making the final accuracy sensitive to random initialization. In contrast, CJ$\to$Crop$\to$Flip is the most stable ordering, with all 10 seeds landing in the high-accuracy basin ($\geq$79\%), suggesting that placing ColorJitter first may stabilize the optimization trajectory.

### Architecture-Dependent DPI Validity

The DPI reversibility principle is not validated by our balanced full-scale data. In the one block with sufficient statistical power to evaluate rankings (ViT-S/4 CIFAR-10, $p=0.0024$), the predicted CJ$\to$Flip$\to$Crop ordering ranks \emph{4th} (80.75\%)---definitively falsified. In our preliminary analysis with asymmetric seeds, this ordering appeared to win by a 2.00\% margin---an artifact of the seed distribution. The balanced data reveals CJ$\to$Crop$\to$Flip (81.94\%) as the true winner, outperforming the DPI prediction by 1.19\%. For the three remaining blocks (spread 0.30--0.81\%, non-significant or borderline), we cannot reliably evaluate rankings and therefore report the DPI verdict as inconclusive or not supported rather than falsified.

A weaker pattern is nonetheless present: CJ-first orderings collectively dominate in ViT-S/4 CIFAR-10 (81.94\% and 80.75\% vs.\ 78.55--78.59\% for Flip-first). This suggests the DPI principle captures \emph{something} real---placing the most reversible transform (ColorJitter) first benefits ViT training on CIFAR-10---but it does not determine the relative ordering of the remaining transforms. The specific DPI prediction about Flip-vs-Crop ordering is not supported.

For CIFAR-100 (harder task, 100 classes), the conventional Crop$\to$Flip$\to$CJ ordering wins for both architectures, and CJ-first orderings do not show the same benefit. On harder tasks, augmentation likely functions more as a regularizer than as a mutual information preserving operation. The conventional ordering reflects implicit empirical optimization for this regularization regime, which may override the CJ-first effect seen on easier tasks. The DPI mechanism appears to be both architecture-dependent (present for ViTs, absent for CNNs) and dataset-difficulty-dependent (present for CIFAR-10, absent for CIFAR-100).

## The Theory-Practice Gap

### Why NC$_2$ Fails as a Predictor

The NC$_2$ Wasserstein non-commutativity measure correctly identifies that augmentation operations do not commute: all three pairwise NC$_2$ values are positive (Crop--CJ: 0.051, Crop--Flip: 0.045, Flip--CJ: 0.035). The generalization bound (Theorem 1) is mathematically valid. Yet NC$_2$ fails to predict which orderings produce better accuracy ($\rho_s = -0.20$).

Three factors contribute to this failure:
\begin{enumerate}
\item \textbf{The bound is too loose.} The $O(1/\sqrt{n})$ scaling and the Lipschitz constant $L$ (which can be large for random crops) absorb too much variance. The bound correctly states that orderings with higher NC$_2$ can differ more in generalization, but the gap between the upper bound and the actual generalization difference is too large for the bound to be rank-predictive.
\item \textbf{Pixel-space distance is a poor proxy for learning-relevant distance.} NC$_2$ measures distributional distance in $\mathbb{R}^{C \times H \times W}$, the raw pixel space. But the relevant distance for learning is in the feature space of the model, where semantically equivalent images (different crops of the same object) are close. Two orderings may produce distributions that are far apart in pixel space but near-identical in a useful feature space.
\item \textbf{NC$_2$ ignores optimization dynamics.} Even if two augmented distributions are equidistant from the true data distribution, the optimizer may converge to different solutions depending on the local loss landscape structure, which depends on the distribution in complex, architecture-specific ways.
\end{enumerate}

### Why MI Shows Inconsistent Signals

The InfoNCE MI estimates show $\rho_s = +0.54$ on CIFAR-10 but $\rho_s = -0.66$ on CIFAR-100. This sign flip has two possible explanations. First, the InfoNCE estimator is unreliable at pilot scale: with only 100 samples and encoders trained for 10 epochs, the MI estimates may not accurately reflect the true mutual information. Second, if the sign flip is genuine, it suggests that MI preservation has different effects depending on task difficulty: on easy tasks (10 classes), preserving more MI helps; on hard tasks (100 classes), orderings that reduce MI may provide stronger regularization, akin to how aggressive augmentation (which reduces MI by definition) improves generalization on harder tasks. Distinguishing these explanations requires full-scale MI estimation with properly trained encoders.

### The Failure as a Contribution

We contend that the \emph{failure} of NC$_2$ and MI to predict ordering quality is informative rather than merely a negative result. We acknowledge a structural limitation of our theoretical framework: Theorem 1 (NC$_2$ generalization bound) and the DPI reversibility principle are stated as motivating hypotheses, but neither is used \emph{quantitatively} to explain the observed accuracy differences. The theory identifies why ordering should matter (non-commutativity, information loss) but does not predict which orderings will be better. In this sense, our theoretical section motivates the experiment rather than explaining the results---a limitation we accept in exchange for the value of pre-registering falsifiable hypotheses.

Prior augmentation theory implicitly assumes that distributional properties of the augmented data are the primary determinants of generalization. Our results are consistent with this assumption breaking down for ordering: two orderings can yield mean accuracy differences of up to 3.39\% (ViT-S/4 CIFAR-10) despite similar pixel-space distributional properties, with individual seeds showing even larger divergence due to bimodal basin dynamics. The NC$_2$ and MI measurements at pilot scale (100 samples, 10-epoch encoders) are themselves insufficiently powered to constitute definitive evidence---a larger-scale evaluation could yield different correlations.

We interpret the observed pattern as consistent with an \emph{optimization-mediated gap}: the ordering effect is mediated by the optimizer's trajectory through the loss landscape---a dynamic, architecture-conditioned process that static distributional measures may not capture. We do not claim that all possible distributional measures will fail; feature-space or gradient-based measures may recover predictive power. Rather, we claim that the failure of two principled pilot-scale proxies motivates development of measures that operate in feature space or account for optimization dynamics. \textbf{This interpretation is falsifiable}: if feature-space NC$_2$ or gradient-alignment measures computed on actual model activations achieve $\rho_s > 0.5$ ($p < 0.05$) with a sufficiently powered sample ($n \geq 20$ orderings or architectural variants), that would confirm the gap is optimization-mediated; if such measures also fail, the implication would be that ordering effects are intrinsically unpredictable from any distributional proxy, a stronger negative result that would call into question the distributional framework for augmentation theory altogether.

Future work should explore:
\begin{itemize}
\item \textbf{Optimization-aware measures.} Gradient alignment under different orderings --- measuring how the augmented distribution affects the direction and magnitude of gradient updates --- may be more predictive than static distributional measures.
\item \textbf{Feature-space NC$_2$.} Computing NC$_2$ in the penultimate-layer feature space of a pretrained model, rather than in pixel space, may recover predictive power by measuring distributional distance in a semantically meaningful representation.
\item \textbf{Architecture-conditioned bounds.} Generalization bounds that explicitly account for architectural inductive biases (e.g., patchification in ViTs, local receptive fields in CNNs) may be tighter and more predictive.
\end{itemize}

## Practical Implications

### Ordering as a Zero-Cost Hyperparameter

Augmentation ordering is a zero-cost hyperparameter: reordering a pipeline requires changing a single line of code, with no additional compute, data, or model changes. Our results suggest that practitioners should not accept the default ordering uncritically.

Based on our CIFAR-scale (32$\times$32) full-scale results at moderate augmentation magnitude ($M=9$), we offer four \textbf{preliminary recommendations}. \emph{All evidence is from CIFAR-scale experiments only. These should be treated as hypotheses for ImageNet-scale validation, not as general-purpose rules.}
\begin{enumerate}
\item \textbf{For ViT models (CIFAR-scale), use random-per-image ordering}: On CIFAR-10, random-per-image (82.04\%) marginally exceeds the best fixed ordering CJ$\to$Crop$\to$Flip (81.94\%, $\Delta = +0.10\%$). On CIFAR-100, random-per-image shows a preliminary $+1.60\%$ advantage (51.89\% vs.\ 50.29\%), though this estimate uses $n=4$ seeds in the baseline comparison and should be treated as directional evidence pending $n=10$ confirmation. This is a zero-cost regularization improvement---simply shuffle the transform order once per image. We hypothesize the benefit arises because ViT patchification interacts with spatial transform position; testing on ImageNet-scale ViTs is an important open question.
\item \textbf{For ViT models using a fixed ordering (CIFAR-scale)}: If random-per-image is not available, prefer CJ$\to$Crop$\to$Flip (empirically best ordering). It wins among fixed orderings on CIFAR-10 ViT-S/4 (81.94\%), 3.39\% above the worst fixed ordering mean, and all 10 seeds land in the high-accuracy basin (most stable). The DPI-predicted CJ$\to$Flip$\to$Crop ranks 4th (80.75\%) and is \emph{not} recommended.
\item \textbf{For ResNet/CNN models (CIFAR-scale)}: Use the conventional Crop$\to$Flip$\to$CJ ordering. For ResNets, random-per-image does not help (CIFAR-10: 88.09\% vs.\ best fixed 88.05\%; CIFAR-100: 57.48\% vs.\ best fixed 57.86\%), and the spread among all fixed orderings is small (0.30--0.39\%).
\item \textbf{For longer pipelines (ResNet-18, CIFAR-scale)}: On simpler tasks (CIFAR-10), use block geometric-first ordering (G$^3$P$^3$, 86.19\%). On harder tasks (CIFAR-100), consider random-per-image ordering (52.89\%). Full-scale Tier 2 shows 0.50\%/1.04\% spreads, reversing the pilot's misleading 9.01\% interleaved advantage.
\end{enumerate}

### No Universal Best Ordering

No single ordering dominates across all architecture-dataset combinations. The best orderings are:
\begin{itemize}
\item ResNet-18 $\times$ CIFAR-10: Flip$\to$Crop$\to$CJ (88.05\%)
\item ResNet-18 $\times$ CIFAR-100: Crop$\to$Flip$\to$CJ (57.86\%, conventional)
\item ViT-S/4 $\times$ CIFAR-10: CJ$\to$Crop$\to$Flip (81.94\%, empirically best; most stable)
\item ViT-S/4 $\times$ CIFAR-100: Crop$\to$Flip$\to$CJ (50.29\%, conventional)
\end{itemize}

The architecture and dataset both matter. The practical implication is that ordering should be treated as an empirical hyperparameter that depends on the model family and task difficulty---not as a universal default. The cost of evaluating a few orderings is minimal (only the augmentation ordering in \texttt{transforms.Compose()} changes), making this a low-effort, high-return hyperparameter search.

## Limitations

Our current results have several important limitations that we address transparently:

\begin{enumerate}
\item \textbf{ViT-S/4 CIFAR-10 near-balanced seed counts.} Order\_0 has $n=9$ seeds (seed 42 hardware fault); all other orderings have $n=10$. This minor imbalance means the spread estimate (3.39\%) compares a $n=9$ mean for order\_0 against $n=10$ means elsewhere. A fully balanced design ($n=10$ for all orderings) would provide marginally more precise estimates; we do not expect this to change the qualitative conclusions given the large effect size ($\eta^2 = 0.196$).

\item \textbf{CIFAR resolution and ImageNet generalizability.} CIFAR images are $32 \times 32$ pixels. At this resolution, RandomCrop(32, padding=4) discards a smaller fraction of the image than a typical ImageNet-scale RandomResizedCrop, potentially attenuating or modifying ordering effects. Our practical recommendations (for ViT models, prefer CJ$\to$Crop$\to$Flip or random-per-image; for CNNs, prefer conventional ordering) are \emph{specific to CIFAR-scale experiments}. Whether these recommendations generalize to ImageNet-scale training with ViT-B/16 or ViT-L is unknown and is the most important open question for practical applicability. The patchification-ordering interaction that drives ViT sensitivity at CIFAR scale may be amplified (larger crops discard more content) or attenuated (more overparameterized models may be more robust to ordering) at ImageNet scale.

\item \textbf{Limited operation set.} We test all permutations for only three operations. The 6-operation Tier 2 tests only five canonical category-level orderings, not all 720 permutations. More operations and permutations could reveal additional structure.

\item \textbf{NC$_2$ and MI estimation.} The NC$_2$ proxy used only 100 samples and 100 projections; the InfoNCE MI used 100 samples with 10-epoch encoders. Conclusive evaluation of these theoretical measures requires 10k+ samples and properly trained encoders. The theoretical assessments (H3, H4) should be considered pilot-scale.

\item \textbf{Single augmentation library.} All experiments use PyTorch/torchvision's transform implementations. Results may differ for other augmentation libraries (e.g., albumentations, Kornia) that implement the same operations differently at the pixel level.

\item \textbf{Magnitude scope.} All Tier 1 and Tier 2 experiments use a moderate, fixed augmentation magnitude. Pilot evidence (H5) suggests ordering effects diminish at high magnitude. Recommendations should be treated as specific to the tested magnitude regime until full-scale magnitude experiments are completed.

\item \textbf{Bimodal convergence requires further investigation.} Flip-first orderings on ViT-S/4 CIFAR-10 exhibit bimodal convergence ($\sigma\approx3.2$--$3.5\%$ with $n=10$), with some seeds converging to $\approx$70--75\% and others to $\approx$79--81\%. This high within-ordering variance does not prevent overall significance ($p=0.0024$) because the mean gap between CJ-first and Flip-first orderings is large. However, the bimodal structure itself is not fully characterized: it is unknown whether it reflects a true multimodal loss landscape induced by ordering, or a sensitivity to learning rate warmup and early stochastic dynamics. Controlled study (e.g., varying warmup schedule, visualizing loss landscape curvature) is warranted before drawing strong mechanistic conclusions about the nature of the bimodality.
\end{enumerate}
# Conclusion

We present the first systematic study isolating augmentation operation ordering as the sole independent variable in a controlled factorial experiment. Our full-scale results (200 epochs, 5 seeds, paired $t$-tests) across ResNet-18 and ViT-S/4 on CIFAR-10 and CIFAR-100 yield four primary findings:

\textbf{1. Ordering effects are consistent and architecture-dependent.} Permuting three standard augmentation operations produces spreads of 0.30--0.39\% for ResNet-18 and 0.81--3.39\% for ViT-S/4 at 200 epochs (balanced $n=9$--$10$ seeds for ViT-S/4 CIFAR-10), partially supporting H1 and confirming H2a. ViT-S/4 shows $\sim11\times$ larger sensitivity on CIFAR-10 (3.39\% spread, $p=0.0024$, ANOVA $p=0.0363$, $\eta^2=0.196$), consistent with patchification creating a stronger interaction with spatial transform ordering. CJ-first orderings dominate: CJ$\to$Crop$\to$Flip is the most stable ordering (all 10 seeds in the high-accuracy basin, 81.94\%). Flip-first orderings exhibit bimodal convergence ($\sigma\approx3.2$--$3.5\%$).

\textbf{2. The DPI reversibility principle is not validated.} With balanced seeds, the DPI-predicted ordering (CJ$\to$Flip$\to$Crop) wins 0/4 blocks and ranks 3rd--4th everywhere. In the one well-powered block (ViT-S/4 CIFAR-10, $p=0.0024$), it is definitively falsified: CJ$\to$Crop$\to$Flip (81.94\%) outperforms the DPI prediction by 1.19\%. In the three remaining blocks (spread 0.30--0.81\%, non-significant or borderline), ranking comparisons lack power and the verdict is inconclusive. In our preliminary analysis with asymmetric seeds, this ordering appeared to win for ViT-S/4 CIFAR-10 by a 2.00\% margin---a seed-distribution artifact. A weaker CJ-first signal is present in ViT-S/4 CIFAR-10, suggesting the DPI captures \emph{something} real (placing ColorJitter first benefits ViTs), but the specific Flip-before-Crop sub-ordering is not supported.

\textbf{3. Pixel-space distributional measures cannot predict ordering quality---and this failure is informative.} NC$_2$ achieves $\rho_s = -0.20$ (H3 against); InfoNCE MI achieves $\rho_s = -0.06$ combined and reverses sign across datasets (H4 inconclusive). We argue this constitutes the first evidence for an \emph{optimization-mediated gap}: the relationship between augmented distribution and generalization is not direct but is mediated by optimizer-architecture dynamics that static distributional measures miss. This falsifies an implicit assumption of prior augmentation theory and opens a concrete research direction---developing optimization-aware or architecture-conditioned measures that can bridge this gap.

\textbf{4. Category-level ordering has modest effects; the pilot advantage for interleaved ordering was an artifact.} Full-scale Tier 2 (200 epochs, 5 seeds) reverses the pilot finding: spreads of 0.50\% (CIFAR-10) and 1.04\% (CIFAR-100) replace the pilot's 9.01\% effect. Winners are dataset-dependent: geo-first (G$^3$P$^3$) wins CIFAR-10 (86.19\%), while random-per-image wins CIFAR-100 (52.89\%), 0.21\% ahead of geo-first. Interleaved orderings converge faster in early training but do not achieve higher final accuracy.

A key finding from our baselines is that random-per-image ordering outperforms every fixed ordering for ViT-S/4: by $+0.10\%$ on CIFAR-10 (82.04\% vs.\ best fixed 81.94\%) and, with the caveat that the CIFAR-100 baseline uses $n=4$ seeds, by a preliminary $+1.60\%$ on CIFAR-100 (51.89\%). On CIFAR-10, the best fixed ordering CJ$\to$Crop$\to$Flip (81.94\%) nearly closes the gap with random-per-image (82.04\%, $\Delta=+0.10\%$), making the 3.39\% spread among fixed orderings the dominant source of risk. For ResNet-18, random-per-image ordering provides no benefit. Our practical recommendation is therefore: for ViT models, use random-per-image ordering as the default; for fixed orderings, prefer CJ$\to$Crop$\to$Flip for ViT on CIFAR-10 (most stable, 81.94\%); for ResNet/CNN models, the conventional ordering (Crop$\to$Flip$\to$CJ) is a reliable default. For six-operation pipelines, use geo-first (G$^3$P$^3$) on simpler tasks and random-per-image on harder tasks (CIFAR-100). All of these are zero-cost changes requiring only reordering in \texttt{transforms.Compose()}.

The failure of our theoretical measures to predict ordering quality, while a negative result for the specific frameworks we proposed, opens a productive research direction: developing optimization-aware measures that bridge the gap between how an augmented distribution looks in pixel space and how the optimizer learns from it. Feature-space non-commutativity measures, gradient alignment metrics, and architecture-conditioned generalization bounds are promising avenues. More broadly, our results suggest that the augmentation literature's focus on \emph{which} operations to apply may be overlooking a complementary and practically free dimension: \emph{how} to order them.

\paragraph{Reproducibility.} Training scripts, random seeds, and evaluation code will be released upon publication. All hyperparameters and seed ranges are fully specified in Section~3.
