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
