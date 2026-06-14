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
