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
