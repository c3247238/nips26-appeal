# 4 The Absorption Metric Does Not Transfer to JumpReLU SAEs

## 4.1 Universal Control Failure

Figure 1 presents the core diagnostic: shuffled-label controls exceed measured absorption rates in all five hierarchy domains on L12-16k ($L_0$=82). First-letter: 15.96% measured vs. 74.6% shuffled (4.7$\times$). City-continent: 6.49% vs. 45.2% (6.9$\times$). City-language: 6.56% vs. 18.0% (2.7$\times$). Animal-class: 1.43% vs. 39.3% (27.5$\times$). City-country: 0.0% vs. 10.3% ($\infty$). The random probe control, expected to produce $< 2\%$ absorption, yields 11.8% on first-letter and 9.2--34.3% across other domains.

![Grouped bar chart: measured absorption rate vs. shuffled control vs. random probe control for 5 hierarchy domains on L12-16k (L0=82). Shuffled controls exceed measured rates in all domains. Dashed line marks the expected random floor of 2%.](figures/control_failure.pdf)

Table 2 quantifies the control failure. The net signal (measured minus shuffled) is negative in every domain, ranging from $-0.103$ (city-country) to $-0.462$ (first-letter). A metric that assigns higher absorption scores to randomized labels than to true hierarchical labels cannot be measuring hierarchy-driven competitive exclusion.

| Domain | $N_{\text{parents}}$ | Probe F1 | Measured | Shuffled | Random | Ratio |
|--------|---------------------|----------|----------|----------|--------|-------|
| First-letter | 25 | 0.817 | 15.96% | 74.6% | 11.8% | **4.7x** |
| City $\to$ Continent | 6 | 0.795 | 6.49% | 45.2% | 12.9% | **6.9x** |
| City $\to$ Language | 18 | 0.745 | 6.56% | 18.0% | 20.8% | **2.7x** |
| Animal $\to$ Class | 6 | 0.696 | 1.43% | 39.3% | 34.3% | **27.5x** |
| City $\to$ Country | 28 | 0.602 | 0.0% | 10.3% | 19.0% | **$\infty$** |

**Table 2:** Control results by domain on L12-16k ($L_0$=82). The Ratio column shows shuffled/measured. All ratios exceed 1.0, confirming universal control failure. The metric produces *more* "absorption" with random labels than with true labels in every domain.

The untrained SAE control (C4) produces 0.0% absorption with mean probe F1 = 0.943, confirming that the measured signal depends on trained SAE structure rather than probe artifacts. The failure lies in the Chanin metric's threshold calibration for JumpReLU feature geometry, not in the probes or the SAE representations.

## 4.2 Confound Decomposition: Hedging Dominates

Figure 2 shows the composition of false negatives shifting across $L_0$ values. At $L_0$=22, where all 25 probes achieve F1=1.0, the 657 false negatives decompose as follows:

- **Hedging:** 648 of 657 (98.6%) --- tokens whose parent information is spread across many latents, none clearing the JumpReLU threshold at this sparsity level, but which resolve at higher $L_0$.
- **Hierarchy-driven absorption:** 9 of 657 (1.4%) --- tokens absorbed at all four tested $L_0$ values with residual norm below the $2\sigma$ threshold.
- **Reconstruction error:** 0 of 657 (0.0%) --- no tokens exceed the $2\sigma$ residual norm threshold at any $L_0$.

![Stacked bar chart: fraction of false negatives classified as hedging (blue) vs. hierarchy-driven (red) at each L0 value {22, 41, 82, 176}. At L0=22, 98.6% of false negatives are hedging. At L0=176, 90.0% are hierarchy-driven.](figures/hedging_decomposition.pdf)

The hedging fraction declines monotonically as $L_0$ increases (Spearman $\rho_s = -1.0$): $L_0$=22: 98.6%, $L_0$=41: 98.2%, $L_0$=82: 95.1%, $L_0$=176: 10.0%. The hierarchy-driven fraction follows the inverse trend ($\rho_s = +1.0$). Reconstruction error is 0.0% at all four $L_0$ values. As sparsity relaxes, hedging tokens are resolved --- their parent features activate once more latents are available --- while the 9 hierarchy-driven tokens persist regardless of sparsity pressure.

These 9 words --- "eight" (letter E), "lower" and "liked" (letter L), "offer" and "often" (letter O), plus 4 additional words identified by the cross-$L_0$ persistence criterion --- represent the floor of genuine competitive exclusion: 9 out of 1,196 words, or 0.75% of the vocabulary. At $L_0$=176, these 9 words account for 9 of the 10 remaining false negatives. This result directly falsifies H2 as proposed (>80% hierarchy-driven at $L_0$=22): the actual figure is 1.4%. The 96.9% hierarchy-driven figure from the pilot was a methodological artifact of single-$L_0$ classification.

## 4.3 First-Letter Replication with Improved Protocol

The improved first-letter experiment uses 1,204 single-token words (50+ per letter for 15 letters, 20+ for 22 letters), up from 25 per letter in the Chanin et al. protocol. On L12-16k at $L_0$=82, the aggregate absorption rate is 15.96% (95% CI: [8.4%, 17.5%]), within the published 15--35% range. Mean probe F1 = 0.817, with 10 of 25 letters passing the F1 $> 0.85$ quality gate. The replication succeeds in magnitude but the universal control failure (Section 4.1) means the interpretation as competitive exclusion is unsupported.

Cross-layer stability is strong: L10-16k produces 13.88%, L12-16k produces 15.96%, and L20-16k produces 13.55% at $L_0$=82 (CV = 8.6%). This consistency across layers 10, 12, and 20 indicates the metric's output depends primarily on the $L_0$ operating point and SAE architecture, not the specific layer.

Per-letter absorption rates range from 0.0% (J, K, Q, V, Y, Z) to 36.76% (P) at $L_0$=82. Letters with higher absorption tend to have lower probe F1 (Spearman $\rho_s = -0.67$, $p < 0.001$), suggesting probe quality partially drives the measured rates. Among the 10 letters passing the F1 $> 0.85$ gate, the maximum absorption rate is 14.75% (D), and 6 of 10 show rates below 10%.

## 4.4 Cross-Domain Rates Are Uninterpretable

Cross-domain absorption rates on L12-16k ($L_0$=82): city-continent 6.49% (95% CI: [0%, 11.5%]), city-language 6.56% (CI: [0%, 4.3%]), animal-class 1.43% (CI: [0%, 3.6%]), and city-country 0.0%. These are the first absorption measurements reported on geographic, linguistic, and taxonomic hierarchies.

The city-continent signal concentrates in a single class: Asia shows 21.62% absorption (probe F1 = 0.895), while other continents range from 0% to 3.3%. The city-language signal concentrates similarly. All rates fall below their shuffled controls (Table 2), so absolute values cannot be interpreted as genuine absorption. The five-domain hierarchy predictor analysis (H6) yields no significant correlations: co-occurrence ratio ($\rho_s = 0.40$, Bonferroni-corrected $p = 1.0$), fan-out ($\rho_s = 0.20$, $p = 1.0$), depth ($\rho_s = -0.58$, $p = 1.0$), parent frequency ($\rho_s = 0.40$, $p = 1.0$), all with $n = 5$ domains. H6 is underpowered and cannot be resolved with the available data.

SAE probes outperform dense probes on city-country (F1 = 0.773 vs. 0.617), confirming the SAE encodes geographic knowledge. The metric limitation is not a representation failure but a calibration mismatch between the Chanin metric's thresholds and JumpReLU activation dynamics.

# 5 The L0-Absorption Phase Transition

## 5.1 Monotonic Decline Across L0

As shown in Figure 3, absorption declines monotonically from 42.85% at $L_0$=22 to 0.84% at $L_0$=176 on L12-16k (Spearman $\rho_s = -1.0$, $p < 0.001$). The four data points: $L_0$=22: 42.85% (95% CI: [40.1%, 45.6%]), $L_0$=41: 37.49% (CI: [34.8%, 40.2%]), $L_0$=82: 14.39% (CI: [12.4%, 16.4%]), $L_0$=176: 0.84% (CI: [0.3%, 1.4%]). The steepest decline occurs between $L_0$=41 and $L_0$=82, where the rate drops by 23.1 percentage points --- a phase transition region in which relaxing sparsity eliminates roughly half the measured absorption per $L_0$ doubling.

![Absorption rate versus L0 operating point for three layers (L10, L12, L20), showing monotonic decline and cross-layer stability. Shaded bands indicate 95% bootstrap CI. The steepest decline occurs between L0=41 and L0=82.](figures/l0_phase_transition.pdf)

The $L_0$=82 rates from the multi-$L_0$ confound decomposition (14.39%, measured on 1,195 words with F1=1.0 probes) are slightly lower than the improved first-letter rates (15.96%, measured on 1,204 words with mean F1=0.817 probes). The difference reflects the probe quality gate: F1=1.0 probes at $L_0$=22 identify a slightly different false-negative set than the $L_0$=82 probes.

Cross-layer stability at $L_0$=82: L10-16k 13.88% (CI: [6.9%, 16.1%]), L12-16k 15.96% (CI: [8.4%, 17.5%]), L20-16k 13.55% (CI: [5.8%, 16.2%]). The coefficient of variation across these three layers is 8.6%, confirming the $L_0$-absorption profile is layer-invariant within measurement precision. All measurements at $L_0$=22 use probes with F1=1.0 for all 25 letters, eliminating probe quality as a confound for the low-$L_0$ versus high-$L_0$ comparison.

## 5.2 Width-L0 Interaction

Table 4 presents the absorption rate surface across the 34-configuration scaling grid.

| | 16k | 32k | 65k | 131k | 262k |
|------|------|------|------|------|------|
| Low $L_0$ ($\leq 20$) | 18.5% | 17.1% | 13.1--26.4% | 27.4% | 25.7% |
| Mid $L_0$ (20--60) | 10.0--23.9% | 26.0% | 13.3--20.4% | -- | -- |
| High $L_0$ ($\geq 60$) | 0.2--1.2% | -- | -- | -- | -- |

**Table 4:** First-letter absorption rates on L12 across the width $\times$ $L_0$ grid. Ranges shown where multiple $L_0$ values fall within a bin. $L_0$ is the dominant factor; wider dictionaries show a secondary positive effect at matched $L_0$.

OLS regression on the full 34-configuration surface yields a significant width $\times$ $L_0$ interaction ($F = 37.75$, $p = 1.24 \times 10^{-6}$, $\Delta R^2 = 0.252$). The GAM finds the interaction non-significant ($p = 1.0$, pseudo-$R^2$ = 0.85 with or without the interaction term), indicating the relationship is approximately linear in log space. The $L_0$ main effect is the dominant predictor ($\rho_s = -0.457$, $p = 0.007$), while the width effect is marginal ($\rho_s = 0.308$, $p = 0.076$, not significant after correction). The OLS coefficients in log space: $\beta_{\log L_0} = -8.41$, $\beta_{\log \text{width}} = -2.04$, $\beta_{\text{interaction}} = +0.52$, total $R^2 = 0.81$. The positive interaction coefficient indicates the $L_0$ effect is partially attenuated at larger widths.

## 5.3 JumpReLU vs. L1-ReLU Architecture Reference

GPT-2 Small with L1-ReLU SAEs shows uniformly high absorption: 64.29% at layer 8, 67.29% at layer 10, and 64.26% at layer 11. No $L_0$-dependent phase transition exists because L1-ReLU SAEs lack configurable $L_0$.

Hartigan's dip test confirms bimodal per-letter distributions for both architectures (JumpReLU at $L_0$=82: dip = 0.188, $p = 0.001$; all L1-ReLU layers: BC $> 0.555$). Both architectures produce bimodal patterns --- some letters are heavily absorbed while others are not --- but JumpReLU exhibits a dramatic $L_0$-dependent phase transition (42.9% $\to$ 0.8%) while L1-ReLU shows uniformly high absorption (61--67%) with no comparable transition. This falsifies H7 as stated (the prediction that only JumpReLU shows bimodality), but reveals a distinct architecture-dependent finding: the $L_0$ phase transition is specific to JumpReLU's hard threshold dynamics.

The comparison is cross-model (Gemma 2 2B, $d_{\text{model}} = 2304$ vs. GPT-2 Small, $d_{\text{model}} = 768$), so model capacity differences confound the architecture comparison. A controlled comparison would require training L1-ReLU SAEs on Gemma 2 2B.

## 5.4 The Nine Persistent Core Words

Nine words show non-zero absorption at all four $L_0$ values (22, 41, 82, 176). At $L_0$=176, these 9 words account for 9 of 10 total false negatives. Each has an identifiable child latent with high activation and magnitude ratio $\geq 1.1$. "Often" (letter O) has absorbing feature 1381 with cosine = 0.153 and activation = 35.1 at $L_0$=176; "eight" (letter E) shows absorbing feature 6740 with cosine = 0.130 and activation = 15.0 at $L_0$=176. "Lower" (letter L) persists with activation = 13.5 and magnitude ratio = 1.26 via feature 5463. These are the strongest candidates for genuine competitive exclusion, pending validation via activation patching (zeroing child features and measuring parent recovery).

# 6 Rate-Distortion Diagnostic: CMI Predicts Absorption Susceptibility

## 6.1 CMI Estimation and the Successive Refinement Connection

The successive refinement theorem (Equitz and Cover, 1991) states that $X$ is successively refinable if and only if $X \to f_{\text{child}} \to f_{\text{parent}}$ forms a Markov chain. When this condition holds, $I(X; f_{\text{parent}} \mid f_{\text{child}}) = 0$: the parent carries no unique information beyond the child, and absorbing it is information-theoretically lossless. When CMI is positive, absorption destroys unique information the parent encodes about the input.

We estimate CMI via $k$-nearest neighbors (Kraskov et al., 2004; $k_{\text{NN}} = 5$) in a $d' = 10$ decoder subspace for each of 25 first-letter features, using 1,092 word activations plus 2,599 corpus token activations (3,691 total samples). The absorbed/non-absorbed partition follows the $L_0$=82 improved first-letter results: absorbed letters ($\alpha \geq 0.10$; $n = 13$) versus non-absorbed ($\alpha < 0.05$; $n = 9$), with 3 intermediate letters excluded from the group comparison.

## 6.2 CMI-Absorption Correlation

Figure 4 shows that absorbed letters cluster at lower CMI values:

- Absorbed ($n = 13$): mean CMI = $0.649 \pm 0.187$
- Non-absorbed ($n = 9$): mean CMI = $0.861 \pm 0.258$
- Mann-Whitney $U = 28.0$, two-sided $p = 0.045$; one-sided $p = 0.023$
- Cohen's $d = -0.924$ (large effect)
- Bootstrap 95% CI on mean difference: $[-0.408, -0.013]$

![Scatter plot of CMI at d'=10 (x-axis) vs. absorption rate (y-axis) for 25 first-letter features. Red points: absorbed letters (rate >= 10%). Blue points: non-absorbed letters (rate < 5%). Absorbed letters cluster at lower CMI values, consistent with rate-distortion theory.](figures/cmi_vs_absorption.pdf)

The Spearman rank correlation across all 25 letters is $\rho_s = -0.383$ ($p = 0.059$). The negative direction is consistent with rate-distortion theory: letters whose parent features carry less unique information beyond the child (CMI $\approx 0.4$--$0.6$) are more susceptible to absorption. Letters J (CMI = 1.21), Z (CMI = 1.18), Y (CMI = 1.11), and U (CMI = 1.13) have both high CMI and zero or near-zero absorption. Letters T (CMI = 0.52), L (CMI = 0.59), and P (CMI = 0.71) have low CMI and high absorption (31--37%).

## 6.3 Phase Transition Scale Prediction

The rate-distortion threshold predicts a critical $L_0$ for each letter: $L_{0,\text{crit}} = \lambda / (\text{CMI} \cdot c)$, where $\lambda$ is the effective sparsity pressure inferred from the empirical half-maximum $L_0$ (22.4). Mean predicted $L_{0,\text{crit}} = 24.7$ (median = 25.4, range [13.7, 42.1]) across all 25 letters.

The rank-order prediction is directionally correct: letters with higher predicted $L_{0,\text{crit}}$ (lower CMI, cheaper to absorb) tend to have higher absorption ($\rho_s = +0.333$, $p = 0.103$). A median split confirms the direction: the high-$L_{0,\text{crit}}$ half has mean absorption of 15.2% versus 10.6% for the low-$L_{0,\text{crit}}$ half (ratio = 1.42, Mann-Whitney $p = 0.169$). The scale match ($L_{0,\text{crit}} = 24.7$ vs. empirical half-max $L_0 = 22.4$, 10.2% relative error) is partly by construction because $\lambda$ is estimated from the empirical half-max; the non-trivial prediction is the rank ordering.

## 6.4 Geometric Constant Degeneration

For unit-normalized Gemma Scope decoders ($\|d_j\| = 1.0$ for all features), the geometric constant $c = \sin^2(\text{angle}(w_P, w_C))$ has mean 0.960 and CV = 2.16%. All parent norms equal 1.0, confirming $\|w_P\|^2 = 1$ for all decoder columns. The constant provides negligible modulation beyond CMI: CMI/$c$ correlates with absorption at $\rho_s = -0.374$ versus $\rho_s = -0.383$ for raw CMI, a change of $-2.2\%$ (bootstrap 95% CI of the difference: [$-0.113$, $+0.085$]). The absorbed/non-absorbed group comparison on $c$ alone shows no significant separation (Mann-Whitney $p = 0.205$).

This degeneration is a useful theoretical simplification: for normalized SAEs, the rate-distortion threshold reduces to $L_{0,\text{crit}} \approx \lambda / \text{CMI}$, eliminating the need to estimate decoder geometry.

## 6.5 Dimension Sensitivity

The negative CMI-absorption correlation holds only at $d' = 10$:

| $d'$ | Spearman $\rho_s$ | $p$-value | Cohen's $d$ |
|------|-------------------|-----------|-------------|
| 10 | $-0.383$ | 0.059 | $-0.924$ |
| 20 | $+0.048$ | 0.818 | $+0.226$ |
| 30 | $+0.299$ | 0.147 | $+0.616$ |
| 50 | $+0.197$ | 0.345 | $+0.499$ |

At $d' \geq 20$, the correlation reverses sign. Bonferroni-corrected $p$ for $d' = 10$ is 0.236 (4 tests), failing the 0.05 significance threshold. This dimension instability is a genuine limitation. At low $d'$, the subspace likely captures the most absorption-relevant decoder directions; at higher $d'$, noise from orthogonal dimensions dilutes the signal. This motivates future work on dimension-agnostic MI estimation methods (MINE, KNIFE) or theoretical derivation of the optimal $d'$.

The CMI-absorption relationship at $d' = 10$ is the strongest information-theoretic signal in our data: a large effect size ($d = -0.924$), significant group separation ($p = 0.045$), and direction consistent with theory. The dimension sensitivity constrains the confidence with which we can claim this as a general diagnostic, but the theoretical prediction --- that low-CMI parent features are information-theoretically cheap to absorb --- receives directional support.

<!-- FIGURES
- Figure 1: gen_control_failure.py, control_failure.pdf --- Grouped bar chart showing measured vs. shuffled vs. random absorption rates across 5 domains
- Figure 2: gen_hedging_decomposition.py, hedging_decomposition.pdf --- Stacked bar chart showing hedging vs. hierarchy-driven fractions across L0 values
- Figure 3: gen_l0_phase_transition.py, l0_phase_transition.pdf --- Line plot showing absorption rate vs. L0 for L10, L12, L20
- Figure 4: gen_cmi_vs_absorption.py, cmi_vs_absorption.pdf --- Scatter plot of CMI vs. absorption rate for 25 letters
- Table 2: inline --- Control results by domain
- Table 4: inline --- L0 x width absorption grid for L12
-->
