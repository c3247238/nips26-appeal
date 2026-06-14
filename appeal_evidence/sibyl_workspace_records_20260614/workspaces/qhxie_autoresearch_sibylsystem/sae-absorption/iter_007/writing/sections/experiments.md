# 4 Experiments

This section reports results for the three research questions: metric validity (Section 4.1--4.5), sparsity dynamics (Section 4.6--4.7), and the exploratory CMI analysis (Section 4.8--4.10). All experiments use Gemma 2 2B with Gemma Scope JumpReLU Sparse Autoencoders (SAEs) as described in Section 3, with GPT-2 Small L1-ReLU SAEs for cross-architecture context. Bootstrap confidence intervals (CIs) use 10,000 resamples with seed 42 throughout.

## 4.1 Universal Control Failure Across Five Domains

Table 2 reports feature absorption rates and control conditions across all five hierarchy domains at $L_0 = 82$ on the L12-16k SAE.

| Domain | Measured Absorption | Shuffled Control | Random Probe | Mean Probe F1 |
|--------|:------------------:|:----------------:|:------------:|:-------------:|
| First-letter | 13.4% | **59.6%** | 9.2% | 0.565 |
| City-continent | 6.5% | **45.2%** | 12.9% | 0.795 |
| City-language | 6.6% | **18.0%** | 20.8% | 0.745 |
| Animal-class | 1.4% | **39.3%** | 34.3% | 0.696 |
| City-country | 0.0% | **10.3%** | 19.0% | 0.602 |

**Table 2.** Cross-domain absorption rates at $L_0 = 82$ with four-control suite. Bold indicates the shuffled-label control exceeds the measured absorption rate in every domain. The untrained SAE control (C4) produces 0% absorption in all domains (not shown). The dense probe control (C3) achieves F1 = 0.929, confirming signal exists in the SAE activations.

The shuffled-label control exceeds measured absorption in all five domains, by ratios ranging from 1.6x (city-country) to 4.7x (first-letter). The random probe control (C1) and untrained SAE control (C4) behave as expected: C1 produces a near-chance absorption rate (9.2%), and C4 produces exactly 0%. The control inversion is therefore specific to the shuffled-label condition.

## 4.2 Structural Explanation: Candidate Explosion in High Dimensions

Figure 1 explains the mechanistic cause of the universal control failure.

![Candidate Explosion in High Dimensions](figures/control_failure.pdf)

**Figure 1.** Candidate count distributions for random unit vectors (gray, $n = 1{,}000$), true $k$-sparse probes (blue, $n = 25$), and shuffled probes (orange, $n = 125$) at cosine threshold $\tau_{\cos} \geq 0.025$. All three distributions overlap heavily, centering around 3,500--3,800 candidates out of 16,384 decoder columns.

At the standard cosine threshold $\tau_{\cos} \geq 0.025$ in $\mathbb{R}^{2304}$, a random unit vector identifies a mean of 3,766 decoder columns as candidates (23.0% of the 16,384 total). True probes identify 3,478 candidates (21.2%) and shuffled probes identify 3,502 (21.4%). These counts are statistically indistinguishable: the candidate identification step provides no discriminative power between real and random probe directions.

With $L_0 = 82$ active latents per token, the probability that at least one candidate latent fires is 1.0000 for all three probe types. The expected number of active candidate latents is 18.9 (random), 17.4 (true), and 17.5 (shuffled). Dead features (3,074 of 16,384, or 18.8%) inflate candidate counts further without contributing signal, since they never fire.

The metric therefore reduces to: absorption rate $\approx$ false-negative rate $\times P(\text{any candidate fires}) \approx$ false-negative rate, because $P(\text{any candidate fires}) \approx 1.0$. Shuffled probes produce worse classification accuracy (lower F1), yielding more false negatives, which the metric counts as higher "absorption."

## 4.3 Threshold Sensitivity

Table 3 reports the 5 $\times$ 4 threshold sensitivity grid varying cosine threshold $\tau_{\cos} \in \{0.01, 0.02, 0.025, 0.03, 0.05\}$ and magnitude gap $\tau_{\text{mag}} \in \{0.5, 1.0, 1.5, 2.0\}$.

| | $\tau_{\cos} = 0.01$ | $\tau_{\cos} = 0.02$ | $\tau_{\cos} = 0.025$ | $\tau_{\cos} = 0.03$ | $\tau_{\cos} = 0.05$ |
|:-:|:-:|:-:|:-:|:-:|:-:|
| **$\tau_{\text{mag}} = 0.5$** | 15.1% | 15.1% | 15.1% | 15.1% | 15.1% |
| **$\tau_{\text{mag}} = 1.0$** | 15.1% | 14.6% | **14.6%** | 14.6% | 14.1% |
| **$\tau_{\text{mag}} = 1.5$** | 15.1% | 14.2% | 13.5% | 13.5% | 13.2% |
| **$\tau_{\text{mag}} = 2.0$** | 15.1% | 13.2% | 12.2% | 12.2% | 11.8% |

**Table 3.** Absorption rate heatmap across the threshold parameter grid. Bold marks the default threshold ($\tau_{\cos} = 0.025$, $\tau_{\text{mag}} = 1.0$). CV = 0.077 across all 20 cells; range = 0.033 (11.8%--15.1%).

Absorption rate is stable across threshold choices (CV = 0.077). Letter-level rankings are highly preserved: mean pairwise Kendall $\tau = 0.977$ (all 190 pairs significant at $p < 0.05$). Magnitude gap has a larger effect than cosine threshold: tightening $\tau_{\text{mag}}$ from 0.5 to 2.0 reduces the rate by 0.030 on average, while tightening $\tau_{\cos}$ from 0.01 to 0.05 reduces it by only 0.010. Cosine threshold 0.01 is saturated (identical rates regardless of magnitude gap). The shuffled-label control (74.6% at default threshold) exceeds the measured rate at all 20 parameter combinations. Control failure is structural, not resolvable by threshold tuning.

## 4.4 Confound Decomposition: Permissive vs. Strict Hedging

At $L_0 = 22$, where all 25 probes achieve F1 = 1.0, we identify 656 false-negative (FN) tokens. Figure 2 shows the decomposition of these FN tokens into hedging and non-hedging categories.

![Hedging Decomposition](figures/hedging_decomposition.pdf)

**Figure 2.** Decomposition of 656 FN tokens at $L_0 = 22$. Permissive hedging (any token ceasing to be FN at a higher $L_0$) classifies 98.6% as hedging. Strict hedging (at least 1 of the $k = 5$ parent-associated latents fires at $L_0 = 176$) classifies only 6.2%.

Table 4 reports the tightened hedging classification.

| Classification | Count | Rate | 95% CI | Shuffled Control |
|:--------------|:-----:|:----:|:------:|:----------------:|
| Strict hedging | 41 | 6.2% | [4.4%, 8.2%] | 3.4% $\pm$ 0.8% |
| Non-hedging | 615 | 93.8% | [91.9%, 95.6%] | --- |
| Permissive hedging (reference) | 647 | 98.6% | --- | --- |

**Table 4.** Tightened hedging classification at $L_0 = 22$ ($n = 656$ FN tokens). Strict hedging requires at least 1 of 5 parent-associated latents to fire at $L_0 = 176$. The strict rate (6.2%) is significantly above the shuffled control mean (3.4%; $z = 3.51$, $p < 0.001$), indicating a real but small hedging signal.

The discrepancy between permissive (98.6%) and strict (6.2%) hedging is large. Of the 656 FN tokens, 93.8% show none of the 5 parent-associated latents firing even at $L_0 = 176$, where the SAE has 8x the capacity of $L_0 = 22$. The permissive rate counts any token that ceases to be FN at a higher $L_0$---a near-tautological criterion, since FN counts drop from 656 at $L_0 = 22$ to 9 at $L_0 = 176$.

Letter G is a pronounced outlier: 19 of 21 FN tokens for G show strict hedging (90.5%), accounting for 46% of all strict-hedging cases despite G contributing only 3.2% of FN tokens. Excluding G, the strict hedging rate for the remaining 20 letters drops to 3.5%. Ten letters (B, D, F, H, J, K, M, N, P, R, T, W) show zero strict hedging.

## 4.5 Activation Patching: Ruling Out Competitive Exclusion

Table 5 reports causal intervention results for the 8 persistent core words---tokens that are FN at all four tested $L_0$ values $\{22, 41, 82, 176\}$.

| Word | Letter | Child Feature | $\cos(d_c, v_p)$ | Parent Recovered? | Control Recovery |
|:----:|:------:|:-------------:|:-----------------:|:-----------------:|:----------------:|
| eight | E | 8174 | 0.216 | No | 0/10 |
| liked | L | 4678 | 0.074 | No | 0/10 |
| lower | L | 14449 | 0.203 | No | 0/10 |
| offer | O | 15092 | 0.223 | No | 0/10 |
| often | O | 3063 | 0.143 | No | 0/10 |
| other | O | 15322 | 0.203 | No | 0/10 |
| under | U | 2810 | 0.246 | No | 0/10 |
| until | U | --- | --- | No | 0/10 |

**Table 5.** Activation patching on 8 persistent core words at $L_0 = 82$. For each word, we zero the highest-cosine child feature and check whether any of the 5 parent-associated latents recover (activation $> 0$). "until" has no absorbing features at $L_0 = 82$ (it is FN but not absorbed by the metric definition). Recovery rate: 0/8 (0%; 95% CI [0%, 36.9%]). Control: zeroing 10 random non-child features produces 0/65 spurious recoveries (0%).

Three patching methods yield identical results: (1) zeroing the child in SAE space and re-decoding/re-encoding, (2) subtracting the child decoder direction from the raw activation and re-encoding, and (3) zeroing all absorbing features simultaneously. None produces parent recovery for any of the 8 words.

The 0/8 recovery result provides metric-independent causal evidence against competitive exclusion for these tokens. The model does not represent first-letter information for these words through the parent latent in a way that the child latent suppresses. Combined with the strict hedging rate of 6.2% (Section 4.4), the evidence indicates that false negatives at $L_0 = 22$ are overwhelmingly driven by information spreading (hedging) rather than active child-to-parent suppression.

## 4.6 Probe Quality Confound

Across the 25 tested letters at $L_0 = 82$, probe F1 and absorption rate are strongly anti-correlated: Spearman $\rho_s = -0.69$ ($p < 0.001$). Letters with low probe quality (C: F1 = 0.71, absorption = 29.0%; T: F1 = 0.72, absorption = 32.9%; S: F1 = 0.74, absorption = 31.4%) show high absorption, while letters with high probe quality (K: F1 = 0.96, absorption = 0%; J: F1 = 0.95, absorption = 0%; Q: F1 = 0.92, absorption = 0%) show zero absorption. The threshold sensitivity analysis independently confirms this confound: Spearman $\rho_s(F1, \text{absorption}) = -0.762$ ($p < 0.0001$).

This confound means that absorption rate at $L_0 = 82$ partially tracks probe classification errors rather than genuine feature dynamics. At $L_0 = 22$, where all probes achieve F1 = 1.0, this confound is eliminated.

## 4.7 L0 Phase Transition

Figure 3 shows the monotonic decline of absorption rate with $L_0$.

![L0 Phase Transition](figures/l0_phase_transition.pdf)

**Figure 3.** Feature absorption rate as a function of $L_0$ operating point on the L12-16k SAE. Error bars are bootstrap 95% CIs. The shaded region marks the $L_0 \approx 40$--80 transition zone. Cross-layer overlay: L10-16k (13.9%) and L20-16k (13.6%) at $L_0 = 82$ are consistent with L12-16k (14.4%).

Table 7 reports absorption rates at each $L_0$ value.

| $L_0$ | Absorption Rate | 95% CI | FN Count |
|:------:|:--------------:|:------:|:--------:|
| 22 | 42.85% | --- | 656 |
| 41 | 37.49% | --- | 488 |
| 82 | 14.39% | --- | 186 |
| 176 | 0.84% | --- | 9 |

**Table 7.** Absorption rate by $L_0$ on L12-16k. Spearman $\rho_s = -1.0$ (perfect monotonic decline). FN counts decrease from 656 to 9 across the tested range.

Absorption declines monotonically from 42.85% at $L_0 = 22$ to 0.84% at $L_0 = 176$. The sharpest drop occurs between $L_0 = 41$ and $L_0 = 82$ (37.49% to 14.39%), identifying the phase transition region at $L_0 \approx 40$--80. Below $L_0 \approx 40$, the SAE is capacity-starved and absorption exceeds 35%. Above $L_0 \approx 80$, sufficient capacity reduces absorption below 15%.

The transition is stable across layers: at $L_0 = 82$, layer 10 yields 13.88%, layer 12 yields 14.39%, and layer 20 yields 13.55% (CV = 8.6%). The phase transition is a property of the $L_0$ operating point, not the layer position.

### Cross-Architecture Context (Confounded)

GPT-2 Small with L1-ReLU SAEs ($d_{\text{SAE}} = 24{,}576$) shows absorption rates of 61.65%--67.29% at layers 8, 10, and 11. This comparison is confounded by model size (768 vs. 2,304 dimensions), SAE architecture (L1-ReLU vs. JumpReLU), $L_0$ differences, vocabulary overlap, and training data. It is reported as context, not as a controlled comparison.

## 4.8 Exploratory CMI Analysis at L0 = 82

At $L_0 = 82$ with subspace dimension $d' = 10$ (pre-registered), absorbed letters ($n = 18$) have lower mean conditional mutual information (CMI) than non-absorbed letters ($n = 7$): 0.649 (std 0.187) vs. 0.861 (std 0.258). The effect size is large: Cohen's $d = -0.924$; Mann-Whitney $p = 0.045$.

The rank correlation between CMI and absorption rate is Spearman $\rho_s = -0.383$ ($p = 0.059$, uncorrected; Bonferroni-corrected $p = 0.236$). After controlling for probe F1, the partial correlation weakens: partial $\rho_s = -0.328$ (permutation $p = 0.118$). Restricting to the 10 letters with probe F1 $> 0.85$ eliminates the signal entirely: $\rho_s = -0.113$ ($p = 0.757$).

Table 6 summarizes the progressive weakening.

| Analysis | $n$ | Spearman $\rho_s$ | 95% CI | $p$-value | Interpretation |
|:---------|:---:|:-----------------:|:------:|:---------:|:--------------|
| Raw $\rho_s$(CMI, absorption) | 25 | $-0.383$ | [$-0.684$, 0.031] | 0.059 | Marginal |
| Raw $\rho_s$(absorption, probe F1) | 25 | $-0.692$ | --- | 0.0001 | Strong confound |
| Partial $\rho_s$(CMI, absorption $\mid$ F1) | 25 | $-0.328$ | [$-0.664$, 0.246] | 0.118 | Weakened |
| Restricted $\rho_s$ (F1 $> 0.85$) | 10 | $-0.113$ | [$-0.789$, 0.808] | 0.757 | Eliminated |

**Table 6.** CMI-absorption correlation at $L_0 = 82$ ($d' = 10$). The signal weakens with each robustness check and vanishes when restricted to letters with high probe quality.

## 4.9 Leave-One-Out Sensitivity

No single letter removal changes $\rho_s$ by more than 0.1 (maximum $|\Delta\rho_s| = 0.088$, letter V). All 25 leave-one-out $\rho_s$ values remain negative, ranging from $-0.471$ to $-0.321$. The jackknife standard error is 0.186; bias-corrected $\rho_s = -0.397$.

Letter V has the largest influence: removing it shifts $\rho_s$ from $-0.383$ to $-0.471$ (strengthening). Removing both S and K---the two letters identified as potential theory contradictions---shifts $\rho_s$ to $-0.505$ ($p = 0.014$), which achieves significance. This post-hoc removal is reported for transparency but constitutes cherry-picking.

The marginal signal at $L_0 = 82$ is stable across individual letter removals but, as shown in Section 4.8, not robust to probe quality controls.

## 4.10 Replication at L0 = 22: Null Result

At $L_0 = 22$, all 25 probes achieve F1 = 1.0, eliminating the probe quality confound. The pre-registered analysis at $d' = 10$ yields Spearman $\rho_s = 0.044$ ($p = 0.835$; Bonferroni-corrected $p = 1.0$). The sign is positive---reversed from the $-0.383$ observed at $L_0 = 82$.

![CMI vs. Absorption at L0 = 82 and L0 = 22](figures/cmi_vs_absorption.pdf)

**Figure 4.** CMI vs. absorption rate per letter at $d' = 10$, comparing $L_0 = 82$ (left panel, points colored by probe F1) and $L_0 = 22$ (right panel, all F1 = 1.0). The negative trend at $L_0 = 82$ vanishes at $L_0 = 22$ where probe quality is uniform.

At higher subspace dimensions, the L0 = 22 correlations are positive and strengthening: $d' = 20$ yields $\rho_s = 0.248$ ($p = 0.232$); $d' = 30$ yields $\rho_s = 0.410$ ($p = 0.042$); $d' = 50$ yields $\rho_s = 0.483$ ($p = 0.014$). Both significant values reverse the theoretical prediction (higher CMI should mean less absorption, not more).

![CMI Dimension Sensitivity](figures/cmi_dimension_sensitivity.pdf)

**Figure 5.** Spearman $\rho_s$ between CMI and absorption rate as a function of subspace dimension $d'$. At $L_0 = 82$, the correlation is negative at $d' = 10$ and reverses sign at $d' \geq 20$. At $L_0 = 22$, the correlation is positive at all $d'$ values. The direction depends on both subspace choice and $L_0$, undermining any theoretical interpretation.

The CMI-absorption association at $L_0 = 82$ was driven by probe quality variation, not by rate-distortion theory. When probe quality is uniform (L0 = 22), the correlation vanishes at the pre-registered dimension and reverses at higher dimensions.

<!-- FIGURES
- Table 2: inline — Cross-domain absorption and control results
- Figure 1: control_failure.pdf — Candidate explosion histogram showing structural cause of control failure
- Table 3: inline — Threshold sensitivity heatmap (5x4 grid)
- Figure 2: hedging_decomposition.pdf — Permissive vs. strict hedging decomposition
- Table 4: inline — Tightened hedging classification results
- Table 5: inline — Activation patching on 8 persistent core words
- Table 7: inline — L0 phase transition absorption rates
- Figure 3: l0_phase_transition.pdf — Monotonic absorption decline with shaded transition zone
- Table 6: inline — CMI correlation summary with progressive weakening
- Figure 4: cmi_vs_absorption.pdf — Two-panel scatter: CMI vs. absorption at L0=82 and L0=22
- Figure 5: cmi_dimension_sensitivity.pdf — CMI sign reversal across subspace dimensions
-->
