# 1 Introduction

Sparse autoencoders (SAEs) decompose polysemantic neural network activations into overcomplete sparse bases of interpretable latents (Bricken et al., 2023; Templeton et al., 2024).
The promise is mechanistic interpretability at scale: each SAE latent should correspond to a single concept, enabling researchers to audit and steer model behavior.
But SAEs suffer from a failure mode that directly undermines this promise.
*Feature absorption* occurs when an SAE latent silently fails to fire because a more specific, co-occurring latent subsumes its role under the sparsity objective (Chanin et al., 2024).
On the first-letter spelling task---where a general "starts with S" latent is absorbed by a specific "September" latent---Chanin et al. measure 15--35% absorption rates across hundreds of Gemma Scope SAEs, and every tested architecture (L1, TopK, JumpReLU, Gated) exhibits the phenomenon.

Three weaknesses block the field's ability to act on these measurements.

**The confound problem.** The strongest published evidence that absorption degrades SAE quality is a correlation of $r = -0.595$ across 54 Gemma Scope SAEs between absorption rate and downstream quality metrics (Chanin et al., 2024; Karvonen et al., 2025).
But this correlation was computed without controlling for $L_0$ (the expected number of active latents per input).
All high-absorption SAEs in that dataset are 1M-width with low $L_0$ (16--58); all low-absorption SAEs are 16k or 65k width with high $L_0$ (137--297).
If absorption is merely a proxy for $L_0$---which independently affects quality through feature hedging (Chanin & Garriga-Alonso, 2025)---the entire absorption-quality narrative collapses into "use SAEs with higher $L_0$."
No prior work has applied formal confound control to this question.

**The single-task problem.** Every absorption measurement in the literature uses a single evaluation task: first-letter spelling.
At least five papers explicitly call out this limitation (Chanin et al., 2024; Karvonen et al., 2025; Korznikov et al., 2025; Bussmann et al., 2025; Li et al., 2025).
The first-letter task has an unusually clean, deterministic hierarchy---"September" is always a member of "starts with S"---that may systematically maximize the sparsity incentive for absorption.
Whether absorption occurs in fuzzier, real-world hierarchies (e.g., city-country, city-continent) is unknown.

**The scaling problem.** No study maps absorption jointly across SAE dictionary width and $L_0$.
Practitioners selecting SAE hyperparameters cannot determine whether there exists a region in (width, $L_0$) space where absorption is reliably low, or whether width and $L_0$ interact nonlinearly to produce absorption phase transitions.

This paper addresses all three weaknesses.

**Contribution 1: Confound resolution via epidemiological causal inference.**
We apply partial correlation analysis, Baron-Kenny mediation, and Rosenbaum sensitivity analysis to 48 Gemma Scope SAEs---methods standard in epidemiology and social science but never previously applied to SAE evaluation.
The result reverses the concern that absorption might be an $L_0$ epiphenomenon.
After controlling for $\log(L_0)$, the partial correlation between absorption and sparse probing F1 *strengthens* from $r = -0.664$ to $r = -0.746$ ($p = 1.2 \times 10^{-9}$), a classical suppression effect: $L_0$ was partially masking absorption's true impact on quality.
Two additional quality metrics---SCR ($r_{\text{partial}} = -0.570$, $p = 6.6 \times 10^{-5}$) and RAVEL TPP ($r_{\text{partial}} = -0.331$, $p = 0.022$)---retain significant associations.
Baron-Kenny mediation analysis finds that absorption fully mediates $L_0$'s effect on SCR (direct effect $c' = -0.003$, n.s.; Sobel $z = 3.62$, $p = 2.9 \times 10^{-4}$) and on TPP (proportion mediated = 0.54).
Rosenbaum sensitivity bounds show that the TPP result can withstand a hidden confounder with a 2.65:1 odds ratio (see Figure 1 for partial correlation comparisons and Figure 2 for the mediation path diagram).

**Contribution 2: Cross-domain absorption measurement on knowledge hierarchies.**
We provide the first measurement of absorption beyond first-letter spelling, using five probe types over 3,552 cities from the RAVEL dataset on GPT-2 Small.
The attempt exposes a critical methodological limitation: the standard dominance-based absorption metric (Chanin et al., 2024) produces 51--85% absorption rates across knowledge domains, but shuffled-hierarchy controls---where city-attribute mappings are randomized---show 100% absorption.
The metric does not discriminate real from randomized hierarchies because a small number of polysemantic "super-absorber" features (particularly Feature 8213) dominate at all false-negative positions regardless of the probe direction.
A cosine-calibrated variant that requires alignment between the dominant feature and the probe direction detects 0% absorption.
This discrepancy reveals that GPT-2 Small's SAE (24k features, 98% dead) does not encode knowledge-hierarchy directions as dedicated latents---a finding with direct implications for how future cross-domain absorption studies should be designed.

**Contribution 3: Absorption scaling surface with significant interaction structure.**
We construct the first empirical absorption phase surface across 420 SAEs from SAEBench (Gemma 2 2B), spanning dictionary widths from 2,304 to 1,048,576 and $L_0$ from 9.3 to 8,277.
A generalized additive model (GAM) with a tensor interaction term yields $R^2 = 0.693$, substantially outperforming both the additive model ($R^2 = 0.620$) and the linear baseline ($R^2 = 0.488$).
The interaction term is highly significant ($p = 3.1 \times 10^{-15}$): absorption cannot be predicted from width or $L_0$ independently.
Gradient analysis identifies a transition zone at $\log_2(L_0) \in [2.7, 3.8]$ (roughly $L_0 \approx 7$--$14$), where absorption rises sharply.
At $L_0 > 14$, absorption is low regardless of width; at $L_0 < 7$, absorption increases dramatically as width scales from 16k to 1M.
These results provide an actionable heuristic: practitioners seeking to minimize absorption should target $L_0 > 14$ and avoid scaling width at low $L_0$ (see Figure 5 for the contour surface and Figure 6 for the phase boundary).

Together, these contributions transform feature absorption from a narrowly validated observation on a single task into a rigorously characterized phenomenon with quantified causal status, documented metric limitations, and an actionable scaling map.
The epidemiological methods introduced here---mediation analysis, propensity matching, Rosenbaum sensitivity bounds---provide a methodological template for establishing causal claims from observational SAE data, a challenge that will recur as the field evaluates an expanding landscape of SAE architectures and quality metrics.

The remainder of this paper is organized as follows.
Section 2 reviews SAE architectures, prior absorption measurements, proposed mitigations, and the unresolved confound problem.
Section 3 details the epidemiological causal inference methods, the cross-domain probe training and absorption measurement protocol, and the GAM-based scaling surface construction.
Section 4 presents results for each contribution in order: confound resolution (Section 4.1), cross-domain absorption (Section 4.2), the scaling surface (Section 4.3), and taxonomy validation (Section 4.4).
Section 5 discusses the causal status of absorption, the mechanism behind the metric failure on knowledge hierarchies, and practical implications of the scaling surface.
Section 6 concludes with actionable recommendations for SAE practitioners and the most pressing directions for follow-up work.

<!-- FIGURES
- Figure 1: gen_fig1_partial_correlations_l0.py, fig1_partial_correlations_l0.pdf — Grouped bar chart comparing partial correlations with and without L0 control across four quality metrics
- Figure 2: gen_fig2_mediation_path.py, fig2_mediation_path.pdf — Mediation path diagram showing L0 -> Absorption -> Quality with standardized coefficients
- Figure 5: gen_fig5_absorption_contour.py, fig5_absorption_contour.pdf — 2D contour plot of absorption scaling surface in (log2(width), log2(L0)) space (referenced ahead)
- Figure 6: gen_fig6_gradient_surface.py, fig6_gradient_surface.pdf — Gradient magnitude surface with phase boundary overlay (referenced ahead)
-->
